"""PPO training for MyoElbowReflexAGamma-v0 (Architecture A+γ).

Reflects VESSL workflow lessons:
    --device  cpu | cuda | auto   (don't hardcode cpu)
    --vec-env dummy | subproc     (subproc unlocks parallel n_envs > 1)

Environment hyperparameters specific to A+γ are exposed at the top level so
they can be sweep-overridden from the command line.

Warm-start support (Stage 0–2 of A→A+γ transfer, see SKILL.md §9):
    --init-from <ppo_zip>          load policy weights from a seed model
    --init-vecnorm <vecnorm.pkl>   load VecNormalize stats alongside it
    --freeze-backbone-steps N      keep backbone frozen for the first N steps
                                   (Stage 1: only γs/γd heads + log_std train)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from _paths_agamma import setup as _setup_path  # noqa: E402

_setup_path()

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

import myosuite_elbow_arch_a_gamma  # noqa: F401, E402  (env registration)
from myosuite_elbow_arch_a_gamma import ENV_ID  # noqa: E402

# Frozen-by-default keys = "backbone" = everything except the two action-head
# tensors and the log_std vector. Matches what transfer_from_A.py initializes.
TRAINABLE_HEAD_KEYS = ("action_net.weight", "action_net.bias", "log_std")


class ProgressCallback(BaseCallback):
    """Push training progress into each env so the curriculum sees it.

    Updates only at rollout boundaries — env._sample_task() reads
    self.progress at reset(), so per-step IPC chatter (16 RPCs/step on
    SubprocVecEnv) is wasted.
    """

    def __init__(self, total_budget: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_budget = int(total_budget)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        progress = min(1.0, self.num_timesteps / max(self.total_budget, 1))
        self.training_env.env_method("set_progress", progress)


class GammaStatsCallback(BaseCallback):
    """Log γ usage and post-reflex muscle input stats to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        for k in (
            "gamma_s_actual_mean",
            "gamma_d_actual_mean",
            "gamma_s_actual_max",
            "gamma_d_actual_max",
            "muscle_input_norm",
            "muscle_input_mean",
        ):
            vals = [info.get(k) for info in infos if info.get(k) is not None]
            if vals:
                self.logger.record(f"agamma/{k}", float(np.mean(vals)))
        return True


class RewardLogCallback(BaseCallback):
    """Per-rollout episode reward stats to a tsv file (mirrors train_A.py).

    Append-mode: re-running with the same tag does not wipe history.
    """

    def __init__(self, log_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        self._fh = open(self.log_path, "a")
        if write_header:
            self._fh.write("timesteps\twall_s\tmean_ep_rew\tmean_ep_len\n")
            self._fh.flush()
        self._t0 = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ep_info = getattr(self.model, "ep_info_buffer", None)
        if not ep_info:
            return
        rews = [e["r"] for e in ep_info]
        lens = [e["l"] for e in ep_info]
        mean_r = float(np.mean(rews)) if rews else float("nan")
        mean_l = float(np.mean(lens)) if lens else float("nan")
        wall = time.time() - self._t0
        self._fh.write(f"{self.num_timesteps}\t{wall:.1f}\t{mean_r:.3f}\t{mean_l:.1f}\n")
        self._fh.flush()

    def _on_training_end(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class BackboneFreezeCallback(BaseCallback):
    """Stage-1 helper: freeze everything except γ heads + log_std for the first
    ``freeze_steps`` timesteps, then unfreeze all parameters.

    SB3's PPO optimizer is created with all policy parameters; setting
    ``requires_grad=False`` blocks gradient flow but keeps the optimizer state
    valid. Re-enabling produces fresh momentum buffers on next update.
    """

    def __init__(self, freeze_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.freeze_steps = int(freeze_steps)
        self._unfrozen = False

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        for name, param in self.model.policy.named_parameters():
            if name in TRAINABLE_HEAD_KEYS:
                param.requires_grad = True
            else:
                param.requires_grad = bool(requires_grad)

    def _on_training_start(self) -> None:
        if self.freeze_steps <= 0:
            return
        self._set_backbone_grad(False)
        if self.verbose:
            n_train = sum(int(p.requires_grad) for p in self.model.policy.parameters())
            n_total = sum(1 for _ in self.model.policy.parameters())
            print(f"[freeze] backbone frozen ({n_train}/{n_total} params trainable)")

    def _on_step(self) -> bool:
        if self.freeze_steps <= 0 or self._unfrozen:
            return True
        if self.num_timesteps >= self.freeze_steps:
            self._set_backbone_grad(True)
            self._unfrozen = True
            if self.verbose:
                print(f"[freeze] unfrozen at step {self.num_timesteps}")
        return True


def make_env_fn(env_kwargs: dict, seed: int = 0):
    def _init():
        import gymnasium as gym
        env = gym.make(ENV_ID, **env_kwargs)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # training
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="AGamma_short")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["cpu", "cuda", "auto"],
                    help="GPU helps little (collection is CPU-bound MuJoCo); "
                         "use cpu when n_envs is small.")
    ap.add_argument("--vec-env", type=str, default="subproc",
                    choices=["dummy", "subproc"])
    # γ-specific env kwargs
    ap.add_argument("--k-th", type=float, default=0.10)
    ap.add_argument("--k-v", type=float, default=1.0)
    ap.add_argument("--tau-gamma", type=float, default=0.100)
    ap.add_argument("--gamma-init", type=float, default=0.5)
    # PPO
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    # warm-start (A → A+γ transfer)
    ap.add_argument("--init-from", type=str, default=None,
                    help="Load policy weights from this PPO zip (produced by "
                         "transfer_from_A.py or a prior A+γ checkpoint).")
    ap.add_argument("--init-vecnorm", type=str, default=None,
                    help="Load VecNormalize stats from this pickle. Recommended "
                         "alongside --init-from.")
    ap.add_argument("--freeze-backbone-steps", type=int, default=0,
                    help="Stage 1 helper: freeze backbone, train only γ heads + "
                         "log_std for the first N steps. Default 0 = off.")
    return ap.parse_args()


def _build_vec_env(args, env_kwargs, seed_offset=0):
    VecCls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    return VecCls(
        [make_env_fn(env_kwargs, seed=args.seed + seed_offset + i)
         for i in range(args.n_envs)]
    )


def main() -> None:
    args = parse_args()
    budget = args.budget if args.budget is not None else args.timesteps

    # Make sure the log directory exists before any tee/redirect or callback writes.
    logs_dir = os.path.join(ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    env_kwargs = dict(
        total_training_budget=budget,
        k_th=args.k_th,
        k_v=args.k_v,
        tau_gamma=args.tau_gamma,
        gamma_init=args.gamma_init,
    )

    vec_env = _build_vec_env(args, env_kwargs)
    if args.init_vecnorm:
        vec_env = VecNormalize.load(args.init_vecnorm, vec_env)
        # SB3 2.7.1: VecNormalize.load preserves saved spaces; rebind to the
        # underlying env so a stale 6-d action_space from an A pickle can't
        # poison policy construction.
        vec_env.action_space = vec_env.venv.action_space
        vec_env.observation_space = vec_env.venv.observation_space
        vec_env.training = True
        vec_env.norm_reward = True
        print(f"[warm] loaded VecNormalize stats from {args.init_vecnorm}")
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            norm_obs_keys=["sensory"],
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

    eval_vec = DummyVecEnv([make_env_fn(env_kwargs, seed=args.seed + 1000)])
    eval_vec = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,
        norm_obs_keys=["sensory"],
        clip_obs=10.0,
        training=False,
    )
    # Sync obs running stats from training vec_env so eval normalizes consistently.
    eval_vec.obs_rms = vec_env.obs_rms

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
    )

    if args.init_from:
        # Load the full PPO state and rebind the env. Keeps all hyperparameters
        # unless explicitly overridden via custom_objects.
        custom_objects = dict(
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            clip_range=0.2,
        )
        model = PPO.load(
            args.init_from,
            env=vec_env,
            device=args.device,
            tensorboard_log=logs_dir,
            seed=args.seed,
            custom_objects=custom_objects,
        )
        print(f"[warm] loaded PPO from {args.init_from} (device={args.device})")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=args.device,
            tensorboard_log=logs_dir,
            seed=args.seed,
        )

    model_dir = os.path.join(ROOT, "models", args.tag)
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    best_dir = os.path.join(model_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    callbacks = [
        ProgressCallback(budget),
        GammaStatsCallback(),
        CheckpointCallback(
            save_freq=max(1, 25_000 // args.n_envs),
            save_path=ckpt_dir,
            name_prefix=f"ppo_{args.tag}",
        ),
        EvalCallback(
            eval_vec,
            best_model_save_path=best_dir,
            eval_freq=max(1, 20_000 // args.n_envs),
            n_eval_episodes=5,
            deterministic=True,
        ),
        RewardLogCallback(os.path.join(logs_dir, f"reward_{args.tag}.tsv")),
    ]
    if args.freeze_backbone_steps > 0:
        callbacks.append(BackboneFreezeCallback(args.freeze_backbone_steps, verbose=1))
    callback_list = CallbackList(callbacks)

    print(
        f"[train_AGamma] tag={args.tag} timesteps={args.timesteps} "
        f"n_envs={args.n_envs} vec={args.vec_env} device={args.device}"
    )
    print(
        f"[train_AGamma] γ-config k_th={args.k_th} k_v={args.k_v} "
        f"τ_γ={args.tau_gamma} γ_init={args.gamma_init}"
    )
    if args.init_from:
        print(f"[train_AGamma] warm-start from {args.init_from}")
    if args.freeze_backbone_steps > 0:
        print(f"[train_AGamma] backbone frozen for first "
              f"{args.freeze_backbone_steps} steps")

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback_list,
        tb_log_name=args.tag,
        reset_num_timesteps=not bool(args.init_from),
    )
    elapsed = time.time() - t0

    model.save(os.path.join(model_dir, f"ppo_{args.tag}_final.zip"))
    vec_env.save(os.path.join(model_dir, f"vec_normalize_{args.tag}.pkl"))

    print(f"[done] trained {args.timesteps} steps in {elapsed:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
