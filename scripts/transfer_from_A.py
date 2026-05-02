"""Transfer A's PPO policy weights into a fresh A+γ PPO model.

Strategy (per A_gamma SKILL.md §9):
    - Copy the entire MLP backbone (mlp_extractor.policy_net / value_net) and
      both feature extractors verbatim wherever shapes match.
    - Copy A's action-net weights (out=6) into the first 6 rows of A+γ's
      action_net (out=18).
    - Initialize the remaining 12 rows (γs/γd heads) small with bias 0
      → γ_target ≈ 0.5 at start.
    - Copy A's log_std into the first 6 dims; set γ log_std smaller.
    - Copy A's VecNormalize obs running stats but **reset the return running
      stats** — A+γ's effort is computed from post-reflex muscle input and
      is on a different magnitude.

A and A+γ share the observation encoder for "sensory" because the observation
space is identical. SB3's MultiInputPolicy builds the same encoder when
feature dims match.

Note: SB3's internals (action_net naming, log_std attribute) are version-
sensitive. Tested against stable-baselines3 == 2.7.1.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from _paths_agamma import setup as _setup_path  # noqa: E402

_setup_path()

import numpy as np  # noqa: E402, F401
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import myosuite_elbow_arch_a_gamma  # noqa: F401, E402
from myosuite_elbow_arch_a_gamma import ENV_ID as ENV_ID_GAMMA  # noqa: E402

# Keys that are *expected* to mismatch between A (6-d action) and A+γ (18-d).
EXPECTED_RESHAPE_KEYS = {"action_net.weight", "action_net.bias", "log_std"}


def make_env_fn(env_kwargs):
    def _init():
        import gymnasium as gym
        env = gym.make(ENV_ID_GAMMA, **env_kwargs)
        return Monitor(env)
    return _init


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", required=True,
                    help="A model zip (e.g. ppo_A_final.zip)")
    ap.add_argument("--source-vecnorm", required=True,
                    help="A VecNormalize pickle (matched with the model)")
    ap.add_argument("--out-model", required=True,
                    help="Where to save the seeded A+γ model")
    ap.add_argument("--out-vecnorm", required=True,
                    help="Where to save the (copied + ret_rms-reset) VecNormalize stats")
    ap.add_argument("--gamma-log-std", type=float, default=-1.5,
                    help="Initial log_std for γs/γd dims")
    ap.add_argument("--gamma-head-init-std", type=float, default=0.01,
                    help="Std for γs/γd weight init")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "auto"],
                    help="Device for the temporary PPO instances. CPU is fine.")
    ap.add_argument("--keep-ret-rms", action="store_true",
                    help="Skip resetting the saved VecNormalize ret_rms. Off by default — "
                         "A+γ effort uses post-reflex muscle input, so reward magnitudes differ.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    env_kwargs = dict(total_training_budget=1)
    vec = DummyVecEnv([make_env_fn(env_kwargs)])

    # Copy the existing VecNormalize stats over verbatim — obs space is identical.
    vec = VecNormalize.load(args.source_vecnorm, vec)
    # SB3 2.7.1's VecNormalize.set_venv preserves the saved pickle's
    # action_space (A's 6-d) instead of inheriting from the new venv. Force the
    # new spaces from the underlying A+γ env so PPO builds the 18-d action_net.
    vec.action_space = vec.venv.action_space
    vec.observation_space = vec.venv.observation_space
    vec.training = True

    if not args.keep_ret_rms:
        # A+γ effort uses post-reflex muscle input → different reward magnitude.
        # Letting A's ret_rms persist would scale the first batches' advantages
        # incorrectly. Fresh ret_rms re-adapts within ~100 steps.
        vec.ret_rms = RunningMeanStd(shape=())
        if hasattr(vec, "returns"):
            vec.returns = np.zeros(vec.num_envs, dtype=np.float64)

    target = PPO(
        "MultiInputPolicy",
        vec,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                           activation_fn=nn.ReLU),
        device=args.device,
        seed=args.seed,
    )
    source = PPO.load(args.source_model, device=args.device)

    src_state = source.policy.state_dict()
    tgt_state = target.policy.state_dict()

    # 1) Copy every key whose shape matches exactly (backbone, value head, log_std prefix).
    copied, deferred, skipped = [], [], []
    for k, v in src_state.items():
        if k not in tgt_state:
            skipped.append((k, "missing in target"))
            continue
        if tgt_state[k].shape == v.shape:
            tgt_state[k] = v.clone()
            copied.append(k)
        else:
            deferred.append((k, tuple(v.shape), tuple(tgt_state[k].shape)))

    # 2) Action net: pad first 6 rows with A weights, init the rest small.
    for key in ("action_net.weight", "action_net.bias"):
        if key not in src_state or key not in tgt_state:
            raise RuntimeError(f"Missing required key {key} in source or target policy.")
        src = src_state[key]
        tgt = tgt_state[key]
        if src.shape[0] != 6:
            raise RuntimeError(f"Expected source {key} dim-0 = 6, got {src.shape}")
        if tgt.shape[0] != 18:
            raise RuntimeError(f"Expected target {key} dim-0 = 18, got {tgt.shape}")
        new = torch.empty_like(tgt)
        new[0:6] = src
        if key.endswith("weight"):
            new[6:18].normal_(0.0, args.gamma_head_init_std)
        else:
            new[6:18].fill_(0.0)  # → γ_target = 0.5 (after [-1,1]→[0,1])
        tgt_state[key] = new

    # 3) log_std: prefix from A, suffix smaller.
    if "log_std" not in src_state or "log_std" not in tgt_state:
        raise RuntimeError("log_std missing in source or target policy.")
    src_ls = src_state["log_std"]
    tgt_ls = tgt_state["log_std"]
    if src_ls.numel() != 6 or tgt_ls.numel() != 18:
        raise RuntimeError(
            f"Unexpected log_std shapes: src={src_ls.shape}, tgt={tgt_ls.shape}"
        )
    new_ls = torch.empty_like(tgt_ls)
    new_ls[0:6] = src_ls
    new_ls[6:18] = float(args.gamma_log_std)
    tgt_state["log_std"] = new_ls

    target.policy.load_state_dict(tgt_state)

    # 4) Validate that the only deferred keys are the ones we explicitly handled.
    unexpected_deferred = [d for d in deferred if d[0] not in EXPECTED_RESHAPE_KEYS]
    if unexpected_deferred or skipped:
        msg = ["Source ↔ target policy structure mismatch:"]
        if unexpected_deferred:
            msg.append("  unexpected shape mismatches (check A's policy_kwargs match "
                       "net_arch=pi=[256,256], vf=[256,256], ReLU):")
            for k, s, t in unexpected_deferred:
                msg.append(f"    - {k}: src={s} tgt={t}")
        if skipped:
            msg.append("  source keys missing in target:")
            for k, why in skipped:
                msg.append(f"    - {k}: {why}")
        raise RuntimeError("\n".join(msg))

    print(f"[transfer] copied {len(copied)} tensors verbatim")
    print(f"[transfer] reshaped action_net + log_std for γ extension")
    if not args.keep_ret_rms:
        print(f"[transfer] reset VecNormalize ret_rms (A+γ reward magnitudes differ)")

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_vecnorm) or ".", exist_ok=True)
    target.save(args.out_model)
    vec.save(args.out_vecnorm)
    print(f"[transfer] wrote: {args.out_model}\n[transfer] wrote: {args.out_vecnorm}")


if __name__ == "__main__":
    sys.exit(main())
