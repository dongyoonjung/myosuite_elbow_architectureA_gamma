"""Evaluate a trained MyoElbowReflexAGamma policy.

Reports reward, final position error, and γ usage statistics.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from _paths_agamma import setup as _setup_path  # noqa: E402

_setup_path()

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import myosuite_elbow_arch_a_gamma  # noqa: F401, E402
from myosuite_elbow_arch_a_gamma import ENV_ID  # noqa: E402


def make_env_fn(env_kwargs: dict):
    def _init():
        import gymnasium as gym
        env = gym.make(ENV_ID, **env_kwargs)
        return Monitor(env)
    return _init


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--vecnorm", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "auto"])
    ap.add_argument("--k-th", type=float, default=0.10)
    ap.add_argument("--k-v", type=float, default=1.0)
    ap.add_argument("--tau-gamma", type=float, default=0.100)
    ap.add_argument("--gamma-init", type=float, default=0.5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    env_kwargs = dict(
        k_th=args.k_th, k_v=args.k_v,
        tau_gamma=args.tau_gamma, gamma_init=args.gamma_init,
    )

    vec = DummyVecEnv([make_env_fn(env_kwargs)])
    vec = VecNormalize.load(args.vecnorm, vec)
    vec.training = False
    vec.norm_reward = False
    vec.seed(args.seed)

    model = PPO.load(args.model, device=args.device)

    rewards, final_errs_deg, gamma_s_means, gamma_d_means = [], [], [], []

    for ep in range(args.episodes):
        obs = vec.reset()
        theta_end = vec.get_attr("theta_end")[0]
        T_episode = vec.get_attr("T_episode")[0]

        ep_r = 0.0
        steps = 0
        gs_log, gd_log = [], []
        final_theta = None
        done = False
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            # SB3 VecEnv legacy 4-tuple: dones = terminated | truncated.
            obs, rew, dones, infos = vec.step(action)
            ep_r += float(rew[0])
            steps += 1
            theta = infos[0].get("theta")
            if theta is not None:
                final_theta = theta
            gs_log.append(infos[0].get("gamma_s_actual_mean", float("nan")))
            gd_log.append(infos[0].get("gamma_d_actual_mean", float("nan")))
            done = bool(dones[0])

        final_err_deg = (
            float(np.rad2deg(abs(final_theta - theta_end)))
            if final_theta is not None
            else float("nan")
        )
        rewards.append(ep_r)
        final_errs_deg.append(final_err_deg)
        gamma_s_means.append(float(np.nanmean(gs_log)) if gs_log else float("nan"))
        gamma_d_means.append(float(np.nanmean(gd_log)) if gd_log else float("nan"))

        print(
            f"ep{ep:02d} r={ep_r:7.2f} steps={steps:3d} "
            f"theta_end={np.rad2deg(theta_end):6.1f}deg "
            f"err={final_err_deg:5.1f}deg "
            f"<γs>={gamma_s_means[-1]:.3f} <γd>={gamma_d_means[-1]:.3f} "
            f"T={T_episode:.2f}s"
        )

    print(
        "\n=== summary ==="
        f"\nmean_r        = {float(np.mean(rewards)):.2f}"
        f"\nmean_final_err= {float(np.mean(final_errs_deg)):.2f}deg"
        f"\nmean_γs       = {float(np.nanmean(gamma_s_means)):.3f}"
        f"\nmean_γd       = {float(np.nanmean(gamma_d_means)):.3f}"
    )


if __name__ == "__main__":
    sys.exit(main())
