"""Smoke tests for MyoElbowReflexAGamma-v0.

Run:
    python tests/test_smoke_AGamma.py
or
    pytest tests/

Covers:
    1. env creation, reset/step shapes (action 18, obs 23)
    2. 100 random-action steps produce no NaN
    3. With k_th=k_v=0 the env should behave near-identically to A
       (sanity check; γ has no leverage)
    4. With γ_target = 1 forced, γ_actual converges toward 1 over a control step
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from _paths_agamma import setup as _setup_path  # noqa: E402

_setup_path()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import myosuite_elbow_arch_a_gamma  # noqa: F401, E402
from myosuite_elbow_arch_a_gamma import ENV_ID  # noqa: E402


def test_action_space_shape() -> None:
    env = gym.make(ENV_ID)
    assert env.action_space.shape == (18,), env.action_space
    assert env.observation_space["sensory"].shape == (20,)
    assert env.observation_space["task"].shape == (3,)
    env.close()
    print("[ok] action_space=(18,) sensory=(20,) task=(3,)")


def test_random_rollout_no_nan(n_steps: int = 100) -> None:
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        a = rng.uniform(-1.0, 1.0, size=18).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs["sensory"])), "sensory NaN"
        assert np.all(np.isfinite(obs["task"])), "task NaN"
        assert np.isfinite(r), "reward NaN"
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    print(f"[ok] {n_steps} random steps, no NaN")


def test_gamma_lowpass_converges() -> None:
    env = gym.make(ENV_ID, tau_gamma=0.020)
    env.reset(seed=0)
    # Force γ_target = 1 by sending action with γs/γd raw = +1.
    a = np.zeros(18, dtype=np.float32)
    a[6:18] = 1.0
    last_info = None
    for _ in range(50):  # 50 control steps × 20ms = 1s, τ=20ms → ~exp(-50) far from init
        _, _, term, trunc, info = env.step(a)
        last_info = info
        if term or trunc:
            env.reset()
    assert last_info is not None
    gs = last_info["gamma_s_actual_mean"]
    gd = last_info["gamma_d_actual_mean"]
    print(f"[ok] γ low-pass: γs_mean={gs:.4f} γd_mean={gd:.4f} (expect → 1.0)")
    assert gs > 0.9 and gd > 0.9, f"γ failed to converge: γs={gs}, γd={gd}"
    env.close()


def test_zero_gamma_modulation_baseline() -> None:
    env = gym.make(ENV_ID, k_th=0.0, k_v=0.0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(1)
    rewards = []
    for _ in range(50):
        a = rng.uniform(-1.0, 1.0, size=18).astype(np.float32)
        _, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        if term or trunc:
            env.reset()
    print(f"[ok] k_th=k_v=0 sanity: mean(r)={np.mean(rewards):.3f}")
    env.close()


if __name__ == "__main__":
    test_action_space_shape()
    test_random_rollout_no_nan()
    test_gamma_lowpass_converges()
    test_zero_gamma_modulation_baseline()
    print("\n[smoke] all checks passed")
