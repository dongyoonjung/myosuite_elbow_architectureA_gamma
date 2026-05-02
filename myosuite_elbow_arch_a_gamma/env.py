"""MyoElbow Architecture A+γ environment.

Extends Architecture A by adding γs and γd targets to the action space.
Policy outputs 18 dims = (α, γs_target, γd_target) × 6. γ_actual follows γ_target
through a 1st-order low-pass with τ_γ. Reflex receives (L, V, F, γs, γd) and
modulates stretch threshold + Ia gain + reciprocal inhibition. Effort is
computed from post-reflex muscle input rather than α alone.

Observation space is identical to Architecture A — γ feeds back implicitly
through its effect on muscle input and resulting sensory signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np

from ._path import ensure_arch_a_on_path

ensure_arch_a_on_path()

from gymnasium import spaces  # noqa: E402

from myosuite_elbow_arch_a.env import MyoElbowReflexEnvA  # noqa: E402

from .reflex import compute_reflex_gamma  # noqa: E402


class MyoElbowReflexEnvAGamma(MyoElbowReflexEnvA):
    """A + γ-motor action extension.

    Action layout (18 dims, all in [-1, 1]):
        [0:6]   α_raw       → α      ∈ [0, 1]
        [6:12]  γs_tgt_raw  → γs_tgt ∈ [0, 1]
        [12:18] γd_tgt_raw  → γd_tgt ∈ [0, 1]
    """

    def __init__(
        self,
        # γ kwargs (new)
        k_th: float = 0.10,
        k_v: float = 1.0,
        tau_gamma: float = 0.100,
        gamma_init: float = 0.5,
        # all other kwargs forwarded to A
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.k_th = float(k_th)
        self.k_v = float(k_v)
        self.tau_gamma = float(tau_gamma)
        self.gamma_init = float(np.clip(gamma_init, 0.0, 1.0))

        # Override action space to 18 dims (A's setup wrote 6).
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )

        self.gamma_s_target = np.full(self.N_MUSCLES, self.gamma_init, dtype=np.float64)
        self.gamma_d_target = np.full(self.N_MUSCLES, self.gamma_init, dtype=np.float64)
        self.gamma_s_actual = self.gamma_s_target.copy()
        self.gamma_d_actual = self.gamma_d_target.copy()

        self._lp_alpha = self.sim_dt / max(self.tau_gamma, 1e-6)

        self.muscle_input_current = np.zeros(self.N_MUSCLES, dtype=np.float64)

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        obs, info = super().reset(seed=seed, options=options)
        self.gamma_s_target.fill(self.gamma_init)
        self.gamma_d_target.fill(self.gamma_init)
        self.gamma_s_actual.fill(self.gamma_init)
        self.gamma_d_actual.fill(self.gamma_init)
        self.muscle_input_current.fill(0.0)
        info = self._extend_info(info)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(18)
        action = np.clip(action, -1.0, 1.0)

        alpha_cmd = 0.5 * (action[0:6] + 1.0)
        self.gamma_s_target = 0.5 * (action[6:12] + 1.0)
        self.gamma_d_target = 0.5 * (action[12:18] + 1.0)
        self.u_desc_current = alpha_cmd

        sub_rewards = []
        terminated = False
        reason: Optional[str] = None

        for _ in range(self.sim_steps_per_control):
            # 1) γ low-pass dynamics (2 ms)
            self._update_gamma_actual()

            # 2) γ-modulated reflex
            u_reflex = self._compute_reflex_with_gamma()

            # 3) post-reflex muscle input
            muscle_input = np.clip(self.u_desc_current + u_reflex, 0.0, 1.0)
            self.muscle_input_current = muscle_input

            # 4) MuJoCo step
            self.data.ctrl[:] = muscle_input
            mujoco.mj_step(self.model, self.data)
            self.t_elapsed += self.sim_dt

            # 5) sensory write
            sens = self._compute_raw_sensory()
            if not np.all(np.isfinite(sens)):
                sens = np.nan_to_num(sens, nan=0.0, posinf=1e3, neginf=-1e3)
            self.sens_buffer.write(sens)

            # 6) sub-reward (effort uses post-reflex muscle input)
            sub_r = self._compute_sub_reward()

            # 7) termination check
            done, reason_i, penalty = self._check_termination()
            if done:
                sub_r += penalty
                sub_rewards.append(sub_r)
                terminated = True
                reason = reason_i
                break
            sub_rewards.append(sub_r)

        self.total_timesteps += self.sim_steps_per_control
        reward = float(np.sum(sub_rewards))
        obs = self._get_obs()
        truncated = False
        info = self._get_info()
        info["termination_reason"] = reason
        info = self._extend_info(info)
        self._last_term_reason = reason
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- internals

    def _update_gamma_actual(self) -> None:
        a = self._lp_alpha
        self.gamma_s_actual += a * (self.gamma_s_target - self.gamma_s_actual)
        self.gamma_d_actual += a * (self.gamma_d_target - self.gamma_d_actual)

    def _compute_reflex_with_gamma(self) -> np.ndarray:
        delayed = self.sens_buffer.read(self.spinal_delay_steps)
        L, V, F = delayed[0:6], delayed[6:12], delayed[12:18]
        return compute_reflex_gamma(
            L=L, V=V, F=F,
            L_opt=self.L_opt, F_max=self.F_max,
            flexors=self.flexors, extensors=self.extensors,
            gamma_s=self.gamma_s_actual,
            gamma_d=self.gamma_d_actual,
            k_th=self.k_th, k_v=self.k_v,
            **self._reflex_cfg,
        )

    def _compute_sub_reward(self) -> float:
        """Mirror A's reward but compute effort from post-reflex muscle input.

        post-reflex effort captures metabolic-equivalent activation including
        spinal contributions, so that runaway γ → reflex chatter is naturally
        penalized.
        """
        theta = float(self.data.qpos[self.joint_qpos_adr])
        theta_d = float(self.data.qvel[self.joint_dof_adr])
        theta_dd = float(self.data.qacc[self.joint_dof_adr])
        t = self.t_elapsed
        T = self.T_episode

        pos_err_sq = (theta - self.theta_end) ** 2
        sigma = max(self.sigma_current, 1e-3)
        kernel = float(np.exp(-0.5 * ((t - T) / sigma) ** 2))
        r_pos = -self._reward_cfg["alpha"] * pos_err_sq * kernel
        r_pos_cont = -self._reward_cfg["alpha_cont"] * pos_err_sq

        vel_gate = 1.0 / (1.0 + np.exp(-10.0 * (t - T)))
        r_vel = -self._reward_cfg["beta"] * (theta_d ** 2) * vel_gate

        r_eff = -self._reward_cfg["gamma"] * float(
            np.sum(self.muscle_input_current ** 2)
        )
        r_jerk = -self._reward_cfg["delta"] * (theta_dd ** 2)

        return float(r_pos + r_pos_cont + r_vel + r_eff + r_jerk)

    def _extend_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info["gamma_s_actual_mean"] = float(self.gamma_s_actual.mean())
        info["gamma_d_actual_mean"] = float(self.gamma_d_actual.mean())
        info["gamma_s_actual_max"] = float(self.gamma_s_actual.max())
        info["gamma_d_actual_max"] = float(self.gamma_d_actual.max())
        info["muscle_input_norm"] = float(np.linalg.norm(self.muscle_input_current))
        info["muscle_input_mean"] = float(self.muscle_input_current.mean())
        return info
