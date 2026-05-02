"""Architecture A+gamma: A inheritance with γ-motor action extension."""

from ._path import ensure_arch_a_on_path

ensure_arch_a_on_path()

from gymnasium.envs.registration import register, registry

from .env import MyoElbowReflexEnvAGamma
from .reflex import compute_reflex_gamma

ENV_ID = "MyoElbowReflexAGamma-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="myosuite_elbow_arch_a_gamma.env:MyoElbowReflexEnvAGamma",
        max_episode_steps=None,
    )

__all__ = ["MyoElbowReflexEnvAGamma", "compute_reflex_gamma", "ENV_ID"]
