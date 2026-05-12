from gymnasium.envs.registration import register, registry

from .envs import OpenArmBimanualReachEnv, OpenArmReachEnv

ENV_ID = "OpenArmReach-v0"
BIMANUAL_ENV_ID = "OpenArmBimanualReach-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="openarm_env.envs:OpenArmReachEnv",
        max_episode_steps=200,
    )

if BIMANUAL_ENV_ID not in registry:
    register(
        id=BIMANUAL_ENV_ID,
        entry_point="openarm_env.envs:OpenArmBimanualReachEnv",
        max_episode_steps=200,
    )

__all__ = ["BIMANUAL_ENV_ID", "ENV_ID", "OpenArmBimanualReachEnv", "OpenArmReachEnv"]
