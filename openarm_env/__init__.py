from gymnasium.envs.registration import register, registry

from .pick_place.envs import OpenArmBimanualPickPlaceEnv
from .reach.envs import OpenArmBimanualReachEnv, OpenArmReachEnv

ENV_ID = "OpenArmReach-v0"
BIMANUAL_ENV_ID = "OpenArmBimanualReach-v0"
PICK_PLACE_ENV_ID = "OpenArmBimanualPickPlace-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="openarm_env.reach.envs:OpenArmReachEnv",
        max_episode_steps=200,
    )

if BIMANUAL_ENV_ID not in registry:
    register(
        id=BIMANUAL_ENV_ID,
        entry_point="openarm_env.reach.envs:OpenArmBimanualReachEnv",
        max_episode_steps=200,
    )

if PICK_PLACE_ENV_ID not in registry:
    register(
        id=PICK_PLACE_ENV_ID,
        entry_point="openarm_env.pick_place.envs:OpenArmBimanualPickPlaceEnv",
        max_episode_steps=450,
    )

__all__ = [
    "BIMANUAL_ENV_ID",
    "ENV_ID",
    "PICK_PLACE_ENV_ID",
    "OpenArmBimanualPickPlaceEnv",
    "OpenArmBimanualReachEnv",
    "OpenArmReachEnv",
]
