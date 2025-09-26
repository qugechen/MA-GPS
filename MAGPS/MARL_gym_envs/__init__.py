"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)



# Customized environments begin: 
register(
    id="Three_Unicycle_Game-v0",
    entry_point="MAGPS.MARL_gym_envs.Three_Unicycle:Three_Unicycle_Game_Env0",
    max_episode_steps=300,
    reward_threshold=1e9,
)



register(
    id="basketball-v0",
    entry_point="MAGPS.MARL_gym_envs.Six_basketball_players:basketball_Env0",
    max_episode_steps=300,
    reward_threshold=1e9,
)

register(
    id="non_linear-v17",
    entry_point="MAGPS.MARL_gym_envs.Drone_racing_final:Double_Drones_RA_non_linear_Game_Env17",
    max_episode_steps=150,
    reward_threshold=1e9,)