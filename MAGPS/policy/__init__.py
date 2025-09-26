"""Policy package."""
# isort:skip_file

from MAGPS.policy.base import BasePolicy
from MAGPS.policy.random import RandomPolicy

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DDPGPolicy",
    "SACPolicy",
    "reach_avoid_game_DDPGPolicy_annealing", # arXiv:2112.12288, implemented using DDPG
    "reach_avoid_game_DDPGPolicy", # Our new method, implemented using DDPG
    "reach_avoid_game_SACPolicy_annealing", # arXiv:2112.12288, implemented using SAC
    "reach_avoid_game_SACPolicy", # Our new method, implemented using SAC
]

