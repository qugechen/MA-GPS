"""Trainer package."""

from MAGPS.trainer.base import BaseTrainer
from MAGPS.trainer.offline import (
    OfflineTrainer,
    offline_trainer,
    offline_trainer_iter,
)
from MAGPS.trainer.offpolicy import (
    OffpolicyTrainer,
    offpolicy_trainer,
    offpolicy_trainer_iter,
)
from MAGPS.trainer.onpolicy import (
    OnpolicyTrainer,
    onpolicy_trainer,
    onpolicy_trainer_iter,
)
from MAGPS.trainer.utils import gather_info, test_episode

__all__ = [
    "BaseTrainer",
    "offpolicy_trainer",
    "offpolicy_trainer_iter",
    "OffpolicyTrainer",
    "onpolicy_trainer",
    "onpolicy_trainer_iter",
    "OnpolicyTrainer",
    "offline_trainer",
    "offline_trainer_iter",
    "OfflineTrainer",
    "test_episode",
    "gather_info",
]
