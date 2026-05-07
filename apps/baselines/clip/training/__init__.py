"""Trainer scaffolding."""

from apps.baselines.clip.training.checkpoint import load_checkpoint, save_checkpoint
from apps.baselines.clip.training.interleave import InterleaveScheduler
from apps.baselines.clip.training.loop import ContrastiveTrainer
from apps.baselines.clip.training.optim import build_adamw, build_cosine_scheduler
from apps.baselines.clip.training.wandb import WandbLogger

__all__ = [
    "ContrastiveTrainer",
    "InterleaveScheduler",
    "WandbLogger",
    "build_adamw",
    "build_cosine_scheduler",
    "load_checkpoint",
    "save_checkpoint",
]
