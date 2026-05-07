"""Weights & Biases helpers — thin wrappers that no-op gracefully when wandb is unavailable.

These take plain arguments rather than a particular config schema so any
trainer can use them without coupling to a top-level ``Config`` type.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.utils.distributed import is_main_process

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

try:
    import wandb

    _is_wandb_available = True
except ImportError:
    logger.warning("wandb is not installed; wandb_log / init_wandb will no-op")
    _is_wandb_available = False


def wandb_run_name(output_dir: str) -> str:
    """Build a stable wandb run name from SLURM env + the experiment's output dir."""
    slurm_str = (
        f"{os.getenv('SLURM_ARRAY_JOB_ID') or os.getenv('SLURM_JOB_ID', '0')}_{os.getenv('SLURM_ARRAY_TASK_ID', '0')}"
    )
    run_str = str(output_dir).replace("/", "__")
    return f"{slurm_str}_{run_str}"


def init_wandb(  # noqa: PLR0913
    *,
    project: str | None,
    entity: str | None,
    output_dir: str,
    log_dir: str | None = None,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
    resume: bool = False,
    log_code: bool = False,
    model: torch.nn.Module | None = None,
) -> str | None:
    """Initialize wandb on the main process only; returns the resolved run_id (or None)."""
    if not _is_wandb_available or not is_main_process():
        return None

    resolved_run_id = run_id or os.getenv("SLURM_JOB_ID") or str(int(time.time()))
    wandb.init(
        name=wandb_run_name(output_dir),
        project=project,
        entity=entity,
        dir=log_dir,
        config=config,
        id=resolved_run_id,
        resume=resume and run_id is not None,
        save_code=False,
    )
    if log_code:
        assert wandb.run is not None
        wandb.run.log_code(
            str(Path.cwd().parent),
            include_fn=lambda path: path.endswith((".py", ".yaml")),
        )
    if model is not None:
        wandb.watch(model, log="all", log_freq=1000)
    return resolved_run_id


def wandb_log(data: dict[str, Any], *, disable_format: bool = False) -> None:
    """Log a dict to wandb on the main process; no-op when wandb is unavailable."""
    if not _is_wandb_available or not is_main_process():
        return
    formatted = (
        data if disable_format else {k.replace("val_", "val/").replace("train_", "train/"): v for k, v in data.items()}
    )
    wandb.log(formatted)
