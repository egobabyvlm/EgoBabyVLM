"""Rank-0-only Weights & Biases initialization and logging helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, cast

from core.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


class WandbLogger:
    """Thin wrapper around ``wandb`` that no-ops on non-main ranks.

    Args:
        project: W&B project name. ``None`` disables logging entirely.
        run_name: Optional run name. Defaults to W&B's auto-generated name.
        config: Resolved training config to log alongside metrics.
        mode: ``"online"`` (default), ``"offline"``, or ``"disabled"``.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str = "online",
    ) -> None:
        self._enabled = project is not None and is_main_process()
        self._wandb: Any | None = None
        self.run: Any | None = None
        if not self._enabled:
            return

        import wandb

        self._wandb = wandb

        env_mode = os.environ.get("WANDB_MODE")
        resolved_mode = cast("Literal['online', 'offline', 'disabled', 'shared']", env_mode or mode)
        self.run = wandb.init(project=project, name=run_name, config=config, mode=resolved_mode)
        logger.info("Initialized W&B run: project=%s, name=%s, mode=%s", project, run_name, resolved_mode)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a flat dict of metrics. No-op on non-main ranks / disabled."""
        if not self._enabled or self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self._enabled or self._wandb is None:
            return
        self._wandb.finish()
