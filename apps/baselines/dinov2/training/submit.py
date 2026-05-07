"""Submitit-based SLURM submission entrypoint for the DINOv2 SSL trainer.

Wraps the upstream ``dinov2.run.train.train.main`` so the OSS-facing
module path matches the rest of ``apps/baselines/``::

    python -m apps.baselines.dinov2.training.submit \\
        --config-file <path> \\
        --partition <slurm_partition> \\
        --output-dir <path> \\
        --ngpus <int>
"""

from __future__ import annotations

import sys

# Importing the upstream package registers it as ``dinov2`` in sys.modules.
import apps.baselines.dinov2.third_party.dinov2  # noqa: F401
from apps.baselines.dinov2.third_party.dinov2.run.train.train import main

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
