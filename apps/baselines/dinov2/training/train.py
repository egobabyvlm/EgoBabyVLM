"""DINOv2 SSL trainer entrypoint.

Re-exports the upstream ``dinov2.train.train.main`` and
``get_args_parser`` so the OSS-facing module path
(``apps.baselines.dinov2.training.train``) matches the rest of the
``apps/baselines/`` layout. Run with::

    python -m apps.baselines.dinov2.training.train --config-file <path>

To submit to SLURM via Submitit, use::

    python -m apps.baselines.dinov2.training.submit --config-file <path>
"""

from __future__ import annotations

import sys

# Importing the upstream package registers it as ``dinov2`` in sys.modules so
# its internal ``from dinov2.X import ...`` calls resolve here.
import apps.baselines.dinov2.third_party.dinov2  # noqa: F401
from apps.baselines.dinov2.third_party.dinov2.train.train import (
    do_train,
    get_args_parser,
    main,
)

__all__ = ["do_train", "get_args_parser", "main"]


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    sys.exit(main(args) or 0)
