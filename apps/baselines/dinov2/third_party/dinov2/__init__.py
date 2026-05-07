# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""``dinov2`` package (third-party, modified copy of facebookresearch/dinov2).

Internal imports inside this tree use the bare upstream form
``from dinov2.X import ...``. To make those imports resolve, we register
this package as ``dinov2`` in ``sys.modules`` on first import.
"""

from __future__ import annotations

import sys

__version__ = "0.0.1"

# Register this tree under the bare ``dinov2`` name so internal imports
# (``from dinov2.X import ...``) resolve to this package instead of any
# external pip-installed ``dinov2`` (which would clash on namespacing).
sys.modules.setdefault("dinov2", sys.modules[__name__])
