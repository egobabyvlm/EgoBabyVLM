# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Vendored DINOv2 ``ExtendedVisionDataset`` base.

Originally wrapped ``iopath.PathManager`` for remote-storage reach. The OSS
port replaces that with a no-op ``LocalPathManager`` covering local-FS paths
only — drop in your own ``PathManager``-shaped object via
``self._path_manager`` if you need remote storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torchvision.datasets import VisionDataset

from .decoders import ImageDataDecoder, TargetDecoder


class LocalPathManager:
    """Minimal stand-in for ``iopath.PathManager`` covering local-FS paths only."""

    @staticmethod
    def get_local_path(path: str) -> str:
        """Return ``path`` unchanged; nothing to fetch for a local path."""
        return path

    @staticmethod
    def open(path: str, mode: str = "r"):
        return Path(path).open(mode)

    @staticmethod
    def glob(root: str, pattern: str):
        return Path(root).glob(pattern)


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._path_manager: LocalPathManager = LocalPathManager()

    @property
    def path_manager(self) -> LocalPathManager:
        return self._path_manager

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            msg = f"can not read image for sample {index}"
            raise RuntimeError(msg) from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
