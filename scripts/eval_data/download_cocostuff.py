#!/usr/bin/env python
"""Download COCO-Stuff (images + stuff/thing segmentation maps) into the cache.

Layout written matches what :class:`evaluation.data.semantic_segmentation.COCOStuffDataset`
expects:

.. code-block::

    {cache_dir}/
        train2017/                          # RGB images
        val2017/
        annotations/
            stuff_train2017.json
            stuff_val2017.json
            stuffthings_val2017.json        # merged stuff+things variant
            ...

Total download size is ~20 GB.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)

DOWNLOADS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "stuffthingmaps_trainval2017.zip": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
    "stuff_annotations_trainval2017.zip": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_annotations_trainval2017.zip",
}


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with progress logging."""
    logger.info("Downloading %s -> %s", url, dest)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Unzipping %s -> %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def main() -> None:
    parser: argparse.ArgumentParser = cache_argparser("cocostuff")
    args = parser.parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("COCO-Stuff already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "COCO-Stuff", env_var="COCOSTUFF_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = args.cache_dir / "_downloads"
    downloads_dir.mkdir(exist_ok=True)

    logger.warning("COCO-Stuff is ~20 GB; this will take a while.")

    for fname, url in DOWNLOADS.items():
        zip_path = downloads_dir / fname
        if not zip_path.exists() or args.force:
            try:
                _download(url, zip_path)
            except Exception as e:  # noqa: BLE001
                logger.error("Download failed for %s: %s", url, e)
                sys.exit(1)

    # Images: extracted into top-level cache dir.
    _unzip(downloads_dir / "train2017.zip", args.cache_dir)
    _unzip(downloads_dir / "val2017.zip", args.cache_dir)

    # Annotations: extracted under cache_dir/annotations.
    _unzip(downloads_dir / "stuff_annotations_trainval2017.zip", args.cache_dir / "annotations")
    _unzip(downloads_dir / "stuffthingmaps_trainval2017.zip", args.cache_dir / "annotations")

    logger.info("Cleaning up zip downloads in %s", downloads_dir)
    shutil.rmtree(downloads_dir)

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "COCO-Stuff", env_var="COCOSTUFF_ROOT")


if __name__ == "__main__":
    main()
