"""Semantic segmentation datasets."""

import json
import logging
import random
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

from evaluation.data.base import SemanticSegmentationDataset, SemanticSegmentationSample

logger = logging.getLogger(__name__)


class COCOStuffDataset(SemanticSegmentationDataset):
    """COCO-Stuff semantic segmentation dataset.

    Provides COCO-image semantic segmentation across 91 stuff categories. The
    "other" catch-all (category 183) is excluded and mapped to ``ignore_index``.
    A coarse mode consolidates the 91 classes into 15 super-categories
    following Ji et al. 2019. When ``instances_json`` is provided, thing
    (object instance) annotations are merged in for the full 171-class setup
    (80 things + 91 stuff).
    """

    is_video_dataset = False

    def __init__(
        self,
        dataset_root: str,
        mode: str = "val",
        image_size: int = 224,
        *,
        image_dir: str | None = None,
        annotation_file: str | None = None,
        instances_json: str | None = None,
        coarse_labels: bool = False,
        subset_fraction: float | None = None,
        subset_seed: int = 42,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_root: Root of the COCO-Stuff dataset (annotations + pixelmap masks).
            mode: ``"train"`` or ``"val"``.
            image_size: Image side length after resize.
            image_dir: Directory with the actual COCO RGB images. Defaults to ``{dataset_root}/{mode}2017``.
            annotation_file: Custom annotation JSON path. Overrides default lookup.
            instances_json: Optional COCO instances JSON for merging in thing categories at runtime.
                Pre-merged annotation files are faster.
            coarse_labels: Consolidate stuff classes into 15 super-categories. Not allowed with ``instances_json``.
            subset_fraction: Sample this fraction of images, in ``(0, 1]``. ``None`` uses all.
            subset_seed: Random seed for subsampling.
            normalize_mean: Per-channel mean. Defaults to ImageNet.
            normalize_std: Per-channel std. Defaults to ImageNet.
        """
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self.image_size = image_size
        self.coarse_labels = coarse_labels

        mean = list(normalize_mean) if normalize_mean is not None else [0.485, 0.456, 0.406]
        std = list(normalize_std) if normalize_std is not None else [0.229, 0.224, 0.225]

        # v2 transforms apply geometric ops jointly to image + mask, color ops to image only.
        if mode == "train":
            self.preprocessor = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.preprocessor = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        if annotation_file is not None:
            json_file = Path(annotation_file)
        elif mode == "train":
            json_file = self.dataset_root / "stuff_train2017_filtered.json"
        else:
            json_file = self.dataset_root / "stuff_val2017.json"

        if image_dir is not None:
            self.img_dir = Path(image_dir)
        else:
            self.img_dir = self.dataset_root / f"{mode}2017"

        if not json_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_file}")

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        logger.info("Loading COCO-Stuff %s annotations from %s", mode, json_file)
        with json_file.open() as f:
            self.data = json.load(f)

        if instances_json is not None:
            if coarse_labels:
                raise ValueError("coarse_labels is not supported when instances_json is provided")
            instances_path = Path(instances_json)
            if not instances_path.exists():
                raise FileNotFoundError(f"Instances JSON not found: {instances_path}")
            logger.info("Loading COCO instances (things) from %s", instances_path)
            with instances_path.open() as f:
                instances_data = json.load(f)
            thing_cat_ids = {cat["id"] for cat in instances_data["categories"]}
            stuff_cat_ids = {cat["id"] for cat in self.data["categories"]}
            if thing_cat_ids & stuff_cat_ids:
                raise ValueError(f"Category ID overlap: {sorted(thing_cat_ids & stuff_cat_ids)}")
            self.data["categories"] = self.data["categories"] + instances_data["categories"]
            max_stuff_ann_id = max((ann["id"] for ann in self.data["annotations"]), default=0)
            for ann in instances_data["annotations"]:
                ann["id"] = max_stuff_ann_id + ann["id"] + 1
            self.data["annotations"] = self.data["annotations"] + instances_data["annotations"]
            logger.info(
                "Merged: %d thing + %d stuff categories, %d total annotations",
                len(thing_cat_ids),
                len(stuff_cat_ids),
                len(self.data["annotations"]),
            )

        self.images = {img["id"]: img for img in self.data["images"]}
        annotations = self.data["annotations"]

        self._annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in annotations:
            self._annotations_by_image[ann["image_id"]].append(ann)

        self.img_ids = list(self._annotations_by_image.keys())

        # Category 183 ("other") is excluded in both modes (mapped to ignore_index).
        self.categories = {cat["id"]: cat for cat in self.data["categories"] if cat["supercategory"] != "other"}
        self.id_to_name = {cat_id: cat["name"] for cat_id, cat in self.categories.items()}

        if self.coarse_labels:
            super_cats = {cat["supercategory"] for cat in self.categories.values()}
            self._coarse_names = sorted(super_cats)
            super_cat_to_id = {sc: i for i, sc in enumerate(self._coarse_names)}

            self._num_classes = len(self._coarse_names)
            self.id_to_continuous = {
                cat["id"]: super_cat_to_id[cat["supercategory"]] for cat in self.categories.values()
            }
            logger.info("Coarse classes (%d): %s", self._num_classes, self._coarse_names)
        else:
            self._num_classes = len(self.categories)
            self.id_to_continuous = {cat_id: i for i, cat_id in enumerate(sorted(self.categories.keys()))}
            self.continuous_to_id = dict(enumerate(sorted(self.categories.keys())))

        valid_img_ids = []
        for img_id in self.img_ids:
            img_info = self.images[img_id]
            img_path = self.img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_path.with_suffix(".png")

            try:
                with Image.open(img_path):
                    pass
                valid_img_ids.append(img_id)
            except (OSError, FileNotFoundError) as e:
                logger.warning("Removing invalid/missing image %s: %s", img_path, e)

        self.img_ids = valid_img_ids

        if subset_fraction is not None:
            if not 0.0 < subset_fraction <= 1.0:
                raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
            if subset_fraction < 1.0:
                n_total = len(self.img_ids)
                n_subset = max(1, int(n_total * subset_fraction))
                rng = random.Random(subset_seed)
                self.img_ids = sorted(rng.sample(self.img_ids, n_subset))
                logger.info(
                    "Subsampled %d/%d images (%.1f%%) with seed=%d",
                    n_subset,
                    n_total,
                    subset_fraction * 100,
                    subset_seed,
                )

        label_mode = f"coarse ({self._num_classes})" if self.coarse_labels else f"fine ({self._num_classes})"
        logger.info("Loaded %d valid %s images with %d %s classes", len(self), mode, self._num_classes, label_mode)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> SemanticSegmentationSample:
        try:
            img_id = self.img_ids[idx]
            img_info = self.images[img_id]

            img_path = self.img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_path.with_suffix(".png")

            image = Image.open(img_path).convert("RGB")
            img_anns = self._annotations_by_image[img_id]

            h, w = img_info["height"], img_info["width"]
            ignore_index = 255
            mask = np.full((h, w), ignore_index, dtype=np.int64)

            for ann in img_anns:
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], dict):
                        rle = ann["segmentation"]
                        if isinstance(rle["counts"], list):
                            # Uncompressed RLE (e.g. crowd annotations) — compress first.
                            rle = mask_utils.frPyObjects(rle, h, w)
                        binary_mask = mask_utils.decode(rle)
                        if binary_mask.shape != (h, w):
                            binary_mask = np.resize(binary_mask, (h, w))
                    else:
                        rles = mask_utils.frPyObjects(ann["segmentation"], h, w)
                        rle = mask_utils.merge(rles)
                        binary_mask = mask_utils.decode(rle)

                    cat_id = ann["category_id"]
                    continuous_id = self.id_to_continuous.get(cat_id)
                    if continuous_id is not None:
                        mask[binary_mask > 0] = continuous_id

            image = tv_tensors.Image(image)
            mask_tensor = tv_tensors.Mask(torch.from_numpy(mask))

            assert self.preprocessor is not None, "preprocessor is required at __getitem__ time"
            image, mask_tensor = self.preprocessor(image, mask_tensor)

            return SemanticSegmentationSample(media=image, mask=mask_tensor.long(), sample_id=img_id)

        except (OSError, FileNotFoundError, KeyError) as e:
            logger.warning("Error loading image at index %d, img_id %d: %s", idx, self.img_ids[idx], e)
            return self.__getitem__((idx + 1) % len(self))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @cached_property
    def class_ids(self) -> list[int]:
        return list(range(self._num_classes))

    @cached_property
    def class_names(self) -> list[str]:
        if self.coarse_labels:
            return list(self._coarse_names)
        return [self.id_to_name[self.continuous_to_id[i]] for i in range(self._num_classes)]
