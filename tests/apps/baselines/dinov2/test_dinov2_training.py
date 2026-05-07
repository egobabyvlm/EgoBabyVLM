"""Smoke tests for the DINOv2 SSL trainer entrypoint + dataset registry."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

# Importing the upstream package registers it as ``dinov2`` in sys.modules so
# its internal ``from dinov2.X`` imports resolve.
import apps.baselines.dinov2.third_party.dinov2  # noqa: F401

_REQUIRES_FVCORE = pytest.mark.skipif(
    importlib.util.find_spec("fvcore") is None,
    reason="fvcore not installed; trainer-entry imports skipped (it's pinned in pixi.toml).",
)


@_REQUIRES_FVCORE
def test_training_entry_imports() -> None:
    """The OSS-facing entry shim re-exports the upstream trainer's public API."""
    train_mod = importlib.import_module("apps.baselines.dinov2.training.train")
    assert callable(train_mod.main)
    assert callable(train_mod.do_train)
    assert callable(train_mod.get_args_parser)


@_REQUIRES_FVCORE
def test_submit_entry_imports() -> None:
    """The Submitit driver shim is importable and exposes a ``main()``."""
    submit_mod = importlib.import_module("apps.baselines.dinov2.training.submit")
    assert callable(submit_mod.main)


@_REQUIRES_FVCORE
def test_argparse_smokes() -> None:
    """``get_args_parser()`` constructs without erroring and accepts the documented flags."""
    from apps.baselines.dinov2.training.train import get_args_parser

    parser = get_args_parser(add_help=False)
    args = parser.parse_args(["--config-file", "/tmp/dummy.yaml", "--no-wandb"])
    assert args.config_file == "/tmp/dummy.yaml"
    assert args.no_wandb is True


# ---- dataset registry -----------------------------------------------------


_EXPECTED_DATASETS = ("BabyView", "Ego4D", "HowToSubset", "ImageNet", "MSCOCO")


@pytest.mark.parametrize("name", _EXPECTED_DATASETS)
def test_dataset_imports(name: str) -> None:
    """Every dataset class shipped under ``data/datasets/`` is importable."""
    mod = importlib.import_module("apps.baselines.dinov2.third_party.dinov2.data.datasets")
    assert hasattr(mod, name), f"{name} missing from data.datasets exports"
    cls = getattr(mod, name)
    assert hasattr(cls, "Split"), f"{name} should expose a Split enum"


def test_make_dataset_resolves_known_names() -> None:
    """``make_dataset`` rejects unknown names and accepts each shipped one."""
    from apps.baselines.dinov2.third_party.dinov2.data.loaders import _parse_dataset_str

    # Each shipped dataset name should resolve without raising. We can't
    # actually instantiate without real data, so we just check the parser
    # branches.
    for name in ("ImageNet", "MSCOCO", "Ego4D", "HowTo", "BabyView"):
        cls, _ = _parse_dataset_str(f"{name};root=/tmp/dummy;extra=/tmp/dummy")
        assert cls is not None

    with pytest.raises(ValueError, match="Unsupported dataset"):
        _parse_dataset_str("NoSuchDataset;root=/tmp/dummy")


def test_local_path_manager_is_no_op() -> None:
    """``LocalPathManager.get_local_path`` returns the input unchanged."""
    from apps.baselines.dinov2.third_party.dinov2.data.datasets.extended import LocalPathManager

    pm = LocalPathManager()
    assert pm.get_local_path("/tmp/foo/bar") == "/tmp/foo/bar"
