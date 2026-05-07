"""Convert an older contrastive-trainer checkpoint to the self-contained ``.pt`` format.

Two on-disk formats are supported, auto-detected from the file contents:

* **PyTorch Lightning ``.ckpt``**: has a top-level ``state_dict`` key
  holding ``MultiModalLitModel``-style keys (``model.image_embed.X`` +
  duplicated ``vision_encoder.X`` views). ``hyper_parameters.args``
  carries the run config.

* **Pure-PyTorch ``.pth``**: has a top-level ``clip_model`` dict with
  already-inner ``image_embed.X`` / ``text_embed.X`` keys.
  ``training_config`` carries the run config.

In both cases, encoder hyperparameters (HF model name, vision arch,
embedding dim, dropout, pooling, etc.) are read from the embedded
config — the user does not have to re-supply them. CLI overrides are
accepted for the rare case where the embedded config is wrong / missing.

Usage::

    egobabyvlm-convert-legacy-ckpt \\
        --legacy-ckpt /path/to/old.{ckpt,pth} \\
        --output /path/to/new.pt

The output is a self-contained ``.pt`` readable by the contrastive-trainer
checkpoint loader; the embedded config has the full ``_target_`` for both
encoders so ``hydra.utils.instantiate`` rebuilds the exact same
architecture.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Hub DINOv2 model names that map to ``HubDINOv2VisionEncoder``. Anything else
#: in the legacy ``cnn_model`` field is treated as a custom SSL checkpoint path.
_HUB_DINOV2_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")


def _rewrite_inner_model_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite ``image_embed.model.{head, X}`` to the new split structure.

    Both legacy formats wrap the DINOv2 backbone under ``image_embed.model.X``
    and add a linear classifier at ``image_embed.model.head.{weight,bias}``
    that acts as the projection. The new ``HubDINOv2VisionEncoder`` /
    ``CustomDINOv2VisionEncoder`` split these into sibling ``backbone`` and
    ``projection`` submodules, so we rewrite:

      * ``image_embed.model.head.{weight,bias}`` → ``image_embed.projection.{weight,bias}``
      * ``image_embed.model.X``                  → ``image_embed.backbone.X``

    The text tower keeps its ``self.model`` wrapper (BERT backbone) in the
    new code so ``text_embed.X`` keys pass through unchanged.
    """
    new: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("image_embed.model.head."):
            new["image_embed.projection." + k[len("image_embed.model.head.") :]] = v
        elif k.startswith("image_embed.model."):
            new["image_embed.backbone." + k[len("image_embed.model.") :]] = v
        else:
            new[k] = v
    return new


def _drop_vocab_fallback_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove ``text_embed.embedding.*`` keys from older vocab-indexed checkpoints.

    The new ``TextEncoder`` always uses the BERT backbone path and has no
    fallback embedding layer.
    """
    return {k: v for k, v in state.items() if not k.startswith("text_embed.embedding.")}


def _strip_lightning_outer_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Lightning ``MultiModalLitModel`` keys to the inner ``MultiModalModel`` view.

    The Lightning module exposes encoders both as ``vision_encoder``/``text_encoder``
    AND as ``model.image_embed`` / ``model.text_embed``. We keep only the inner
    view (which is what the new ``MultiModalModel`` exposes) and drop the
    duplicates.
    """
    new: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(("model.image_embed.", "model.text_embed.")):
            new[k[len("model.") :]] = v
        elif k == "model.logit_neg_log_temperature":
            new["logit_neg_log_temperature"] = v
        # Drop duplicates (vision_encoder.*, text_encoder.*) and unrelated keys.
    return new


def convert_legacy_state_dict(
    legacy_state: dict[str, torch.Tensor],
    *,
    is_lightning: bool,
) -> dict[str, torch.Tensor]:
    """Convert an older state_dict (Lightning or pure-PyTorch) to the self-contained format."""
    inner = _strip_lightning_outer_prefix(legacy_state) if is_lightning else legacy_state
    return _drop_vocab_fallback_keys(_rewrite_inner_model_keys(inner))


def _infer_vision_encoder_target(cnn_model: str) -> tuple[str, dict[str, Any]]:
    """Pick the right vision encoder class + ctor args from the legacy ``cnn_model`` field.

    Hub DINOv2 names dispatch to ``HubDINOv2VisionEncoder``; anything else is
    treated as a custom SSL teacher checkpoint path and dispatched to
    ``CustomDINOv2VisionEncoder`` (the matching ``config.yaml`` sits three
    directories up from the checkpoint, per DINOv2's standard output layout).
    """
    if cnn_model in _HUB_DINOV2_NAMES:
        return "apps.baselines.clip.modeling.HubDINOv2VisionEncoder", {"model_name": cnn_model}

    ckpt = Path(cnn_model)
    config_path = ckpt.parent.parent.parent / "config.yaml"
    return (
        "apps.baselines.clip.modeling.CustomDINOv2VisionEncoder",
        {"checkpoint_path": str(ckpt), "config_path": str(config_path)},
    )


def _build_embedded_config(legacy_config: dict[str, Any], legacy_ckpt_path: Path) -> dict[str, Any]:
    """Reconstruct the self-contained config from the source run's training_config / hyperparameters."""
    embedding_dim = int(legacy_config.get("embedding_dim", 512))
    dropout = float(legacy_config.get("dropout_o", 0.1))
    pooling = str(legacy_config.get("pooling", "cls"))
    hf_model_name = legacy_config["hf_text_encoder"]
    cnn_model = legacy_config["cnn_model"]
    image_size = int(legacy_config.get("image_size", 224))
    normalize_features = bool(legacy_config.get("normalize_features", False))
    temperature = float(legacy_config.get("temperature", 0.07))
    fix_temperature = bool(legacy_config.get("fix_temperature", False))

    vision_target, vision_extra = _infer_vision_encoder_target(cnn_model)
    vision_cfg = {
        "_target_": vision_target,
        "embedding_dim": embedding_dim,
        "freeze": False,
        **vision_extra,
    }
    if vision_target.endswith("HubDINOv2VisionEncoder"):
        vision_cfg["image_size"] = image_size

    return {
        "model": {
            "embedding_dim": embedding_dim,
            "normalize_features": normalize_features,
            "temperature": temperature,
            "fix_temperature": fix_temperature,
            "text_encoder": {
                "_target_": "apps.baselines.clip.modeling.TextEncoder",
                "hf_model_name": hf_model_name,
                "embedding_dim": embedding_dim,
                "dropout": dropout,
                "freeze": False,
                "pooling": pooling,
            },
            "vision_encoder": vision_cfg,
        },
        "_converted_from_legacy": True,
        "_legacy_ckpt_path": str(legacy_ckpt_path),
        "_legacy_training_config": _coerce_to_primitives(legacy_config),
    }


def _legacy_args_to_dict(legacy_args: object) -> dict[str, Any]:
    """Lightning's hyper_parameters['args'] is a Namespace; pure-PyTorch's training_config is a dict."""
    if isinstance(legacy_args, dict):
        return legacy_args
    return vars(legacy_args)


def _coerce_to_primitives(value: object) -> object:
    """Recursively coerce a value to OmegaConf-friendly primitives.

    Legacy training configs sometimes contain non-primitive objects (e.g. an
    optimizer class reference for ``OPTIMIZER = torch.optim.AdamW``). OmegaConf
    rejects those, which would prevent the converted checkpoint's embedded
    config from round-tripping through ``OmegaConf.create``. We stringify
    anything that isn't already a primitive.
    """
    if isinstance(value, dict):
        return {k: _coerce_to_primitives(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_to_primitives(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def convert_checkpoint_payload(legacy: dict[str, Any], legacy_ckpt_path: Path) -> dict[str, Any]:
    """Convert an in-memory legacy checkpoint dict to the new self-contained payload.

    Splits into two cases by inspecting the dict keys; raises if neither shape
    matches.
    """
    if "state_dict" in legacy and "hyper_parameters" in legacy:
        # PyTorch Lightning .ckpt
        legacy_state = legacy["state_dict"]
        legacy_config = _legacy_args_to_dict(legacy["hyper_parameters"]["args"])
        new_state = convert_legacy_state_dict(legacy_state, is_lightning=True)
        epoch = int(legacy.get("epoch", 0))
        step = int(legacy.get("global_step", 0))
    elif "clip_model" in legacy:
        # Pure-PyTorch .pth (interleaved_dino / interleaved_lm_dino / filtered_triple)
        legacy_state = legacy["clip_model"]
        legacy_config = _legacy_args_to_dict(legacy["training_config"])
        new_state = convert_legacy_state_dict(legacy_state, is_lightning=False)
        epoch = int(legacy.get("epoch", 0))
        step = int(legacy.get("step_count", 0))
    else:
        raise ValueError(
            "Unrecognized legacy checkpoint format: expected 'state_dict' (Lightning) or "
            f"'clip_model' (pure-PyTorch); got top-level keys {list(legacy.keys())[:10]}",
        )

    embedded = _build_embedded_config(legacy_config, legacy_ckpt_path)
    return {
        "model_state_dict": new_state,
        "mlm_head_state_dict": None,
        "ssl_state_dict": None,
        "optimizer_state_dicts": {},
        "scheduler_state": None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": float(legacy.get("best_val_loss", float("inf"))),
        "config": embedded,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-ckpt",
        type=Path,
        required=True,
        help="Path to the legacy checkpoint file (Lightning .ckpt or pure-PyTorch .pth).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the new .pt checkpoint.")
    args = parser.parse_args()

    logger.info("Loading legacy checkpoint from %s", args.legacy_ckpt)
    legacy = torch.load(args.legacy_ckpt, map_location="cpu", weights_only=False)
    payload = convert_checkpoint_payload(legacy, args.legacy_ckpt)

    n_in = len(legacy.get("state_dict", legacy.get("clip_model", {})))
    n_out = len(payload["model_state_dict"])
    logger.info("Converted %d legacy keys → %d new keys", n_in, n_out)
    logger.info("Embedded config (model): %s", payload["config"]["model"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info("Wrote self-contained checkpoint to %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
