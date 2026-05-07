"""Unit tests for the older-checkpoint → self-contained-format converter."""

from __future__ import annotations

import torch

from apps.baselines.clip.scripts.convert_legacy_checkpoint import (
    _infer_vision_encoder_target,
    _strip_lightning_outer_prefix,
    convert_legacy_state_dict,
)


def test_strip_outer_lit_module_prefix() -> None:
    """Lightning state_dict has both `vision_encoder.X` and `model.image_embed.X`
    (the inner-MultiModalModel view); the converter keeps only the latter."""
    legacy = {
        "vision_encoder.model.cls_token": torch.zeros(1, 1, 768),
        "model.image_embed.model.cls_token": torch.zeros(1, 1, 768),
        "text_encoder.projection.weight": torch.zeros(2, 2),
        "model.text_embed.projection.weight": torch.zeros(2, 2),
        "model.logit_neg_log_temperature": torch.tensor(2.0),
    }
    inner = _strip_lightning_outer_prefix(legacy)
    assert set(inner.keys()) == {
        "image_embed.model.cls_token",
        "text_embed.projection.weight",
        "logit_neg_log_temperature",
    }


def test_lightning_full_round_trip_remaps_head_to_projection() -> None:
    """Lightning .ckpt: outer-prefix stripped + image_embed.model.head → projection."""
    legacy = {
        "model.image_embed.model.head.weight": torch.zeros(64, 768),
        "model.image_embed.model.head.bias": torch.zeros(64),
        "model.image_embed.model.cls_token": torch.zeros(1, 1, 768),
        "model.text_embed.projection.weight": torch.zeros(64, 768),
        "model.logit_neg_log_temperature": torch.tensor(2.0),
        "vision_encoder.model.cls_token": torch.zeros(1, 1, 768),  # duplicate, dropped
    }
    new = convert_legacy_state_dict(legacy, is_lightning=True)
    assert "image_embed.projection.weight" in new
    assert "image_embed.projection.bias" in new
    assert "image_embed.backbone.cls_token" in new
    assert "text_embed.projection.weight" in new
    assert "logit_neg_log_temperature" in new
    assert "vision_encoder.model.cls_token" not in new  # duplicate dropped


def test_pure_pytorch_full_round_trip() -> None:
    """Pure-PyTorch .pth: keys are already in `image_embed.X` form, just split
    out the head → projection."""
    legacy = {
        "logit_neg_log_temperature": torch.tensor(2.0),
        "image_embed.model.head.weight": torch.zeros(64, 768),
        "image_embed.model.head.bias": torch.zeros(64),
        "image_embed.model.cls_token": torch.zeros(1, 1, 768),
        "text_embed.projection.weight": torch.zeros(64, 768),
    }
    new = convert_legacy_state_dict(legacy, is_lightning=False)
    assert "image_embed.projection.weight" in new
    assert "image_embed.projection.bias" in new
    assert "image_embed.backbone.cls_token" in new
    assert "text_embed.projection.weight" in new
    assert "logit_neg_log_temperature" in new


def test_drop_vocab_fallback_embedding() -> None:
    """Older vocab-indexed checkpoints carried a fallback embedding layer for the
    no-pretrained-text path; the new TextEncoder doesn't, so those weights are dropped."""
    legacy = {
        "text_embed.embedding.weight": torch.zeros(100, 64),
        "text_embed.projection.weight": torch.zeros(64, 32),
        "image_embed.model.head.weight": torch.zeros(64, 32),
    }
    new = convert_legacy_state_dict(legacy, is_lightning=False)
    assert "text_embed.embedding.weight" not in new
    assert "text_embed.projection.weight" in new
    assert "image_embed.projection.weight" in new


def test_infer_vision_target_hub() -> None:
    target, kwargs = _infer_vision_encoder_target("dinov2_vitb14")
    assert target.endswith("HubDINOv2VisionEncoder")
    assert kwargs == {"model_name": "dinov2_vitb14"}


def test_infer_vision_target_custom_ssl_path() -> None:
    target, kwargs = _infer_vision_encoder_target(
        "/data/runs/dino_howto/eval/training_500/teacher_checkpoint.pth",
    )
    assert target.endswith("CustomDINOv2VisionEncoder")
    assert kwargs["checkpoint_path"].endswith("teacher_checkpoint.pth")
    assert kwargs["config_path"].endswith("/data/runs/dino_howto/config.yaml")
