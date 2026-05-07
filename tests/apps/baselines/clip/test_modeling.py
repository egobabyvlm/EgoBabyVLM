"""Unit tests for the modeling components.

Marked ``integration`` (not run by default CI) because every test instantiates
a HuggingFace BERT backbone — that requires the model files to be cached
locally or network access to download them.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration


def test_text_encoder_shapes() -> None:
    from apps.baselines.clip.modeling import TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    proj, hidden = te(["hello world", "a cat sits on a mat"])
    assert proj.shape == (2, 128)
    assert hidden.shape[0] == 2
    assert hidden.shape[2] == 768  # bert-base hidden size


def test_text_encoder_pooling_modes() -> None:
    from apps.baselines.clip.modeling import TextEncoder

    cls = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0, pooling="cls")
    mean = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0, pooling="mean")
    proj_cls, _ = cls(["hello world"])
    proj_mean, _ = mean(["hello world"])
    assert proj_cls.shape == proj_mean.shape == (1, 128)


def test_text_encoder_invalid_pooling_raises() -> None:
    from apps.baselines.clip.modeling import TextEncoder

    with pytest.raises(ValueError, match="pooling must be"):
        TextEncoder("bert-base-uncased", pooling="bogus")  # type: ignore[arg-type]


def test_random_vit_vision_encoder_shapes() -> None:
    from apps.baselines.clip.modeling import RandomViTVisionEncoder

    ve = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    out = ve(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 128)
    assert ve.output_dim == 768
    assert ve.arch == "vitb14"
    assert ve.image_size == 224


def test_multimodal_model_contrastive_loss() -> None:
    from apps.baselines.clip.modeling import MultiModalModel, RandomViTVisionEncoder, TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    ve = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    mm = MultiModalModel(ve, te, normalize_features=True, temperature=0.07)

    out = mm.compute_contrastive_loss(torch.randn(4, 3, 224, 224), [f"caption {i}" for i in range(4)])
    assert torch.isfinite(out.loss)
    assert out.logits_per_image.shape == (4, 4)
    assert out.logits_per_text.shape == (4, 4)
    assert 0.0 <= out.image_accuracy.item() <= 1.0


def test_mlm_head_shapes_and_loss() -> None:
    from apps.baselines.clip.modeling import MLMHead, TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    mlm = MLMHead(te)

    _, hidden = te(["hello world", "another caption"])
    preds = mlm(hidden)
    assert preds.shape[:2] == hidden.shape[:2]
    assert preds.shape[2] == te.config.vocab_size

    labels = torch.full(hidden.shape[:2], -100, dtype=torch.long)
    labels[0, 1] = 1234
    labels[1, 2] = 5678
    loss = MLMHead.loss(preds, labels)
    accuracy = MLMHead.accuracy(preds, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= accuracy.item() <= 1.0
