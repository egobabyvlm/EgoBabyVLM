"""Unit tests for the self-contained checkpoint format.

Marked ``integration`` because the test instantiates a HuggingFace BERT
backbone (777 MB checkpoint) — too heavy for the default ``pytest -q`` run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def test_save_load_round_trip(tmp_path: Path) -> None:
    from apps.baselines.clip.modeling import (
        MultiModalModel,
        RandomViTVisionEncoder,
        TextEncoder,
    )
    from apps.baselines.clip.training import (
        InterleaveScheduler,
        build_adamw,
        load_checkpoint,
        save_checkpoint,
    )

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    ve = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    mm = MultiModalModel(ve, te)
    opt = build_adamw(mm.parameters(), lr=1e-3)
    sched = InterleaveScheduler({"contrastive": 1})
    cfg = OmegaConf.create({"foo": "bar", "model": {"embedding_dim": 128}})

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=mm,
        optimizers={"contrastive": opt},
        scheduler=sched,
        config=cfg,
        epoch=3,
        step=42,
        best_val_loss=0.5,
    )

    te2 = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    ve2 = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    mm2 = MultiModalModel(ve2, te2)
    opt2 = build_adamw(mm2.parameters(), lr=1e-3)
    sched2 = InterleaveScheduler({"contrastive": 1})

    payload = load_checkpoint(
        ckpt_path,
        model=mm2,
        optimizers={"contrastive": opt2},
        scheduler=sched2,
    )
    assert payload["epoch"] == 3
    assert payload["step"] == 42
    assert payload["best_val_loss"] == 0.5
    assert payload["config"] == {"foo": "bar", "model": {"embedding_dim": 128}}

    for (n1, p1), (_, p2) in zip(mm.named_parameters(), mm2.named_parameters(), strict=True):
        assert torch.allclose(p1, p2), f"{n1} mismatch"
