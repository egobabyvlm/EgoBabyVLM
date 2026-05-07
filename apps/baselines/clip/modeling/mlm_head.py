"""BERT MLM head for the interleaved-LM and triple training modes.

Wraps :class:`transformers.models.bert.modeling_bert.BertOnlyMLMHead`. The
head consumes hidden states from the text encoder's BERT backbone and emits
vocab-size logits, used to compute the masked-LM cross-entropy loss
alongside the contrastive loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

if TYPE_CHECKING:
    from apps.baselines.clip.modeling.text_encoder import TextEncoder

# HuggingFace's standard ignore_index for masked-LM labels: positions with this
# value are excluded from the loss and accuracy computation.
MLM_IGNORE_INDEX = -100


class MLMHead(nn.Module):
    """BERT MLM prediction head + cross-entropy loss helper.

    Args:
        text_encoder: The :class:`TextEncoder` whose backbone hidden states
            this head will consume. Its ``config`` provides the vocab size,
            hidden size, and tie-weights setting.
    """

    def __init__(self, text_encoder: TextEncoder) -> None:
        super().__init__()
        self.head = BertOnlyMLMHead(text_encoder.config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return ``(B, L, vocab_size)`` prediction logits."""
        return self.head(hidden_states)

    @staticmethod
    def loss(prediction_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Masked cross-entropy loss with ``ignore_index=-100`` (HF convention)."""
        return F.cross_entropy(
            prediction_scores.view(-1, prediction_scores.size(-1)),
            labels.view(-1),
            ignore_index=MLM_IGNORE_INDEX,
        )

    @staticmethod
    def accuracy(prediction_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Token-level accuracy over the masked positions only."""
        predictions = prediction_scores.argmax(dim=-1)
        mask = labels != MLM_IGNORE_INDEX
        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction_scores.device)
        return (predictions == labels).float()[mask].mean()
