"""Cosine similarity helpers."""

import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Compute pairwise cosine similarity.

    Args:
        a: Tensor of shape ``(N, D)``.
        b: Tensor of shape ``(M, D)``.
        normalize: L2-normalize rows before the matmul.

    Returns:
        Similarity matrix of shape ``(N, M)``.
    """
    if normalize:
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
    return a @ b.T


def cosine_pairwise(a: torch.Tensor, b: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Compute element-wise cosine similarity.

    Args:
        a: Tensor of shape ``(N, D)``.
        b: Tensor of shape ``(N, D)``.
        normalize: L2-normalize rows before the dot product.

    Returns:
        Similarity scores of shape ``(N,)``.
    """
    if normalize:
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
    return (a * b).sum(dim=-1)
