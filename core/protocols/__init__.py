"""Structural protocols for the evaluation pipeline."""

from core.protocols.feature_extractor import (
    FeatureExtractor,
    ImageFeatureExtractor,
    MultiModalFeatureExtractor,
    TextFeatureExtractor,
    VideoFeatureExtractor,
)

__all__ = [
    "FeatureExtractor",
    "ImageFeatureExtractor",
    "MultiModalFeatureExtractor",
    "TextFeatureExtractor",
    "VideoFeatureExtractor",
]
