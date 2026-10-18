"""Feature extraction: registry, extractors, pooling and descriptor caching."""

from irbench.features.base import (
    EXTRACTORS,
    ExtractorInfo,
    FeatureExtractor,
    create_extractor,
    extractor_info,
    list_extractors,
    register_extractor,
)

__all__ = [
    "EXTRACTORS",
    "ExtractorInfo",
    "FeatureExtractor",
    "create_extractor",
    "extractor_info",
    "list_extractors",
    "register_extractor",
]
