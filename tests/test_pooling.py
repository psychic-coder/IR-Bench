"""Pooling shapes and numerics, and extractor output contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from irbench.features import create_extractor, list_extractors
from irbench.features.pooling import (
    gap,
    gem,
    patch_gap,
    patch_gem,
    pool_conv,
    pool_tokens,
    rmac,
    rmac_regions,
)


@pytest.fixture
def fmap() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.relu(torch.randn(3, 16, 7, 9)) + 1e-3


def test_gem_p1_equals_gap(fmap: torch.Tensor) -> None:
    assert torch.allclose(gem(fmap, p=1.0), gap(fmap), atol=1e-6)


def test_gem_large_p_approaches_max(fmap: torch.Tensor) -> None:
    x = fmap.double()  # x**p overflows float32 for large p
    mx = x.amax(dim=(-2, -1))
    assert torch.allclose(gem(x, p=200.0), mx, rtol=0.05)


def test_gem_is_monotone_in_p(fmap: torch.Tensor) -> None:
    g1, g3, g6 = gem(fmap, 1.0), gem(fmap, 3.0), gem(fmap, 6.0)
    assert torch.all(g1 <= g3 + 1e-6) and torch.all(g3 <= g6 + 1e-6)


def test_gem_rejects_non_positive_p(fmap: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        gem(fmap, p=0.0)


@pytest.mark.parametrize("method", ["gap", "gem", "rmac"])
def test_conv_pooling_shapes(fmap: torch.Tensor, method: str) -> None:
    assert pool_conv(fmap, method).shape == (3, 16)


def test_rmac_regions_cover_the_map() -> None:
    regions = rmac_regions(7, 9, levels=3)
    assert regions[0] == (0, 0, 7, 9)
    for y0, x0, h, w in regions:
        assert y0 >= 0 and x0 >= 0 and y0 + h <= 7 and x0 + w <= 9 and h > 0 and w > 0
    assert len(regions) > 1 + 1 + 4  # global + level 1 + level 2 at least


def test_rmac_on_square_map_matches_sum_of_normalised_region_maxima() -> None:
    x = torch.rand(1, 4, 6, 6)
    expected = torch.zeros(1, 4)
    for y0, x0, h, w in rmac_regions(6, 6):
        v = x[:, :, y0 : y0 + h, x0 : x0 + w].amax(dim=(-2, -1))
        expected += v / (v.norm(dim=1, keepdim=True) + 1e-6)
    assert torch.allclose(rmac(x), expected)


def test_token_poolings() -> None:
    tokens = torch.randn(2, 1 + 10, 8)
    assert torch.equal(pool_tokens(tokens, "cls"), tokens[:, 0])
    assert torch.allclose(patch_gap(tokens), tokens[:, 1:].mean(dim=1))
    assert patch_gem(tokens).shape == (2, 8)
    assert torch.all(patch_gem(tokens) > 0)


def test_registry_lists_all_families() -> None:
    families = {e.family for e in list_extractors()}
    assert families == {"classical", "cnn", "transformer"}
    names = {e.name for e in list_extractors()}
    assert {"hog", "hsv_hist", "resnet50", "efficientnet_b0", "dinov2_vits14", "clip_vitb32"} <= names


def _images() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)),
        Image.fromarray(rng.integers(0, 255, (160, 120, 3), dtype=np.uint8)),
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)),
    ]


@pytest.mark.parametrize(
    ("name", "kwargs", "dim"),
    [
        ("hog", {}, 1764),
        ("hsv_hist", {}, 512),
        ("resnet50", {"pooling": "gem", "image_size": 96, "pretrained": False}, 2048),
        ("resnet50", {"pooling": "rmac", "image_size": 96, "pretrained": False, "layer": "layer3"}, 1024),
        ("efficientnet_b0", {"pooling": "gap", "image_size": 96, "pretrained": False}, 1280),
        ("dinov2_vits14", {"pooling": "patch_gem", "image_size": 84, "pretrained": False}, 384),
        ("clip_vitb32", {"pooling": "cls", "pretrained": False}, 512),
    ],
)
def test_extractor_contract(name: str, kwargs: dict, dim: int) -> None:
    ext = create_extractor(name, **kwargs)
    out = ext.extract(_images())
    assert out.shape == (3, dim) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert ext.extract([]).shape == (0, dim)


def test_extraction_is_deterministic_and_order_preserving() -> None:
    ext = create_extractor("resnet50", pooling="gem", image_size=64, pretrained=False)
    imgs = _images()
    a = ext.extract(imgs)
    b = ext.extract(imgs[::-1])[::-1]
    assert np.allclose(a, b, atol=1e-5)


def test_multiscale_changes_descriptor_and_validation() -> None:
    single = create_extractor("resnet50", pooling="gem", image_size=64, pretrained=False)
    multi = create_extractor(
        "resnet50", pooling="gem", image_size=64, pretrained=False, scales=[1.0, 0.7, 1.4]
    )
    imgs = _images()
    assert multi.extract(imgs).shape == single.extract(imgs).shape
    assert multi.config()["scales"] == [1.0, 0.7, 1.4]
    with pytest.raises(ValueError):
        create_extractor("hog", scales=[1.0, 0.5])
    with pytest.raises(ValueError):
        create_extractor("resnet50", pooling="cls", pretrained=False)
