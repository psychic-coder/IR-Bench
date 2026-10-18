"""Global pooling operators.

Convolutional feature maps ``x`` have shape ``(B, C, H, W)``; transformer token sequences
have shape ``(B, 1 + N, C)`` with the CLS token first. Every function returns ``(B, C)``.

* GAP     : f_c = 1/|X_c| * sum_x x
* GeM     : f_c = (1/|X_c| * sum_x x^p)^(1/p)          (p = 1 -> GAP, p -> inf -> max)
* R-MAC   : sum over multi-scale regions R of l2(max_{x in R} x), then l2
* CLS     : the transformer class token
* patch-GeM / patch-GAP : GeM / mean over the patch tokens (CLS excluded)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CONV_POOLINGS = ("gap", "gem", "rmac")
TOKEN_POOLINGS = ("cls", "patch_gem", "patch_gap")
CLASSICAL_POOLINGS = ("none",)


def gap(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling over the spatial dimensions of ``(B, C, H, W)``."""
    return x.mean(dim=(-2, -1))


def gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalized-mean pooling over ``(B, C, H, W)`` with a fixed exponent ``p``.

    Activations are clamped to ``eps`` so the fractional power is defined. For ReLU
    feature maps (all CNN backbones here) the clamp only affects exact zeros, which is
    why ``gem(x, p=1)`` equals :func:`gap` up to ``eps``.
    """
    if p <= 0:
        raise ValueError("GeM exponent p must be positive")
    return x.clamp(min=eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)


def rmac_regions(h: int, w: int, levels: int = 3, overlap: float = 0.4) -> list[tuple[int, int, int, int]]:
    """Square region grid of Tolias et al. (2016), returned as ``(y0, x0, size_h, size_w)``.

    Level ``l`` uses squares of side ``2 * min(H, W) / (l + 1)`` laid out uniformly with
    ~``overlap`` overlap. The number of regions along the longer side is chosen to make
    the overlap as close as possible to the target, exactly as in the reference
    implementation (``cirtorch.layers.functional.rmac``).
    """
    regions: list[tuple[int, int, int, int]] = []
    short = min(h, w)
    ovr = overlap
    steps = torch.arange(1, 7, dtype=torch.float64)  # candidate extra regions on long side
    b = (max(h, w) - short) / steps
    idx = int(torch.argmin(torch.abs(((short**2 - short * b) / short**2) - ovr)).item())
    wd = hd = 0
    if h < w:
        wd = idx + 1
    elif h > w:
        hd = idx + 1
    regions.append((0, 0, h, w))  # level 0: the full map (global max pooling)
    for level in range(1, levels + 1):
        side = math.floor(2 * short / (level + 1))
        if side < 1:
            continue
        half = math.floor(side / 2 - 1)
        n_w = level + wd
        n_h = level + hd
        bw = (w - side) / (n_w - 1) if n_w > 1 else 0.0
        bh = (h - side) / (n_h - 1) if n_h > 1 else 0.0
        for iy in range(n_h):
            y0 = max(0, int(math.floor(half + iy * bh)) - half)
            for ix in range(n_w):
                x0 = max(0, int(math.floor(half + ix * bw)) - half)
                regions.append((y0, x0, min(side, h - y0), min(side, w - x0)))
    return regions


def rmac(x: torch.Tensor, levels: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Regional maximum activations of convolutions over ``(B, C, H, W)``."""
    _, _, h, w = x.shape
    out = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    for y0, x0, rh, rw in rmac_regions(h, w, levels=levels):
        region = x[:, :, y0 : y0 + rh, x0 : x0 + rw]
        v = region.amax(dim=(-2, -1))
        out = out + v / (v.norm(dim=1, keepdim=True) + eps)
    return out


def cls_token(tokens: torch.Tensor) -> torch.Tensor:
    """Return the class token from ``(B, 1 + N, C)``."""
    return tokens[:, 0]


def patch_gem(tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """GeM over patch tokens ``(B, 1 + N, C)``; negative activations are clamped to ``eps``."""
    patches = tokens[:, 1:]
    return patches.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


def patch_gap(tokens: torch.Tensor) -> torch.Tensor:
    """Mean of patch tokens ``(B, 1 + N, C)`` (signed, no clamping)."""
    return tokens[:, 1:].mean(dim=1)


def pool_conv(x: torch.Tensor, method: str, p: float = 3.0) -> torch.Tensor:
    """Dispatch a convolutional pooling by name."""
    if method == "gap":
        return gap(x)
    if method == "gem":
        return gem(x, p=p)
    if method == "rmac":
        return rmac(x)
    raise ValueError(f"Unknown conv pooling '{method}'. Valid: {CONV_POOLINGS}")


def pool_tokens(tokens: torch.Tensor, method: str, p: float = 3.0) -> torch.Tensor:
    """Dispatch a token pooling by name."""
    if method == "cls":
        return cls_token(tokens)
    if method == "patch_gem":
        return patch_gem(tokens, p=p)
    if method == "patch_gap":
        return patch_gap(tokens)
    raise ValueError(f"Unknown token pooling '{method}'. Valid: {TOKEN_POOLINGS}")


def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalisation."""
    return F.normalize(x, p=2, dim=-1, eps=eps)
