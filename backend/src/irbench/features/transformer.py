"""Transformer backbones (Hugging Face): DINOv2 ViT-S/14 and CLIP ViT-B/32."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, Dinov2Config, Dinov2Model

from irbench.features.base import register_extractor
from irbench.features.pooling import TOKEN_POOLINGS, pool_tokens
from irbench.features.torch_base import TorchExtractor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@register_extractor
class DINOv2SmallExtractor(TorchExtractor):
    """DINOv2 ViT-S/14 (``facebook/dinov2-small``), aspect-preserving input snapped to 14 px.

    Positional embeddings are interpolated by the model for non-224 inputs, so the
    backbone sees the whole (resized) image rather than a centre crop.
    """

    name = "dinov2_vits14"
    family = "transformer"
    poolings = TOKEN_POOLINGS
    default_pooling = "cls"
    supports_multiscale = True
    patch_multiple = 14
    default_image_size = 448
    description = "DINOv2 ViT-S/14 self-supervised, CLS token or patch-token GeM/mean"

    def __init__(self, model_id: str = "facebook/dinov2-small", **kwargs: Any) -> None:
        self.model_id = model_id
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return 384

    def _extra_config(self) -> dict[str, Any]:
        return {**super()._extra_config(), "model_id": self.model_id}

    def _build_model(self) -> nn.Module:
        if self.pretrained:
            return Dinov2Model.from_pretrained(self.model_id)
        config = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            patch_size=14,
            image_size=518,
        )
        return Dinov2Model(config)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        tokens = self.model(pixel_values=batch).last_hidden_state
        return pool_tokens(tokens, self.pooling, self.gem_p)


@register_extractor
class CLIPViTB32Extractor(TorchExtractor):
    """CLIP ViT-B/32 image tower (``openai/clip-vit-base-patch32``) at a fixed 224x224 input.

    ``cls`` returns the projected image embedding (512-d, the space CLIP was trained in);
    patch poolings use the post-layernorm patch tokens (768-d).
    """

    name = "clip_vitb32"
    family = "transformer"
    poolings = TOKEN_POOLINGS
    default_pooling = "cls"
    supports_multiscale = False
    fixed_size = (224, 224)
    default_image_size = 224
    mean = CLIP_MEAN
    std = CLIP_STD
    description = "CLIP ViT-B/32 image encoder (224x224), projected CLS or patch tokens"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", **kwargs: Any) -> None:
        self.model_id = model_id
        kwargs["image_size"] = 224
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return 512 if self.pooling == "cls" else 768

    def _extra_config(self) -> dict[str, Any]:
        return {**super()._extra_config(), "model_id": self.model_id}

    def _build_model(self) -> nn.Module:
        if self.pretrained:
            return CLIPVisionModelWithProjection.from_pretrained(self.model_id)
        return CLIPVisionModelWithProjection(CLIPVisionConfig())

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=batch)
        if self.pooling == "cls":
            return out.image_embeds
        tokens = self.model.vision_model.post_layernorm(out.last_hidden_state)
        return pool_tokens(tokens, self.pooling, self.gem_p)
