"""Multimodal embedding layer used by :class:`Model.PETRG_3D.PETRG3D`.

The layer maintains a learnable embedding table that concatenates:

1. The LLM's own input embeddings (tied via ``self.weight``).
2. Four pairs of learned embeddings for the wrapper tokens
   ``<ct>/</ct>``, ``<pet>/</pet>``, ``<region>/</region>`` and
   ``<template>/</template>``.
3. Per-forward CT and PET visual features that fill the positions of the
   dynamic ``<image_ct*>`` / ``<image_pet*>`` placeholder tokens.

The tokenizer IDs and the row order of this concatenated embedding table are
kept in sync -- the forward pass therefore simply converts ``text_input`` to a
one-hot matrix and multiplies it by the concatenated table.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from .helpers import PerceiverResampler
from .vit_3d import ViT


class MyEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pretrained_visual_encoder: Optional[str] = None,
        pretrained_adapter: Optional[str] = None,
        perceiver_num: int = 128,
        vis_dim: int = 768,
        patch_size: int = 32,
        frame_patch_size: int = 4,
        enable_pet: bool = True,
        pet_vis_dim: int = 768,
        pet_perceiver_num: int = 128,
        finetune_ct_feature_extractor: bool = False,
    ):
        super().__init__()
        self.enable_pet = enable_pet
        self.finetune_ct_feature_extractor = finetune_ct_feature_extractor
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Tied to ``lang_model.get_input_embeddings().weight`` in the parent
        # module, so initialization here is only a placeholder.
        self.weight = nn.Parameter(
            torch.randn((num_embeddings, embedding_dim)), requires_grad=True
        )

        # Learnable embeddings for the wrapper special tokens. The order below
        # MUST match the order in which they are added to the tokenizer
        # (``<ct>/</ct>`` then ``<pet>/</pet>`` then ``<region>/</region>``
        # then ``<template>/</template>``) -- both orders are asserted at
        # forward time.
        self.image_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True
        )
        self.pet_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True
        )
        # NOTE(compat): kept to remain backward-compatible with checkpoints
        # trained before the ``use_regions`` pathway was removed.
        self.region_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True
        )
        self.template_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True
        )

        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.vis_dim = vis_dim

        # 3D ViT backbone shared between CT and PET (only the Perceiver + FC
        # heads are modality-specific).
        self.vision_encoder = ViT(
            image_size=512,
            frames=512,
            image_patch_size=patch_size,
            frame_patch_size=frame_patch_size,
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location="cpu")
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)

        if not self.finetune_ct_feature_extractor:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("INFO: CT feature extractor (vision_encoder) is frozen.")
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            print("INFO: CT feature extractor (vision_encoder) is trainable.")

        self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        # NOTE: ``pretrained_adapter`` (RadFM's perceiver/fc weights) is no longer
        # loaded because we increased ``perceiver_num`` from 32 to 128, which is
        # incompatible in shape. The adapter is re-trained from scratch.
        print(
            f"INFO: PerceiverResampler (CT) initialized from scratch with "
            f"{perceiver_num} latents."
        )

        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)

        if self.enable_pet:
            self.pet_vis_dim = pet_vis_dim
            self.pet_perceiver = PerceiverResampler(
                dim=self.pet_vis_dim, num_latents=pet_perceiver_num
            )
            self.pet_fc = nn.Linear(self.pet_vis_dim, self.embedding_dim)
            print(
                f"INFO: PerceiverResampler (PET) initialized from scratch with "
                f"{pet_perceiver_num} latents."
            )

    def _encode_volume(self, vol: torch.Tensor, perceiver: nn.Module) -> torch.Tensor:
        """Encode a (B, S, 3, H, W, D) volume and return (B, S*num_latents, vis_dim)."""
        B, S = vol.shape[0], vol.shape[1]
        feat = rearrange(vol, "b S c h w d -> (b S) c h w d")
        feat, _ = self.vision_encoder(feat)
        feat = rearrange(feat, "(b s) v d -> b s v d", b=B, s=S)
        feat = feat.unsqueeze(2)
        feat = perceiver(feat)
        n = feat.shape[2]
        feat = rearrange(feat, "b s n d -> (b s n) d")
        feat = rearrange(feat, "(b T) d -> b T d", b=B, T=n * S)
        return feat

    def forward(
        self,
        vision_x: Dict[str, torch.Tensor],
        text_input: torch.Tensor,
    ) -> torch.Tensor:
        """Run the multimodal embedding lookup.

        Args:
            vision_x: ``{"ct_image": tensor, "pet_image": tensor}`` where each
                tensor has shape ``(B, S, 3, H, W, D)`` (``S == 1`` in this
                release -- the whole volume is fed as a single chunk).
            text_input: ``(B, L)`` LongTensor of token IDs.

        Returns:
            ``(B, L, embedding_dim)`` mixed textual + visual embeddings suitable
            for ``LLM(inputs_embeds=...)``.
        """
        ct_input = vision_x["ct_image"]
        B = ct_input.shape[0]

        ct_feat = self._encode_volume(ct_input, self.perceiver)
        image_embedding = self.fc(ct_feat)  # (B, n_ct, embedding_dim)

        if self.enable_pet and "pet_image" in vision_x:
            pet_input = vision_x["pet_image"]
            pet_feat = self._encode_volume(pet_input, self.pet_perceiver)
            pet_mapped = self.pet_fc(pet_feat)  # (B, n_pet, embedding_dim)
            image_embedding = torch.cat([image_embedding, pet_mapped], dim=1)

        # Concatenation order MUST match the tokenizer's additional_special_tokens
        # order: <ct>/</ct>, <pet>/</pet>, <region>/</region>, <template>/</template>,
        # then the dynamic <image_ct*> / <image_pet*> slots.
        embedding_weight = torch.cat(
            [
                self.weight,
                self.image_token_weight,
                self.pet_token_weight,
                self.region_token_weight,
                self.template_token_weight,
            ],
            dim=0,
        )
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)
        embedding_weight = torch.cat([embedding_weight, image_embedding], dim=1)

        # Sanity check: every token ID we may see must map to a valid row in the
        # concatenated embedding matrix.
        assert int(text_input.max().item()) < embedding_weight.shape[1], (
            f"token id {int(text_input.max().item())} exceeds embedding table "
            f"size {embedding_weight.shape[1]}"
        )

        text_one_hot = torch.nn.functional.one_hot(
            text_input, embedding_weight.shape[1]
        ).to(image_embedding.dtype).to(text_input.device)
        return torch.matmul(text_one_hot, embedding_weight)
