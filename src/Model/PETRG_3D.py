"""PETRG-3D main model definition."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .my_embedding_layer import MyEmbedding


class PETRG3D(nn.Module):
    """PETRG-3D model.

    Wraps:
      * an LLM (loaded via ``AutoModelForCausalLM``) with a LoRA adapter
        attached through PEFT,
      * a shared 3D ViT visual encoder followed by two
        ``PerceiverResampler + FC`` heads -- one for CT, one for PET --
        implemented in :class:`Model.my_embedding_layer.MyEmbedding`.
    """

    def __init__(
        self,
        text_tokenizer_path: str,
        lang_model_path: str,
        pretrained_visual_encoder: Optional[str],
        pretrained_adapter: Optional[str] = None,
        max_img_size: int = 1,
        image_num: int = 128,
        use_fast: bool = False,
        enable_pet: bool = True,
        enable_thinking: bool = False,
        finetune_ct_feature_extractor: bool = False,
        training_stage: str = "joint",
    ):
        super().__init__()
        self.enable_pet = enable_pet
        self.enable_thinking = enable_thinking
        self.finetune_ct_feature_extractor = finetune_ct_feature_extractor
        self.training_stage = training_stage

        self.lang_model = AutoModelForCausalLM.from_pretrained(
            lang_model_path,
            trust_remote_code=True,
        )

        model_type = str(getattr(self.lang_model.config, "model_type", "")).lower()
        # GLM-family tokenizers require the fast implementation.
        self.use_fast = True if "glm" in model_type else bool(use_fast)

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_path,
            use_fast=self.use_fast,
            trust_remote_code=True,
        )

        # Step 1. Record the size of the LLM's native vocabulary before adding
        # any placeholder tokens so that we can correctly concatenate embeddings
        # later on.
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": []})
        self.lang_model.resize_token_embeddings(len(self.text_tokenizer))
        self.hidden_dim = self.lang_model.get_input_embeddings().embedding_dim
        self.voc_size = self.lang_model.get_input_embeddings().num_embeddings

        # Step 2. Add placeholder tokens for the modality wrappers and the visual
        # feature slots injected by :class:`MyEmbedding`.
        #
        # NOTE(compat): ``<region>/</region>`` are kept even though the public
        # release does not use region-level features, so that checkpoints trained
        # before the cleanup still load with ``strict=True``.
        special_token = {
            "additional_special_tokens": [
                "<ct>", "</ct>",
                "<pet>", "</pet>",
                "<region>", "</region>",
                "<template>", "</template>",
            ]
        }

        self.image_padding_tokens = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = f"<image_ct{i * image_num + j}>"
                padding += tok
                special_token["additional_special_tokens"].append(tok)
            self.image_padding_tokens.append(padding)

        self.pet_padding_tokens = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = f"<image_pet{i * image_num + j}>"
                padding += tok
                special_token["additional_special_tokens"].append(tok)
            self.pet_padding_tokens.append(padding)

        self.text_tokenizer.add_special_tokens(special_token)

        # Step 3. Wrap the LLM with a LoRA adapter.
        lora_kwargs = dict(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        if "glm" in model_type:
            # PEFT cannot auto-map the LoRA target modules for GLM variants.
            target_modules = [
                "query_key_value",
                "q_proj", "k_proj", "v_proj", "o_proj",
                "dense_h_to_4h", "dense_4h_to_h",
            ]
            peft_config = LoraConfig(**lora_kwargs, target_modules=target_modules)
        else:
            peft_config = LoraConfig(**lora_kwargs)
        self.lang_model = get_peft_model(self.lang_model, peft_config)
        self.lang_model.print_trainable_parameters()

        try:
            self.lang_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except Exception:
            self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()

        # Step 4. Build the multimodal embedding layer.
        self.embedding_layer = MyEmbedding(
            pretrained_visual_encoder=pretrained_visual_encoder,
            pretrained_adapter=pretrained_adapter,
            enable_pet=self.enable_pet,
            embedding_dim=self.hidden_dim,
            num_embeddings=self.voc_size,
            perceiver_num=image_num,
            pet_perceiver_num=image_num,
            finetune_ct_feature_extractor=self.finetune_ct_feature_extractor,
        )
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.loss_function = nn.CrossEntropyLoss()

        self._setup_trainable_params()

    # ------------------------------------------------------------------
    # Trainable-parameter schedules for the three supported training modes.
    # ------------------------------------------------------------------
    def _setup_trainable_params(self):
        if self.training_stage == "stage1":
            print("=" * 50)
            print("STAGE 1: Visual-Language Alignment")
            print("  Trainable: Vision Encoder + Perceiver + FC")
            print("  Frozen: LLM (including LoRA)")
            print("=" * 50)
            for param in self.lang_model.parameters():
                param.requires_grad = False
            for param in self.embedding_layer.vision_encoder.parameters():
                param.requires_grad = True
            for param in self.embedding_layer.perceiver.parameters():
                param.requires_grad = True
            for param in self.embedding_layer.fc.parameters():
                param.requires_grad = True
            if self.enable_pet:
                for param in self.embedding_layer.pet_perceiver.parameters():
                    param.requires_grad = True
                for param in self.embedding_layer.pet_fc.parameters():
                    param.requires_grad = True

        elif self.training_stage == "stage2":
            print("=" * 50)
            print("STAGE 2: Instruction Tuning")
            print("  Trainable: Perceiver + FC + LLM (LoRA)")
            print("  Frozen: Vision Encoder")
            print("=" * 50)
            for param in self.embedding_layer.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.embedding_layer.perceiver.parameters():
                param.requires_grad = True
            for param in self.embedding_layer.fc.parameters():
                param.requires_grad = True
            if self.enable_pet:
                for param in self.embedding_layer.pet_perceiver.parameters():
                    param.requires_grad = True
                for param in self.embedding_layer.pet_fc.parameters():
                    param.requires_grad = True
            for name, param in self.lang_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

        else:  # "joint"
            print("=" * 50)
            print("JOINT TRAINING: All modules train together")
            print(" (CT vision encoder follows `finetune_ct_feature_extractor`)")
            print("=" * 50)

        self._print_trainable_params()

    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"\nTrainable params: {trainable:,} || "
            f"Total params: {total:,} || "
            f"Trainable%: {100 * trainable / total:.2f}%\n"
        )

    # ------------------------------------------------------------------
    # Forward / generation entry points.
    # ------------------------------------------------------------------
    def forward(
        self,
        lang_x: torch.Tensor,
        vision_x: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        input_embedding = self.embedding_layer(vision_x, lang_x)
        model_dtype = next(self.lang_model.parameters()).dtype
        output = self.lang_model(
            inputs_embeds=input_embedding.to(model_dtype),
            attention_mask=attention_mask,
            labels=labels,
        )
        return dict(loss=output["loss"], logits=output["logits"])

    def generate(self, lang_x: torch.Tensor, vision_x: Dict[str, torch.Tensor]):
        with torch.no_grad():
            input_embedding = self.embedding_layer(vision_x, lang_x)
            if self.enable_thinking:
                # Qwen3 "thinking" preset (see the Qwen3 HuggingFace model card).
                generation = self.lang_model.generate(
                    inputs_embeds=input_embedding,
                    do_sample=True,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    temperature=0.6,
                    repetition_penalty=1.05,
                    max_new_tokens=2048,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                )
            else:
                generation = self.lang_model.generate(
                    inputs_embeds=input_embedding,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.05,
                    max_new_tokens=1024,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                )
            report = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True)
            return report


# Backwards-compatible alias: previous versions of the codebase named the class
# ``Reg2RG`` (after the upstream repository it was forked from). Checkpoints only
# store parameter names, so the alias is kept only for any user code that still
# imports the old class name.
Reg2RG = PETRG3D
