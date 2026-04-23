"""Default training argument schema for PETRG-3D.

All paths are left as ``None`` / placeholders by default; concrete values
should be supplied through the shell configuration scripts under ``configs/``
or on the command line.
"""

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or HF repo of the backbone LLM."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or HF repo of the tokenizer (often the same as lang_encoder_path)."},
    )
    pretrained_visual_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to RadFM's 3D ViT checkpoint (RadFM_vit3d.pth)."},
    )
    pretrained_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "(Unused in the released config; kept for backwards compatibility.)"},
    )
    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a previous PETRG-3D checkpoint to continue training from."},
    )
    finetune_ct_feature_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to finetune the CT feature extractor (3D ViT) during training."},
    )
    training_stage: str = field(
        default="joint",
        metadata={"help": "Training stage: 'joint' (all modules), 'stage1' (vision+adapter), or 'stage2' (adapter+LLM)."},
    )


@dataclass
class DataArguments:
    # Main data paths for PET/CT.
    petct_image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing paired {id}_0000.nii.gz (CT) and {id}_0001.nii.gz (PET) volumes."},
    )
    petct_report_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing Chinese JSON reports named {id}.json."},
    )

    # Image preprocessing.
    z_chunk_size: int = field(default=64)
    z_chunk_stride: int = field(default=32)
    xy_size: int = field(default=256)
    pet_clip_percentile: float = field(default=99.5)
    pet_apply_log: bool = field(default=True)

    # Text / LLM handling.
    max_seq: int = field(default=2048)
    use_fast: bool = field(default=False, metadata={"help": "Use the fast tokenizer implementation."})
    use_template: bool = field(default=False, metadata={"help": "Wrap the answer into the Chinese report template."})
    template_path: Optional[str] = field(default=None, metadata={"help": "Path to the JSON template file."})
    enable_thinking: bool = field(default=False, metadata={"help": "Enable Qwen3-style 'thinking' generation."})
    enable_augmentation: bool = field(default=False, metadata={"help": "Enable on-the-fly volume augmentation."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to adopt during training: 'no', 'steps' or 'epoch'."},
    )
    output_dir: Optional[str] = field(
        default="./outputs/petrg_3d_run",
        metadata={"help": "Base directory where checkpoints and logs are written."},
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pin_memory: bool = field(default=True)
