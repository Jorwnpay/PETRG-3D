"""Default inference argument schema for PETRG-3D."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or HF repo of the backbone LLM."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or HF repo of the tokenizer."},
    )
    pretrained_visual_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to RadFM's 3D ViT checkpoint (RadFM_vit3d.pth)."},
    )
    pretrained_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "(Unused; kept for backwards compatibility.)"},
    )
    ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a trained PETRG-3D checkpoint (.bin or .safetensors)."},
    )


@dataclass
class DataArguments:
    petct_image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing paired CT/PET .nii.gz volumes for inference."},
    )
    petct_report_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing the matched ground-truth JSON reports."},
    )
    result_path: Optional[str] = field(
        default="./results/petrg_3d_reports.csv",
        metadata={"help": "Output CSV path; parent directories are created automatically."},
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
    use_template: bool = field(default=False, metadata={"help": "Wrap the prompt with the Chinese report template."})
    template_path: Optional[str] = field(default=None, metadata={"help": "Path to the JSON template file."})
    enable_thinking: bool = field(default=False, metadata={"help": "Enable Qwen3-style 'thinking' generation."})
