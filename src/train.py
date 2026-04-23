# Disable wandb logging unconditionally; use transformers' report_to if needed.
import os
os.environ.setdefault("WANDB_DISABLED", "true")

import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from transformers import Trainer

from Model.PETRG_3D import PETRG3D
from Dataset.petct_dataset_train import PETCTDataset_Train
from args.train.default import ModelArguments, DataArguments, TrainingArguments


@dataclass
class DataCollator(object):
    """Collate a batch of PET/CT + text samples produced by PETCTDataset_Train."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        lang_xs, vision_xs, attention_masks, labels = tuple(
            [instance[key] for instance in instances]
            for key in ('lang_x', 'vision_x', 'attention_mask', 'label')
        )

        lang_xs = torch.cat([x.unsqueeze(0) for x in lang_xs], dim=0)
        attention_masks = torch.cat([x.unsqueeze(0) for x in attention_masks], dim=0)
        labels = torch.cat([x.unsqueeze(0) for x in labels], dim=0)

        ct_images = torch.cat([v['ct_image'].unsqueeze(0) for v in vision_xs], dim=0)
        pet_images = torch.cat([v['pet_image'].unsqueeze(0) for v in vision_xs], dim=0)

        return dict(
            lang_x=lang_xs,
            vision_x={'ct_image': ct_images, 'pet_image': pet_images},
            attention_mask=attention_masks,
            labels=labels,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
def main():
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("================= Arguments ================")
    print(f"data_args: {data_args}")
    print(f"model_args: {model_args}")
    print(f"training_args: {training_args}")
    print("==========================================")

    print("Setup Data")
    Train_dataset = PETCTDataset_Train(
        tokenizer_path=model_args.tokenizer_path,
        image_folder=data_args.petct_image_folder,
        report_folder=data_args.petct_report_folder,
        z_chunk_size=data_args.z_chunk_size,
        z_chunk_stride=data_args.z_chunk_stride,
        xy_size=(data_args.xy_size, data_args.xy_size),
        max_seq=data_args.max_seq,
        pet_clip_percentile=data_args.pet_clip_percentile,
        pet_apply_log=data_args.pet_apply_log,
        use_fast=data_args.use_fast,
        use_template=data_args.use_template,
        template_path=data_args.template_path,
        enable_thinking=data_args.enable_thinking,
        enable_augmentation=data_args.enable_augmentation,
    )

    print("Setup Model")
    model = PETRG3D(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
        use_fast=data_args.use_fast,
        enable_pet=True,
        enable_thinking=data_args.enable_thinking,
        finetune_ct_feature_extractor=model_args.finetune_ct_feature_extractor,
        training_stage=model_args.training_stage,
    )
    
    # Load pretrained model weights if specified
    if model_args.pretrained_model_path is not None:
        print(f"Loading pretrained model weights from: {model_args.pretrained_model_path}")
        ckpt = torch.load(model_args.pretrained_model_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)
        print("Pretrained model weights loaded successfully!")

    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      args = training_args,
                      data_collator=DataCollator(),
                      )

    trainer.train()
    trainer.save_state()
      
if __name__ == "__main__":
    main()