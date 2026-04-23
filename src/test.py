import os
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
import transformers
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader

from Model.PETRG_3D import PETRG3D
from Dataset.petct_dataset_test import PETCTDataset_Test
from args.test.default import ModelArguments, DataArguments

@dataclass
class DataCollator(object):
    """Collate a batch of PET/CT + text samples produced by PETCTDataset_Test."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        acc_nums, lang_xs, vision_xs, questions, gt_reports = tuple(
            [instance[key] for instance in instances]
            for key in ('acc_num', 'lang_x', 'vision_x', 'question', 'gt_report')
        )

        lang_xs = torch.cat([x.unsqueeze(0) for x in lang_xs], dim=0)
        ct_images = torch.cat([v['ct_image'].unsqueeze(0) for v in vision_xs], dim=0)
        pet_images = torch.cat([v['pet_image'].unsqueeze(0) for v in vision_xs], dim=0)

        return dict(
            acc_num=acc_nums,
            lang_x=lang_xs,
            vision_x={'ct_image': ct_images, 'pet_image': pet_images},
            question=questions,
            gt_report=gt_reports,
        )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(42)
# 预处理数据

def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    (model_args, data_args) = parser.parse_args_into_dataclasses()
    print(model_args.ckpt_path)
    if_enable_thinking = getattr(data_args, 'enable_thinking', False)
    if if_enable_thinking:
        print("Enable thinking")
    else:
        print("Disable thinking")
    
    # 判断结果保存路径是否存在，不存在则创建（父目录也会被自动创建）
    result_parent = os.path.dirname(os.path.abspath(data_args.result_path))
    os.makedirs(result_parent, exist_ok=True)
    if os.path.exists(data_args.result_path):
        df = pd.read_csv(data_args.result_path)
        inferenced_id = df["AccNum"].tolist()
    else:
        df = pd.DataFrame(columns=["AccNum", "Question", "GT_report", "Pred_report"])
        df.to_csv(data_args.result_path, index=False)
        inferenced_id = []

    print("Setup Data")
    Test_dataset = PETCTDataset_Test(
        tokenizer_path=model_args.tokenizer_path,
        image_folder=data_args.petct_image_folder,
        report_folder=data_args.petct_report_folder,
        inferenced_id=inferenced_id,
        z_chunk_size=data_args.z_chunk_size,
        z_chunk_stride=data_args.z_chunk_stride,
        xy_size=(data_args.xy_size, data_args.xy_size),
        max_seq=data_args.max_seq,
        pet_clip_percentile=data_args.pet_clip_percentile,
        pet_apply_log=data_args.pet_apply_log,
        use_fast=data_args.use_fast,
        use_template=data_args.use_template,
        template_path=data_args.template_path,
        enable_thinking=if_enable_thinking,
    )

    Test_dataloader = DataLoader(
        Test_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=DataCollator(),
        drop_last=False,
    )

    print("Setup Model")

    model = PETRG3D(
        text_tokenizer_path=model_args.tokenizer_path,
        lang_model_path=model_args.lang_encoder_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
        use_fast=data_args.use_fast,
        enable_pet=True,
        enable_thinking=if_enable_thinking,
    )

    print(f"Loading ckpt from {model_args.ckpt_path}")
    # 在 PyTorch 2.6+ 中，torch.load 默认 weights_only=True，这可能导致加载包含非权重数据（如模型结构）的旧检查点失败。
    # 由于我们加载的是自己训练的可信模型，这里显式设置 weights_only=False 以兼容旧的检查点格式。
    if model_args.ckpt_path.endswith(".safetensors"):
        ckpt = load_safetensors(model_args.ckpt_path, device='cpu')
    else:
        ckpt = torch.load(model_args.ckpt_path, map_location='cpu', weights_only=False)
    # in case the state dict is not from peft
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=True)
    print("load ckpt")
    model = model.cuda()
    print("model to cuda")

    model.eval()

    for sample in tqdm.tqdm(Test_dataloader):
        acc_num = sample["acc_num"][0]
        question = sample["question"][0]
        lang_x = sample["lang_x"].cuda()
        vision_x = {modality: tensor.cuda() for modality, tensor in sample["vision_x"].items()}
        gt_report = sample["gt_report"][0]

        pred_report = model.generate(lang_x, vision_x)
        pred_report = pred_report[0]

        print('AccNum: ', acc_num)
        print('GT_report: ', gt_report)
        print('Pred_report: ', pred_report)
         
        new_data = pd.DataFrame([[acc_num, question, gt_report, pred_report]], 
                                columns=["AccNum", "Question", "GT_report", "Pred_report"])
        new_data.to_csv(data_args.result_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    main()
