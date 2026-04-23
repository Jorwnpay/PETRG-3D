import os
import glob
import json
import math
import inspect
from typing import Dict, List, Tuple, Any

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms
from transformers import AutoConfig, AutoTokenizer


def _resolve_message_type(tokenizer_path: str, override: str = None) -> str:
    """Return the prompt-formatting style ('chatml' or 'human_design').

    ``human_design`` is the [INST]/[/INST]-style prompt used by Llama-2 Chinese
    Chat checkpoints; every other supported backbone uses the ChatML-style
    template emitted by ``tokenizer.apply_chat_template``. Callers may override
    the auto-detection by passing a concrete value.
    """
    if override is not None:
        return override
    try:
        model_type = str(getattr(AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True), "model_type", "")).lower()
    except Exception:
        model_type = ""
    if "llama" in model_type:
        return "human_design"
    return "chatml"


def threshold_ct(x):
    return x > -1000


class PETCTDataset_Train(Dataset):
    def __init__(
        self,
        tokenizer_path: str,
        image_folder: str,
        report_folder: str,
        template_path: str = None,
        z_chunk_size: int = 64,
        z_chunk_stride: int = 32,
        xy_size: Tuple[int, int] = (256, 256),
        max_img_size: int = 1,
        image_num: int = 128,
        max_seq: int = 4096,
        pet_clip_percentile: float = 99.5,
        pet_apply_log: bool = True,
        use_fast: bool = False,
        use_template: bool = False,
        enable_thinking: bool = False,
        enable_augmentation: bool = False,
        message_type: str = None,
    ) -> None:
        super().__init__()
        self.use_template = use_template
        self.enable_thinking = enable_thinking
        self.enable_augmentation = enable_augmentation
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast, trust_remote_code=True)
        # 记录扩展前词表大小，用于标签屏蔽保证仅监督基础词表
        self.core_vocab_size = self.tokenizer.vocab_size

        # 特殊 token：CT 与 PET 包裹符和占位符
        special_token = {
            "additional_special_tokens": ["<ct>", "</ct>", "<pet>", "</pet>", "<region>", "</region>", "<template>", "</template>"]
        }

        # 创建CT图像的特征占位符，<image_ct1>, <image_ct2>, ...
        self.image_padding_tokens: List[str] = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = "<image_ct" + str(i * image_num + j) + ">"
                padding += tok
                special_token["additional_special_tokens"].append(tok) # 把新加入的token存放到special_token列表中
            self.image_padding_tokens.append(padding)

        # 创建PET图像的特征占位符，<image_pet1>, <image_pet2>, ...
        self.pet_padding_tokens: List[str] = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = "<image_pet" + str(i * image_num + j) + ">"
                padding += tok
                special_token["additional_special_tokens"].append(tok) # 把新加入的token存放到special_token列表中
            self.pet_padding_tokens.append(padding)
        

        '''
        NOTE：（1）教程中说使用add_special_tokens加入新的token后，需要用model.resize_token_embeddings(len(tokenizer))来更新模型的embedding size，
        但在目前的代码中，没有找到任何调用model.resize_token_embeddings(len(tokenizer))的地方，因此这里表示怀疑。
        答：后面在my_embedding中将会处理；
        （2）此外还有一个问题，在当前的代码中为什么要加入special token呢？
        答：因为特征不属于词表中的token，因此需要加入到special_token中，以供后续使用。
        '''
        self.tokenizer.add_special_tokens(special_token) # 把特殊token加入到tokenizer的词表中
        
        # 保持与现有实现一致
        '''
        NOTE：这里需要确认：Qwen和LLaMA的pad_token_id、bos_token_id、eos_token_id是一致的吗？
        答：完全不同！
        '''
        self.max_seq = max_seq
        # 兼容脚本传参中残留的引号，去除首尾 ' 或 "
        self.image_folder = image_folder#.strip('\'"') if isinstance(image_folder, str) else image_folder
        self.report_folder = report_folder#.strip('\'"') if isinstance(report_folder, str) else report_folder
        self.z_chunk_size = z_chunk_size
        self.z_chunk_stride = z_chunk_stride
        self.xy_size = xy_size
        self.image_num = image_num
        self.pet_clip_percentile = pet_clip_percentile
        self.pet_apply_log = pet_apply_log

        # 采样列表
        self.samples = self._prepare_samples() 
        print('Number of PET/CT training samples:', len(self.samples))

        # 'chatml' for Qwen/ChatGLM/Gemma/Mistral etc., 'human_design' for Llama-2.
        self.message_type = _resolve_message_type(tokenizer_path, message_type)
        if self.message_type == 'human_design':
            # Llama-2 Chinese Chat ships without bos/eos/pad tokens set.
            self.tokenizer.pad_token_id = 0
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2

        # 体素与图像处理
        # 不再进行resize和z轴分块，直接使用原始图像
        self.ct_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.pet_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 数据增强变换
        if self.enable_augmentation:
            self.aug_transforms = transforms.Compose([
                transforms.RandFlipd(keys=["ct", "pet"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["ct", "pet"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["ct", "pet"], prob=0.5, spatial_axis=2),
                transforms.RandRotate90d(keys=["ct", "pet"], prob=0.5, max_k=3),
                # 可以根据需要开启以下增强
                # transforms.RandAffined(
                #     keys=["ct", "pet"],
                #     prob=0.3,
                #     rotate_range=(0.1, 0.1, 0.1),
                #     scale_range=(0.1, 0.1, 0.1),
                #     mode=("bilinear", "bilinear"),
                #     padding_mode="zeros"
                # ),
            ])

        if self.use_template:
            self.template_path = template_path
            self.report_template = json.load(open(template_path, "r"))

    def _prepare_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        
        # 递归扫描，适配子目录结构
        ct_paths = glob.glob(os.path.join(self.image_folder, "*_0000.nii.gz"), recursive=True)
        pet_paths = glob.glob(os.path.join(self.image_folder, "*_0001.nii.gz"), recursive=True)
        json_paths = glob.glob(os.path.join(self.report_folder, "*.json"), recursive=True)

        # 全局索引：分别递归扫描 CT/PET/JSON，按 patient_id 取交集，避免依赖相同目录结构
        def pid_from(path: str, suffix: str) -> str:
            base = os.path.basename(path)
            if base.endswith(suffix):
                return base[: -len(suffix)]
            return os.path.splitext(base)[0]

        ct_index: Dict[str, str] = {}
        for p in ct_paths:
            pid = pid_from(p, "_0000.nii.gz")
            if pid not in ct_index:
                ct_index[pid] = p

        pet_index: Dict[str, str] = {}
        for p in pet_paths:
            pid = pid_from(p, "_0001.nii.gz")
            if pid not in pet_index:
                pet_index[pid] = p

        json_index: Dict[str, str] = {}
        for p in json_paths:
            pid = pid_from(p, ".json")
            if pid not in json_index:
                json_index[pid] = p

        common = set(ct_index.keys()) & set(pet_index.keys()) & set(json_index.keys())
        for pid in sorted(common):
            samples.append({
                "patient_id": pid,
                "ct": ct_index[pid],
                "pet": pet_index[pid],
                "report": json_index[pid],
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _split_chunks(self, vol: np.ndarray, pad_value: float = -1000) -> List[np.ndarray]:
        # vol: (H, W, D)
        H, W, D = vol.shape
        chunks: List[np.ndarray] = []
        if self.z_chunk_size >= D:
            start = 0
            end = D
            pad = self.z_chunk_size - (end - start)
            if pad > 0:
                pad_before = pad // 2
                pad_after = pad - pad_before
                # 若图像的Z轴长度小于z_chunk_size，则填充到Z轴长度
                # NOTE: 这里原来是用数组的边缘值填充，但用边缘值是不合理的，我改成了用固定值填充，对于CT数据，填充-1000，对于PET数据，填充0
                # vol_pad = np.pad(vol, ((0, 0), (0, 0), (pad_before, pad_after)), mode='edge')
                vol_pad = np.pad(vol, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant', constant_values=(pad_value, pad_value))
                chunks.append(vol_pad)
            else:
                chunks.append(vol[:, :, start:end])
            return chunks

        step = self.z_chunk_stride
        for z in range(0, D - self.z_chunk_size + 1, step):
            chunk = vol[:, :, z:z + self.z_chunk_size]
            chunks.append(chunk)
        # 最后一块若未覆盖到尾部，则补一块尾部
        if len(chunks) == 0 or (chunks[-1].shape[-1] < self.z_chunk_size or (D - (len(chunks) - 1) * step - self.z_chunk_size) > 0):
            chunk = vol[:, :, D - self.z_chunk_size:D]
            chunks.append(chunk)
        return chunks

    def _normalize_ct(self, vol: np.ndarray) -> np.ndarray:
        # 假定预处理已裁剪到 [-1000, 1000]
        vol = np.clip(vol, -1000.0, 1000.0)
        # --- NOTE: 参照之前的CT的方式，把CT HU值归一化到[-1, 1]，以对齐训练期间的数据范围 ---
        # vol = (vol + 1000.0) / 2000.0
        vol = vol / 1000.0
        return vol.astype(np.float32)

    def _normalize_pet(self, vol: np.ndarray) -> np.ndarray:
        v = vol.astype(np.float32)
        if self.pet_apply_log:
            v = np.log1p(np.maximum(v, 0.0))
        if self.pet_clip_percentile is not None:
            hi = np.percentile(v, self.pet_clip_percentile)
            if hi <= 0:
                hi = float(v.max() + 1e-6)
            v = np.clip(v, 0.0, hi)
        vmax = float(v.max() + 1e-6)
        v = v / vmax
        return v.astype(np.float32)

    def _load_report_answer(self, path: str, if_load_gender: bool = False) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 仅保留“检查所见”作为答案
        answer = data.get("检查所见", "")
        if answer is None:
            answer = ""
        
        # 清洗并添加结束标志
        clean_answer = str(answer).strip() + " 【报告结束】"

        if if_load_gender:
            if "病人信息" in data:
                gender = data.get("病人信息", {}).get("性别", "")
            elif "性别" in data:
                gender = data.get("性别", "")
            else:
                gender = ""
            return clean_answer, str(gender).strip()

        # 简单清洗
        return clean_answer

    def _text_add_ct_tokens(self, text: str) -> str:
        text = '<ct>' + self.image_padding_tokens[0] + '</ct>' + '。' + text
        return text

    def _text_add_pet_tokens(self, text: str) -> str:
        text = '<pet>' + self.pet_padding_tokens[0] + '</pet>' + '。' + text
        return text

    def _text_add_petct_tokens(self, text: str) -> str:
        text = self._text_add_ct_tokens(text)
        text = self._text_add_pet_tokens(text)
        return text

    def _resize_and_pad(self, vol: np.ndarray, target_size: Tuple[int, int], pad_value: float) -> np.ndarray:
        H, W, D = vol.shape
        target_H, target_W = target_size

        # Determine resize scale
        if W / H > target_W / target_H:
            # Width is the limiting factor
            scale = target_W / W
            new_W = target_W
            new_H = int(round(H * scale))
        else:
            # Height is the limiting factor
            scale = target_H / H
            new_H = target_H
            new_W = int(round(W * scale))

        zoom_factor = (new_H / H, new_W / W, 1)
        resized_vol = scipy.ndimage.zoom(vol, zoom_factor, order=1)

        res_H, res_W, _ = resized_vol.shape
        
        pad_h_total = target_H - res_H
        pad_w_total = target_W - res_W
        
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        padded_vol = np.pad(
            resized_vol,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
        
        return padded_vol

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        ct_img = nib.load(sample["ct"]).get_fdata()
        pet_img = nib.load(sample["pet"]).get_fdata()

        # 直接使用原始图像，不进行resize和分块
        # 归一化处理
        ct_img = self._normalize_ct(ct_img) # [284, 228, 284]
        pet_img = self._normalize_pet(pet_img) # [284, 228, 284]

        # 添加通道维度 (1, H, W, D)
        ct_img = ct_img[np.newaxis, ...]
        pet_img = pet_img[np.newaxis, ...]

        # 数据增强
        if self.enable_augmentation:
            data_dict = {"ct": ct_img, "pet": pet_img}
            data_dict = self.aug_transforms(data_dict)
            ct_img = data_dict["ct"]
            pet_img = data_dict["pet"]

        # 转换为Tensor
        ct_tensor = self.ct_transform(ct_img)
        pet_tensor = self.pet_transform(pet_img)

        # 重复到3通道，保持与现有 ViT3D 输入一致
        ct_tensor = ct_tensor.repeat(3, 1, 1, 1)  # (3, H, W, D)  # [3, 284, 228, 284]
        pet_tensor = pet_tensor.repeat(3, 1, 1, 1)  # (3, H, W, D)

        # Pad图像使其尺寸能被patch size整除
        # ViT3D需要: H和W能被32整除，D能被4整除
        C, H, W, D = ct_tensor.shape
        
        # 计算需要padding的大小
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        pad_d = (4 - D % 4) % 4
        
        # 进行padding (pad的顺序是从最后一个维度开始：D, W, H)
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            # padding格式：(left, right, top, bottom, front, back)
            ct_tensor = torch.nn.functional.pad(
                ct_tensor, 
                (0, pad_d, 0, pad_w, 0, pad_h), 
                mode='constant', 
                value=0
            )
            pet_tensor = torch.nn.functional.pad(
                pet_tensor, 
                (0, pad_d, 0, pad_w, 0, pad_h), 
                mode='constant', 
                value=0
            )

        # 增加批维度，shape变为 (1, 3, H', W', D')，表示1个图像（不分块）
        ct_tensor = ct_tensor.unsqueeze(0) # [1, 3, 288, 256, 284]
        pet_tensor = pet_tensor.unsqueeze(0)

        # 文本
        answer, gender = self._load_report_answer(sample["report"], if_load_gender=True)  # 仅检查所见
        if self.use_template:
            gender_flag = None
            if gender == "男":
                gender_flag = "M"
            elif gender == "女":
                gender_flag = "F"
            else:
                print(f"性别为{gender}，不合法")
            hospital_flag = sample["ct"].split("/")[-1][:2]
            style_template = self.report_template[hospital_flag][gender_flag]
            instruction = f"报告模板：<template>{style_template}</template>\\n请根据提供的全身PET和CT的影像特征，参考提供的医院特定风格的健康患者的报告模板，生成中文影像学报告的“检查所见”部分。"
        else:
            instruction = "已提供该患者的全身CT与PET影像整体信息。请根据影像信息，生成中文影像学报告的“检查所见”部分。"

        if 'chatml' in self.message_type:
            # The content of user combines images and instruction. The image placeholders are replaced with actual image tokens in the model.
            messages = [
                {
                    "role": "system",
                    "content": "你是一名核医学影像专家，尤其精通淋巴瘤的诊断与分析。"
                },
                {
                    "role": "user",
                    "content": f"<ct>{self.image_padding_tokens[0]}</ct>\\n<pet>{self.pet_padding_tokens[0]}</pet>\\n{instruction}"
                }
            ]
            
            # The answer is added to the messages for the assistant to learn from.
            messages.append({"role": "assistant", "content": answer})

            # Tokenization is handled here using the chat template.
            self.tokenizer.padding_side = "right"

            try:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": False,
                    "enable_thinking": self.enable_thinking,
                }
                text = self.tokenizer.apply_chat_template(**template_kwargs)
            except:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": False,
                }   
                text = self.tokenizer.apply_chat_template(**template_kwargs)
            
            text_tensor = self.tokenizer(
                text,
                max_length=self.max_seq,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            # Find the start of the assistant's answer for loss calculation
            # We need to tokenize the user part of the prompt separately to find its length
            prompt_messages = messages[:-1] # Exclude assistant's answer
            
            try:
                prompt_template_kwargs = {
                    "conversation": prompt_messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": self.enable_thinking,
                }
                prompt_text = self.tokenizer.apply_chat_template(**prompt_template_kwargs)
            except:
                prompt_template_kwargs = {
                    "conversation": prompt_messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                prompt_text = self.tokenizer.apply_chat_template(**prompt_template_kwargs)
            
            prompt_tensor = self.tokenizer(
                prompt_text,
                max_length=self.max_seq,
                truncation=True,
                return_tensors="pt",
            )
            prompt_len = torch.sum(prompt_tensor["attention_mask"][0])
            
            label = input_ids.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            # Mask out the prompt part, only calculate loss on the answer
            label[:prompt_len] = -100
            # Also mask special image tokens if they are in the labels
            label[label >= self.core_vocab_size] = -100

            return {
                'lang_x': input_ids,
                'vision_x': {
                    'ct_image': ct_tensor,  # (1, 3, H, W, D) - 原始图像，不再分块
                    'pet_image': pet_tensor,
                },
                'mask_x': {},
                'region2area': {},
                'attention_mask': attention_mask,
                'label': label,
            }
        
        elif 'human_design' in self.message_type:
            prompt = self._text_add_petct_tokens(instruction)
            
            self.tokenizer.padding_side = "right"
            text_tensor = self.tokenizer(
                prompt + ' ' + answer,
                max_length=self.max_seq,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            effective_length = torch.sum(attention_mask)
            if effective_length < self.max_seq:
                input_ids[effective_length] = self.tokenizer.eos_token_id
            else:
                input_ids[self.max_seq - 1] = self.tokenizer.eos_token_id

            prompt_tensor = self.tokenizer(
                prompt,
                max_length=self.max_seq,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_len = torch.sum(prompt_tensor["attention_mask"][0])

            label = input_ids.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[label >= self.core_vocab_size] = -100
            label[:prompt_len] = -100

            return {
                'lang_x': input_ids,
                'vision_x': {
                    'ct_image': ct_tensor,  # (1, 3, H, W, D) - 原始图像，不再分块
                    'pet_image': pet_tensor,
                },
                'mask_x': {},
                'region2area': {},
                'attention_mask': attention_mask,
                'label': label,
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test PETCTDataset_Train with local paths."
    )
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--report_folder", required=True)
    parser.add_argument("--template_path", default=None)
    parser.add_argument("--use_template", action="store_true")
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()

    dataset = PETCTDataset_Train(
        tokenizer_path=args.tokenizer_path,
        image_folder=args.image_folder,
        report_folder=args.report_folder,
        template_path=args.template_path,
        use_template=args.use_template,
        use_fast=args.use_fast,
    )
    prompt_len_list = []
    for data in dataset:
        prompt_len_list.append(torch.sum(data["attention_mask"]).item())

    print(max(prompt_len_list))