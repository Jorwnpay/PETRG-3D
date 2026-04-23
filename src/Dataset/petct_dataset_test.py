import os
import glob
import json
import inspect
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms
from transformers import AutoConfig, AutoTokenizer


def threshold_ct(x):
    return x > -1000


def _resolve_message_type(tokenizer_path: str, override: str = None) -> str:
    """See :func:`petct_dataset_train._resolve_message_type`."""
    if override is not None:
        return override
    try:
        model_type = str(getattr(AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True), "model_type", "")).lower()
    except Exception:
        model_type = ""
    if "llama" in model_type:
        return "human_design"
    return "chatml"


class PETCTDataset_Test(Dataset):
    def __init__(
        self,
        tokenizer_path: str,
        image_folder: str,
        report_folder: str,
        inferenced_id: List[str],
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
        message_type: str = None,
    ) -> None:
        super().__init__()
        self.use_template = use_template
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)

        # 特殊 token：与训练保持一致
        special_token = {
            "additional_special_tokens": ["<ct>", "</ct>", "<pet>", "</pet>", "<region>", "</region>", "<template>", "</template>"]
        }
        # 创建CT图像的特征占位符，<image_ct0>, <image_ct1>, ...
        self.image_padding_tokens: List[str] = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = "<image_ct" + str(i * image_num + j) + ">"
                padding += tok
                special_token["additional_special_tokens"].append(tok)
            self.image_padding_tokens.append(padding)

        self.pet_padding_tokens: List[str] = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = "<image_pet" + str(i * image_num + j) + ">"
                padding += tok
                special_token["additional_special_tokens"].append(tok)
            self.pet_padding_tokens.append(padding)

        self.tokenizer.add_special_tokens(special_token)

        self.max_seq = max_seq
        self.image_folder = image_folder
        self.report_folder = report_folder
        self.inferenced_id = set(inferenced_id or [])
        self.z_chunk_size = z_chunk_size
        self.z_chunk_stride = z_chunk_stride
        self.xy_size = xy_size
        self.image_num = image_num
        self.pet_clip_percentile = pet_clip_percentile
        self.pet_apply_log = pet_apply_log

        self.samples = self._prepare_samples()

        self.message_type = _resolve_message_type(tokenizer_path, message_type)
        if self.message_type == 'human_design':
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

        if self.use_template:
            self.report_template = json.load(open(template_path, "r"))

    def _prepare_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        ct_paths = glob.glob(os.path.join(self.image_folder, "*_0000.nii.gz"), recursive=True)
        pet_paths = glob.glob(os.path.join(self.image_folder, "*_0001.nii.gz"), recursive=True)
        json_paths = glob.glob(os.path.join(self.report_folder, "*.json"), recursive=True)

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
            if pid in self.inferenced_id:
                continue
            samples.append({
                "accnum": pid,
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
        if if_load_gender:
            if "病人信息" in data:
                gender = data.get("病人信息", {}).get("性别", "")
            elif "性别" in data:
                gender = data.get("性别", "")
            else:
                gender = ""
            return str(answer).strip(), str(gender).strip()

        # 简单清洗
        return str(answer).strip()

    def _text_add_ct_tokens(self, text: str) -> str:
        return '<ct>' + self.image_padding_tokens[0] + '</ct>' + '。' + text

    def _text_add_pet_tokens(self, text: str) -> str:
        return '<pet>' + self.pet_padding_tokens[0] + '</pet>' + '。' + text

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
        item = self.samples[index]
        ct_img = nib.load(item["ct"]).get_fdata() # [336, 184, 328]
        pet_img = nib.load(item["pet"]).get_fdata() # [336, 184, 328]

        # 直接使用原始图像，不进行resize和分块
        # 归一化处理
        ct_img = self._normalize_ct(ct_img)
        pet_img = self._normalize_pet(pet_img)

        # 添加通道维度 (1, H, W, D)
        ct_img = ct_img[np.newaxis, ...]
        pet_img = pet_img[np.newaxis, ...]

        # 转换为Tensor
        ct_tensor = self.ct_transform(ct_img)
        pet_tensor = self.pet_transform(pet_img)

        # 重复到3通道，保持与现有 ViT3D 输入一致
        ct_tensor = ct_tensor.repeat(3, 1, 1, 1)  # (3, H, W, D) # [3, 336, 184, 328]
        pet_tensor = pet_tensor.repeat(3, 1, 1, 1)  # (3, H, W, D) # [3, 336, 184, 328]

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
        ct_tensor = ct_tensor.unsqueeze(0) # [1, 3, 352, 192, 328]
        pet_tensor = pet_tensor.unsqueeze(0)

        gt_answer, gender = self._load_report_answer(item["report"], if_load_gender=True)  # 仅供保存对照
        if self.use_template:
            gender_flag = None
            if gender == "男":
                gender_flag = "M"
            elif gender == "女":
                gender_flag = "F"
            else:
                print(f"性别为{gender}，不合法")
            hospital_flag = item["ct"].split("/")[-1][:2]
            style_template = self.report_template[hospital_flag][gender_flag]
            instruction = f"报告模板：<template>{style_template}</template>\\n请根据提供的全身PET和CT的影像特征，参考提供的医院特定风格的健康患者的报告模板，生成中文影像学报告的“检查所见”部分。"
        else:
            instruction = "已提供该患者的全身CT与PET影像整体信息。请根据影像信息，生成中文影像学报告的“检查所见”部分。"

        if 'chatml' in self.message_type:
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
            
            
            # sig = inspect.signature(self.tokenizer.apply_chat_template)
            
            # if "enable_thinking" in sig.parameters:
            #     template_kwargs["enable_thinking"] = self.enable_thinking

            try:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": self.enable_thinking,
                }
                text = self.tokenizer.apply_chat_template(**template_kwargs)
            except:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                }   
                text = self.tokenizer.apply_chat_template(**template_kwargs)

            text_tensor = self.tokenizer(text, max_length=self.max_seq, truncation=True, return_tensors="pt")
            text_input = text_tensor["input_ids"][0]
            
            # For Qwen, the 'question' can be the structured messages list for better inspection
            question_to_return = json.dumps(messages, ensure_ascii=False)

        elif 'human_design' in self.message_type:
            prompt = self._text_add_petct_tokens(instruction)
            text_tensor = self.tokenizer(prompt, max_length=self.max_seq, truncation=True, return_tensors="pt")
            text_input = text_tensor["input_ids"][0]
            question_to_return = prompt

        return {
            'acc_num': item['accnum'],
            'lang_x': text_input,
            'vision_x': {'ct_image': ct_tensor, 'pet_image': pet_tensor},
            'mask_x': {},
            'region2area': {},
            'question': question_to_return,
            'gt_report': gt_answer,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test PETCTDataset_Test with local paths."
    )
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--report_folder", required=True)
    parser.add_argument("--template_path", default=None)
    parser.add_argument("--use_template", action="store_true")
    args = parser.parse_args()

    dataset = PETCTDataset_Test(
        tokenizer_path=args.tokenizer_path,
        image_folder=args.image_folder,
        report_folder=args.report_folder,
        inferenced_id=[],
        template_path=args.template_path,
        use_template=args.use_template,
    )
    print(dataset[0])