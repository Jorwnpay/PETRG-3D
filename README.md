# PETRG-3D: PET/CT Report Generation with a 3D Vision-Language Model

> Official implementation of *"PETRG-3D: PET/CT Report Generation with a 3D Vision-Language Model"* (CVPR 2026 Findings).

PETRG-3D is a multimodal LLM that consumes paired 3D CT and PET volumes and generates a structured Chinese radiology report. The architecture couples a shared 3D ViT visual encoder (initialized from RadFM's ViT-3D), two `PerceiverResampler + FC` projection heads (one per modality) and an LLM decoder with a LoRA adapter. We support six interchangeable Chinese-chat LLM backbones out of the box.

---

## 1. Repository layout

```
PETRG-3D-CVPR/
├── configs/
│   ├── train/              # training configs (one per LLM backbone)
│   └── test/               # inference configs
├── ds_configs/
│   └── deepspeed_zero2.json
├── evaluation/
│   ├── CN_nlg_evaluation.py        # BLEU / ROUGE / CIDEr / METEOR (Chinese)
│   ├── clean_csv_report.py         # strip the end-of-report marker and dump per-patient JSONs
│   ├── extract_pet_ct_labels.py    # LLM-based 24-region PET/CT label extraction
│   ├── cal_petrg_ces.py            # Macro-F1 clinical efficacy scores from the labels above
│   └── local_metrics/bleu/         # vendored BLEU implementation (from the Reg2RG repo)
├── results/
│   └── PETRG-3D-qwen3-8B/          # example paper results (predictions + metric CSVs)
├── scripts/
│   ├── train.sh / test.sh          # main launch wrappers
│   └── sinfer.sh / srun.sh         # example SLURM wrappers
└── src/
    ├── args/train/default.py
    ├── args/test/default.py
    ├── Dataset/petct_dataset_{train,test}.py
    ├── Model/PETRG_3D.py           # main model (class PETRG3D)
    ├── Model/my_embedding_layer.py # multimodal embedding layer
    ├── Model/vit_3d.py             # RadFM ViT-3D backbone
    ├── train.py
    └── test.py
```

---

## 2. Installation

```bash
conda create -n petrg3d python=3.10 -y
conda activate petrg3d
pip install -r requirements.txt
```

The code is tested with PyTorch 2.8.0 + CUDA 12.1 on 2× A800 (80G). Multi-GPU training uses `torchrun` + DeepSpeed ZeRO-2 (see `ds_configs/deepspeed_zero2.json`).

### Pretrained dependencies

1. Download RadFM's 3D ViT checkpoint and place it somewhere accessible:
  - `RadFM_vit3d.pth` — [RadFM on HuggingFace](https://huggingface.co/chaoyi-wu/RadFM).
2. Download the Chinese-chat LLM of your choice. The released configs cover:
  - [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (main results), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414), [Gemma-2-9B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Gemma-2-9B-Chinese-Chat), [Mistral-7B-v0.3-Chinese-Chat](https://huggingface.co/shenzhi-wang/Mistral-7B-v0.3-Chinese-Chat), [Llama2-Chinese-7b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat).

---

## 3. Dataset: `AutoPet-RG-lym`

Our **AutoPET-Lym-135** dataset is available at：[https://huggingface.co/datasets/jwppku/AutoPET-RG-Lym](https://huggingface.co/datasets/jwppku/AutoPET-RG-Lym)

### Expected directory layout

```
petrg_lym (your PETRG dataset for model training)/
├── train/
│   ├── images/    # <id>_0000.nii.gz (CT) and <id>_0001.nii.gz (PET)
│   └── reports/          # <id>.json with at least a "检查所见" field
├── valid/
│   ├── images/
│   ├── reports/
│   └── labels/           # ground-truth 24-region label JSONs (for the CES metric)
└── template.json         # report template (see `configs/train/*.sh`)
```

---

## 4. Training

All training launches go through `scripts/train.sh <config_name>`, which `source`'s `configs/train/<config_name>.sh` and then calls `torchrun`.

Before running, edit `configs/train/qwen3_joint_template.sh` (or the equivalent for another backbone) to point the placeholder paths at your local copies **or** export the corresponding environment variables, e.g.:

```bash
export LANG_ENCODER_PATH=/path/to/Qwen3-8B
export PRETRAINED_VISUAL_ENCODER=/path/to/RadFM_vit3d.pth
export PETRG_LYM_TRAIN_IMAGES=/path/to/petrg_lym/train/images_mv_leg
export PETRG_LYM_TRAIN_REPORTS=/path/to/petrg_lym/train/reports
export PETRG_LYM_TEMPLATE=/path/to/petrg_lym/template.json
export PETRG3D_OUTPUT_DIR=./outputs

cd scripts
bash train.sh qwen3_joint_template
```

A SLURM launcher template is provided as `scripts/srun.sh` — customize the partition, `--gres` and the conda activation call for your cluster.

### Available configs


| Config                   | LLM backbone                 | Default `use_fast` |
| ------------------------ | ---------------------------- | ------------------ |
| `qwen3_joint_template`   | Qwen3-8B *(main result)*     | `False`            |
| `qwen25_joint_template`  | Qwen2.5-7B-Instruct          | `False`            |
| `gemma_joint_template`   | Gemma-2-9B-Chinese-Chat      | `False`            |
| `glm_joint_template`     | GLM-4-9B-0414                | `True` (required)  |
| `llama2_joint_template`  | Llama2-Chinese-7b-Chat       | `False`            |
| `mistral_joint_template` | Mistral-7B-v0.3-Chinese-Chat | `False`            |


---

## 5. Inference

```bash
export LANG_ENCODER_PATH=/path/to/Qwen3-8B
export PRETRAINED_VISUAL_ENCODER=/path/to/RadFM_vit3d.pth
export PETRG3D_CKPT=/path/to/petrg3d-qwen3-8B/model.safetensors
export AUTOPET_LYM_IMAGES=/path/to/autopet_rg_lym/images_mv_leg
export AUTOPET_LYM_REPORTS=/path/to/autopet_rg_lym/reports
export AUTOPET_LYM_TEMPLATE=/path/to/autopet_rg_lym/template.json
export PETRG3D_RESULT=./results/PETRG-3D-qwen3-8B/valid_reports.csv

cd scripts
bash test.sh qwen3_template_autopet
```

Inference loops over the CSV at `--result_path` so it can be resumed safely. The generated predictions are written to the `Pred_report` column of that CSV.

> **Note on reproducibility**: generation uses `do_sample=True` with a positive temperature, so the token-level outputs are not bit-wise reproducible even with a fixed random seed. The reported numbers are averaged over the whole validation set, which is stable within noise.

### Released checkpoints

The paper's main checkpoints is released at: [https://huggingface.co/jwppku/PETRG-3D-qwen3-8B](https://huggingface.co/jwppku/PETRG-3D-qwen3-8B).

---

## 6. Evaluation

**Clean the generated reports** and dump per-patient JSONs that only contain the 检查所见 field:
```bash
python evaluation/clean_csv_report.py \
    -i results/PETRG-3D-qwen3-8B/valid_reports.csv \
    --to_json \
```

### 6.1 Chinese NLG metrics (BLEU / ROUGE / CIDEr / METEOR)

```bash
python evaluation/CN_nlg_evaluation.py \
    -i results/PETRG-3D-qwen3-8B/valid_reports.csv \
    -g GT_report
    -p Cleaned_Pred_report
```

The bundled `evaluation/local_metrics/bleu/nmt_bleu.py` provides a local BLEU implementation; CIDEr is taken from `pycocoevalcap` and ROUGE from `rouge-score` (both installed via `requirements.txt`). METEOR requires an additional NLTK data download (see the script's error message) and is optional.

### 6.2 Clinical efficacy scores (CES, Macro-F1 over PET / CT status)

CES evaluates whether the predicted report agrees with the ground truth on 24 anatomical regions along two axes (PET tracer uptake and CT finding type). Concretely:
1. **Extract 24-region labels** from the cleaned reports using an OpenAI-compatible LLM (we used Gemini-class models for the paper; any JSON-emitting instruct model works): 
  ```bash
  export LLM_API_KEY="your-api-key"
  python evaluation/extract_pet_ct_labels.py \
      -i results/PETRG-3D-qwen3-8B/json_pred \
      -o results/PETRG-3D-qwen3-8B/labels_llm \
      --model "gemini-3-pro" \
      --base_url "https://generativelanguage.googleapis.com/v1beta/openai/" \
      -w 10
  ```
2. **Compute Macro-F1** against the held-out ground-truth labels:
  ```bash
   python evaluation/cal_petrg_ces.py \
       --labels_dir /path/to/petrg_lym/valid/labels \
       --preds_dir  results/PETRG-3D-qwen3-8B/labels_llm \
       --save_dir   results/PETRG-3D-qwen3-8B
  ```

`results/PETRG-3D-qwen3-8B/` ships with the paper-quality CSVs produced by the steps above for the `valid` and `autopet` splits (e.g. `complete_CN_nlg_evaluation_results.csv` and `PETRG-CES-*Results*.csv`), which you can use as reference numbers.

---

## 7. Acknowledgements

PETRG-3D is built on top of several excellent open-source projects:

- [RadFM](https://github.com/chaoyi-wu/RadFM) — 3D ViT visual encoder.
- [Reg2RG](https://github.com/chenzhixuan/Reg2RG) — the codebase this repo is forked from. The BLEU implementation in `evaluation/local_metrics/bleu/` is copied verbatim from that repository.
- The `PerceiverResampler` implementation is adapted from [lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch).

---

## 8. Citation

```bibtex
@article{jiao2025vision,
  title={Vision-Language Models for Automated 3D PET/CT Report Generation},
  author={Jiao, Wenpei and Shang, Kun and Li, Hui and Yan, Ke and Zhang, Jiajin and Yang, Guangjie and Guo, Lijuan and Wan, Yan and Yang, Xing and Jin, Dakai and others},
  journal={arXiv preprint arXiv:2511.20145},
  year={2025}
}
```

## 9. License

This project is released under the Apache-2.0 License (see `LICENSE`). 