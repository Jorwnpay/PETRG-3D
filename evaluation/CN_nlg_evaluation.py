#!/usr/bin/env python3
"""
中文NLG评估脚本（BLEU / ROUGE / CIDEr / 可选METEOR）

特性：
- 中文分词：字符级（默认）或结巴分词
- 指标：逐样本均值±方差，以及整体（corpus-level）
- 完全本地实现，与 `local_metrics` 对齐
- 命令行参数指定 CSV 路径与列名
"""

import sys
import os
import argparse
import random
import json
from typing import Callable, List, Dict, Any

import numpy as np
import pandas as pd

# Append the bundled ``local_metrics`` directory (sibling of this file) so the
# local BLEU implementation can be imported without any external dependency.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "local_metrics", "bleu"))

from nmt_bleu import compute_bleu  # local BLEU (copied from the Reg2RG repo)
from pycocoevalcap.cider.cider import Cider  # CIDEr from the pycocoevalcap PyPI package


# NLTK is optional; only needed for METEOR.
try:
    import nltk
    nltk_data_path = os.path.join(_HERE, "local_metrics", "nltk_data")
    if os.path.exists(nltk_data_path):
        nltk.data.path.append(nltk_data_path)
except Exception:
    nltk = None


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_zh_tokenizer(mode: str) -> Callable[[str], List[str]]:
    """构造中文分词函数。

    mode: 'char' | 'jieba'
    返回：callable(text) -> List[str]
    """
    mode = (mode or 'char').lower()

    if mode == 'jieba':
        try:
            import jieba  # type: ignore

            def zh_tokenize(text: str) -> List[str]:
                text = '' if text is None else str(text)
                return [tok for tok in jieba.lcut(text) if tok and tok.strip()]

            return zh_tokenize
        except Exception:
            # 回退到字符级
            pass

    # 默认：字符级分词（稳健、依赖最少）
    def zh_tokenize(text: str) -> List[str]:
        text = '' if text is None else str(text)
        return [ch for ch in text if ch and ch.strip()]

    return zh_tokenize


class LocalBleuEvaluatorCN:
    def __init__(self, zh_tokenize: Callable[[str], List[str]]):
        self.zh_tokenize = zh_tokenize

    def compute(self, predictions: List[str], references: List[List[str]], max_order: int = 4, smooth: bool = True) -> Dict[str, Any]:
        # references: [[ref_str], ...]
        tokenized_references: List[List[List[str]]] = [
            [[tok for tok in self.zh_tokenize(r)] for r in ref_group] for ref_group in references
        ]
        tokenized_predictions: List[List[str]] = [self.zh_tokenize(p) for p in predictions]

        bleu, precisions, bp, ratio, translation_length, reference_length = compute_bleu(
            reference_corpus=tokenized_references,
            translation_corpus=tokenized_predictions,
            max_order=max_order,
            smooth=smooth,
        )

        return {
            "bleu": bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "translation_length": translation_length,
            "reference_length": reference_length,
        }


class LocalRougeEvaluatorCN:
    def __init__(self, zh_tokenize: Callable[[str], List[str]]):
        # rouge_score 允许自定义 tokenizer（需实现 tokenize 方法）
        from rouge_score import rouge_scorer

        class _CNTokenizer:
            def __init__(self, fn: Callable[[str], List[str]]):
                self._fn = fn

            def tokenize(self, text: str) -> List[str]:  # noqa: D401
                return self._fn(text)

        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=False,
            tokenizer=_CNTokenizer(zh_tokenize),
        )

    def compute(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        if isinstance(references[0], str):
            references = [[references[0]]]

        rouge1_scores: List[float] = []
        rouge2_scores: List[float] = []
        rougeL_scores: List[float] = []

        for refs, pred in zip(references, predictions):
            try:
                ref_scores = []
                for ref in refs:
                    scores = self.scorer.score(ref, pred)
                    ref_scores.append(scores)

                avg_rouge1 = float(np.mean([s['rouge1'].fmeasure for s in ref_scores]))
                avg_rouge2 = float(np.mean([s['rouge2'].fmeasure for s in ref_scores]))
                avg_rougeL = float(np.mean([s['rougeL'].fmeasure for s in ref_scores]))

                rouge1_scores.append(avg_rouge1)
                rouge2_scores.append(avg_rouge2)
                rougeL_scores.append(avg_rougeL)
            except Exception:
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)

        return {
            "rouge1": float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
            "rouge2": float(np.mean(rouge2_scores)) if rouge2_scores else 0.0,
            "rougeL": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        }


class LocalMeteorEvaluatorCN:
    """简单版中文 METEOR：仅使用精确匹配（不依赖英文词干/同义词）。

    说明：nltk 的 meteor_score 对中文不完美，这里用中文分词后的 token 列表参与评分。
    若缺少 NLTK 或数据包，将自动禁用。
    """

    def __init__(self, zh_tokenize: Callable[[str], List[str]]):
        if nltk is None:
            raise RuntimeError("NLTK 不可用，无法启用 METEOR")
        try:
            from nltk.translate import meteor_score as meteor_module  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"无法导入 nltk.meteor_score: {exc}")

        self.meteor_module = meteor_module
        self.zh_tokenize = zh_tokenize

    def compute(self, predictions: List[str], references: List[List[str]], alpha: float = 0.9, beta: float = 3, gamma: float = 0.5) -> Dict[str, float]:
        if isinstance(references[0], str):
            references = [[references[0]]]

        scores: List[float] = []
        for refs, pred in zip(references, predictions):
            try:
                tokenized_refs = [self.zh_tokenize(r) for r in refs]
                tokenized_pred = self.zh_tokenize(pred)
                score = self.meteor_module.meteor_score(
                    tokenized_refs,
                    tokenized_pred,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                scores.append(float(score))
            except Exception:
                scores.append(0.0)

        return {"meteor": float(np.mean(scores)) if scores else 0.0}


def compute_cider_cn(predictions: List[str], references: List[List[str]], zh_tokenize: Callable[[str], List[str]]) -> Dict[str, float]:
    """使用中文预分词后再计算 CIDEr（通过空格拼接避免英文 PTB 误切分）。"""
    cider_scorer = Cider()

    # 预分词并空格拼接
    gts: Dict[str, List[str]] = {}
    res: Dict[str, List[str]] = {}

    for idx, (pred, ref_group) in enumerate(zip(predictions, references)):
        ref_texts = [" ".join(zh_tokenize(r)) for r in ref_group]
        pred_text = " ".join(zh_tokenize(pred))
        gts[str(idx)] = ref_texts
        res[str(idx)] = [pred_text]

    score, scores = cider_scorer.compute_score(gts, res)
    mean_cider = float(np.mean(scores)) if isinstance(scores, (list, tuple, np.ndarray)) and len(scores) else float(score)
    std_cider = float(np.std(scores)) if isinstance(scores, (list, tuple, np.ndarray)) and len(scores) else 0.0
    return {"mean": mean_cider, "std": std_cider, "cider": float(score)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中文报告生成评估（BLEU/ROUGE/CIDEr/METEOR）")
    parser.add_argument("-i", "--csv_path", type=str, required=True, help="包含参考与预测的CSV路径")
    parser.add_argument("-g", "--gt_col", type=str, default="GT_report", help="参考文本列名")
    parser.add_argument("-p", "--pred_col", type=str, default="Pred_report", help="预测文本列名")
    parser.add_argument("--tokenizer", type=str, choices=["char", "jieba"], default="jieba", help="中文分词方式")
    parser.add_argument("--metrics", nargs="+", default=["all"], choices=["bleu", "meteor", "rouge", "cider", "all"], help="需要评估的指标，可多选或 all")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--progress_every", type=int, default=100, help="进度打印步长")
    parser.add_argument("--save_csv", type=str, default="", help="结果CSV保存路径（默认与输入同目录）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    zh_tokenize = build_zh_tokenizer(args.tokenizer)

    compute_bleu_flag = ("all" in args.metrics) or ("bleu" in args.metrics)
    compute_meteor_flag = ("all" in args.metrics) or ("meteor" in args.metrics)
    compute_rouge_flag = ("all" in args.metrics) or ("rouge" in args.metrics)
    compute_cider_flag = ("all" in args.metrics) or ("cider" in args.metrics)

    # 读取数据
    df = pd.read_csv(args.csv_path)
    assert args.gt_col in df.columns, f"找不到参考列: {args.gt_col}"
    assert args.pred_col in df.columns, f"找不到预测列: {args.pred_col}"

    gt_list = df[args.gt_col].astype(str).fillna("").tolist()
    pred_list = df[args.pred_col].astype(str).fillna("").tolist()

    references: List[List[str]] = [[gt] for gt in gt_list]
    predictions: List[str] = pred_list

    print("初始化评估器（中文）...")
    bleu = LocalBleuEvaluatorCN(zh_tokenize) if compute_bleu_flag else None

    meteor = None
    if compute_meteor_flag:
        try:
            meteor = LocalMeteorEvaluatorCN(zh_tokenize)
            print("✓ METEOR 可用")
        except Exception as e:
            print(f"✗ METEOR 不可用：{e}")
            meteor = None

    rouge = None
    if compute_rouge_flag:
        try:
            rouge = LocalRougeEvaluatorCN(zh_tokenize)
            print("✓ ROUGE 可用")
        except Exception as e:
            print(f"✗ ROUGE 不可用：{e}")
            rouge = None

    if bleu is not None:
        print("✓ BLEU 可用")

    print(f"样本数: {len(predictions)}")
    progress_every = max(1, int(args.progress_every))

    # =============== 逐样本指标 ===============
    bleu_scores_per_n: Dict[int, List[float]] = {n: [] for n in range(1, 5)} if bleu is not None else {}
    meteor_scores: List[float] = []
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougeL_scores: List[float] = []

    if bleu is not None:
        print("计算 BLEU（逐样本）...")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if i % progress_every == 0:
                print(f"BLEU 进度: {i}/{len(predictions)}")
            for n in range(1, 5):
                try:
                    score_obj = bleu.compute(
                        predictions=[pred], references=[ref], max_order=n, smooth=True
                    )
                    bleu_scores_per_n[n].append(float(score_obj["bleu"]))
                except Exception as e:
                    print(f"BLEU-{n} 计算错误（样本 {i}）: {e}")
                    bleu_scores_per_n[n].append(0.0)

        for n in range(1, 5):
            mean_bleu = float(np.mean(bleu_scores_per_n[n])) if bleu_scores_per_n[n] else 0.0
            std_bleu = float(np.std(bleu_scores_per_n[n])) if bleu_scores_per_n[n] else 0.0
            print(f"BLEU-{n}: mean = {mean_bleu:.4f}, std = {std_bleu:.4f}")

    if meteor is not None:
        print("\n计算 METEOR（逐样本）...")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if i % progress_every == 0:
                print(f"METEOR 进度: {i}/{len(predictions)}")
            try:
                result = meteor.compute(predictions=[pred], references=[ref])
                meteor_scores.append(float(result['meteor']))
            except Exception as e:
                print(f"METEOR 计算错误（样本 {i}）: {e}")
                meteor_scores.append(0.0)

        mean_meteor = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        std_meteor = float(np.std(meteor_scores)) if meteor_scores else 0.0
        print(f"METEOR: mean = {mean_meteor:.4f}, std = {std_meteor:.4f}")
    else:
        mean_meteor = 0.0
        std_meteor = 0.0

    if rouge is not None:
        print("\n计算 ROUGE（逐样本）...")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if i % progress_every == 0:
                print(f"ROUGE 进度: {i}/{len(predictions)}")
            try:
                result = rouge.compute(predictions=[pred], references=[ref])
                rouge1_scores.append(float(result['rouge1']))
                rouge2_scores.append(float(result['rouge2']))
                rougeL_scores.append(float(result['rougeL']))
            except Exception as e:
                print(f"ROUGE 计算错误（样本 {i}）: {e}")
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)

        mean_rouge1 = float(np.mean(rouge1_scores)) if rouge1_scores else 0.0
        std_rouge1 = float(np.std(rouge1_scores)) if rouge1_scores else 0.0
        mean_rouge2 = float(np.mean(rouge2_scores)) if rouge2_scores else 0.0
        std_rouge2 = float(np.std(rouge2_scores)) if rouge2_scores else 0.0
        mean_rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0
        std_rougeL = float(np.std(rougeL_scores)) if rougeL_scores else 0.0

        print(f"ROUGE-1: mean = {mean_rouge1:.4f}, std = {std_rouge1:.4f}")
        print(f"ROUGE-2: mean = {mean_rouge2:.4f}, std = {std_rouge2:.4f}")
        print(f"ROUGE-L: mean = {mean_rougeL:.4f}, std = {std_rougeL:.4f}")
    else:
        mean_rouge1 = mean_rouge2 = mean_rougeL = 0.0
        std_rouge1 = std_rouge2 = std_rougeL = 0.0

    print("\n计算 CIDEr（逐样本分布与整体）...")
    try:
        cider_result = compute_cider_cn(predictions, references, zh_tokenize) if compute_cider_flag else {"mean": 0.0, "std": 0.0, "cider": 0.0}
        mean_cider = float(cider_result.get("mean", 0.0))
        std_cider = float(cider_result.get("std", 0.0))
        print(f"CIDEr: mean = {mean_cider:.4f}, std = {std_cider:.4f}")
    except Exception as e:
        print(f"CIDEr 计算失败：{e}")
        mean_cider = 0.0
        std_cider = 0.0

    # =============== 整体指标（corpus-level） ===============
    print("\n=== 整体评估（corpus-level） ===")
    if bleu is not None:
        for n in range(1, 5):
            try:
                overall = bleu.compute(predictions=predictions, references=references, max_order=n, smooth=True)
                print(f"Overall BLEU-{n}: {overall['bleu']:.4f}")
            except Exception as e:
                print(f"Overall BLEU-{n} 计算失败: {e}")

    if meteor is not None:
        try:
            overall_meteor = meteor.compute(predictions=predictions, references=references)
            print(f"Overall METEOR: {overall_meteor['meteor']:.4f}")
        except Exception as e:
            print(f"Overall METEOR 计算失败: {e}")
    else:
        print("Overall METEOR: 不可用")

    if rouge is not None:
        try:
            overall_rouge = rouge.compute(predictions=predictions, references=references)
            print(f"Overall ROUGE-1: {overall_rouge['rouge1']:.4f}")
            print(f"Overall ROUGE-2: {overall_rouge['rouge2']:.4f}")
            print(f"Overall ROUGE-L: {overall_rouge['rougeL']:.4f}")
        except Exception as e:
            print(f"Overall ROUGE 计算失败: {e}")
    else:
        print("Overall ROUGE: 不可用")

    # =============== 保存结果 ===============
    print("\n=== 保存结果 ===")
    results: Dict[str, str] = {}

    # 逐样本均值（转百分比字符串）
    if bleu is not None:
        for n in range(1, 5):
            mean_bleu = float(np.mean(bleu_scores_per_n[n])) if bleu_scores_per_n[n] else 0.0
            results[f"BLEU-{n}"] = f"{mean_bleu * 100:.2f}"

    results["METEOR"] = f"{float(np.mean(meteor_scores)) * 100:.2f}" if meteor_scores else f"{0.0:.2f}"
    results["ROUGE-1"] = f"{mean_rouge1 * 100:.2f}"
    results["ROUGE-2"] = f"{mean_rouge2 * 100:.2f}"
    results["ROUGE-L"] = f"{mean_rougeL * 100:.2f}"
    results["CIDEr"] = f"{mean_cider * 100:.2f}"

    # 整体 BLEU/METEOR/ROUGE（若可用）
    if bleu is not None:
        for n in range(1, 5):
            try:
                overall = bleu.compute(predictions=predictions, references=references, max_order=n, smooth=True)
                results[f"Overall_BLEU-{n}"] = f"{overall['bleu'] * 100:.2f}"
            except Exception:
                results[f"Overall_BLEU-{n}"] = "N/A"

    if meteor is not None:
        try:
            overall_meteor = meteor.compute(predictions=predictions, references=references)
            results["Overall_METEOR"] = f"{overall_meteor['meteor'] * 100:.2f}"
        except Exception:
            results["Overall_METEOR"] = "N/A"
    else:
        results["Overall_METEOR"] = "N/A"

    if rouge is not None:
        try:
            overall_rouge = rouge.compute(predictions=predictions, references=references)
            results["Overall_ROUGE-1"] = f"{overall_rouge['rouge1'] * 100:.2f}"
            results["Overall_ROUGE-2"] = f"{overall_rouge['rouge2'] * 100:.2f}"
            results["Overall_ROUGE-L"] = f"{overall_rouge['rougeL'] * 100:.2f}"
        except Exception:
            results["Overall_ROUGE-1"] = results["Overall_ROUGE-2"] = results["Overall_ROUGE-L"] = "N/A"
    else:
        results["Overall_ROUGE-1"] = results["Overall_ROUGE-2"] = results["Overall_ROUGE-L"] = "N/A"

    # 保存到 CSV
    save_dir = os.path.dirname(args.csv_path)
    save_path = args.save_csv.strip() or os.path.join(save_dir, "complete_CN_nlg_evaluation_results.csv")
    pd.DataFrame(results, index=[0]).to_csv(save_path, index=False)
    print(f"结果已保存到: {save_path}")
    print("\n=== 评估完成 ===")


if __name__ == "__main__":
    main()


