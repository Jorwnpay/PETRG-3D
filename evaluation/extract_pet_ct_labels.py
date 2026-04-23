"""Extract 24-region PET / CT status labels from Chinese PETCT reports.

For every JSON report in ``--reports_dir``, we prompt an external LLM with a
structured instruction (see :data:`SYSTEM_PROMPT`) and store the resulting JSON
(one per patient) under ``--save_dir``. The LLM is accessed through any
OpenAI-compatible Chat Completions endpoint, so both the API key, base URL and
model name can be chosen freely via the CLI (or the matching environment
variables).

Example::

    export LLM_API_KEY=sk-...
    python evaluation/extract_pet_ct_labels.py \\
        --reports_dir ./results/PETRG-3D-qwen3-8B/json_pred \\
        --save_dir    ./results/PETRG-3D-qwen3-8B/labels_llm \\
        --base_url    https://api.openai.com/v1 \\
        --model       gpt-4o-mini

The output JSON per patient follows the schema consumed by
``evaluation/cal_petrg_ces.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Ordered list of the 24 regions that a PET/CT report is decomposed into.
ORGANS: List[str] = [
    "脑、颅骨与脑膜",
    "眼眶、鼻腔与副鼻窦",
    "咽部间隙、扁桃体与喉",
    "甲状腺与主要唾液腺（腮腺、下颌下腺）",
    "颈部淋巴结",
    "肺与胸膜",
    "纵隔与肺门（含淋巴结）",
    "心脏与心包",
    "腋窝与胸壁",
    "乳腺",
    "肝脏",
    "胆囊与胆道",
    "脾脏",
    "胰腺",
    "肾脏",
    "肾上腺",
    "胃肠道（食管、胃、肠）",
    "腹膜后间隙（含淋巴结）",
    "腹膜、肠系膜与网膜",
    "盆腔脏器（膀胱、子宫/附件 或 前列腺/精囊腺）",
    "盆腔与腹股沟淋巴结",
    "脊柱",
    "骨盆与四肢骨",
    "肌肉与皮下组织",
]


SYSTEM_PROMPT = """你是一名专业的核医学影像医生，你的任务是从给定的PET/CT“检查所见”报告文本中，提取24个关键器官的放射性示踪剂摄取状态和CT结构/密度状态。

一、任务1：提取PET摄取状态
1.状态分类: 针对每个器官，将其摄取状态归类为以下字典中的5个序号之一：
{
"1": "高度异常摄取：明确描述了摄取增高、示踪剂浓聚等词语，对应Deauville Score 5，明确提示恶性病变。",
"2": "轻度/可疑异常摄取：对应Deauville Score 3（在某些情况下可疑）或轻度高于肝脏。",
"3": "生理性/背景摄取：明确提示生理性摄取。"
"4": "摄取缺损/减低：明确描述了放射性减低、放射性缺损等词语。",
"5": "正常：描述了未见示踪剂浓集、未见异常摄取、放射性分布均匀等，或根据归一化原则推断为正常。"
}
2.SUV值描述: 如果报告中描述了该结构的SUV信息，请在`suv`字段中摘录SUVmax或SUVmean的具体值，例如"suv": "SUVmax: 4.5"；否则置为空字符串""。注意，如果某个区域存在多个SUV信息，请选择其中最大的一个作为SUVmax。

二、任务2：提取CT结构/密度状态
1.状态分类: 针对每个器官，将其CT发现归类为以下字典中的8个序号之一：
{
"1": "淋巴结肿大：指淋巴结在形态或大小上的异常。",
"2": "局灶性病变：包括结节、肿块。",
"3": "肺实质异常：包括磨玻璃影、索条影、斑片影。",
"4": "壁/膜增厚：指一个正常解剖结构（通常是壁、膜或间质）的厚度增加。",
"5": "钙化：指组织内出现的高密度钙盐沉积。",
"6": "骨骼病变：如成骨性、溶骨性病变。",
"7": "其他异常：所有不属于上述异常的其余异常。",
"8": "正常：未描述或明确描述无异常，或根据归一化原则推断为正常。"
}
2.提取描述: 如果状态为前7种“异常”描述，请在`ct_description`字段中简要摘录异常的具体描述；否则置为空字符串""。

三、归一化规则:
（1）默认正常: 如果报告中完全没有提及某个器官，则该器官的状态应被归类为“正常”。
（2）层级正常: 如果一个大的类别（如“肺与胸膜”）被描述为正常，其下的所有子区域都应被视为“正常”。
（3）隐含正常: 如果只提到了某个器官的局部异常（如“颈部2区摄取增高”），那么该解剖区域的其他未提及部分应被视为“正常”，此时整个器官的状态应根据最显著的发现来判断（此例中为“摄取增高”）。

四、输出格式:
请严格按照以下JSON格式输出结果，不要添加任何额外的解释。注意输出的pet_status和ct_status字段的格式应严格为单个数字。

JSON模板:
{
    "脑、颅骨与脑膜": {"pet_status": "...", "suv": "...", "ct_status": "...", "ct_description": "..."},
    ...（其余23个器官同格式）
}
"""


def _clean_text(text: str) -> str:
    if "`" in text:
        text = text.replace("`", "")
    return text.strip()


def call_llm(
    client: OpenAI,
    model: str,
    findings_text: str,
    max_retries: int = 3,
    retry_delay: int = 30,
) -> Optional[Dict[str, Any]]:
    """Call an OpenAI-compatible chat completion endpoint and parse its JSON output."""

    user_prompt = f"**待处理的报告文本:**\n{findings_text}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content.strip()

            json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"First 500 chars of payload:\n{json_str[:500]}")
                return None

        except Exception as e:
            print(f"LLM call failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)

    print(f"Giving up after {max_retries} retries.")
    return None


def _process_single_report(
    file_path: str,
    save_dir: str,
    client: OpenAI,
    model: str,
) -> str:
    patient_id = os.path.basename(file_path)[:-5]
    output_path = os.path.join(save_dir, f"{patient_id}.json")

    if os.path.exists(output_path):
        return f"[skip] already processed: {patient_id}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        return f"[error] read {file_path}: {e}"

    findings = _clean_text(str(report.get("检查所见", "")))
    if not findings:
        return f"[warn] empty findings: {patient_id}"

    result = call_llm(client, model, findings)
    if result is None:
        return f"[error] model call/parsing failed: {patient_id}"

    # Fill missing organs with a "normal" default so downstream scripts can
    # assume every report has all 24 entries.
    final_result: Dict[str, Dict[str, str]] = {}
    for organ in ORGANS:
        item = result.get(organ, {}) or {}
        final_result[organ] = {
            "pet_status": str(item.get("pet_status", "5")),
            "suv": str(item.get("suv", "")),
            "ct_status": str(item.get("ct_status", "8")),
            "ct_description": str(item.get("ct_description", "")),
        }

    os.makedirs(save_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    return f"[ok] {patient_id}"


def process_reports(
    reports_dir: str,
    save_dir: str,
    client: OpenAI,
    model: str,
    max_workers: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    files = [
        os.path.join(reports_dir, name)
        for name in sorted(os.listdir(reports_dir))
        if name.endswith(".json")
    ]
    print(f"Found {len(files)} reports in {reports_dir}; using {max_workers} workers.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_report, fp, save_dir, client, model): fp
            for fp in files
        }
        iterable = as_completed(futures)
        if TQDM_AVAILABLE:
            iterable = tqdm(iterable, total=len(files), desc="Extracting labels")

        results: List[str] = []
        for fut in iterable:
            try:
                results.append(fut.result())
            except Exception as exc:
                fp = futures[fut]
                results.append(f"[exception] {fp}: {exc}")

    print("\n--- Summary ---")
    ok = sum(1 for r in results if r.startswith("[ok]"))
    skip = sum(1 for r in results if r.startswith("[skip]"))
    bad = len(files) - ok - skip
    print(f"total={len(files)}  ok={ok}  skipped={skip}  failed={bad}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 24-region PET/CT labels using an LLM.")
    parser.add_argument("-i", "--reports_dir", required=True, help="Directory of input JSON reports (one per patient).")
    parser.add_argument("-o", "--save_dir", required=True, help="Directory to write extracted labels to.")
    parser.add_argument("-w", "--workers", type=int, default=5, help="Number of concurrent worker threads.")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"), help="OpenAI-compatible model name.")
    parser.add_argument("--base_url", default=os.environ.get("LLM_BASE_URL"), help="OpenAI-compatible chat completion endpoint URL.")
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY"), help="OpenAI-compatible API key (recommend: set the LLM_API_KEY env variable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.reports_dir):
        raise FileNotFoundError(f"reports_dir does not exist: {args.reports_dir}")
    if not args.api_key:
        raise ValueError("No API key provided. Pass --api_key or set the LLM_API_KEY environment variable.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=args.api_key)
    process_reports(args.reports_dir, args.save_dir, client, args.model, args.workers)


if __name__ == "__main__":
    main()
