"""Clean generated reports in an inference CSV.

The PETRG-3D generation prompt wraps the output with a dedicated end-of-report
marker (``【报告结束】`` by default). Before feeding the predictions to a downstream
evaluator, we trim anything after that marker and optionally dump per-patient
JSON files with the cleaned text -- the format consumed by
``extract_pet_ct_labels.py`` and by ``CN_nlg_evaluation.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def clean_report_text(text: str, marker: str = "【报告结束】") -> str:
    """Return ``text`` truncated at the first occurrence of ``marker``."""
    if not isinstance(text, str):
        return ""
    end_index = text.find(marker)
    return text[:end_index] if end_index != -1 else text


def process_report_csv(
    input_path: str,
    output_path: str,
    to_json: bool = False,
    json_dir: str | None = None,
    target_column: str = "Pred_report",
    id_column: str = "AccNum",
    marker: str = "【报告结束】",
) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise KeyError(f"column {target_column!r} not found in {input_path}; available columns: {list(df.columns)}")

    df["Cleaned_Pred_report"] = df[target_column].apply(lambda t: clean_report_text(t, marker))

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote cleaned CSV -> {output_path}")

    if to_json:
        if not json_dir:
            raise ValueError("`--to_json` requires --json_dir to be set.")
        os.makedirs(json_dir, exist_ok=True)
        if id_column not in df.columns:
            raise KeyError(f"column {id_column!r} not found; required when --to_json is set")
        for _, row in df.iterrows():
            patient_id = row[id_column]
            cleaned = row["Cleaned_Pred_report"]
            with open(os.path.join(json_dir, f"{patient_id}.json"), "w", encoding="utf-8") as f:
                json.dump({"检查所见": cleaned}, f, ensure_ascii=False)
        print(f"Wrote per-patient JSONs under {json_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean predicted reports in an inference CSV.")
    parser.add_argument("-i", "--input_path", required=True, help="Input CSV produced by src/test.py.")
    parser.add_argument("-o", "--output_path", default=None, help="Where to write the cleaned CSV (defaults to overwriting the input).")
    parser.add_argument("--to_json", action="store_true", help="Also dump per-patient JSONs containing only the cleaned 检查所见 field.")
    parser.add_argument("--json_dir", default=None, help="Directory for the per-patient JSONs; defaults to <input_dir>/json_pred.")
    parser.add_argument("--target_column", default="Pred_report")
    parser.add_argument("--id_column", default="AccNum")
    parser.add_argument("--marker", default="【报告结束】")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.input_path
    json_dir = args.json_dir or os.path.join(os.path.dirname(os.path.abspath(args.input_path)), "json_pred") if args.to_json else None

    process_report_csv(
        input_path=args.input_path,
        output_path=output_path,
        to_json=args.to_json,
        json_dir=json_dir,
        target_column=args.target_column,
        id_column=args.id_column,
        marker=args.marker,
    )


if __name__ == "__main__":
    main()
