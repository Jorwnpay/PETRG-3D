import os
import json
import csv
from sklearn.metrics import f1_score, classification_report
from collections import defaultdict

# --- 1. 配置和常量定义 (与 cal_petrg_diag_ces.py 保持一致) ---

# 标签的有效范围 (映射回 1-based string 供后续逻辑使用)
VALID_PET_STATUSES = {'1', '2', '3', '4', '5'}
VALID_CT_STATUSES = {str(i) for i in range(1, 9)}

CT_NAMES = {
    "1": "淋巴结肿大",
    "2": "局灶性病变",
    "3": "肺实质异常",
    "4": "壁/膜增厚",
    "5": "钙化",
    "6": "骨骼病变",
    "7": "其他异常",
    "8": "正常"
}

# 方案一：所有类别的标签
PET_LABELS_ALL = ['1', '2', '3', '4', '5']
# CT 所有类别列表 (8 类)
CT_LABELS_ALL = ['1', '2', '3', '4', '5', '6', '7', '8']

# 方案三：仅异常类别的标签
PET_LABELS_ABNORMAL = ['1', '2', '3', '4']
# CT 异常类别列表 (7 类)
CT_LABELS_ABNORMAL = ['1', '2', '3', '4', '5', '6', '7']

# 用于生成可读报告的名称映射
PET_NAMES = {
    "1": "高度异常摄取", "2": "轻度/可疑异常摄取", "3": "生理性/背景摄取", "4": "摄取缺损/减低", "5": "正常"
}


# --- 2. 需求一：数据验证 ---

def validate_data_directory(directory_path: str) -> list:
    """
    遍历指定目录中的所有JSON文件，验证pet_status和ct_status。
    返回一个包含所有错误信息的列表。
    """
    errors = []
    if not os.path.exists(directory_path):
        errors.append(f"错误：目录不存在: {directory_path}")
        return errors

    for filename in os.listdir(directory_path):
        if not filename.endswith('.json'):
            continue
        
        patient_id = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            errors.append(f"文件解析失败: {file_path} (不是有效的JSON)")
            continue
        except Exception as e:
            errors.append(f"无法读取文件: {file_path}, 错误: {e}")
            continue

        for location, values in data.items():
            pet_status = values.get('pet_status')
            ct_status = values.get('ct_status')

            if pet_status is None:
                errors.append(f"[{patient_id} - {location}]: 'pet_status' 键缺失")
            elif pet_status not in VALID_PET_STATUSES:
                errors.append(f"[{patient_id} - {location}]: 'pet_status' 值无效: '{pet_status}'")

            if ct_status is None:
                errors.append(f"[{patient_id} - {location}]: 'ct_status' 键缺失")
            elif ct_status not in VALID_CT_STATUSES: # 验证时仍检查 1-10
                errors.append(f"[{patient_id} - {location}]: 'ct_status' 值无效: '{ct_status}'")
                
    return errors

def run_validation(labels_dir: str, preds_dir: str):
    """执行数据验证并打印报告"""
    print("--- 1. 开始数据验证 ---")
    
    label_errors = validate_data_directory(labels_dir)
    pred_errors = validate_data_directory(preds_dir)
    
    if not label_errors and not pred_errors:
        print("✅ 所有数据均有效！")
    else:
        print(f"\n❌ 在 'json_labels' 中发现 {len(label_errors)} 个错误:")
        for error in label_errors:
            print(f"  - {error}")
            
        print(f"\n❌ 在 'json_preds' 中发现 {len(pred_errors)} 个错误:")
        for error in pred_errors:
            print(f"  - {error}")
    
    print("--- 验证结束 ---\n")
    return not label_errors and not pred_errors # 返回数据是否完全有效


# --- 3. 需求二：指标评估 ---

def collect_labels_and_preds(labels_dir: str, preds_dir: str):
    """
    收集所有匹配的、有效的真实标签和预测标签。
    """
    y_true_pet, y_pred_pet = [], []
    y_true_ct, y_pred_ct = [], []
    
    missing_preds = 0
    missing_locations = 0

    if not os.path.exists(labels_dir) or not os.path.exists(preds_dir):
        print(f"错误：无法找到目录 {labels_dir} 或 {preds_dir}")
        return None

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
    
    for filename in label_files:
        patient_id = filename.split('.')[0]
        label_file_path = os.path.join(labels_dir, filename)
        pred_file_path = os.path.join(preds_dir, filename)

        if not os.path.exists(pred_file_path):
            missing_preds += 1
            continue
            
        try:
            with open(label_file_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            with open(pred_file_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
        except Exception:
            continue

        for location, label_values in label_data.items():
            
            if location not in pred_data:
                missing_locations += 1
                continue
            
            pred_values = pred_data[location]

            label_pet = label_values.get('pet_status')
            label_ct = label_values.get('ct_status')
            pred_pet = pred_values.get('pet_status')
            pred_ct = pred_values.get('ct_status')

            pred_pet = str(pred_pet) if isinstance(pred_pet, int) else pred_pet
            pred_ct = str(pred_ct) if isinstance(pred_ct, int) else pred_ct

            # 验证 *原始* 标签是否有效
            if label_pet not in VALID_PET_STATUSES or \
               label_ct not in VALID_CT_STATUSES or \
               pred_pet not in VALID_PET_STATUSES or \
               pred_ct not in VALID_CT_STATUSES:
                continue

            # 添加到列表中
            y_true_pet.append(label_pet)
            y_pred_pet.append(pred_pet)
            
            y_true_ct.append(label_ct)
            y_pred_ct.append(pred_ct)

    if missing_preds > 0:
        print(f"警告：有 {missing_preds} 个真实标签文件在 'json_preds' 中找不到对应的预测文件。")
    if missing_locations > 0:
        print(f"警告：在预测文件中累计缺失了 {missing_locations} 个解剖位置的标签。")
        
    if not y_true_pet:
        print("错误：未能收集到任何有效的标签对。请检查您的文件。")
        return None

    print(f"成功收集到 {len(y_true_pet)} 个有效的 (位置-标签) 对用于评估。")
    return y_true_pet, y_pred_pet, y_true_ct, y_pred_ct


def calculate_and_save_metrics(summary_file_path, detail_file_path, y_true_pet, y_pred_pet, y_true_ct, y_pred_ct):
    """
    (需求3) 计算指标，打印总结，并保存到 CSV 文件。
    """
    print("\n--- 2. 开始评估指标 ---")

    # --- 方案一：Macro-F1 (所有类别) ---
    print("\n### 方案一：Macro-F1 (所有类别) ###")
    
    pet_macro_f1_all = f1_score(
        y_true_pet, y_pred_pet, 
        labels=PET_LABELS_ALL, 
        average='macro', 
        zero_division=0
    ) * 100
    # (需求2) 使用新的 CT_LABELS_ALL (8类)
    ct_macro_f1_all = f1_score(
        y_true_ct, y_pred_ct, 
        labels=CT_LABELS_ALL, 
        average='macro', 
        zero_division=0
    ) * 100
    
    print(f"  PET (5类) Macro-F1: {pet_macro_f1_all:.2f}")
    print(f"  CT (8类, 重映射) Macro-F1: {ct_macro_f1_all:.2f}")

    # --- 方案三：Macro-F1 (仅异常类别) ---
    print("\n### 方案三：Macro-F1 (仅异常类别) ###")
    
    pet_macro_f1_abnormal = f1_score(
        y_true_pet, y_pred_pet, 
        labels=PET_LABELS_ABNORMAL, 
        average='macro', 
        zero_division=0
    ) * 100
    # (需求2) 使用新的 CT_LABELS_ABNORMAL (7类)
    ct_macro_f1_abnormal = f1_score(
        y_true_ct, y_pred_ct, 
        labels=CT_LABELS_ABNORMAL, 
        average='macro', 
        zero_division=0
    ) * 100
    
    print(f"  PET (异常4类) Macro-F1: {pet_macro_f1_abnormal:.2f}")
    print(f"  CT (异常7类, 重映射) Macro-F1: {ct_macro_f1_abnormal:.2f}")

    # --- (需求3) 保存宏平均结果到 CSV ---
    results_summary_file = summary_file_path
    print(f"\n--- 3. 保存宏平均F1分数到 {results_summary_file} ---")
    try:
        with open(results_summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "PET-Macro-F1-All", "CT-Macro-F1-All",
                "PET-Macro-F1-Abnormal", "CT-Macro-F1-Abnormal"
            ])
            writer.writerow([
                f"{pet_macro_f1_all:.2f}", f"{ct_macro_f1_all:.2f}",
                f"{pet_macro_f1_abnormal:.2f}", f"{ct_macro_f1_abnormal:.2f}"
            ])
        print(f"✅ 成功保存到 {results_summary_file}")
    except Exception as e:
        print(f"❌ 保存 {results_summary_file} 失败: {e}")

    # --- (需求3) 保存详细分类报告到 CSV ---
    results_detail_file = detail_file_path
    print(f"\n--- 4. 保存详细报告到 {results_detail_file} ---")

    # 获取字典格式的报告
    pet_report_dict = classification_report(
        y_true_pet, y_pred_pet,
        labels=PET_LABELS_ALL,
        target_names=[PET_NAMES[l] for l in PET_LABELS_ALL],
        zero_division=0,
        output_dict=True
    )
    # (需求2) 使用重映射后的新标签和新名称
    ct_report_dict = classification_report(
        y_true_ct, y_pred_ct,
        labels=CT_LABELS_ALL, 
        target_names=[CT_NAMES[l] for l in CT_LABELS_ALL],
        zero_division=0,
        output_dict=True
    )
    
    try:
        with open(results_detail_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # --- PET 报告 ---
            writer.writerow([f"PET-labels ({len(PET_LABELS_ALL)}类)", "precision", "recall", "f1-score", "support"])
            for label_name, metrics in pet_report_dict.items():
                if label_name == "accuracy":
                    # accuracy 是一个浮点数
                    total_support = int(pet_report_dict['macro avg']['support'])
                    writer.writerow([label_name, "", "", f"{metrics:.4f}", total_support])
                elif isinstance(metrics, dict):
                    # 其他行是字典
                    writer.writerow([
                        label_name,
                        f"{metrics['precision']*100:.2f}",
                        f"{metrics['recall']*100:.2f}",
                        f"{metrics['f1-score']*100:.2f}",
                        int(metrics['support'])
                    ])
            
            writer.writerow([]) # 空行分隔

            # --- CT 报告 ---
            writer.writerow([f"CT-labels ({len(CT_LABELS_ALL)}类, 已重映射)", "precision", "recall", "f1-score", "support"])
            for label_name, metrics in ct_report_dict.items():
                if label_name == "accuracy":
                    total_support = int(ct_report_dict['macro avg']['support'])
                    writer.writerow([label_name, "", "", f"{metrics:.4f}", total_support])
                elif isinstance(metrics, dict):
                    writer.writerow([
                        label_name,
                        f"{metrics['precision']*100:.2f}",
                        f"{metrics['recall']*100:.2f}",
                        f"{metrics['f1-score']*100:.2f}",
                        int(metrics['support'])
                    ])
                    
        print(f"✅ 成功保存到 {results_detail_file}")
    except Exception as e:
        print(f"❌ 保存 {results_detail_file} 失败: {e}")

    print("\n--- 评估结束 ---")


# --- 4. 主执行函数 ---

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="PET/CT clinical efficacy (Macro-F1) evaluation from LLM-extracted label JSONs."
    )
    parser.add_argument("--labels_dir", required=True, help="Directory of ground-truth per-patient label JSONs.")
    parser.add_argument("--preds_dir", required=True, help="Directory of predicted per-patient label JSONs (e.g. from extract_pet_ct_labels.py).")
    parser.add_argument("--save_dir", default=None, help="Directory to write the summary / detail CSVs (default: parent of --preds_dir).")
    parser.add_argument("--summary_name", default="PETRG-CES-Results.csv")
    parser.add_argument("--detail_name", default="PETRG-CES-Detail-Results.csv")
    return parser.parse_args()


def main():
    args = _parse_args()
    save_dir = args.save_dir or os.path.dirname(os.path.abspath(args.preds_dir))
    os.makedirs(save_dir, exist_ok=True)
    summary_file_path = os.path.join(save_dir, args.summary_name)
    detail_file_path = os.path.join(save_dir, args.detail_name)

    run_validation(args.labels_dir, args.preds_dir)

    print("--- Collecting label/prediction pairs ... ---")
    data_collections = collect_labels_and_preds(args.labels_dir, args.preds_dir)

    if data_collections:
        y_true_pet, y_pred_pet, y_true_ct, y_pred_ct = data_collections
        calculate_and_save_metrics(summary_file_path, detail_file_path, y_true_pet, y_pred_pet, y_true_ct, y_pred_ct)
    else:
        print("\nEvaluation aborted: no valid label/prediction pairs collected.")


if __name__ == "__main__":
    main()