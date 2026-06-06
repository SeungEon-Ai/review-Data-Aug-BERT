###############################################################################
#
#   리뷰 데이터 — 결과 통합 + 통계 검정
#
#   3개 실험 결과 JSON을 합쳐서:
#     - 4가지 조건 비교표
#     - Paired t-test + Wilcoxon + Cohen's d
#     - 최종 종합 결과 저장
#
#   선행: review_01_base.py, review_02_eda.py, review_03_gpt.py 실행 완료
#
###############################################################################

from google.colab import drive
drive.mount("/content/drive")

import json
import numpy as np
from scipy import stats

DRIVE_FOLDER = "/content/drive/MyDrive/thesis"

# ===================== 결과 로드 =====================
print("=" * 60)
print("  결과 로드")
print("=" * 60)

with open(f"{DRIVE_FOLDER}/review_result_base.json") as f:
    base_data = json.load(f)
with open(f"{DRIVE_FOLDER}/review_result_eda.json") as f:
    eda_data = json.load(f)
with open(f"{DRIVE_FOLDER}/review_result_gpt.json") as f:
    gpt_data = json.load(f)

# fold별 F1 추출
results = {
    "base": base_data["cv_results"]["base"]["folds"],
    "class_weight": base_data["cv_results"]["class_weight"]["folds"],
    "eda": eda_data["cv_results"]["folds"],
    "gpt": gpt_data["cv_results"]["folds"],
}

test_results = {
    "base": base_data["test_results"]["base"],
    "class_weight": base_data["test_results"]["class_weight"],
    "eda": eda_data["test_results"],
    "gpt": gpt_data["test_results"],
}

# ===================== 통계 검정 =====================
print("\n" + "=" * 60)
print("  통계 검정")
print("=" * 60)

def run_test(name_a, name_b):
    f1_a = [r["f1"] for r in results[name_a]]
    f1_b = [r["f1"] for r in results[name_b]]
    t_stat, t_pval = stats.ttest_rel(f1_b, f1_a)
    try:
        w_stat, w_pval = stats.wilcoxon(f1_b, f1_a)
    except:
        w_pval = None
    diff = np.array(f1_b) - np.array(f1_a)
    d = float(np.mean(diff)/np.std(diff)) if np.std(diff) > 0 else 0
    sig = "***" if t_pval<0.001 else "**" if t_pval<0.01 else "*" if t_pval<0.05 else "n.s."
    return {
        "comparison": f"{name_a} vs {name_b}",
        "a_f1": f"{np.mean(f1_a):.4f}±{np.std(f1_a):.4f}",
        "b_f1": f"{np.mean(f1_b):.4f}±{np.std(f1_b):.4f}",
        "t_pval": float(t_pval), "w_pval": float(w_pval) if w_pval else None,
        "cohens_d": d, "significance": sig,
    }

comparisons = [
    run_test("base", "class_weight"),
    run_test("base", "eda"),
    run_test("base", "gpt"),
    run_test("eda", "gpt"),
    run_test("class_weight", "eda"),
    run_test("class_weight", "gpt"),
]

for c in comparisons:
    print(f"  {c['comparison']:<25} {c['a_f1']} → {c['b_f1']}  "
          f"p={c['t_pval']:.6f} d={c['cohens_d']:.3f} {c['significance']}")

# ===================== 종합 비교표 =====================
print(f"\n{'='*60}")
print(f"  CV 10-Fold (Last Epoch, Mean ± SD)")
print(f"{'='*60}")
print(f"  {'Condition':<15} {'F1':>14} {'Accuracy':>14} {'Precision':>14} {'Recall':>14}")
print(f"  {'─'*71}")
for cond in ["base", "class_weight", "eda", "gpt"]:
    f1s = [r["f1"] for r in results[cond]]
    accs = [r["accuracy"] for r in results[cond]]
    precs = [r["precision"] for r in results[cond]]
    recs = [r["recall"] for r in results[cond]]
    print(f"  {cond:<15} {np.mean(f1s):.4f}±{np.std(f1s):.4f} "
          f"{np.mean(accs):.4f}±{np.std(accs):.4f} "
          f"{np.mean(precs):.4f}±{np.std(precs):.4f} "
          f"{np.mean(recs):.4f}±{np.std(recs):.4f}")

print(f"\n  {'─'*71}")
print(f"  HELD-OUT TEST")
print(f"  {'─'*71}")
print(f"  {'Condition':<15} {'F1':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
for cond in ["base", "class_weight", "eda", "gpt"]:
    r = test_results[cond]
    print(f"  {cond:<15} {r['f1']:.4f}     {r['accuracy']:.4f}     "
          f"{r['precision']:.4f}     {r['recall']:.4f}")

# ===================== 저장 =====================
final = {
    "cv_summary": {
        cond: {
            "f1": f"{np.mean([r['f1'] for r in results[cond]]):.4f}±{np.std([r['f1'] for r in results[cond]]):.4f}",
            "accuracy": f"{np.mean([r['accuracy'] for r in results[cond]]):.4f}±{np.std([r['accuracy'] for r in results[cond]]):.4f}",
            "precision": f"{np.mean([r['precision'] for r in results[cond]]):.4f}±{np.std([r['precision'] for r in results[cond]]):.4f}",
            "recall": f"{np.mean([r['recall'] for r in results[cond]]):.4f}±{np.std([r['recall'] for r in results[cond]]):.4f}",
        } for cond in ["base", "class_weight", "eda", "gpt"]
    },
    "test_results": test_results,
    "statistical_tests": comparisons,
}

with open(f"{DRIVE_FOLDER}/review_result_final.json", "w") as f:
    json.dump(final, f, indent=2, default=str)

print(f"\n  저장: {DRIVE_FOLDER}/review_result_final.json")
print(f"  완료!")
