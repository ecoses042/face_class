"""
analyze_regression_to_cls.py
기존 regression 예측값을 3-class classification으로 변환하여 성능을 분석한다.

실행:
    python src/analyze_regression_to_cls.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
CLASS_NAMES = ["young (<20)", "adult (20-60)", "senior (60+)"]
OUTPUT_DIR = RESULTS_DIR / "classification_analysis" / "regression_cls"

MODELS = {
    "fcn_regressor": RESULTS_DIR / "fcn_regressor" / "predictions.csv",
    "cnn_small":     RESULTS_DIR / "cnn_small"     / "predictions.csv",
    "cnn_vgg":       RESULTS_DIR / "cnn_vgg"        / "predictions.csv",
}


def age_to_class(age: float) -> int:
    """age < 20 → 0, 20 ≤ age < 60 → 1, age ≥ 60 → 2"""
    if age < 20:
        return 0
    elif age < 60:
        return 1
    return 2


def plot_confusion_matrix(cm: np.ndarray, model_name: str, out_path: Path) -> None:
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
        ax.set_yticklabels(CLASS_NAMES, rotation=0)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_model(model_name: str, csv_path: Path) -> dict | None:
    if not csv_path.exists():
        warnings.warn(f"[SKIP] predictions file not found: {csv_path}", stacklevel=2)
        return None

    df = pd.read_csv(csv_path)
    if "true_age" not in df.columns or "pred_age" not in df.columns:
        warnings.warn(
            f"[SKIP] {csv_path} does not have 'true_age' / 'pred_age' columns", stacklevel=2
        )
        return None

    y_true = df["true_age"].apply(age_to_class).to_numpy()
    y_pred = df["pred_age"].apply(age_to_class).to_numpy()

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(report[CLASS_NAMES[i]]["precision"], 4),
            "recall":    round(report[CLASS_NAMES[i]]["recall"],    4),
            "f1":        round(report[CLASS_NAMES[i]]["f1-score"],  4),
            "support":   int(report[CLASS_NAMES[i]]["support"]),
        }
        for i in range(3)
    }

    metrics = {
        "accuracy":    round(acc,         4),
        "macro_f1":    round(macro_f1,    4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class":   per_class,
    }

    # Save metrics JSON
    (OUTPUT_DIR / f"metrics_{model_name}.json").write_text(json.dumps(metrics, indent=2))

    # Save confusion matrix PNG
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm, model_name, OUTPUT_DIR / f"confusion_matrix_{model_name}.png")

    print(f"\n=== {model_name} ===")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0,
    ))

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, csv_path in MODELS.items():
        metrics = analyze_model(model_name, csv_path)
        if metrics is not None:
            rows.append({
                "model":       model_name,
                "accuracy":    metrics["accuracy"],
                "macro_f1":    metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            })

    if not rows:
        print("경고: 처리된 모델이 없습니다. predictions.csv 파일을 확인하세요.")
        return

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    print("\n" + "=" * 60)
    print("Classification Performance Summary (regression → 3-class)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print(f"\n저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
