"""
evaluate.py
predictions.csv를 읽어 MAE/RMSE/연령대별 분석 및 시각화를 출력한다.

실행:
    python src/evaluate.py --pred results/baseline_deepface/predictions.csv
    python src/evaluate.py --pred results/fcn_regressor/predictions.csv
    python src/evaluate.py --compare  # 두 모델 비교
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

AGE_BINS = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    return {
        "n":    len(true),
        "MAE":  float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "ME":   float(np.mean(err)),
    }


def age_group_mae(true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi in AGE_BINS:
        mask = (true >= lo) & (true < hi)
        if mask.sum() == 0:
            continue
        m = compute_metrics(true[mask], pred[mask])
        rows.append({"age_group": f"{lo}~{hi}", **m})
    return pd.DataFrame(rows)


def plot_scatter(true, pred, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, alpha=0.3, s=10)
    lim = (0, max(true.max(), pred.max()) + 5)
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True Age"); ax.set_ylabel("Predicted Age")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_error_dist(errors, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Error (pred - true)"); ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def evaluate_single(pred_csv: Path, label: str):
    df = pd.read_csv(pred_csv)
    true = df["true_age"].values.astype(float)
    pred = df["pred_age"].values.astype(float)

    m = compute_metrics(true, pred)
    print(f"\n=== {label} ===")
    print(f"  n={m['n']:,}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  ME={m['ME']:+.2f}")

    ag = age_group_mae(true, pred)
    print("\n  [연령대별 MAE]")
    print(ag[["age_group", "MAE", "n"]].to_string(index=False))

    out_dir = pred_csv.parent
    plot_scatter(true, pred, f"{label} — True vs Pred", out_dir / "scatter.png")
    plot_error_dist(pred - true, f"{label} — Error Distribution", out_dir / "error_dist.png")

    return m, ag


def compare(dirs: list[Path]):
    rows = []
    for d in dirs:
        m_file = d / "metrics.json"
        if not m_file.exists():
            continue
        m = json.loads(m_file.read_text())
        rows.append({"model": d.name, **m})
    if not rows:
        print("metrics.json 파일을 찾을 수 없습니다.")
        return
    df = pd.DataFrame(rows)[["model", "MAE", "RMSE", "ME", "n_samples"]]
    out = RESULTS_DIR / "comparison" / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== 모델 비교 ===")
    print(df.to_string(index=False))
    print(f"\n  저장: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=None,
                        help="평가할 predictions.csv 경로")
    parser.add_argument("--compare", action="store_true",
                        help="results/ 하위 모든 모델 비교")
    args = parser.parse_args()

    if args.compare:
        dirs = [RESULTS_DIR / "baseline_deepface", RESULTS_DIR / "fcn_regressor",
                RESULTS_DIR / "cnn_vgg", RESULTS_DIR / "cnn_small"]
        compare(dirs)
    elif args.pred:
        label = args.pred.parent.name
        evaluate_single(args.pred, label)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
