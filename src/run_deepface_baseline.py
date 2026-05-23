"""
run_deepface_baseline.py
DeepFace 기본 age prediction을 Validation 이미지에 실행하고 MAE/RMSE를 측정한다.

실행:
    python src/run_deepface_baseline.py [--frontal-only] [--limit N]
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import PROC_DIR, RESULTS_DIR


RESULT_DIR = RESULTS_DIR / "baseline_deepface"


def predict_age(img_path: str) -> float | None:
    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=["age"],
            enforce_detection=False,
            silent=True,
        )
        return float(result[0]["age"])
    except Exception:
        return None


def run(frontal_only: bool, limit: int | None):
    df = pd.read_csv(PROC_DIR / "metadata.csv")

    # Validation + 이미지 있는 행만
    val = df[
        (df["split"] == "validation") &
        (df["image_path"].notna()) &
        (df["image_path"] != "")
    ].copy()

    if frontal_only:
        val = val[val["is_frontal"] == 1]
        print(f"정면 필터 적용: {len(val):,}개")
    else:
        print(f"전체 Validation: {len(val):,}개")

    if limit:
        val = val.head(limit)
        print(f"샘플 제한: {limit}개")

    print("DeepFace baseline 실행 중...")
    pred_ages, true_ages, paths = [], [], []

    for _, row in tqdm(val.iterrows(), total=len(val), desc="예측"):
        pred = predict_age(row["image_path"])
        if pred is None:
            continue
        pred_ages.append(pred)
        true_ages.append(row["photo_age"])
        paths.append(row["image_path"])

    pred_ages = np.array(pred_ages)
    true_ages = np.array(true_ages)
    errors    = pred_ages - true_ages

    mae  = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    me   = np.mean(errors)   # 방향성 편향

    print(f"\n=== 결과 ({len(pred_ages):,}개) ===")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  ME  : {me:+.2f}  (+ = 과대 예측)")

    # 연령대별 MAE
    age_bins = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]
    print("\n  [연령대별 MAE]")
    for lo, hi in age_bins:
        mask = (true_ages >= lo) & (true_ages < hi)
        if mask.sum() == 0:
            continue
        g_mae = np.mean(np.abs(errors[mask]))
        print(f"  {lo:2d}~{hi:2d}세: MAE={g_mae:.2f}  (n={mask.sum()})")

    # 결과 저장
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        "image_path": paths,
        "true_age":   true_ages,
        "pred_age":   pred_ages,
        "error":      errors,
    })
    results_df.to_csv(RESULT_DIR / "predictions.csv", index=False)

    metrics = {
        "n_samples": int(len(pred_ages)),
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "ME":   round(float(me), 4),
        "frontal_only": frontal_only,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n  저장 완료: {RESULT_DIR}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontal-only", action="store_true",
                        help="정면 얼굴(is_frontal=1)만 평가")
    parser.add_argument("--limit", type=int, default=None,
                        help="빠른 테스트용 샘플 수 제한 (예: --limit 100)")
    args = parser.parse_args()
    run(frontal_only=args.frontal_only, limit=args.limit)


if __name__ == "__main__":
    main()
