"""
predict.py
JPG/PNG 이미지 한 장을 받아 두 모델의 나이 예측 결과를 출력한다.

실행:
    python src/predict.py --image path/to/face.jpg

출력 예시:
    이미지: face.jpg
    ── DeepFace Baseline : 34세
    ── FCN Regressor     : 28.4세
"""

import argparse
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # TF는 CPU 사용 (GPU JIT 에러 방지)
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR
from models import AgeRegressor
from utils import extract_face_crop

CKPT_PATH = RESULTS_DIR / "fcn_regressor" / "checkpoints" / "best_model.pt"
EMBED_DIM  = 512


# ── DeepFace import (무거워서 필요 시 로드) ──────────────────────────
def _deepface():
    from deepface import DeepFace
    return DeepFace


# ── 모델 1: DeepFace Baseline ────────────────────────────────────────
def predict_baseline(img_path: str) -> float | None:
    DeepFace = _deepface()
    try:
        crop = extract_face_crop(img_path)
        result = DeepFace.analyze(
            img_path=crop,
            actions=["age"],
            enforce_detection=False,
            silent=True,
        )
        return float(result[0]["age"])
    except Exception as e:
        print(f"  [Baseline 오류] {e}")
        return None


# ── 모델 2: FCN Regressor ────────────────────────────────────────────
def extract_embedding(img_path: str) -> np.ndarray | None:
    """
    이미지를 1280px로 리사이즈 → opencv 얼굴 검출 → 크롭(20% 패딩) → ArcFace 임베딩.
    extract_embeddings.py 학습 파이프라인과 동일한 방식.
    """
    DeepFace = _deepface()

    try:
        crop = extract_face_crop(img_path)
        result = DeepFace.represent(
            img_path=crop,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
        )
        return np.array(result[0]["embedding"], dtype=np.float32)

    except Exception as e:
        print(f"  [임베딩 오류] {e}")
        return None


def load_fcn() -> AgeRegressor:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"체크포인트 없음: {CKPT_PATH}")
    model = AgeRegressor(EMBED_DIM)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()
    return model


def predict_fcn(img_path: str, model: AgeRegressor) -> float | None:
    emb = extract_embedding(img_path)
    if emb is None:
        return None
    with torch.no_grad():
        x = torch.tensor(emb).unsqueeze(0)
        return float(model(x).item())


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="얼굴 이미지 나이 예측")
    parser.add_argument("--image", required=True, help="입력 이미지 경로 (JPG/PNG)")
    args = parser.parse_args()

    img_path = str(Path(args.image).resolve())
    if not Path(img_path).exists():
        print(f"파일을 찾을 수 없습니다: {img_path}")
        sys.exit(1)

    print(f"\n이미지: {Path(args.image).name}")
    print("=" * 40)

    # Baseline
    baseline_age = predict_baseline(img_path)
    if baseline_age is not None:
        print(f"  DeepFace Baseline : {baseline_age:.0f}세")
    else:
        print("  DeepFace Baseline : 예측 실패")

    # FCN
    try:
        model = load_fcn()
        fcn_age = predict_fcn(img_path, model)
        if fcn_age is not None:
            print(f"  FCN Regressor     : {fcn_age:.1f}세")
        else:
            print("  FCN Regressor     : 예측 실패 (얼굴 검출 불가)")
    except FileNotFoundError as e:
        print(f"  FCN Regressor     : {e}")

    print()


if __name__ == "__main__":
    main()
