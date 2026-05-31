"""
predict_all.py
test_picture/ 안의 모든 이미지에 대해 4개 모델의 나이 예측을 비교한다.

실행:
    python src/predict_all.py [--img-dir test_picture] [--out results/test_predictions.png]
"""

import argparse
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # TF CPU 전용 (XLA GPU 에러 방지)
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR
from models import AgeRegressor

FCN_CKPT       = RESULTS_DIR / "fcn_regressor"    / "checkpoints" / "best_model.pt"
SMALL_CNN_RAW_CKPT   = RESULTS_DIR / "small_cnn_aihub"       / "checkpoints" / "best_model.keras"
SMALL_CNN_CROP_CKPT  = RESULTS_DIR / "small_cnn_aihub_crops" / "checkpoints" / "best_model.keras"
VGG_CNN_RAW_CKPT     = RESULTS_DIR / "cnn_aihub"             / "checkpoints" / "best_model.keras"
VGG_CNN_CROP_CKPT    = RESULTS_DIR / "cnn_aihub_crops"       / "checkpoints" / "best_model.keras"
IMG_SIZE = 128
EMBED_DIM = 512


# ── 공통 유틸 ────────────────────────────────────────────────────────
def _deepface():
    from deepface import DeepFace
    return DeepFace


def _load_and_resize(img_path: str, max_dim: int = 1280) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.array(img)


# ── Model 1: DeepFace Baseline ───────────────────────────────────────
def predict_deepface(img_path: str) -> float | None:
    try:
        result = _deepface().analyze(
            img_path=img_path, actions=["age"],
            enforce_detection=False, silent=True,
        )
        return float(result[0]["age"])
    except Exception:
        return None


# ── Model 2: FCN Regressor ───────────────────────────────────────────
def _extract_embedding(img_path: str) -> np.ndarray | None:
    DeepFace = _deepface()
    try:
        img_arr = _load_and_resize(img_path)
        sw, sh = img_arr.shape[1], img_arr.shape[0]
        face_crop = None
        try:
            faces = DeepFace.extract_faces(
                img_path=img_arr, detector_backend="opencv", enforce_detection=True,
            )
            if faces:
                fa = faces[0]["facial_area"]
                x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                px, py = w * 0.2, h * 0.2
                x1, y1 = max(0, int(x - px)), max(0, int(y - py))
                x2, y2 = min(sw, int(x + w + px)), min(sh, int(y + h + py))
                face_crop = img_arr[y1:y2, x1:x2]
        except Exception:
            pass
        inp = face_crop if face_crop is not None else img_arr
        result = DeepFace.represent(
            img_path=inp, model_name="ArcFace",
            enforce_detection=False, detector_backend="skip",
        )
        return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception:
        return None


_fcn_model = None


def predict_fcn(img_path: str) -> float | None:
    global _fcn_model
    if _fcn_model is None:
        if not FCN_CKPT.exists():
            return None
        m = AgeRegressor(EMBED_DIM)
        m.load_state_dict(torch.load(FCN_CKPT, map_location="cpu"))
        m.eval()
        _fcn_model = m
    emb = _extract_embedding(img_path)
    if emb is None:
        return None
    with torch.no_grad():
        return float(_fcn_model(torch.tensor(emb).unsqueeze(0)).item())


# ── Model 3 & 4: Keras CNN 모델 공통 로더 ───────────────────────────
# tf.keras.models.load_model 이 batch_shape 역직렬화 오류를 내므로
# 아키텍처를 직접 재구성한 뒤 .keras 파일에서 가중치만 로드한다.
_keras_models: dict = {}


def _predict_keras(img_path: str, ckpt: Path, key: str, builder) -> float | None:
    import tensorflow as tf

    if key not in _keras_models:
        if not ckpt.exists():
            return None
        model = builder(IMG_SIZE)
        # 가중치를 .keras zip에서 직접 추출해 로드
        import zipfile, tempfile, os
        with zipfile.ZipFile(str(ckpt)) as zf:
            names = zf.namelist()
            w_name = next((n for n in names if n.endswith(".weights.h5") or n == "model.weights.h5"), None)
            if w_name is None:
                return None
            with tempfile.TemporaryDirectory() as tmp:
                zf.extract(w_name, tmp)
                model.load_weights(os.path.join(tmp, w_name))
        _keras_models[key] = model

    model = _keras_models[key]
    try:
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, 0)
        return float(model.predict(x, verbose=0)[0][0])
    except Exception:
        return None


def _face_crop_image(img_path: str) -> Image.Image:
    """DeepFace opencv 검출 → 크롭 → fallback: 전체 이미지. extract_face_crops.py 와 동일."""
    DeepFace = _deepface()
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(1280 / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    img_arr = np.array(img)
    sw, sh = img.size
    try:
        faces = DeepFace.extract_faces(img_path=img_arr, detector_backend="opencv",
                                       enforce_detection=True)
        if faces:
            fa = faces[0]["facial_area"]
            x, y, fw, fh = fa["x"], fa["y"], fa["w"], fa["h"]
            px, py = fw * 0.2, fh * 0.2
            x1, y1 = max(0, int(x - px)), max(0, int(y - py))
            x2, y2 = min(sw, int(x + fw + px)), min(sh, int(y + fh + py))
            return img.crop((x1, y1, x2, y2))
    except Exception:
        pass
    return img


def predict_small_cnn_raw(img_path: str) -> float | None:
    from models_cnn import build_small_cnn_regressor
    return _predict_keras(img_path, SMALL_CNN_RAW_CKPT, "small_raw", build_small_cnn_regressor)


def predict_small_cnn_crops(img_path: str) -> float | None:
    from models_cnn import build_small_cnn_regressor
    # 학습과 동일한 얼굴 크롭 전처리 적용
    face = _face_crop_image(img_path)
    face_resized = face.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.array(face_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)
    import tensorflow as tf
    key = "small_crops"
    if key not in _keras_models:
        if not SMALL_CNN_CROP_CKPT.exists():
            return None
        import zipfile, tempfile, os as _os
        model = build_small_cnn_regressor(IMG_SIZE)
        with zipfile.ZipFile(str(SMALL_CNN_CROP_CKPT)) as zf:
            w_name = next((n for n in zf.namelist() if "weights.h5" in n), None)
            if w_name is None:
                return None
            with tempfile.TemporaryDirectory() as tmp:
                zf.extract(w_name, tmp)
                model.load_weights(_os.path.join(tmp, w_name))
        _keras_models[key] = model
    try:
        return float(_keras_models[key].predict(x, verbose=0)[0][0])
    except Exception:
        return None


def predict_vgg_cnn_crops(img_path: str) -> float | None:
    from models_cnn import build_cnn_regressor
    return _predict_keras(img_path, VGG_CNN_CROP_CKPT, "vgg_crops", build_cnn_regressor)


# ── 결과 시각화 ──────────────────────────────────────────────────────
MODELS = [
    ("DeepFace\nBaseline",      predict_deepface),
    ("FCN\nRegressor",          predict_fcn),
    ("Small CNN\n(raw)",        predict_small_cnn_raw),
    ("Small CNN\n(crops)",      predict_small_cnn_crops),
    ("VGG-CNN\n(crops)",        predict_vgg_cnn_crops),
]
MODEL_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#E377C2", "#C44E52"]


def run(img_dir: Path, out_path: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix in exts])
    if not images:
        print(f"이미지 없음: {img_dir}")
        return

    n_imgs   = len(images)
    n_models = len(MODELS)

    # ── 예측 수행 ────────────────────────────────────────────────────
    results: dict[str, list] = {name: [] for name, _ in MODELS}

    for img_path in images:
        print(f"\n▶ {img_path.name}")
        for name, fn in MODELS:
            pred = fn(str(img_path))
            label = name.replace("\n", " ")
            if pred is not None:
                print(f"  {label:20s}: {pred:.1f}세")
            else:
                print(f"  {label:20s}: 예측 실패")
            results[name].append(pred)

    # ── Figure: 이미지 행 × 모델 예측 막대 ──────────────────────────
    fig = plt.figure(figsize=(4 + n_models * 2, n_imgs * 3.2))
    fig.suptitle("모델별 나이 예측 비교", fontsize=14, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(n_imgs, n_models + 1,
                          width_ratios=[2] + [1] * n_models,
                          hspace=0.5, wspace=0.3)

    for i, img_path in enumerate(images):
        # 이미지 표시
        ax_img = fig.add_subplot(gs[i, 0])
        pil = Image.open(img_path).convert("RGB")
        ax_img.imshow(pil)
        ax_img.set_title(img_path.name, fontsize=8)
        ax_img.axis("off")

        # 모델별 막대
        for j, (name, _) in enumerate(MODELS):
            ax = fig.add_subplot(gs[i, j + 1])
            pred = results[name][i]

            if pred is not None:
                bar = ax.bar([""], [pred], color=MODEL_COLORS[j], width=0.5)
                ax.bar_label(bar, fmt="%.1f", padding=2, fontsize=9)
                ax.set_ylim(0, 85)
            else:
                ax.text(0.5, 0.5, "실패", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.set_ylim(0, 85)

            if i == 0:
                ax.set_title(name, fontsize=8, color=MODEL_COLORS[j], fontweight="bold")
            ax.set_ylabel("나이 (세)" if j == 0 else "")
            ax.tick_params(labelbottom=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ 결과 저장: {out_path}")

    # ── 요약 테이블 출력 ─────────────────────────────────────────────
    col_w = 14
    header = f"{'이미지':<20}" + "".join(f"{n.replace(chr(10),' '):>{col_w}}" for n, _ in MODELS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for i, img_path in enumerate(images):
        row = f"{img_path.name:<20}"
        for name, _ in MODELS:
            p = results[name][i]
            row += f"{(f'{p:.1f}세' if p is not None else '실패'):>{col_w}}"
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path,
                        default=Path(__file__).parent.parent / "test_picture")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent.parent / "results" / "test_predictions.png")
    args = parser.parse_args()
    run(args.img_dir, args.out)


if __name__ == "__main__":
    main()
