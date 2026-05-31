"""
extract_face_crops.py
split_meta.csv의 모든 이미지에서 얼굴을 검출·크롭하여 dataset/face_crops/ 에 저장한다.
extract_embeddings.py 와 동일한 검출 파이프라인을 사용하므로
CNN 학습 시 FCN 과 동일한 얼굴 기준 입력을 보장한다.

실행:
    python src/extract_face_crops.py [--size 128] [--out-dir dataset/face_crops]

출력:
    dataset/face_crops/{filename}.jpg   (128×128 크롭 이미지)
    dataset/embeddings/split_meta_crops.csv  (split_meta + crop_path 컬럼)
"""

import argparse
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import EMBED_DIR, PROC_DIR

MAX_DIM = 1280


def _resize_for_detection(img: Image.Image):
    w, h = img.size
    scale = min(MAX_DIM / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def _detect_and_crop(img_small: Image.Image) -> Image.Image | None:
    """opencv 얼굴 검출 후 20% 패딩 크롭. 실패 시 None."""
    from deepface import DeepFace

    img_arr = np.array(img_small)
    sw, sh = img_small.size
    try:
        faces = DeepFace.extract_faces(
            img_path=img_arr,
            detector_backend="opencv",
            enforce_detection=True,
        )
        if not faces:
            return None
        fa = faces[0]["facial_area"]
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
        pad_x, pad_y = w * 0.2, h * 0.2
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(sw, int(x + w + pad_x))
        y2 = min(sh, int(y + h + pad_y))
        return img_small.crop((x1, y1, x2, y2))
    except Exception:
        return None


def crop_one(img_path: str, out_size: int) -> Image.Image:
    """
    이미지 로드 → 리사이즈 → 얼굴 검출 → 크롭.
    검출 실패 시 리사이즈 이미지 전체를 fallback으로 사용.
    """
    img = Image.open(img_path).convert("RGB")
    img_small = _resize_for_detection(img)
    face = _detect_and_crop(img_small)
    result = face if face is not None else img_small
    return result.resize((out_size, out_size), Image.BILINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",    type=int,  default=128,
                        help="저장할 크롭 이미지 크기 (default: 128)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("dataset/face_crops"),
                        help="크롭 이미지 저장 디렉토리")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_meta = pd.read_csv(EMBED_DIR / "split_meta.csv")
    metadata   = pd.read_csv(PROC_DIR  / "metadata.csv")
    merged = split_meta.merge(
        metadata[["filename", "photo_age", "image_path"]],
        on="filename", how="left",
    )
    merged = merged[(merged["image_path"].notna()) & (merged["image_path"] != "")].copy()

    print(f"총 {len(merged):,}개 이미지 처리 시작 → {out_dir}")
    print(f"크롭 크기: {args.size}×{args.size}")

    crop_paths, detected = [], 0
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        out_path = out_dir / f"{row['filename']}.jpg"
        if out_path.exists():
            crop_paths.append(str(out_path))
            detected += 1
            continue

        try:
            img_small = Image.open(row["image_path"]).convert("RGB")
            img_small = _resize_for_detection(img_small)
            face = _detect_and_crop(img_small)
            if face is not None:
                detected += 1
            result = face if face is not None else img_small
            result.resize((args.size, args.size), Image.BILINEAR).save(out_path, "JPEG", quality=95)
            crop_paths.append(str(out_path))
        except Exception as e:
            print(f"\n  [오류] {row['filename']}: {e}")
            crop_paths.append("")

    merged["crop_path"] = crop_paths
    valid = merged[merged["crop_path"] != ""]
    out_csv = EMBED_DIR / "split_meta_crops.csv"
    valid[["filename", "fcn_split", "photo_age", "crop_path"]].to_csv(out_csv, index=False)

    print(f"\n완료: {len(valid):,}개 저장 (얼굴 검출 성공: {detected:,}개)")
    print(f"  낙관 fallback 포함: {len(valid) - detected:,}개")
    print(f"  메타 저장: {out_csv}")
    print("\n다음 단계:")
    print("  python src/train_small_cnn_aihub.py --use-crops")
    print("  python src/train_cnn_aihub.py --use-crops")


if __name__ == "__main__":
    main()
