"""
extract_embeddings.py
Validation 이미지에서 DeepFace 임베딩을 추출하고 person_id 기준으로 분할한다.

실행:
    python src/extract_embeddings.py [--model-name ArcFace]

출력:
    dataset/embeddings/
        train_embeddings.npy  valid_embeddings.npy  test_embeddings.npy
        train_labels.npy      valid_labels.npy      test_labels.npy
        split_meta.csv        (각 샘플의 filename / person_id / split)
"""

import argparse
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import PROC_DIR, EMBED_DIR, RANDOM_SEED, VALID_RATIO, TEST_RATIO


def split_by_person(df: pd.DataFrame) -> pd.DataFrame:
    """person_id 기준으로 train/valid/test 분리."""
    rng = np.random.default_rng(RANDOM_SEED)
    person_ids = df["person_id"].unique()
    rng.shuffle(person_ids)

    n = len(person_ids)
    n_test  = max(1, int(n * TEST_RATIO))
    n_valid = max(1, int(n * VALID_RATIO))

    test_ids  = set(person_ids[:n_test])
    valid_ids = set(person_ids[n_test:n_test + n_valid])

    def assign(pid):
        if pid in test_ids:  return "test"
        if pid in valid_ids: return "valid"
        return "train"

    df = df.copy()
    df["fcn_split"] = df["person_id"].apply(assign)
    return df


MAX_DIM = 1280  # 검출 전 리사이즈 최대 크기 (속도 확보)


def _resize_for_detection(img):
    """긴 쪽이 MAX_DIM을 넘으면 비율 유지로 리사이즈."""
    from PIL import Image as PILImage
    w, h = img.size
    scale = min(MAX_DIM / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), PILImage.BILINEAR)
    return img, scale


def _detect_and_crop(img_array: np.ndarray, iw: int, ih: int) -> np.ndarray | None:
    """opencv로 얼굴 검출 후 원본 크기 기준으로 20% 패딩 크롭."""
    try:
        faces = DeepFace.extract_faces(
            img_path=img_array,
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
        x2 = min(iw, int(x + w + pad_x))
        y2 = min(ih, int(y + h + pad_y))
        from PIL import Image as PILImage
        return np.array(PILImage.fromarray(img_array).crop((x1, y1, x2, y2)))
    except Exception:
        return None


def extract_one(img_path: str, model_name: str, **_) -> np.ndarray | None:
    """
    이미지를 1280px로 리사이즈 → opencv 얼굴 검출 → 크롭(20% 패딩) → ArcFace 임베딩.
    predict.py와 완전히 동일한 파이프라인으로 학습-추론 분포를 일치시킨다.
    검출 실패 시 리사이즈 이미지 전체로 fallback.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert("RGB")
        img_small, _ = _resize_for_detection(img)
        img_arr = np.array(img_small)
        sw, sh = img_small.size

        face_crop = _detect_and_crop(img_arr, sw, sh)
        input_img = face_crop if face_crop is not None else img_arr

        result = DeepFace.represent(
            img_path=input_img,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",
        )
        return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="ArcFace",
                        help="DeepFace 임베딩 모델 (ArcFace / Facenet512 / VGG-Face)")
    parser.add_argument("--frontal-only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(PROC_DIR / "metadata.csv")
    val = df[
        (df["split"] == "validation") &
        (df["image_path"].notna()) &
        (df["image_path"] != "")
    ].copy()

    if args.frontal_only:
        val = val[val["is_frontal"] == 1]
        print(f"정면 필터: {len(val):,}개")
    else:
        print(f"전체 Validation 이미지: {len(val):,}개")

    # person_id 기준 분할
    val = split_by_person(val)
    print(val["fcn_split"].value_counts().to_string())

    # 임베딩 추출
    print(f"\nDeepFace 임베딩 추출 (모델: {args.model_name}) ...")
    embeddings, labels, filenames, splits = [], [], [], []

    for _, row in tqdm(val.iterrows(), total=len(val)):
        emb = extract_one(row["image_path"], args.model_name)
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append(float(row["age_past"]))
        filenames.append(row["filename"])
        splits.append(row["fcn_split"])

    embeddings = np.array(embeddings)
    labels     = np.array(labels, dtype=np.float32)
    splits     = np.array(splits)

    print(f"\n추출 완료: {len(embeddings):,}개  임베딩 차원: {embeddings.shape[1]}")

    # 분할별 저장
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    meta_rows = []

    for split in ("train", "valid", "test"):
        mask = splits == split
        np.save(EMBED_DIR / f"{split}_embeddings.npy", embeddings[mask])
        np.save(EMBED_DIR / f"{split}_labels.npy",     labels[mask])
        print(f"  {split}: {mask.sum():,}개  → {EMBED_DIR}")
        for fn, sp in zip(np.array(filenames)[mask], splits[mask]):
            meta_rows.append({"filename": fn, "fcn_split": sp})

    pd.DataFrame(meta_rows).to_csv(EMBED_DIR / "split_meta.csv", index=False)

    # 임베딩 차원 기록
    (EMBED_DIR / "embedding_dim.txt").write_text(str(embeddings.shape[1]))
    print(f"\n[완료] 다음 단계: python src/train_fcn.py")


if __name__ == "__main__":
    main()
