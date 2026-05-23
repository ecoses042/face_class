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


def extract_one(img_path: str, model_name: str,
                bbox_x: float = 0, bbox_y: float = 0,
                bbox_w: float = 0, bbox_h: float = 0) -> np.ndarray | None:
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        # bbox가 유효하면 얼굴 영역 크롭 (여유 20% 추가)
        if bbox_w > 0 and bbox_h > 0:
            pad_x = bbox_w * 0.2
            pad_y = bbox_h * 0.2
            x1 = max(0, int(bbox_x - pad_x))
            y1 = max(0, int(bbox_y - pad_y))
            x2 = min(iw, int(bbox_x + bbox_w + pad_x))
            y2 = min(ih, int(bbox_y + bbox_h + pad_y))
            img = img.crop((x1, y1, x2, y2))

        result = DeepFace.represent(
            img_path=np.array(img),
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
        emb = extract_one(
            row["image_path"], args.model_name,
            bbox_x=row.get("bbox_x", 0), bbox_y=row.get("bbox_y", 0),
            bbox_w=row.get("bbox_w", 0), bbox_h=row.get("bbox_h", 0),
        )
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append(float(row["photo_age"]))
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
