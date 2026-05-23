"""
preprocess.py
원본 zip(라벨 + 이미지)을 풀고 metadata.csv를 생성한다.

실행:
    python src/preprocess.py [--skip-images]

출력:
    dataset/processed/labels/      ← JSON 파일
    dataset/processed/images/      ← PNG 파일 (이미지 zip이 있을 때만)
    dataset/processed/metadata.csv ← 전체 메타데이터
"""

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from config import (
    FIELD_AGE_NOW, FIELD_AGE_PAST, FIELD_ANNOT, FIELD_BIRTH,
    FIELD_BOX, FIELD_FORMAT, FIELD_GENDER, FIELD_ID, FIELD_LANDMARK,
    FRONTAL_SYM_THRESH, MAX_AGE, MIN_AGE, MIN_BOX_SIZE,
    PROC_DIR, PROC_IMAGES, PROC_LABELS, RAW_DIR,
    IMG_SUBDIR, LABEL_SUBDIR,
)


# ── 1. zip 압축 해제 ──────────────────────────────────────────────

def extract_zips(src_dir: Path, dst_dir: Path, prefix: str, desc: str) -> int:
    """src_dir 내 prefix로 시작하는 zip을 dst_dir에 일괄 해제."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(src_dir.glob(f"{prefix}*.zip"))
    if not zips:
        print(f"  [skip] {desc}: zip 없음 ({src_dir})")
        return 0

    print(f"  {desc}: {len(zips)}개 zip 해제 → {dst_dir}")
    count = 0
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name:
                    continue
                dst_path = dst_dir / name
                if dst_path.exists():
                    continue
                with zf.open(member) as src, open(dst_path, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
    print(f"    → {count}개 파일 추출 완료")
    return count


def extract_all(skip_images: bool = False):
    for split in ("Training", "Validation"):
        split_dir = RAW_DIR / split

        # 라벨
        label_src = split_dir / LABEL_SUBDIR
        prefix = "TL_" if split == "Training" else "VL_"
        extract_zips(label_src, PROC_LABELS / split, prefix, f"{split} 라벨")

        if skip_images:
            continue

        # 이미지
        img_src = split_dir / IMG_SUBDIR
        prefix = "TS_" if split == "Training" else "VS_"
        extract_zips(img_src, PROC_IMAGES / split, prefix, f"{split} 이미지")


# ── 2. JSON 파싱 ──────────────────────────────────────────────────

def is_frontal(landmarks: list, box: dict) -> bool:
    """
    5점 랜드마크 좌우 대칭으로 정면 여부 판별.
    landmarks 순서: [코, 왼눈, 오른눈, 왼입꼬리, 오른입꼬리] (추정).
    대칭 비율 = |코x - 눈중간x| / 두눈간격
    """
    if len(landmarks) < 3:
        return True  # 랜드마크 부족 → 필터링 포기

    pts = [list(p) for p in landmarks]
    # 눈 두 점 중 y값이 높은(얼굴 위쪽) 두 점을 눈으로 가정
    by_y = sorted(pts, key=lambda p: p[1])
    eye1, eye2 = by_y[0], by_y[1]
    nose = by_y[2]

    eye_mid_x = (eye1[0] + eye2[0]) / 2
    eye_dist  = abs(eye1[0] - eye2[0])

    if eye_dist < 1:
        return True

    sym_ratio = abs(nose[0] - eye_mid_x) / eye_dist
    return sym_ratio < FRONTAL_SYM_THRESH


def parse_json(json_path: Path, split: str) -> dict | None:
    """JSON 한 파일을 파싱해 레코드 dict 반환. 유효하지 않으면 None."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    annots = data.get(FIELD_ANNOT, [])
    if not annots:
        return None

    annot = annots[0]
    box   = annot.get(FIELD_BOX, {})
    lm    = annot.get(FIELD_LANDMARK, [])

    # bbox 최소 크기 필터
    if box.get("w", 0) < MIN_BOX_SIZE or box.get("h", 0) < MIN_BOX_SIZE:
        return None

    age_now  = data.get(FIELD_AGE_NOW, -1)
    age_past = data.get(FIELD_AGE_PAST, 0)
    # 실제 촬영 시점 나이
    photo_age = age_now - age_past

    if not (MIN_AGE <= photo_age <= MAX_AGE):
        return None

    frontal = is_frontal(lm, box)
    fmt     = data.get(FIELD_FORMAT, "png")
    fname   = json_path.stem  # 확장자 제외

    # 이미지 경로 (있으면 절대경로, 없으면 빈 문자열)
    img_path = PROC_IMAGES / split / f"{fname}.{fmt}"
    img_path_str = str(img_path) if img_path.exists() else ""

    return {
        "filename":   fname,
        "split":      split.lower(),   # training / validation
        "person_id":  data.get(FIELD_ID),
        "birth":      data.get(FIELD_BIRTH),
        "age_now":    age_now,
        "age_past":   age_past,
        "photo_age":  photo_age,
        "gender":     data.get(FIELD_GENDER, ""),
        "bbox_x":     box.get("x", 0),
        "bbox_y":     box.get("y", 0),
        "bbox_w":     box.get("w", 0),
        "bbox_h":     box.get("h", 0),
        "is_frontal": int(frontal),
        "image_path": img_path_str,
        "label_path": str(json_path),
    }


# ── 3. metadata.csv 생성 ─────────────────────────────────────────

def build_metadata() -> pd.DataFrame:
    records = []
    for split in ("Training", "Validation"):
        label_dir = PROC_LABELS / split
        if not label_dir.exists():
            print(f"  [skip] {split} 라벨 폴더 없음")
            continue

        jsons = list(label_dir.glob("*.json"))
        print(f"  {split}: JSON {len(jsons):,}개 파싱 중...")
        for jp in jsons:
            rec = parse_json(jp, split)
            if rec:
                records.append(rec)

    df = pd.DataFrame(records)
    print(f"\n  총 유효 레코드: {len(df):,}개")
    return df


def save_metadata(df: pd.DataFrame):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "metadata.csv"
    df.to_csv(out, index=False)
    print(f"  저장 완료: {out}")

    print("\n  [통계]")
    print(f"  - 정면(is_frontal=1): {df['is_frontal'].sum():,}개")
    print(f"  - 비정면:              {(df['is_frontal'] == 0).sum():,}개")
    print(f"  - 이미지 존재:         {(df['image_path'] != '').sum():,}개")
    print(f"  - photo_age 분포:\n{df['photo_age'].describe().round(1)}")
    print(f"  - split 분포:\n{df['split'].value_counts()}")


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true",
                        help="이미지 zip 해제를 건너뜀 (라벨만 처리)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="zip 해제 단계를 완전히 건너뜀")
    args = parser.parse_args()

    print("=== Step 1: zip 압축 해제 ===")
    if args.skip_extract:
        print("  (skip)")
    else:
        extract_all(skip_images=args.skip_images)

    print("\n=== Step 2: JSON 파싱 및 metadata.csv 생성 ===")
    df = build_metadata()
    save_metadata(df)

    print("\n[완료] 다음 단계: python src/split_dataset.py")


if __name__ == "__main__":
    main()
