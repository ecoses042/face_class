"""
train_small_cnn_aihub.py
소형 CNN (~3.3M) 으로 AIHub 이미지에서 나이 회귀를 학습한다.
FCN 과 동일한 split_meta.csv 분할을 사용한다.

모델 출처: Kaggle neneti/deeplearning-final-project-cnn
  Conv32 → Conv64 → Conv128 (valid padding) → Dense(128) → Dense(1)

실행:
    python src/train_small_cnn_aihub.py [--img-size 128] [--epochs 50] [--lr 1e-3] [--batch-size 32]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR
from models_cnn import build_small_cnn_regressor
from pipeline_aihub import load_splits, make_dataset, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size",   type=int,   default=128)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--use-crops",  action="store_true",
                        help="얼굴 크롭 이미지 사용 (extract_face_crops.py 선행 필요)")
    args = parser.parse_args()

    tag = "crops" if args.use_crops else "raw"
    print(f"데이터 로딩 ({'얼굴 크롭' if args.use_crops else '원본 이미지'}) ...")
    train_df, valid_df, test_df = load_splits(use_crops=args.use_crops)
    print(f"  train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}")

    train_ds = make_dataset(train_df, args.img_size, args.batch_size, shuffle=True)
    valid_ds = make_dataset(valid_df, args.img_size, args.batch_size, shuffle=False)
    test_ds  = make_dataset(test_df,  args.img_size, args.batch_size, shuffle=False)

    model = build_small_cnn_regressor(img_size=args.img_size)
    model.summary()

    run_training(
        model, train_ds, valid_ds, test_ds, test_df,
        result_dir=RESULTS_DIR / f"small_cnn_aihub_{tag}",
        title=f"Small CNN-AIHub (3.3M, {tag})",
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
