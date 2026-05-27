"""
train_cnn.py
Adience fold_4 데이터셋으로 VGG-like CNN age classifier를 학습한다.

실행:
    python src/train_cnn.py [--data-dir PATH] [--epochs 30] [--lr 1e-3] [--batch-size 32]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RANDOM_SEED, RESULTS_DIR
from models_cnn import build_cnn_model


KEYS = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]
AGE_CLASSES = {key: idx for idx, key in enumerate(KEYS)}


def string_to_tuple(value):
    value = str(value).strip().strip("()")
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        return None
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return None


def format_face_id(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def directory_path(data_dir: Path, row) -> str:
    return str(
        data_dir
        / "aligned"
        / str(row["user_id"])
        / f"landmark_aligned_face.{format_face_id(row['face_id'])}.{row['original_image']}"
    )


def set_proportion(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    min_val = int(df[label_col].value_counts().min())
    if min_val == 0:
        raise ValueError("No samples available for at least one age class.")
    parts = [
        group.sample(n=min_val, random_state=RANDOM_SEED)
        for _, group in df.groupby(label_col, sort=True)
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def load_dataframe(data_dir: Path) -> pd.DataFrame:
    fold_file = data_dir / "fold_4_data.txt"
    if not fold_file.exists():
        raise FileNotFoundError(f"Adience metadata file not found: {fold_file}")

    df = pd.read_csv(fold_file, sep="\t")
    required = {"age", "user_id", "face_id", "original_image"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {fold_file}: {sorted(missing)}")

    df = df.dropna(subset=["age"]).copy()
    df["age_tuple"] = df["age"].apply(string_to_tuple)
    df = df[df["age_tuple"].isin(KEYS)].copy()
    if df.empty:
        raise ValueError("No rows matched the supported Adience age classes.")

    df["label"] = df["age_tuple"].map(AGE_CLASSES).astype("int64")
    df["path"] = df.apply(lambda row: directory_path(data_dir, row), axis=1)

    exists = df["path"].map(lambda path: Path(path).exists())
    if not exists.all():
        print(f"경고: 누락된 이미지 {int((~exists).sum()):,}개 제외")
        df = df[exists].copy()
    if df.empty:
        raise ValueError("No image files were found for the filtered metadata rows.")

    df = set_proportion(df)
    return df


def split_dataframe(df: pd.DataFrame, group_col: str = "user_id"):
    """Split by user_id to avoid identity leakage across train/valid/test."""
    group_labels = df.groupby(group_col)["label"].agg(lambda labels: labels.value_counts().idxmax())
    n_groups = len(group_labels)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique {group_col} values for train/valid/test split.")

    valid_groups = set()
    test_groups = set()
    rare_labels = []

    for label, groups in group_labels.groupby(group_labels):
        shuffled = groups.index.to_series().sample(frac=1.0, random_state=RANDOM_SEED + int(label))
        if len(shuffled) < 3:
            rare_labels.append(int(label))
            continue

        n_test = max(1, round(len(shuffled) * 0.1))
        n_valid = max(1, round(len(shuffled) * 0.1))
        if n_test + n_valid >= len(shuffled):
            n_test = 1
            n_valid = 1

        test_groups.update(shuffled.iloc[:n_test])
        valid_groups.update(shuffled.iloc[n_test:n_test + n_valid])

    if rare_labels:
        print(f"경고: group 수가 3개 미만인 label은 train에만 배치됨: {rare_labels}")

    test_df = df[df[group_col].isin(test_groups)].copy()
    valid_df = df[df[group_col].isin(valid_groups)].copy()
    train_df = df[~df[group_col].isin(test_groups | valid_groups)].copy()

    for name, split_df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        if split_df.empty:
            raise ValueError(f"{name} split is empty.")

    return train_df, valid_df, test_df


def split_summary(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().sort_index()
    return {str(label): int(counts.get(label, 0)) for label in range(len(KEYS))}


def make_dataset(df: pd.DataFrame, img_size: int, batch_size: int, shuffle: bool):
    import tensorflow as tf

    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].astype("int64").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/kaggle/input/datasets/alfredhhw/adiencegender/AdienceGender"),
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    result_dir = RESULTS_DIR / "cnn_adience"
    ckpt_dir = result_dir / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"데이터 로딩: {args.data_dir}")
    df = load_dataframe(args.data_dir)
    train_df, valid_df, test_df = split_dataframe(df)
    print(f"train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}")
    print(f"label 분포 train={split_summary(train_df)}")
    print(f"label 분포 valid={split_summary(valid_df)}")
    print(f"label 분포 test ={split_summary(test_df)}")

    train_ds = make_dataset(train_df, args.img_size, args.batch_size, shuffle=True)
    valid_ds = make_dataset(valid_df, args.img_size, args.batch_size, shuffle=False)
    test_ds = make_dataset(test_df, args.img_size, args.batch_size, shuffle=False)

    model = build_cnn_model(img_size=args.img_size, num_classes=len(KEYS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    print(f"\n학습 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\n=== Test 결과 ===")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"  Loss    : {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")

    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "test_loss": round(float(test_loss), 4),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "train_label_counts": split_summary(train_df),
        "valid_label_counts": split_summary(valid_df),
        "test_label_counts": split_summary(test_df),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("CNN Training Curve")
    fig.tight_layout()
    fig.savefig(result_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    print(f"\n  저장 완료: {result_dir}")


if __name__ == "__main__":
    main()
