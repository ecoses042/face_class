"""
pipeline_aihub.py
AIHub 이미지 기반 나이 회귀 학습의 공통 파이프라인.
train_cnn_aihub.py / train_small_cnn_aihub.py 에서 공유한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import EMBED_DIR, PROC_DIR, RANDOM_SEED, RESULTS_DIR


def load_splits(use_crops: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split_meta.csv (또는 split_meta_crops.csv) 기준으로 train/valid/test를 반환한다.

    Args:
        use_crops: True면 얼굴 크롭 이미지(split_meta_crops.csv)를 사용한다.
    """
    if use_crops:
        csv_path = EMBED_DIR / "split_meta_crops.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} 없음. 먼저 python src/extract_face_crops.py 를 실행하세요."
            )
        merged = pd.read_csv(csv_path)
        path_col = "crop_path"
    else:
        split_meta = pd.read_csv(EMBED_DIR / "split_meta.csv")
        metadata   = pd.read_csv(PROC_DIR  / "metadata.csv")
        merged = split_meta.merge(
            metadata[["filename", "photo_age", "image_path"]],
            on="filename", how="left",
        )
        path_col = "image_path"

    merged = merged[(merged[path_col].notna()) & (merged[path_col] != "")].copy()
    merged = merged.rename(columns={path_col: "image_path"})

    train_df = merged[merged["fcn_split"] == "train"].reset_index(drop=True)
    valid_df = merged[merged["fcn_split"] == "valid"].reset_index(drop=True)
    test_df  = merged[merged["fcn_split"] == "test" ].reset_index(drop=True)
    return train_df, valid_df, test_df


def make_dataset(df: pd.DataFrame, img_size: int, batch_size: int, shuffle: bool):
    """DataFrame → tf.data.Dataset (이미지 로드 + 전처리 포함)."""
    import tensorflow as tf

    paths  = df["image_path"].astype(str).to_numpy()
    labels = df["photo_age"].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_callbacks(result_dir: Path, patience: int):
    """ModelCheckpoint / EarlyStopping / ReduceLROnPlateau 콜백을 반환한다."""
    import tensorflow as tf

    ckpt_dir = result_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best_model.keras"),
            monitor="val_loss", mode="min", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=patience, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            patience=max(3, patience // 2), factor=0.5, min_lr=1e-5,
        ),
    ]


def run_training(
    model,
    train_ds,
    valid_ds,
    test_ds,
    test_df: pd.DataFrame,
    result_dir: Path,
    title: str,
    epochs: int,
    patience: int,
    lr: float,
) -> dict:
    """모델 컴파일 → 학습 → 평가 → 결과 저장을 수행하고 metrics dict를 반환한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    result_dir.mkdir(parents=True, exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=build_callbacks(result_dir, patience),
    )

    # ── 테스트 평가 ──────────────────────────────────────────────
    preds = model.predict(test_ds, verbose=0).flatten()
    trues = test_df["photo_age"].to_numpy().astype("float32")

    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    me   = float(np.mean(preds - trues))

    print(f"\n=== Test 결과 ({title}) ===")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  ME  : {me:+.2f}")

    pd.DataFrame({"true_age": trues, "pred_age": preds, "error": preds - trues}).to_csv(
        result_dir / "predictions.csv", index=False
    )

    metrics = {
        "n_samples": int(len(trues)),
        "MAE":  round(mae,  4),
        "RMSE": round(rmse, 4),
        "ME":   round(me,   4),
        "n_train": int(len(test_df)),   # test set size stored; full split info in print
        "n_valid": len(valid_ds),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Loss curve ───────────────────────────────────────────────
    hist = history.history
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["loss"],     label="Train MAE (loss)")
    ax.plot(hist["val_loss"], label="Val MAE (loss)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (years)")
    ax.legend(); ax.set_title(f"{title} — Training Curve")
    fig.tight_layout()
    fig.savefig(result_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # ── Scatter + Error distribution ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(trues, preds, alpha=0.4, s=20)
    lim = max(float(trues.max()), float(preds.max())) + 5
    ax.plot([0, lim], [0, lim], "r--")
    ax.set_xlabel("True Age"); ax.set_ylabel("Predicted Age")
    ax.set_title(f"{title} — True vs Pred")
    ax = axes[1]
    ax.hist(preds - trues, bins=30, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Error (pred - true)"); ax.set_ylabel("Count")
    ax.set_title(f"{title} — Error Distribution")
    fig.tight_layout()
    fig.savefig(result_dir / "scatter.png", dpi=150)
    plt.close(fig)

    print(f"  저장 완료: {result_dir}")
    return {"MAE": mae, "RMSE": rmse, "ME": me}
