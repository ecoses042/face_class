"""
train_small_cnn_classifier.py
소형 CNN (~3.3M)으로 AIHub 이미지에서 나이 3-class 분류를 학습한다.
클래스: young (<20) / adult (20-60) / senior (60+)

실행:
    python src/train_small_cnn_classifier.py [--img-size 128] [--epochs 30] [--lr 1e-3] [--batch-size 32]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# sample_weight 경로가 GPU XLA Triton autotuner와 충돌하는 문제 우회
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR
from models_cnn import build_small_cnn_classifier
from pipeline_aihub import (
    age_to_class,
    build_callbacks,
    compute_class_weights,
    load_splits,
    make_dataset_cls,
)

# ---------------------------------------------------------------------------
CLASS_NAMES = ["young (<20)", "adult (20-60)", "senior (60+)"]
RESULT_DIR  = RESULTS_DIR / "cnn_small_classifier"


def build_cls_callbacks(result_dir: Path, patience: int):
    """EarlyStopping / ReduceLROnPlateau / Checkpoint on val_accuracy."""
    import tensorflow as tf

    ckpt_dir = result_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best_model.keras"),
            monitor="val_accuracy", mode="max", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=patience, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            patience=max(3, patience // 2), factor=0.5, min_lr=1e-5,
        ),
    ]


def plot_confusion_matrix(cm, title, out_path):
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
        ax.set_yticklabels(CLASS_NAMES, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size",   type=int,   default=128)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    import tensorflow as tf
    tf.config.optimizer.set_jit(False)  # sample_weight + GPU XLA Triton 충돌 우회

    print("데이터 로딩 (원본 이미지, use_crops=False) ...")
    train_df, valid_df, test_df = load_splits(use_crops=False)
    print(f"  train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}")

    # Class distribution
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        cls_counts = df["age_past"].apply(age_to_class).value_counts().sort_index()
        print(f"  {split_name} class distribution: {cls_counts.to_dict()}")

    cw = compute_class_weights(train_df, age_col="age_past")
    print(f"  클래스 가중치: {cw}")

    train_ds = make_dataset_cls(train_df, args.img_size, args.batch_size, shuffle=True,  age_col="age_past", class_weights=cw)
    valid_ds = make_dataset_cls(valid_df, args.img_size, args.batch_size, shuffle=False, age_col="age_past")
    test_ds  = make_dataset_cls(test_df,  args.img_size, args.batch_size, shuffle=False, age_col="age_past")

    model = build_small_cnn_classifier(img_size=args.img_size, num_classes=3)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=build_cls_callbacks(RESULT_DIR, args.patience),
    )

    # Test predictions
    probs     = model.predict(test_ds, verbose=0)   # (N, 3)
    y_pred    = probs.argmax(axis=1)
    y_true    = test_df["age_past"].apply(age_to_class).to_numpy()
    true_ages = test_df["age_past"].to_numpy()

    acc         = float(accuracy_score(y_true, y_pred))
    macro_f1    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print(f"\n=== Test 결과 (Small CNN Classifier) ===")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2],
                                 target_names=CLASS_NAMES, zero_division=0))

    # predictions.csv
    pd.DataFrame({
        "true_class": y_true,
        "pred_class": y_pred,
        "true_age":   true_ages,
    }).to_csv(RESULT_DIR / "predictions.csv", index=False)

    # metrics.json
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                    target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(report[CLASS_NAMES[i]]["precision"], 4),
            "recall":    round(report[CLASS_NAMES[i]]["recall"],    4),
            "f1":        round(report[CLASS_NAMES[i]]["f1-score"],  4),
            "support":   int(report[CLASS_NAMES[i]]["support"]),
        }
        for i in range(3)
    }
    metrics = {
        "n_samples":   int(len(y_true)),
        "accuracy":    round(acc,         4),
        "macro_f1":    round(macro_f1,    4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class":   per_class,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Loss curve
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["loss"],     label="Train Loss")
    axes[0].plot(hist["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title("Small CNN Classifier — Loss")
    axes[1].plot(hist["accuracy"],     label="Train Acc")
    axes[1].plot(hist["val_accuracy"], label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].set_title("Small CNN Classifier — Accuracy")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm, "Small CNN Classifier", RESULT_DIR / "confusion_matrix.png")

    print(f"\n  저장 완료: {RESULT_DIR}")


if __name__ == "__main__":
    main()
