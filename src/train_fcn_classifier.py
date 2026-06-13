"""
train_fcn_classifier.py
사전 추출된 임베딩으로 FCN age classifier (3-class)를 학습한다.

실행:
    python src/train_fcn_classifier.py [--epochs 100] [--lr 1e-3] [--batch-size 64] [--patience 15]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from config import EMBED_DIR, RESULTS_DIR
from models import AgeClassifier

# ---------------------------------------------------------------------------
RESULT_DIR = RESULTS_DIR / "fcn_classifier"
CKPT_DIR   = RESULT_DIR / "checkpoints"
CLASS_NAMES = ["young (<20)", "adult (20-60)", "senior (60+)"]
N_CLASSES   = 3


def age_to_class(age: float) -> int:
    """age < 20 → 0, 20 ≤ age < 60 → 1, age ≥ 60 → 2"""
    if age < 20:
        return 0
    elif age < 60:
        return 1
    return 2


def load_split(split: str):
    emb = np.load(EMBED_DIR / f"{split}_embeddings.npy")
    lbl = np.load(EMBED_DIR / f"{split}_labels.npy")  # continuous ages
    cls = np.array([age_to_class(a) for a in lbl], dtype=np.int64)
    return torch.tensor(emb, dtype=torch.float32), torch.tensor(cls, dtype=torch.long), lbl


def make_loader(split: str, batch_size: int, shuffle: bool) -> tuple[DataLoader, np.ndarray]:
    X, y_cls, y_age = load_split(split)
    loader = DataLoader(TensorDataset(X, y_cls), batch_size=batch_size, shuffle=shuffle)
    return loader, y_age


def compute_class_weights(y_cls: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y_cls, minlength=N_CLASSES).float()
    total  = float(len(y_cls))
    weights = total / (N_CLASSES * counts)
    return weights


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for X, y in loader:
        logits = model(X.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(dim=1).numpy()
    trues  = labels.numpy()
    acc    = float(accuracy_score(trues, preds))
    return acc, preds, probs, trues


def plot_confusion_matrix(cm, out_path):
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
    ax.set_title("FCN Classifier — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--patience",   type=int,   default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    emb_dim = int((EMBED_DIR / "embedding_dim.txt").read_text().strip())
    print(f"임베딩 차원: {emb_dim}")

    train_loader, _    = make_loader("train", args.batch_size, shuffle=True)
    valid_loader, _    = make_loader("valid", args.batch_size, shuffle=False)

    # Compute class weights from training labels
    _, train_cls, _ = load_split("train")
    class_weights = compute_class_weights(train_cls)
    print(f"클래스 가중치: {class_weights.tolist()}")

    print(f"train: {len(train_loader.dataset):,}  valid: {len(valid_loader.dataset):,}")

    model     = AgeClassifier(emb_dim, num_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc, no_improve = -1.0, 0
    history = []

    print(f"\n학습 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValAcc':>8}")
    print("-" * 32)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _, _ = evaluate(model, valid_loader, device)
        scheduler.step(-val_acc)  # ReduceLROnPlateau expects a metric to minimize

        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_acc:>8.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (best val acc: {best_acc:.4f})")
                break

    # Test evaluation
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pt", map_location=device))
    test_loader, test_ages = make_loader("test", args.batch_size, shuffle=False)
    test_acc, preds, probs, trues = evaluate(model, test_loader, device)

    print(f"\n=== Test 결과 ===")
    print(f"  Accuracy: {test_acc:.4f}")
    print(classification_report(trues, preds, labels=[0, 1, 2],
                                 target_names=CLASS_NAMES, zero_division=0))

    macro_f1    = float(f1_score(trues, preds, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(trues, preds, average="weighted", zero_division=0))

    # Save results
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "true_class":  trues,
        "pred_class":  preds,
        "true_age":    test_ages,
        "pred_prob_0": probs[:, 0],
        "pred_prob_1": probs[:, 1],
        "pred_prob_2": probs[:, 2],
    })
    pred_df.to_csv(RESULT_DIR / "predictions.csv", index=False)

    report = classification_report(trues, preds, labels=[0, 1, 2],
                                    target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(report[CLASS_NAMES[i]]["precision"], 4),
            "recall":    round(report[CLASS_NAMES[i]]["recall"],    4),
            "f1":        round(report[CLASS_NAMES[i]]["f1-score"],  4),
            "support":   int(report[CLASS_NAMES[i]]["support"]),
        }
        for i in range(N_CLASSES)
    }
    metrics = {
        "n_samples":   int(len(trues)),
        "accuracy":    round(test_acc,    4),
        "macro_f1":    round(macro_f1,    4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class":   per_class,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Loss curve
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    ax.plot(hist_df["epoch"], hist_df["val_acc"],    label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("FCN Classifier — Training Curve")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(trues, preds, labels=[0, 1, 2])
    plot_confusion_matrix(cm, RESULT_DIR / "confusion_matrix.png")

    print(f"\n  저장 완료: {RESULT_DIR}")


if __name__ == "__main__":
    main()
