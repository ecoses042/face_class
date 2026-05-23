"""
train_fcn.py
사전 추출된 임베딩으로 FCN age regressor를 학습한다.

실행:
    python src/train_fcn.py [--epochs 100] [--lr 1e-3] [--batch-size 64]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from config import EMBED_DIR, RESULTS_DIR
from models import AgeRegressor

CKPT_DIR = RESULTS_DIR / "fcn_regressor" / "checkpoints"


def load_split(split: str):
    emb = np.load(EMBED_DIR / f"{split}_embeddings.npy")
    lbl = np.load(EMBED_DIR / f"{split}_labels.npy")
    return torch.tensor(emb), torch.tensor(lbl)


def make_loader(split: str, batch_size: int, shuffle: bool) -> DataLoader:
    X, y = load_split(split)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


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
    preds, trues = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu())
        trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return mae, rmse, preds, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Early stopping patience")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 임베딩 차원 자동 감지
    emb_dim = int((EMBED_DIR / "embedding_dim.txt").read_text().strip())
    print(f"임베딩 차원: {emb_dim}")

    train_loader = make_loader("train", args.batch_size, shuffle=True)
    valid_loader = make_loader("valid", args.batch_size, shuffle=False)

    print(f"train: {len(train_loader.dataset):,}  valid: {len(valid_loader.dataset):,}")

    model     = AgeRegressor(emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.HuberLoss(delta=5.0)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mae, no_improve = float("inf"), 0
    history = []

    print(f"\n학습 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValMAE':>8}  {'ValRMSE':>9}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mae, val_rmse, _, _ = evaluate(model, valid_loader, device)
        scheduler.step(val_mae)

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mae": val_mae, "val_rmse": val_rmse})

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_mae:>8.2f}  {val_rmse:>9.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (best val MAE: {best_mae:.2f})")
                break

    # 테스트 평가
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pt", map_location=device))
    test_loader = make_loader("test", args.batch_size, shuffle=False)
    test_mae, test_rmse, preds, trues = evaluate(model, test_loader, device)

    print(f"\n=== Test 결과 ===")
    print(f"  MAE : {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    me = float(np.mean(preds - trues))
    print(f"  ME  : {me:+.2f}")

    # 결과 저장
    result_dir = RESULTS_DIR / "fcn_regressor"
    result_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    pd.DataFrame({"true_age": trues, "pred_age": preds,
                  "error": preds - trues}).to_csv(result_dir / "predictions.csv", index=False)

    metrics = {"n_samples": int(len(trues)), "MAE": round(test_mae, 4),
               "RMSE": round(test_rmse, 4), "ME": round(me, 4)}
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # loss curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["epoch"], hist_df["val_mae"], label="Val MAE")
    ax.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss", alpha=0.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.legend(); ax.set_title("FCN Training Curve")
    fig.tight_layout()
    fig.savefig(result_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    print(f"\n  저장 완료: {result_dir}")
    print(f"[완료] python src/evaluate.py --compare 로 baseline과 비교 가능")


if __name__ == "__main__":
    main()
