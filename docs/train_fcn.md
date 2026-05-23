# train_fcn.py

## 목적

사전 추출된 임베딩(.npy)으로 FCN age regressor를 학습하고 test 성능을 측정한다.

## 실행

```bash
.venv/bin/python src/train_fcn.py
.venv/bin/python src/train_fcn.py --epochs 200 --lr 5e-4 --batch-size 32
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 100 | 최대 학습 epoch |
| `--lr` | 1e-3 | 초기 학습률 |
| `--batch-size` | 64 | 배치 크기 |
| `--patience` | 15 | Early stopping patience |

## 학습 설정

- Loss: HuberLoss (delta=5.0) — 큰 오차에 덜 민감
- Optimizer: Adam + weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: val MAE 기준

## 출력

| 파일 | 설명 |
|------|------|
| `results/fcn_regressor/checkpoints/best_model.pt` | 최고 val MAE 체크포인트 |
| `results/fcn_regressor/predictions.csv` | test 예측 결과 |
| `results/fcn_regressor/metrics.json` | MAE / RMSE / ME |
| `results/fcn_regressor/loss_curve.png` | 학습 곡선 |
