# train_cnn_aihub.py

## 목적

AIHub 안면 인식 에이징 데이터셋으로 VGG-like CNN age regressor를 학습한다.  
`train_fcn.py`(FCN AgeRegressor)와 **동일한** train/valid/test split(`split_meta.csv`)을 사용하여 두 모델을 공정하게 비교한다.

## 입력 데이터

| 파일 | 역할 |
|------|------|
| `dataset/embeddings/split_meta.csv` | FCN과 동일한 train/valid/test 분할 (4,355건) |
| `dataset/processed/metadata.csv` | 이미지 경로 및 `photo_age` 레이블 |
| `dataset/processed/images/` | 전처리된 얼굴 PNG 이미지 |

## 모델

`models_cnn.build_cnn_regressor` — VGG-like CNN (~33.6M 파라미터)  
출력: `Dense(1)` (activation 없음, 연속 나이 예측)

## 실행

```bash
python src/train_cnn_aihub.py \
  [--img-size 128] \
  [--epochs 30] \
  [--lr 1e-3] \
  [--batch-size 32] \
  [--patience 5]
```

## 출력

| 경로 | 내용 |
|------|------|
| `results/cnn_aihub/metrics.json` | MAE, RMSE, ME, 샘플 수 |
| `results/cnn_aihub/predictions.csv` | true_age / pred_age / error |
| `results/cnn_aihub/loss_curve.png` | Train/Val MAE 곡선 |
| `results/cnn_aihub/checkpoints/best_model.keras` | val_mae 기준 best 모델 |

## FCN과의 비교

두 모델 모두 동일한 4,355개 AIHub 샘플(split_meta.csv)을 사용하므로 MAE/RMSE를 직접 비교할 수 있다.

| 모델 | 입력 | 특징 |
|------|------|------|
| FCN AgeRegressor | DeepFace 임베딩 (512-d 벡터) | 가볍고 빠름 |
| CNN Regressor | 원본 얼굴 이미지 (128×128) | end-to-end 학습 |
