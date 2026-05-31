# pipeline_aihub.py

## 목적

AIHub 이미지 기반 나이 회귀 학습 스크립트들의 공통 파이프라인 모듈.  
`train_cnn_aihub.py` / `train_small_cnn_aihub.py` 에서 공유한다.

## 제공 함수

| 함수 | 설명 |
|------|------|
| `load_splits()` | `split_meta.csv` 기준 train/valid/test DataFrame 반환 |
| `make_dataset(df, img_size, batch_size, shuffle)` | DataFrame → tf.data.Dataset |
| `build_callbacks(result_dir, patience)` | ModelCheckpoint / EarlyStopping / ReduceLROnPlateau |
| `run_training(...)` | 컴파일 → 학습 → 평가 → 결과 저장 |

## `run_training` 인터페이스

```python
run_training(
    model,        # tf.keras.Model (회귀, Dense(1) 출력)
    train_ds,     # tf.data.Dataset
    valid_ds,
    test_ds,
    test_df,      # 정답 레이블(photo_age) 포함 DataFrame
    result_dir,   # 결과 저장 경로 (Path)
    title,        # 플롯 제목 문자열
    epochs,
    patience,
    lr,
)
```

## 저장 파일

| 파일 | 내용 |
|------|------|
| `{result_dir}/metrics.json` | MAE, RMSE, ME |
| `{result_dir}/predictions.csv` | true_age / pred_age / error |
| `{result_dir}/loss_curve.png` | Train/Val MAE 곡선 |
| `{result_dir}/scatter.png` | True vs Pred + Error Distribution |
| `{result_dir}/checkpoints/best_model.keras` | val_loss 기준 best 모델 |
