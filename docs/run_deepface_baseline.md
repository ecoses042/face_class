# run_deepface_baseline.py

## 목적

DeepFace 기본 age prediction 기능을 Validation 이미지에 실행하여 baseline 성능을 측정한다.

## 실행

```bash
# 전체 Validation 평가
.venv/bin/python src/run_deepface_baseline.py

# 정면 얼굴만 평가
.venv/bin/python src/run_deepface_baseline.py --frontal-only

# 빠른 테스트 (N개 샘플만)
.venv/bin/python src/run_deepface_baseline.py --limit 100
```

## 출력

| 파일 | 설명 |
|------|------|
| `results/baseline_deepface/predictions.csv` | 이미지별 true_age / pred_age / error |
| `results/baseline_deepface/metrics.json` | MAE, RMSE, ME, 샘플 수 |

## 평가 지표

| 지표 | 설명 |
|------|------|
| MAE | 평균 절대 오차 |
| RMSE | 평균 제곱근 오차 (큰 오차에 민감) |
| ME | 평균 오차 (양수=과대 예측, 음수=과소 예측) |
