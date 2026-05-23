# evaluate.py

## 목적

`predictions.csv`를 읽어 MAE/RMSE/연령대별 분석 및 산점도/오차 분포 그래프를 생성한다.

## 실행

```bash
# baseline 평가
.venv/bin/python src/evaluate.py --pred results/baseline_deepface/predictions.csv

# FCN 모델 평가
.venv/bin/python src/evaluate.py --pred results/fcn_regressor/predictions.csv

# 두 모델 비교
.venv/bin/python src/evaluate.py --compare
```

## 출력 파일

| 파일 | 설명 |
|------|------|
| `scatter.png` | True vs Predicted Age 산점도 |
| `error_dist.png` | 오차 분포 히스토그램 |
| `results/comparison/model_comparison.csv` | `--compare` 시 모델별 지표 비교표 |
