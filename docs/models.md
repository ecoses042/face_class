# models.py

## 목적

DeepFace 임베딩 위에 얹는 FCN age regressor 모델을 정의한다.

## 구조

```
embedding (512-d)
→ Linear(512→256) → BatchNorm → ReLU → Dropout(0.3)
→ Linear(256→128) → BatchNorm → ReLU → Dropout(0.2)
→ Linear(128→64)  → ReLU
→ Linear(64→1)    → 예측 나이 (scalar)
```

DeepFace 모델 자체는 고정(학습 없음). 이 FCN만 학습된다.
