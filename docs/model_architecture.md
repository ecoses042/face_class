# 모델 구조 설명

## 전체 파이프라인

```
얼굴 이미지 (PNG)
        │
        ▼
┌───────────────────────────────┐
│   DeepFace  (ArcFace)         │  ← 고정 (학습 없음)
│   사전학습된 얼굴 인식 모델    │
│   입력: 112 × 112 × 3         │
│   출력: 512-d embedding vector │
└───────────────────────────────┘
        │
        ▼  512-d float32 벡터
┌───────────────────────────────┐
│   FCN Age Regressor           │  ← 학습 대상
│   (AgeRegressor in models.py) │
└───────────────────────────────┘
        │
        ▼  scalar
    예측 나이 (photo_age)
```

---

## 1. DeepFace ArcFace (Feature Extractor)

### 역할

얼굴 이미지를 512차원 임베딩 벡터로 변환한다.  
이 벡터는 얼굴의 신원(identity) 정보를 압축한 표현이다.

### 학습 방식 (원본)

ArcFace는 **Additive Angular Margin Loss**로 학습된 얼굴 인식 모델이다.  
같은 인물의 임베딩은 가깝게, 다른 인물은 멀게 배치되도록 학습되어 얼굴 특징을 잘 분리한다.

```
Loss = -log( e^(s·cos(θ_yi + m)) / (e^(s·cos(θ_yi + m)) + Σ e^(s·cos(θ_j))) )
       ↑ margin m을 각도에 더해 클래스 간 분리를 강화
```

### 입출력

| 항목 | 값 |
|------|----|
| 입력 크기 | 112 × 112 × 3 (RGB) |
| 출력 차원 | 512-d L2 정규화 벡터 |
| 가중치 파일 | `~/.deepface/weights/arcface_weights.h5` (137 MB) |
| 학습 데이터 | MS-Celeb-1M (약 1000만 장, 10만 명) |

### 본 프로젝트에서의 역할

DeepFace는 **고정(frozen)** 상태로 사용한다.  
이미지마다 한 번씩만 추출해 `.npy`로 저장하므로, FCN 학습 시 재계산하지 않는다.

---

## 2. FCN Age Regressor

### 역할

ArcFace 임베딩(512-d)을 입력받아 나이(scalar)를 예측하는 회귀 모델이다.

### 레이어 구조

```
입력: 512-d

Linear(512 → 256)
BatchNorm1d(256)        ← 학습 안정화, 빠른 수렴
ReLU
Dropout(p=0.3)          ← 과적합 방지

Linear(256 → 128)
BatchNorm1d(128)
ReLU
Dropout(p=0.2)

Linear(128 → 64)
ReLU

Linear(64 → 1)          ← 나이 scalar 출력
```

### 설계 근거

| 선택 | 이유 |
|------|------|
| BatchNorm | 임베딩 스케일 편차를 흡수하고 gradient vanishing 방지 |
| Dropout 0.3 → 0.2 | 앞 레이어에 더 강한 정규화, 뒤로 갈수록 완화 |
| HuberLoss (δ=5) | MSE 대비 큰 오차(이상치)에 덜 민감 |
| 출력에 활성함수 없음 | 나이는 연속값이므로 linear output |

### 파라미터 수

```
Linear(512→256):  512×256 + 256 = 131,328
BN(256):          256×2         =     512
Linear(256→128):  256×128 + 128 =  32,896
BN(128):          128×2         =     256
Linear(128→64):   128×64  + 64  =   8,256
Linear(64→1):     64×1    + 1   =      65
─────────────────────────────────────────
총 파라미터:                       173,313  (~173K)
```

---

## 3. 학습 설정

| 항목 | 값 |
|------|----|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5, min_lr=1e-5) |
| Loss | HuberLoss (delta=5.0) |
| Early stopping | val MAE 기준, patience=15 |
| Batch size | 64 |
| Max epochs | 100 |

---

## 4. Baseline vs Proposed 비교

| 항목 | Baseline (DeepFace) | Proposed (DeepFace + FCN) |
|------|--------------------|-----------------------------|
| 추가 학습 | 없음 | FCN 173K 파라미터 |
| 입력 | 이미지 직접 | 사전 추출된 512-d 임베딩 |
| 추론 속도 | 느림 (이미지마다 DeepFace) | 빠름 (임베딩 → FCN forward) |
| 한국인 특화 | 없음 | FCN이 한국인 데이터로 보정 |
| 예상 장점 | 별도 학습 불필요 | 연령대별 편향 보정 가능 |

---

## 5. 나이 정답값 정의

```
photo_age = age_now - age_past
```

- `age_now`: 데이터 수집 시점(2022년) 기준 나이
- `age_past`: 해당 사진이 몇 년 전 촬영인지 (0이면 최근 촬영)
- `photo_age`: **사진 속 얼굴의 실제 나이** → 모델이 예측해야 하는 값
