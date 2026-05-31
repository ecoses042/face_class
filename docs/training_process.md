# 학습 과정 문서

## 전체 파이프라인 개요

```
AIHub 원본 데이터
    │
    ▼
preprocess.py          → dataset/processed/metadata.csv
                          dataset/processed/images/
    │
    ▼
extract_embeddings.py  → dataset/embeddings/
                          {train,valid,test}_embeddings.npy
                          {train,valid,test}_labels.npy
                          split_meta.csv
                          embedding_dim.txt
    │
    ├──▶ train_fcn.py      → FCN AgeRegressor (회귀)
    │
    └──▶ train_cnn_aihub.py → VGG-like CNN Regressor (회귀)

Adience fold_4
    │
    └──▶ train_cnn.py      → VGG-like CNN Classifier (분류)
```

---

## 1. FCN AgeRegressor (`train_fcn.py`)

### 데이터
- **출처**: AIHub 안면 인식 에이징 데이터셋
- **입력**: DeepFace 임베딩 벡터 (512-d float32 numpy array)
- **레이블**: `photo_age` (연속 나이, 1–80세)
- **분할**: `split_meta.csv` 기준 (train 3,550 / valid 412 / test 393)

### 모델 (`models.AgeRegressor`)
| 레이어 | 출력 크기 |
|--------|----------|
| Linear(512→256) + ReLU + BN + Dropout(0.3) | 256 |
| Linear(256→128) + ReLU + BN + Dropout(0.2) | 128 |
| Linear(128→1) | 1 |

- **총 파라미터**: ~200K
- **프레임워크**: PyTorch

### 학습 설정
| 항목 | 값 |
|------|-----|
| Optimizer | Adam (weight_decay=1e-4) |
| Loss | HuberLoss (δ=5.0) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Initial LR | 1e-3 |
| Epochs | 최대 100 (early stopping patience=15) |
| Batch size | 64 |

### 실행 명령
```bash
python src/train_fcn.py --epochs 100 --lr 1e-3 --batch-size 64
```

### 결과 저장 위치
```
results/fcn_regressor/
├── checkpoints/best_model.pt
├── metrics.json          # MAE, RMSE, ME
├── predictions.csv       # true_age, pred_age, error
└── loss_curve.png
```

---

## 2. VGG-like CNN Regressor (`train_cnn_aihub.py`)

### 데이터
- **출처**: AIHub 안면 인식 에이징 데이터셋 (FCN과 동일)
- **입력**: 얼굴 PNG 이미지 → 128×128×3 리사이즈 → [0, 1] 정규화
- **레이블**: `photo_age` (연속 나이, 1–80세)
- **분할**: `split_meta.csv` 기준 (FCN과 **동일**: train 3,550 / valid 412 / test 393)

### 모델 (`models_cnn.build_cnn_regressor`)
| 블록 | 구성 | 출력 크기 |
|------|------|----------|
| Block 1 | Conv64×2 + MaxPool | 64×64×64 |
| Block 2 | Conv128×2 + MaxPool | 32×32×128 |
| Block 3 | Conv256×3 + MaxPool | 16×16×256 |
| Block 4 | Conv512×3 + MaxPool | 8×8×512 |
| Block 5 | Conv512×3 + MaxPool | 4×4×512 |
| FC | Dense(2048)+Dropout(0.5) → Dense(1024)+Dropout(0.5) → Dense(1) | 1 |

- **총 파라미터**: 33,593,153 (~128 MB)
- **프레임워크**: TensorFlow/Keras

### 학습 설정
| 항목 | 값 |
|------|-----|
| Optimizer | Adam |
| Loss | Huber (δ=5.0) |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5, min=1e-5) |
| Initial LR | 1e-3 |
| Epochs | 최대 30 (early stopping patience=5) |
| Batch size | 32 |
| GPU | NVIDIA GeForce RTX 3060 Ti |

### 실행 명령
```bash
python src/train_cnn_aihub.py --epochs 30 --batch-size 32 --patience 5
```

### 결과 저장 위치
```
results/cnn_aihub/
├── checkpoints/best_model.keras
├── metrics.json          # MAE, RMSE, ME, n_train, n_valid, n_test
├── predictions.csv       # true_age, pred_age, error
└── loss_curve.png
```

---

## 3. VGG-like CNN Classifier (`train_cnn.py`)

### 데이터
- **출처**: Adience 데이터셋 fold_4
- **입력**: 얼굴 JPEG 이미지 → 128×128×3 → [0, 1] 정규화
- **레이블**: 8개 나이 구간 클래스 (0–7)
  - 0: (0,2), 1: (4,6), 2: (8,12), 3: (15,20)
  - 4: (25,32), 5: (38,43), 6: (48,53), 7: (60,100)
- **분할**: user_id 기반 분할 (identity leakage 방지)

### 모델 (`models_cnn.build_cnn_model`)
- FCN Regressor와 동일한 conv 블록 구조
- 마지막 레이어: `Dense(8, activation='softmax')`

### 학습 설정
| 항목 | 값 |
|------|-----|
| Optimizer | Adam |
| Loss | sparse_categorical_crossentropy |
| Initial LR | 1e-3 |
| Epochs | 최대 30 (early stopping patience=5) |
| Batch size | 32 |

### 실행 명령
```bash
python src/train_cnn.py --data-dir <Adience_data_dir> --epochs 30
```

### 결과 저장 위치
```
results/cnn_adience/
├── checkpoints/best_model.keras
├── metrics.json          # test_accuracy, test_loss, split 분포
└── accuracy_curve.png
```

---

## 모델 비교 요약

| 모델 | 데이터셋 | 태스크 | 입력 | 파라미터 | 평가지표 |
|------|---------|--------|------|---------|---------|
| DeepFace baseline | AIHub | 회귀 | 이미지 | ~34M (frozen) | MAE, RMSE |
| FCN AgeRegressor | AIHub | 회귀 | 임베딩 (512-d) | ~200K | MAE, RMSE |
| CNN Regressor | AIHub | 회귀 | 이미지 (128×128) | ~33.6M | MAE, RMSE |
| CNN Classifier | Adience | 분류 | 이미지 (128×128) | ~33.6M | Accuracy |

> **참고**: DeepFace baseline과 FCN은 동일한 split_meta.csv를 사용하므로 MAE/RMSE를 직접 비교 가능.  
> CNN Classifier는 다른 데이터셋(Adience)을 사용하므로 직접 수치 비교는 불가.
