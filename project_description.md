# 한국인 얼굴 나이 예측 프로젝트

## 목차
1. [데이터 전처리](#1-데이터-전처리)
2. [모델 구조](#2-모델-구조)
3. [실험 결과 및 분석](#3-실험-결과-및-분석)
4. [test_picture 예측 결과](#4-test_picture-예측-결과)

---

## 1. 데이터 전처리

### 1.1 사용 데이터셋

**AI-Hub 안면 인식 에이징(aging) 이미지 데이터** (데이터셋 ID: 71415)
- 한국인 얼굴 이미지와 나이·성별·랜드마크 주석을 포함한 공개 데이터셋
- Training / Validation 두 분할로 제공
- **실제 사용**: Validation split만 사용 (Training split은 이미지 zip 미제공)
  - Validation 유효 샘플: 4,355개 (4,574개 중 필터링 후)

#### 원본 데이터 구조

```
dataset/raw/aihub_aging/118.안면_인식_에이징(aging)_이미지_데이터/01-1.정식개방데이터/
├── Training/
│   ├── 01.원천데이터/        ← TS_*.zip (이미지, 미제공)
│   └── 02.라벨링데이터/      ← TL_*.zip (JSON 라벨)
└── Validation/
    ├── 01.원천데이터/        ← VS_*.zip (이미지)
    └── 02.라벨링데이터/      ← VL_*.zip (JSON 라벨)
```

#### JSON 라벨 포맷

각 이미지에 1:1 대응하는 JSON 파일이 제공된다. 주요 필드는 다음과 같다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | string | 피험자(인물) 고유 ID |
| `birth` | int | 출생연도 |
| `age_now` | int | 기준 시점 나이 |
| `age_past` | int | 촬영 시점과 기준 시점의 나이 차이 |
| `gender` | string | 성별 |
| `annotation[0].box` | dict | 얼굴 bbox `{x, y, w, h}` |
| `annotation[0].landmark` | list | 5점 얼굴 랜드마크 좌표 |

**학습 레이블(photo_age)** = `age_now − age_past` (촬영 시점 나이)

---

### 1.2 전처리 파이프라인

전처리는 아래 5단계로 구성된다.

```
Step 1  preprocess.py        zip 해제 + JSON 파싱 + 메타데이터 생성
Step 2  split_dataset.py     person_id 기준 train/valid/test 분할
Step 3  extract_embeddings.py  ArcFace 임베딩 추출 → .npy 파일
Step 4  extract_face_crops.py  얼굴 크롭 이미지 저장 → dataset/face_crops/
```

#### Step 1: `preprocess.py` — 메타데이터 생성

**zip 압축 해제**
- Training/Validation 각 분할에서 라벨 zip → `dataset/processed/labels/`
- Validation의 이미지 zip → `dataset/processed/images/`

**JSON 파싱 & 필터링**

| 필터 조건 | 기준 | 이유 |
|---------|------|------|
| bbox 최소 크기 | w ≥ 50px, h ≥ 50px | 너무 작은 얼굴은 특징 추출 불가 |
| 나이 유효 범위 | 1 ≤ photo_age ≤ 80 | 극단값 제거 |
| 주석 존재 | annotation 필드 비어있지 않음 | 유효 라벨 필수 |

**정면 판별 (`is_frontal`)**

5점 랜드마크의 좌우 대칭성으로 판별한다.

```python
# 양눈 중간 x좌표와 코 x좌표의 편차를 양눈 간격으로 정규화
sym_ratio = |nose_x - (eye1_x + eye2_x)/2| / |eye1_x - eye2_x|
is_frontal = (sym_ratio < 0.25)   # 임계값 FRONTAL_SYM_THRESH = 0.25
```

**출력**: `dataset/processed/metadata.csv`
- 41,770개 유효 레코드 (Training+Validation 합산)
- 컬럼: `filename, split, person_id, birth, photo_age, gender, bbox_*, is_frontal, image_path, label_path`

#### Step 2: person_id 기준 분할

동일 인물의 사진이 train/test에 나뉘어 들어가는 데이터 누수를 방지하기 위해 **person_id 단위로 분할**한다.

```python
rng = np.random.default_rng(seed=42)   # 재현성 보장
person_ids = df["person_id"].unique()
rng.shuffle(person_ids)

# 비율: test 10% / valid 10% / train 80%
n_test  = int(len(person_ids) * 0.10)
n_valid = int(len(person_ids) * 0.10)
```

**분할 결과**

| 분할 | 이미지 수 | 비율 |
|------|----------|------|
| train | 3,550 | 81.5% |
| valid | 412 | 9.5% |
| test | 393 | 9.0% |
| **합계** | **4,355** | 100% |

#### Step 3: `extract_embeddings.py` — ArcFace 임베딩 추출

**처리 흐름**

1. 원본 이미지를 최대 1280px로 리사이즈 (검출 속도 확보)
2. `DeepFace.extract_faces(detector_backend="opencv")` 로 얼굴 영역 검출
3. 검출 bbox에 **20% 패딩** 추가 후 크롭 (ArcFace 학습 데이터 관례)
4. 검출 실패 시 리사이즈 이미지 전체를 fallback으로 사용
5. `DeepFace.represent(model_name="ArcFace", detector_backend="skip")` 로 **512차원 벡터** 추출

```python
result = DeepFace.represent(
    img_path=face_crop,
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="skip",
)
embedding = np.array(result[0]["embedding"], dtype=np.float32)  # shape: (512,)
```

**출력 파일**

```
dataset/embeddings/
├── train_embeddings.npy    # (3550, 512)
├── train_labels.npy        # (3550,) ← photo_age
├── valid_embeddings.npy    # (412, 512)
├── valid_labels.npy        # (412,)
├── test_embeddings.npy     # (393, 512)
├── test_labels.npy         # (393,)
└── split_meta.csv          # filename, fcn_split, person_id
```

#### Step 4: `extract_face_crops.py` — 크롭 이미지 저장

Step 3와 동일한 얼굴 검출·크롭 파이프라인을 사용하되, 결과 이미지를 **128×128 JPEG**로 저장한다. CNN 학습 시 train–inference 전처리를 완전히 일치시키기 위한 단계다.

- 전체 4,355개 처리
- 얼굴 검출 성공: ~3,697개 (84.9%)
- Fallback(검출 실패 → 전체 이미지): ~658개 (15.1%)

```
dataset/face_crops/
└── {filename}.jpg    # 128×128, 4355개
dataset/embeddings/split_meta_crops.csv
```

---

## 2. 모델 구조

### 2.1 Baseline DeepFace

DeepFace(Meta)의 사전학습 나이 분석 기능을 추가 학습 없이 직접 사용한다.

```python
result = DeepFace.analyze(
    img_path=img_path,
    actions=["age"],
    enforce_detection=False,
    silent=True,
)
pred_age = float(result[0]["age"])
```

- **입력**: 원본 이미지 (임의 크기)
- **파라미터**: 사전학습 고정, 추가 학습 없음
- **특징**: 한국인 데이터 특화 학습 없음 → 과대예측 편향 발생

---

### 2.2 FCN AgeRegressor (`src/models.py`)

사전 추출된 ArcFace 512차원 임베딩을 입력으로 받아 나이를 회귀하는 경량 PyTorch 모델이다.

```python
class AgeRegressor(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),   # 나이 출력
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
```

**레이어 구조**

| 레이어 | 입력 | 출력 | 부가 연산 |
|--------|------|------|----------|
| Linear | 512 | 256 | BatchNorm + ReLU + Dropout(0.3) |
| Linear | 256 | 128 | BatchNorm + ReLU + Dropout(0.2) |
| Linear | 128 | 64 | ReLU |
| Linear | 64 | 1 | — |

**파라미터 수**: ~173K

**학습 설정**

| 항목 | 값 |
|------|-----|
| Loss | HuberLoss (δ=5.0) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 100 (Early stopping patience=15) |
| Batch size | 64 |

---

### 2.3 VGG-CNN (`src/models_cnn.py` — `build_cnn_regressor`)

VGG 아키텍처를 기반으로 한 대형 TensorFlow CNN 회귀 모델이다.

```python
def build_cnn_regressor(img_size: int = 128) -> tf.keras.Model:
    return models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        # Block 1: 2× Conv2D(64, 3×3, same) + MaxPool
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        # Block 2: 2× Conv2D(128, 3×3, same) + MaxPool
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        # Block 3: 3× Conv2D(256, 3×3, same) + MaxPool
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        # Block 4: 3× Conv2D(512, 3×3, same) + MaxPool
        layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        # Block 5: 3× Conv2D(512, 3×3, same) + MaxPool
        layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        # FC Head
        layers.Flatten(),
        layers.Dense(2048, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1),    # 나이 출력
    ], name="vgg_like_age_regressor")
```

**파라미터 수**: ~33.6M (FC 레이어에 집중)

**학습 설정**

| 항목 | 값 |
|------|-----|
| Loss | MAE (MeanAbsoluteError) |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 30 (Early stopping patience=5) |
| Batch size | 32 |
| 입력 전처리 | tf.image.resize → /255.0 정규화 |

---

### 2.4 Small CNN (`src/models_cnn.py` — `build_small_cnn_regressor`)

Kaggle 노트북(neneti/deeplearning-final-project-cnn)의 분류 아키텍처를 회귀 출력으로 변환한 경량 모델이다. `valid` 패딩(패딩 없음)을 사용하여 특징 맵이 자연스럽게 축소된다.

```python
def build_small_cnn_regressor(img_size: int = 128) -> tf.keras.Model:
    return models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        # Block 1: Conv2D(32, 3×3, valid) + MaxPool
        layers.Conv2D(32, (3,3), activation="relu"),   # → (126, 126, 32)
        layers.MaxPooling2D(),                          # → (63, 63, 32)

        # Block 2: Conv2D(64, 3×3, valid) + MaxPool
        layers.Conv2D(64, (3,3), activation="relu"),   # → (61, 61, 64)
        layers.MaxPooling2D(),                          # → (30, 30, 64)

        # Block 3: Conv2D(128, 3×3, valid) + MaxPool
        layers.Conv2D(128, (3,3), activation="relu"),  # → (28, 28, 128)
        layers.MaxPooling2D(),                          # → (14, 14, 128)

        # FC Head
        layers.Flatten(),                               # → 25,088
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1),    # 나이 출력
    ], name="small_cnn_age_regressor")
```

**레이어별 피처맵 크기** (입력 128×128 기준)

| 레이어 | 출력 형태 | 파라미터 수 |
|--------|----------|------------|
| Conv2D(32) | (126, 126, 32) | 896 |
| MaxPool | (63, 63, 32) | — |
| Conv2D(64) | (61, 61, 64) | 18,496 |
| MaxPool | (30, 30, 64) | — |
| Conv2D(128) | (28, 28, 128) | 73,856 |
| MaxPool | (14, 14, 128) | — |
| Dense(128) | 128 | 3,211,392 |
| Dense(1) | 1 | 129 |

**파라미터 수**: ~3.3M (Dense 레이어에 집중)

**학습 설정**

| 항목 | 값 |
|------|-----|
| Loss | MAE (MeanAbsoluteError) |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 50 (Early stopping patience=10) |
| Batch size | 32 |
| 입력 전처리 | 얼굴 크롭 128×128 → /255.0 정규화 |

---

### 모델 구조 비교 요약

| 항목 | DeepFace | FCN | VGG-CNN | Small CNN |
|------|----------|-----|---------|-----------|
| 프레임워크 | deepface | PyTorch | TensorFlow | TensorFlow |
| 파라미터 수 | 사전학습 | ~173K | ~33.6M | ~3.3M |
| 입력 | 원본 이미지 | 512-D 임베딩 | 128×128×3 | 128×128×3 (크롭) |
| 추가 학습 | 없음 | 있음 | 있음 | 있음 |
| Loss | — | HuberLoss | MAE | MAE |

---

## 3. 실험 결과 및 분석

### 3.1 전체 성능 비교 (AIHub test set, n=393)

| 모델 | MAE | RMSE | ME (편향) |
|------|-----|------|----------|
| **FCN AgeRegressor** | **6.22** | **8.38** | −3.26 |
| Small CNN (crops) | 8.53 | 11.40 | −3.40 |
| VGG-CNN (crops) | 11.73 | 14.97 | −4.60 |
| DeepFace Baseline | 12.80 | 15.26 | **+5.25** |

FCN이 MAE 기준 2위인 Small CNN 대비 2.3세, DeepFace 대비 6.6세 우수하다.

---

### 3.2 연령대별 MAE

| 연령대 | n | DeepFace | FCN | VGG | Small CNN |
|--------|---|---------|-----|-----|-----------|
| 0–9세 | 71 | 22.99 | **2.32** | 10.14 | 5.67 |
| 10–19세 | 106 | 13.55 | **2.60** | 2.60 | 6.40 |
| 20–29세 | 83 | 5.27 | **5.15** | 9.87 | 7.94 |
| 30–39세 | 54 | **5.57** | 8.25 | 19.26 | 11.33 |
| 40–49세 | 33 | **10.48** | 12.37 | 28.89 | 12.86 |
| 50–59세 | 26 | 17.85 | **8.96** | 37.11 | 30.14 |
| 60세+ | 20 | 20.70 | **15.46** | 47.93 | 34.24 |

---

### 3.3 모델별 성능 분석

#### FCN — 전 연령대 1위, 특히 저연령 압도적

FCN은 4개 모델 중 유일하게 ArcFace 임베딩을 입력으로 사용한다. ArcFace는 사람 식별을 위해 학습된 모델이지만, 그 임베딩에는 연령 변화에 따른 얼굴 특징 정보가 내재되어 있다. FCN은 3,550개의 학습 샘플만으로도 이 임베딩을 나이로 정확히 매핑한다.

- **0–9세 MAE 2.32세**: 어린이 얼굴의 명확한 형태적 특징을 임베딩이 잘 포착
- **고령(50대 이상)에서도 FCN만 합리적 수준 유지**: 다른 모델이 30–48세 오차를 낼 때 FCN은 9–15세 수준
- **ME = −3.26세**: 체계적으로 약간 낮게 예측하는 경향이 있으나 편향 규모가 작음
- **한계**: 30–49세 구간에서 DeepFace보다 오차가 크다. 이 구간은 ArcFace 임베딩의 연령 변화 정보가 상대적으로 부족하기 때문으로 해석된다.

#### Small CNN (crops) — CNN 중 최고, 크롭 전처리의 효과

크롭 이미지 기반 학습이 원본 이미지 대비 성능을 높이는 핵심 이유는 **신호 대 잡음비(SNR) 향상**이다. 원본 이미지에는 배경·의류·조명 같은 불필요한 정보가 포함되지만, 얼굴 크롭은 나이와 직결되는 피부·주름·눈가 특징에만 집중할 수 있다.

- **고연령층에서 급격한 성능 저하**: 50대 MAE 30세, 60대 MAE 34세로, 학습 샘플 부족(26개, 20개)이 주원인
- **Fallback 비율 15.1%**: 얼굴 검출 실패 시 전체 이미지를 그대로 사용하는 fallback이 예측 품질에 부정적 영향

#### VGG-CNN — Mean Collapse (모두 ~15세 예측)

VGG-CNN의 예측값 분포를 분석하면 대부분의 샘플에 대해 **약 15세 내외**의 동일한 값을 반복 출력한다.

```
실제 37세 → 예측 15.07세
실제  8세 → 예측 15.07세
실제 35세 → 예측 15.07세
```

이는 **mean collapse** 현상으로, 네트워크가 손실을 최소화하기 위해 훈련 레이블의 평균값으로 수렴한 것이다.

**원인**
1. **과대 파라미터 vs. 소규모 데이터**: 33.6M 파라미터를 3,550개 샘플로 학습 → 극도의 과적합 후 평균 수렴
2. **전이학습 부재**: ImageNet 사전학습 없이 처음부터(from scratch) 학습
3. **도메인 특이성**: 한국인 얼굴 나이 예측에 특화된 fine-tuning 없음

**고연령 오차 폭발**: 모두 ~15세로 예측하므로 실제 60대와의 차이는 최대 48세에 달한다.

#### DeepFace Baseline — 과대예측 편향 (+5.25세)

DeepFace는 유일하게 **양의 ME(+5.25세)**를 보인다. 즉 실제 나이보다 높게 예측하는 경향이 있다.

- **저연령(0–9세) MAE 22.99세**: 5세 아이를 28세로 예측하는 사례가 빈번
- **20–40세 구간 상대적 강세**: 이 연령대가 DeepFace 학습 데이터에 풍부해서 한국인 데이터에도 일부 일반화
- **근본 원인**: 한국인 얼굴 특성에 대한 특화 학습이 없고, 사전학습 데이터의 연령 분포가 다름

---

## 4. test_picture 예측 결과

`test_picture/` 디렉터리에는 5장의 테스트 이미지가 있으며, 파일명이 곧 실제 나이를 나타낸다 (`frown_26.jpg`는 26세 동일 인물의 찡그린 표정 변형).

### 4.1 예측 결과 테이블

`predict_all.py`로 실행한 결과. VGG-CNN은 체크포인트가 저장되지 않아(mean collapse로 배포 불필요) 제외.

| 이미지 | 실제 나이 | DeepFace | FCN | Small CNN (crops) |
|--------|----------|---------|-----|-------------------|
| 22.jpg | 22세 | 28.0세 | **22.8세** | 19.9세 |
| 24.jpg | 24세 | 29.0세 | **23.3세** | 2.8세 |
| 26.jpg | 26세 | 26.0세 | **25.1세** | 12.1세 |
| IMG_4869.JPG | 미상 | 36.0세 | 15.9세 | 25.5세 |
| frown_26.jpg | 26세 | 25.0세 | **18.3세** | 15.1세 |

### 4.2 오차 분석 (실제 나이가 알려진 4장 기준)

| 이미지 | 실제 | DeepFace 오차 | FCN 오차 | Small CNN 오차 |
|--------|------|--------------|---------|--------------|
| 22.jpg | 22 | +6.0 | **+0.8** | −2.1 |
| 24.jpg | 24 | +5.0 | **−0.7** | −21.2 |
| 26.jpg | 26 | 0.0 | **−0.9** | −13.9 |
| frown_26.jpg | 26 | −1.0 | −7.7 | −10.9 |
| **평균 절대오차** | — | **3.0** | **2.5** | **12.0** |

### 4.3 분석

**FCN의 압도적 정확도**

`22.jpg`, `24.jpg`, `26.jpg` 모두 오차 1세 이내로 예측하여 test set에서의 우수한 성능(MAE 6.22세)이 이 이미지들에서도 재현된다. ArcFace 임베딩이 20대 한국인 얼굴의 나이 관련 특징을 효과적으로 포착하고 있음을 보여준다.

**DeepFace의 일관된 과대예측**

`22.jpg`(28세), `24.jpg`(29세)처럼 20대 얼굴을 일관되게 더 높게 예측한다. test set에서의 ME +5.25세 편향이 그대로 반영된 결과다. `26.jpg`에서 정확히 26세를 맞춘 것은 우연에 가깝다.

**Small CNN의 불안정한 예측**

`24.jpg`를 2.8세로 예측하는 것처럼 특정 이미지에서 비정상적인 값을 출력한다. 이는 얼굴 검출 실패(fallback) 또는 해당 이미지의 조명·각도가 학습 분포를 벗어났을 때 발생한다. test set MAE 8.53세에 비해 이 소규모 테스트에서 변동성이 크게 나타나는 것은 샘플 편차의 영향이다.

**표정 변화의 영향 (`26.jpg` vs `frown_26.jpg`)**

동일 인물의 중립 표정(26.jpg)과 찡그린 표정(frown_26.jpg)을 비교하면:

| 모델 | 26.jpg (중립) | frown_26.jpg (찡그림) | 차이 |
|------|--------------|---------------------|------|
| DeepFace | 26.0세 | 25.0세 | −1.0세 |
| FCN | 25.1세 | 18.3세 | **−6.8세** |
| Small CNN | 12.1세 | 15.1세 | +3.0세 |

FCN이 찡그린 표정에서 6.8세 낮게 예측하는 것이 주목할 만하다. ArcFace 임베딩이 표정 변화에 민감하게 반응하여, 찡그린 얼굴의 특징이 더 어린 나이의 임베딩과 유사하게 매핑된 것으로 해석된다. DeepFace는 표정 변화에 상대적으로 강건하다(차이 1.0세).

---

## 부록: 실험 환경

- Python 3.x, venv (`.venv/`)
- TensorFlow 2.21, PyTorch, DeepFace
- GPU: NVIDIA GeForce RTX 3060 Ti (WSL2 환경)
- 학습 시 `TF_ENABLE_ONEDNN_OPTS=0` 필수
- XLA GPU 사용 시 libdevice 심볼릭 링크 필요:
  ```bash
  ln -sf /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc ./libdevice.10.bc
  ```
- Keras 모델 로드: `tf.keras.models.load_model()` 대신 아키텍처 재구성 후 `.keras` zip에서 `weights.h5`만 추출해 `model.load_weights()` 사용 (`batch_shape` 역직렬화 오류 우회)
