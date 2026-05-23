# 데이터 변형 및 학습 파이프라인

## 전체 구조

```
AI-Hub 원본 zip
      │
      ▼
[Step 1] preprocess.py       → dataset/processed/metadata.csv
                                dataset/processed/images/
                                dataset/processed/labels/
      │
      ▼
[Step 2] extract_embeddings.py → dataset/embeddings/{train,valid,test}_embeddings.npy
                                  dataset/embeddings/{train,valid,test}_labels.npy
      │
      ├──▶ [Step 3a] run_deepface_baseline.py → results/baseline_deepface/
      │
      └──▶ [Step 3b] train_fcn.py             → results/fcn_regressor/
                          │
                          ▼
                    evaluate.py --compare      → results/comparison/
```

---

## Step 1: 전처리 (`preprocess.py`)

### 1-1. zip 압축 해제

AI-Hub 데이터는 피험자 ID 단위로 묶인 zip 파일로 제공된다.

```
dataset/raw/aihub_aging/.../
  Training/
    01.원천데이터/   → TS_*.zip  (이미지)
    02.라벨링데이터/ → TL_*.zip  (JSON 라벨)
  Validation/
    01.원천데이터/   → VS_*.zip
    02.라벨링데이터/ → VL_*.zip
```

각 zip을 `dataset/processed/images/` 와 `dataset/processed/labels/` 에 일괄 해제한다.  
이미 존재하는 파일은 덮어쓰지 않고 건너뛴다.

### 1-2. JSON 라벨 파싱

각 이미지마다 대응하는 JSON 파일이 있으며, 주요 필드는 다음과 같다.

| JSON 필드     | 의미                     |
|--------------|--------------------------|
| `id`         | 피험자 고유 ID            |
| `birth`      | 출생연도                  |
| `age_now`    | 현재(기준) 나이           |
| `age_past`   | 촬영 시점과 현재의 나이 차 |
| `annotation[0].box` | 얼굴 bbox (x, y, w, h) |
| `annotation[0].landmark` | 5점 랜드마크 좌표 |

**촬영 시점 나이 계산:**
```
photo_age = age_now - age_past
```

### 1-3. 필터링 조건

파싱 단계에서 다음 조건을 만족하지 않으면 제외한다.

| 조건 | 기준값 | 이유 |
|------|--------|------|
| bbox 최소 크기 | w ≥ 50px, h ≥ 50px | 너무 작은 얼굴 영역 제외 |
| 나이 범위 | 1 ≤ photo_age ≤ 80 | 이상값 제거 |

### 1-4. 정면 판별 (`is_frontal`)

5점 랜드마크(코, 눈 2점, 입꼬리 2점)를 이용해 정면 여부를 판단한다.

```
대칭 비율 = |코 x좌표 - 양눈 중간 x좌표| / 양눈 간격
is_frontal = (대칭 비율 < 0.25)
```

비율이 낮을수록 얼굴이 정면을 향한다. 임계값 0.25는 `config.py`의 `FRONTAL_SYM_THRESH`로 조정 가능하다.

### 1-5. 출력: `metadata.csv`

| 컬럼 | 내용 |
|------|------|
| `filename` | 파일명 (확장자 제외) |
| `split` | `training` / `validation` |
| `person_id` | 피험자 ID |
| `photo_age` | 촬영 시점 나이 (학습 레이블) |
| `bbox_x/y/w/h` | 얼굴 bbox 좌표 |
| `is_frontal` | 정면 여부 (0/1) |
| `image_path` | 이미지 절대 경로 (없으면 빈 문자열) |

**최종 레코드 수:** Training 37,196개 + Validation 4,574개 = 41,770개

---

## Step 2: 임베딩 추출 (`extract_embeddings.py`)

### 2-1. 사용 데이터

AI-Hub의 **Validation** split(4,574개)만 사용한다.  
Training split은 이미지 zip이 제공되지 않아 임베딩 추출이 불가하다.  
이미지 경로가 비어 있는 219개를 제외하면 **4,355개**가 유효 샘플이다.

### 2-2. Person ID 기준 데이터 분할

데이터 누수를 막기 위해 **피험자(person_id) 단위**로 train/valid/test를 분리한다.  
같은 사람의 사진이 train과 test에 동시에 들어가는 것을 원천 차단한다.

```python
RANDOM_SEED = 42
TEST_RATIO  = 0.1   # 피험자의 10% → test
VALID_RATIO = 0.1   # 피험자의 10% → valid
# 나머지 80% → train
```

**결과 분포:**

| split | 피험자 수 | 이미지 수 |
|-------|----------|----------|
| train | ~78명    | 3,550개  |
| valid | ~9명     | 412개    |
| test  | 9명      | 393개    |

### 2-3. 얼굴 크롭

> **핵심 처리 단계.** 원본 이미지는 전신 또는 반신 사진으로, 얼굴이 전체 픽셀의 1~4%에 불과하다.  
> ArcFace는 정렬된 얼굴 영역을 입력으로 요구하므로, bbox 정보로 얼굴을 크롭한 뒤 임베딩을 추출해야 한다.

```python
# bbox에 여유(padding) 20% 추가 후 크롭
pad_x = bbox_w * 0.2
pad_y = bbox_h * 0.2
x1 = max(0, int(bbox_x - pad_x))
y1 = max(0, int(bbox_y - pad_y))
x2 = min(img_w, int(bbox_x + bbox_w + pad_x))
y2 = min(img_h, int(bbox_y + bbox_h + pad_y))
face_crop = image.crop((x1, y1, x2, y2))
```

패딩 20%를 추가하는 이유: ArcFace 학습에 사용된 데이터셋(CASIA, MS1M 등)의 얼굴 크롭 관례를 따른다.

### 2-4. ArcFace 임베딩 추출

얼굴 크롭 이미지를 DeepFace의 ArcFace 모델에 입력해 512차원 임베딩을 추출한다.

```python
DeepFace.represent(
    img_path=face_crop_array,  # numpy array (크롭된 얼굴)
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="skip",   # 얼굴 검출 생략 (이미 크롭됨)
)
```

- `enforce_detection=False`: 얼굴 미검출 시 예외 없이 진행
- `detector_backend="skip"`: 크롭 이미지를 그대로 112×112로 리사이즈해 모델 입력

**ArcFace 모델 특성:**
- 입력: 112×112 RGB 이미지
- 출력: 512차원 벡터 (L2-미정규화 상태로 반환)
- 학습 목적: 동일인 임베딩을 가깝게, 다른 사람 임베딩을 멀리 배치 (face recognition)

> **주의사항:** ArcFace는 얼굴 인식용 모델이므로 나이 정보를 직접 학습한 것이 아니다.  
> 나이 관련 특징은 face identity 특징에 내재되어 있다고 가정하며, FCN이 이를 나이로 매핑한다.

### 2-5. GPU 관련 이슈

TensorFlow + CUDA 조합에서 XLA JIT 컴파일 시 `libdevice.10.bc` 파일을 찾지 못하는 문제가 발생할 수 있다.

```
error: libdevice not found at ./libdevice.10.bc
JIT compilation failed. [Op:Rsqrt]
```

**해결:** `CUDA_VISIBLE_DEVICES=-1` 환경변수로 CPU 모드 강제 실행

```bash
CUDA_VISIBLE_DEVICES=-1 python src/extract_embeddings.py --model-name ArcFace
```

또는 근본 해결:
```bash
XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit" python ...
```

### 2-6. 출력 파일

```
dataset/embeddings/
  train_embeddings.npy  # shape: (3550, 512), dtype: float32
  train_labels.npy      # shape: (3550,),     dtype: float32  ← photo_age
  valid_embeddings.npy  # shape: (412,  512)
  valid_labels.npy      # shape: (412,)
  test_embeddings.npy   # shape: (393,  512)
  test_labels.npy       # shape: (393,)
  embedding_dim.txt     # "512"
  split_meta.csv        # filename, fcn_split
```

---

## Step 3a: DeepFace Baseline (`run_deepface_baseline.py`)

별도 학습 없이 DeepFace의 내장 나이 예측 기능을 바로 사용한다.

```python
DeepFace.analyze(
    img_path=image_path,
    actions=["age"],
    enforce_detection=False,
    silent=True,
)
```

- 입력: **원본 전체 이미지** (얼굴 크롭 없음, DeepFace 내부에서 자동 검출)
- 대상: Validation 전체 4,355개

**결과:**

| 지표 | 값 |
|------|----|
| MAE  | 15.94세 |
| RMSE | 18.79세 |
| ME   | +12.34 (실제보다 나이를 많이 예측하는 경향) |

---

## Step 3b: FCN 학습 (`train_fcn.py`)

### 3b-1. 모델 아키텍처 (`AgeRegressor`)

ArcFace 임베딩(512차원)을 나이(스칼라)로 매핑하는 완전연결 네트워크.

```
Input: (512,)
  → Linear(512 → 256) + BatchNorm1d + ReLU + Dropout(0.3)
  → Linear(256 → 128) + BatchNorm1d + ReLU + Dropout(0.2)
  → Linear(128 → 64)  + ReLU
  → Linear(64  → 1)
Output: scalar (예측 나이)
```

총 파라미터 수: 약 165,000개

### 3b-2. 학습 설정

| 항목 | 값 |
|------|----|
| 손실 함수 | Huber Loss (delta=5.0) |
| 옵티마이저 | Adam (lr=1e-3, weight_decay=1e-4) |
| LR 스케줄러 | ReduceLROnPlateau (patience=5, factor=0.5, min_lr=1e-5) |
| 배치 크기 | 64 |
| 최대 epoch | 100 |
| Early stopping | patience=15 (val MAE 기준) |
| 디바이스 | CUDA (GPU) |

**Huber Loss를 사용한 이유:**  
나이 예측에서 일부 샘플은 10~20세 이상의 큰 오차를 낼 수 있다.  
MSE는 이런 이상치(outlier)에 민감하게 반응해 학습이 불안정해지는 반면,  
Huber Loss는 오차 delta(5.0) 이하에서는 MSE처럼, 이상의 오차에서는 MAE처럼 동작해 더 안정적이다.

### 3b-3. 학습 흐름

```
매 epoch:
  1. train_loader로 forward → Huber loss → backward → Adam step
  2. valid_loader로 MAE, RMSE 계산
  3. ReduceLROnPlateau로 lr 조정
  4. best val MAE 갱신 시 best_model.pt 저장
  5. patience 초과 시 early stopping

학습 종료 후:
  best_model.pt 로드 → test_loader로 최종 평가
```

### 3b-4. 학습 결과

| Epoch | Train Loss | Val MAE | Val RMSE |
|-------|-----------|---------|---------|
| 1     | 49.98     | 8.83    | 12.77   |
| 10    | 12.87     | 6.94    | 9.71    |
| 19    | —         | —       | —       |

Early stopping at **epoch 19** (best val MAE: **6.58**)

**Test 최종 성능:**

| 지표 | 값 |
|------|----|
| MAE  | **9.27세** |
| RMSE | **12.44세** |
| ME   | -5.10 (실제보다 약간 어리게 예측) |

---

## 최종 모델 비교

| 모델 | MAE | RMSE | ME | n |
|------|-----|------|----|---|
| DeepFace Baseline | 15.94 | 18.79 | +12.34 | 4,355 |
| **FCN Regressor** | **9.27** | **12.44** | -5.10 | 393 |

FCN이 MAE 기준 **41.8% 개선.**

**비교 시 유의사항:**
- Baseline은 4,355개 전체 Validation, FCN은 393개 test set 기준이다.
- Baseline의 ME(+12.34)는 뚜렷한 과대예측 bias를 나타낸다. DeepFace 내장 age estimator가 한국인 얼굴에 맞게 보정되지 않았을 가능성이 높다.
- FCN의 ME(-5.10)는 약간의 과소예측 경향이 있으나 bias가 훨씬 작다.

---

## 재현 방법

```bash
# 1. 전처리 (zip 해제 + metadata.csv 생성)
python src/preprocess.py

# 2. 임베딩 추출 (CPU 모드 권장)
CUDA_VISIBLE_DEVICES=-1 python src/extract_embeddings.py --model-name ArcFace

# 3a. DeepFace Baseline 실행
python src/run_deepface_baseline.py

# 3b. FCN 학습
python src/train_fcn.py

# 4. 비교 리포트
python src/evaluate.py --compare
```

---

## 알려진 이슈 및 개선 가능성

| 항목 | 현재 상태 | 개선 방향 |
|------|----------|----------|
| 임베딩 모델 | ArcFace (face recognition 목적) | 나이 특화 모델(AgeDB fine-tuned) 사용 |
| 얼굴 정렬 | 없음 (bbox 크롭만) | 5점 랜드마크 기반 affine alignment 추가 |
| 학습 데이터 | Validation 3,550개 | AI-Hub Training 37,196개 이미지 활용 |
| 모델 복잡도 | FCN 3층 | 더 깊은 네트워크 또는 앙상블 |
| GPU JIT 에러 | CPU 우회로 해결 | `XLA_FLAGS` 환경변수 또는 TF GPU 설정 수정 |
