# 프로젝트 정리: DeepFace 기반 한국인 얼굴 나이 예측

## 1. 프로젝트 목표

본 프로젝트의 목표는 **한국인 정면 얼굴 이미지로부터 나이를 예측하는 모델**을 구축하는 것이다.
이를 위해 **AI-Hub 안면 인식 에이징 이미지 데이터**를 사용하며, DeepFace 라이브러리의 사전학습 얼굴 분석 모델을 활용한다.

모델링 방향은 크게 두 가지로 나눈다.

1. **DeepFace 기본 나이 예측 모델 사용**

   * DeepFace의 `analyze()` 기능을 이용하여 나이를 바로 예측한다.
   * 별도 학습 없이 baseline 성능을 확인하는 목적이다.

2. **DeepFace embedding + FCN/MLP 회귀 모델 학습**

   * DeepFace를 직접 fine-tuning하지 않고, 얼굴 이미지를 embedding vector로 변환한다.
   * 추출된 embedding을 입력으로 하는 별도의 FCN layer를 학습한다.
   * 이 방식은 DeepFace를 **feature extractor / embedding layer**처럼 사용하는 구조이다.

---

## 2. 전체 실험 구조

```text
AI-Hub 안면 인식 에이징 이미지 데이터
        ↓
데이터 정제 및 전처리
        ↓
정면 얼굴 이미지 필터링
        ↓
나이 라벨 정리
        ↓
────────────────────────────
실험 1: DeepFace 기본 age prediction
실험 2: DeepFace embedding + FCN regressor
────────────────────────────
        ↓
성능 평가 및 결과 비교
```

---

## 3. 모델링 방식

## 3.1 Baseline: DeepFace 기본 모델 사용

첫 번째 방식은 DeepFace의 기본 age prediction 기능을 그대로 사용하는 것이다.

```python
from deepface import DeepFace

result = DeepFace.analyze(
    img_path="sample.jpg",
    actions=["age"],
    enforce_detection=False
)

pred_age = result[0]["age"]
```

이 방식의 목적은 다음과 같다.

```text
별도 학습 없이 DeepFace가 한국인 얼굴 나이 예측에서 어느 정도 성능을 보이는지 확인
```

즉, 이 방식은 **baseline**이다.

---

## 3.2 Proposed: DeepFace embedding + FCN age regressor

두 번째 방식은 DeepFace를 얼굴 특징 추출기로 사용하는 것이다.

```text
Face Image
→ DeepFace pretrained model
→ Face embedding vector
→ FCN / MLP regression head
→ Predicted age
```

예시 구조는 다음과 같다.

```text
DeepFace embedding, e.g. 128-d or 512-d
→ Linear
→ ReLU
→ Dropout
→ Linear
→ Predicted age
```

PyTorch 구조 예시는 다음과 같다.

```python
import torch
import torch.nn as nn

class AgeRegressor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
```

이 방식에서 학습되는 것은 DeepFace 전체 모델이 아니라, 뒤에 붙인 FCN 회귀 모델이다.

| 구성 요소          | 역할                  | 학습 여부 |
| -------------- | ------------------- | ----- |
| DeepFace model | 얼굴 embedding 추출     | 고정    |
| FCN / MLP head | embedding으로부터 나이 예측 | 학습    |
| Loss function  | 실제 나이와 예측 나이 비교     | -     |

---

## 4. 데이터셋

사용 데이터셋은 다음과 같다.

```text
AI-Hub 안면 인식 에이징 이미지 데이터
```

이 데이터셋에서 필요한 정보는 다음과 같다.

| 항목          | 사용 목적                           |
| ----------- | ------------------------------- |
| 얼굴 이미지      | 모델 입력                           |
| 나이 라벨       | 정답값                             |
| 얼굴 위치 정보    | 얼굴 crop에 사용                     |
| landmark 정보 | 얼굴 정렬에 사용 가능                    |
| 인물 ID       | train/validation/test split에 사용 |

특히 중요한 점은 **동일 인물이 train과 test에 동시에 들어가지 않도록 분리**하는 것이다.

```text
좋지 않은 방식:
같은 사람의 이미지 A → train
같은 사람의 이미지 B → test

권장 방식:
사람 ID 기준으로 train / validation / test 분리
```

이렇게 해야 모델이 특정 인물을 기억하는 것이 아니라, 실제로 얼굴 특징을 기반으로 나이를 예측하는지 평가할 수 있다.

---

## 5. 권장 파일 구조

프로젝트 파일 구조는 다음과 같이 나누는 것이 좋다.

```text
age_prediction_deepface/
│
├── dataset/
│   ├── raw/
│   │   └── aihub_aging/
│   │       ├── images/
│   │       └── labels/
│   │
│   ├── processed/
│   │   ├── images/
│   │   ├── metadata.csv
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   └── test.csv
│   │
│   └── embeddings/
│       ├── train_embeddings.npy
│       ├── valid_embeddings.npy
│       ├── test_embeddings.npy
│       ├── train_labels.npy
│       ├── valid_labels.npy
│       └── test_labels.npy
│
├── results/
│   ├── baseline_deepface/
│   │   ├── predictions.csv
│   │   └── metrics.json
│   │
│   ├── fcn_regressor/
│   │   ├── checkpoints/
│   │   │   └── best_model.pt
│   │   ├── predictions.csv
│   │   ├── metrics.json
│   │   └── loss_curve.png
│   │
│   └── comparison/
│       ├── model_comparison.csv
│       └── error_analysis.csv
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── split_dataset.py
│   ├── run_deepface_baseline.py
│   ├── extract_embeddings.py
│   ├── train_fcn.py
│   ├── evaluate.py
│   ├── models.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 6. 각 디렉토리 역할

### 6.1 `dataset/`

데이터 관련 파일을 저장하는 공간이다.

```text
dataset/
├── raw/
├── processed/
└── embeddings/
```

#### `dataset/raw/`

AI-Hub에서 받은 원본 데이터를 저장한다.

```text
dataset/raw/aihub_aging/images/
dataset/raw/aihub_aging/labels/
```

이 폴더의 데이터는 가능하면 수정하지 않는다.

---

#### `dataset/processed/`

전처리된 데이터를 저장한다.

예상 파일:

```text
metadata.csv
train.csv
valid.csv
test.csv
```

`metadata.csv`는 전체 이미지 정보를 담는다.

예시:

```csv
image_path,person_id,age,gender,face_bbox,is_frontal
dataset/processed/images/000001.jpg,P001,23,M,"[x,y,w,h]",1
dataset/processed/images/000002.jpg,P002,41,F,"[x,y,w,h]",1
```

---

#### `dataset/embeddings/`

DeepFace로 추출한 embedding을 저장한다.

```text
train_embeddings.npy
valid_embeddings.npy
test_embeddings.npy
train_labels.npy
valid_labels.npy
test_labels.npy
```

FCN 학습 시에는 이미지를 매번 DeepFace에 넣지 않고, 미리 추출한 embedding을 사용한다.

---

### 6.2 `results/`

실험 결과를 저장한다.

```text
results/
├── baseline_deepface/
├── fcn_regressor/
└── comparison/
```

#### `results/baseline_deepface/`

DeepFace 기본 age prediction 결과 저장.

```text
predictions.csv
metrics.json
```

예시:

```csv
image_path,true_age,pred_age,error
sample001.jpg,25,31,6
sample002.jpg,42,39,3
```

---

#### `results/fcn_regressor/`

DeepFace embedding + FCN 모델 결과 저장.

```text
checkpoints/best_model.pt
predictions.csv
metrics.json
loss_curve.png
```

---

#### `results/comparison/`

두 실험 결과를 비교한다.

```text
model_comparison.csv
error_analysis.csv
```

예시:

```csv
model,MAE,RMSE
DeepFace Baseline,8.42,10.15
DeepFace Embedding + FCN,6.31,8.02
```

---

### 6.3 `src/`

소스코드 디렉토리이다.

```text
src/
├── config.py
├── preprocess.py
├── split_dataset.py
├── run_deepface_baseline.py
├── extract_embeddings.py
├── train_fcn.py
├── evaluate.py
├── models.py
└── utils.py
```

각 파일 역할은 다음과 같다.

| 파일                         | 역할                                 |
| -------------------------- | ---------------------------------- |
| `config.py`                | 경로, batch size, learning rate 등 설정 |
| `preprocess.py`            | 원본 이미지와 라벨 전처리                     |
| `split_dataset.py`         | train/valid/test 분리                |
| `run_deepface_baseline.py` | DeepFace 기본 나이 예측 실행               |
| `extract_embeddings.py`    | DeepFace embedding 추출              |
| `train_fcn.py`             | FCN 회귀 모델 학습                       |
| `evaluate.py`              | MAE, RMSE 등 평가                     |
| `models.py`                | FCN 모델 정의                          |
| `utils.py`                 | 공통 유틸 함수                           |

---

## 7. 평가 지표

나이 예측은 회귀 문제이므로 다음 지표를 사용한다.

| 지표                 | 의미                     |
| ------------------ | ---------------------- |
| MAE                | 실제 나이와 예측 나이의 평균 절대 오차 |
| RMSE               | 큰 오차에 더 민감한 평균 제곱근 오차  |
| Error by age group | 연령대별 오차 분석             |

기본 평가는 다음 두 모델을 비교한다.

```text
1. DeepFace 기본 age prediction
2. DeepFace embedding + FCN regressor
```

---

## 8. 실험 순서

```text
Step 1. AI-Hub 데이터 다운로드
Step 2. 원본 데이터 구조 확인
Step 3. 이미지 경로와 나이 라벨을 metadata.csv로 정리
Step 4. 정면 얼굴 이미지 필터링
Step 5. train / valid / test split 생성
Step 6. DeepFace baseline 실행
Step 7. DeepFace embedding 추출
Step 8. FCN age regressor 학습
Step 9. test set 평가
Step 10. baseline과 FCN 모델 성능 비교
```

---

## 9. README에 들어갈 프로젝트 설명 초안

```markdown
# Korean Face Age Prediction using DeepFace

This project aims to predict age from Korean frontal face images using the AI-Hub facial aging image dataset. We compare two approaches based on the DeepFace library.

## Approach 1: DeepFace Baseline

We first use the built-in age prediction function of DeepFace without additional training. This serves as a baseline for evaluating how well a pretrained DeepFace model performs on Korean face images.

## Approach 2: DeepFace Embedding + FCN Regressor

In the second approach, we use DeepFace as a fixed feature extractor. Face images are converted into embedding vectors using a pretrained DeepFace model. Then, a fully connected neural network is trained on top of these embeddings to predict age.

## Dataset

We use the AI-Hub facial aging image dataset. The dataset is preprocessed to extract frontal face images and valid age labels. The train, validation, and test splits are created based on identity to prevent data leakage.

## Evaluation

The models are evaluated using MAE and RMSE. We also analyze prediction errors by age group.
```

---

## 10. 프로젝트 방향성 요약

> 본 프로젝트는 AI-Hub 안면 인식 에이징 이미지 데이터를 활용하여 한국인 정면 얼굴 기반 나이 예측 모델을 구축한다. 모델링은 두 가지 방식으로 진행한다. 첫째, DeepFace 라이브러리의 기본 age prediction 기능을 그대로 사용하여 baseline 성능을 측정한다. 둘째, DeepFace를 얼굴 embedding 추출기로 사용하고, 추출된 embedding vector 위에 별도의 FCN 회귀 모델을 학습하여 나이를 예측한다. 데이터는 원본 데이터, 전처리 데이터, embedding 데이터로 분리하여 관리하며, 실험 결과는 baseline과 FCN 모델별로 저장하고 최종적으로 성능을 비교한다.
