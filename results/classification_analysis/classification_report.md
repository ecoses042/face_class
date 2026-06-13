# 나이 분류(Classification) 성능 분석 보고서

## 1. 실험 개요

본 실험은 기존에 regression task로 학습된 얼굴 나이 예측 모델들을 3-class classification 관점에서 재분석하고, classification task로 처음부터 재학습한 모델과 성능을 비교한다.

- **데이터셋**: AI-Hub 안면 인식 에이징 이미지 (4,355개, train/valid/test = 3,550/412/393)
- **레이블**: `age_past` (촬영 당시 실제 나이)
- **실험 1**: 기존 regression 모델의 예측값 → classification 변환 성능
- **실험 2**: classification으로 처음부터(from scratch) 재학습한 모델 성능

---

## 2. 클래스 정의 및 데이터 분포

| 클래스 | 나이 범위 | 레이블 |
|--------|----------|--------|
| 0 | age < 20 | young |
| 1 | 20 ≤ age < 60 | adult |
| 2 | age ≥ 60 | senior |

### 데이터 분포 (age_past 기준)

| Split | young (<20) | adult (20~60) | senior (60+) | 합계 |
|-------|------------|--------------|-------------|------|
| Train | 1,830 (51.5%) | 1,711 (48.2%) | **9 (0.3%)** | 3,550 |
| Valid | 193 (46.8%) | 206 (50.0%) | 13 (3.2%) | 412 |
| Test  | 177 (45.0%) | 196 (49.9%) | 20 (5.1%) | 393 |

> ⚠️ **심각한 클래스 불균형**: senior 클래스가 훈련셋에 단 9개(0.3%)만 존재하며, 이 9개는 모두 **동일한 인물(person_id 0934, 1960년생)** 의 사진이다. 레이블 자체는 정확하나(파일명·JSON 100% 일치), 인물 다양성이 없어 학습에 한계가 있다.

---

## 3. 실험 1 — Regression 예측값 → Classification 변환 성능

기존 regression 모델의 예측값(`pred_age`)을 동일한 기준으로 클래스로 변환하여 classification 성능을 측정하였다.

### 3.1 결과 요약

| 모델 | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
| **FCN Regressor** | **0.796** | **0.545** | **0.775** |
| CNN Small | 0.768 | 0.506 | 0.760 |
| CNN VGG | 0.573 | 0.243 | 0.417 |

### 3.2 클래스별 상세 성능

**FCN Regressor (최고 성능)**

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| young (<20) | — | — | — | 177 |
| adult (20~60) | — | — | — | 196 |
| senior (60+) | ~0.0 | ~0.0 | ~0.0 | 20 |

> 모든 regression 모델에서 senior 클래스의 recall이 사실상 0에 수렴한다. Regression 모델은 60세 이상 샘플을 거의 본 적이 없어(train 9개) 해당 범위의 나이를 심각하게 과소 예측한다.

### 3.3 Confusion Matrix

| | | FCN Regressor | CNN Small | CNN VGG |
|-|-|:---:|:---:|:---:|
| | 시각화 | `regression_cls/confusion_matrix_fcn_regressor.png` | `regression_cls/confusion_matrix_cnn_small.png` | `regression_cls/confusion_matrix_cnn_vgg.png` |

---

## 4. 실험 2 — Classification 재학습 모델 성능

동일한 아키텍처에서 출력층을 3-class로 변경하고, CrossEntropyLoss + class weighting(senior 가중치 ≈ 131.5)을 적용하여 처음부터 재학습하였다.

### 4.1 결과 요약

| 모델 | Accuracy | Macro F1 | Weighted F1 | senior F1 |
|------|----------|----------|-------------|-----------|
| **FCN Classifier** | **0.773** | **0.709** | **0.776** | **0.545** |
| Small CNN Classifier | 0.486 | 0.330 | 0.460 | 0.000 |
| VGG CNN Classifier | 0.499 | 0.222 | 0.332 | 0.000 |

### 4.2 클래스별 상세 성능

**FCN Classifier**

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| young (<20) | 0.796 | 0.859 | 0.826 | 177 |
| adult (20~60) | 0.820 | 0.699 | 0.755 | 196 |
| senior (60+) | **0.429** | **0.750** | **0.545** | 20 |

**Small CNN Classifier**

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| young (<20) | 0.460 | 0.678 | 0.549 | 177 |
| adult (20~60) | 0.540 | 0.357 | 0.430 | 196 |
| senior (60+) | 0.000 | 0.000 | 0.000 | 20 |

**VGG CNN Classifier**

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| young (<20) | 0.000 | 0.000 | 0.000 | 177 |
| adult (20~60) | 0.499 | 1.000 | 0.665 | 196 |
| senior (60+) | 0.000 | 0.000 | 0.000 | 20 |

> VGG CNN은 모든 샘플을 adult로 예측하는 **degenerate solution**으로 수렴하였다.

### 4.3 Confusion Matrix 및 학습 곡선

| 모델 | Confusion Matrix | 학습 곡선 |
|------|-----------------|----------|
| FCN Classifier | `fcn_classifier/confusion_matrix.png` | `fcn_classifier/loss_curve.png` |
| Small CNN Classifier | `cnn_small_classifier/confusion_matrix.png` | `cnn_small_classifier/loss_curve.png` |
| VGG CNN Classifier | `cnn_vgg_classifier/confusion_matrix.png` | `cnn_vgg_classifier/loss_curve.png` |

---

## 5. 비교 분석

### 5.1 Regression 변환 vs. Classification 재학습

| 모델 | 방식 | Accuracy | Macro F1 | senior F1 |
|------|------|----------|----------|-----------|
| FCN Regressor (변환) | Reg → Cls | 0.796 | 0.545 | ~0.00 |
| **FCN Classifier (재학습)** | From scratch | 0.773 | **0.709** | **0.545** |
| CNN Small (변환) | Reg → Cls | 0.768 | 0.506 | ~0.00 |
| CNN Small Classifier (재학습) | From scratch | 0.486 | 0.330 | 0.000 |
| CNN VGG (변환) | Reg → Cls | 0.573 | 0.243 | ~0.00 |
| CNN VGG Classifier (재학습) | From scratch | 0.499 | 0.222 | 0.000 |

### 5.2 핵심 인사이트

**1. FCN은 재학습이 유의미한 개선을 가져왔다**
- Macro F1: 0.545 → **0.709** (+0.164)
- **senior recall: ~0 → 0.75** — class weighting의 효과로 소수 클래스 인식에 성공
- Accuracy는 소폭 하락(0.796 → 0.773)했으나, 이는 senior 클래스를 일부 오분류하는 대가로 발생한 것이며 macro F1 기준으로는 명백히 개선됨

**2. CNN 모델(이미지 기반)은 재학습 효과 없음**
- Small CNN, VGG CNN 모두 재학습 후 성능이 오히려 하락
- 회귀 변환 방식(Reg → Cls)이 CNN 재학습보다 일관되게 우수
- CNN VGG는 degenerate solution(전부 adult 예측)으로 수렴
- **원인**: 이미지에서 직접 나이를 분류하기 위한 학습 데이터가 절대적으로 부족 (특히 senior 9개), ArcFace 임베딩 기반 FCN보다 이미지에서 미세한 노화 특징을 포착하기 어려움

**3. 임베딩 기반 모델의 우수성**
- FCN은 사전학습된 ArcFace 임베딩(512-dim)을 활용하여 적은 데이터로도 효과적인 분류가 가능
- raw 이미지 기반 CNN은 충분한 데이터 없이는 classification task에서도 embedding 기반 모델을 넘기 어렵다

---

## 6. 한계점

### 6.1 Senior 클래스 데이터 부족 (핵심 한계)
- 훈련 데이터에 senior(60+) 샘플이 **단 9개**, 이 중 **모두 동일 인물(0934)**
- person diversity 부재: 모델은 한 인물의 60세 외모만 학습
- class weight를 아무리 높여도(≈131.5) 1인의 9장 사진으로는 일반화 불가
- FCN Classifier의 senior F1=0.545는 실제로는 과적합에 가까울 수 있음

### 6.2 CNN 이미지 모델의 학습 환경 제약
- Small CNN, VGG CNN 모두 GPU XLA Triton autotuner 호환성 문제로 인해 CPU에서 학습 (RTX 3060 Ti 미활용)
- 학습 시간이 GPU 대비 10-20배 소요되어 epoch 수 제한
- 더 많은 epoch + GPU 학습 시 성능 개선 여지 있음

### 6.3 클래스 경계의 임의성
- `~20 / 20~60 / 60~` 경계는 과제 요건에 따른 것으로, 데이터 분포를 고려한 최적 경계가 아님
- 55세 이상 샘플이 전체의 약 3.5%에 불과하여, 경계를 50세로 낮추면 더 균형 잡힌 학습 가능

---

## 7. 결론

| 항목 | 결론 |
|------|------|
| 최고 성능 모델 | **FCN Classifier (재학습)** — Macro F1 0.709 |
| Regression 변환 vs 재학습 | FCN은 재학습이 유리, CNN은 변환이 오히려 우수 |
| Senior 클래스 인식 | FCN Classifier만 유의미하게 인식 (recall=0.75) |
| CNN 이미지 모델 평가 | 이미지 기반 분류는 현 데이터 규모로 한계 명확 |

> **결론**: 나이 분류 task에서 ArcFace 임베딩 기반 FCN Classifier가 가장 우수한 성능을 보였다. Classification 재학습은 FCN에 한해 유의미하며, 특히 senior 클래스 recall을 0에서 0.75로 끌어올리는 데 성공하였다. 그러나 training senior 샘플이 9개(단일 인물)에 불과하다는 근본적 한계로 인해, 실제 응용 시 senior 예측의 신뢰도는 제한적임을 명시한다.
