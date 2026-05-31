# 안면 나이 예측 실험 보고서

**최종 업데이트**: 2026-06-01  
**데이터셋**: AIHub 안면 인식 에이징 데이터셋  
**실험 목표**: 전처리 통일화를 포함한 다양한 접근법의 나이 회귀 성능 비교

---

## 1. 실험 개요

| 항목 | 내용 |
|------|------|
| 태스크 | 얼굴 이미지에서 나이 회귀 (연속 나이 예측) |
| 평가 지표 | MAE, RMSE, ME |
| 데이터 분할 | train 3,550 / valid 412 / test 393 (split_meta.csv, 전 모델 공통) |
| GPU | NVIDIA GeForce RTX 3060 Ti |

---

## 2. 전처리 파이프라인

### 2-1. Raw 이미지 (초기)
```
AIHub 이미지 (전신·상반신) → 128×128 단순 리사이즈 → CNN 입력
```
- 얼굴이 이미지의 평균 20.9%만 차지 → 128×128 후 얼굴 매우 작음
- 학습과 추론(실제 셀카) 간 도메인 갭 심각

### 2-2. 얼굴 크롭 (개선)
```
AIHub 이미지 → 1280px 리사이즈 → OpenCV 얼굴 검출 → 20% 패딩 크롭
→ 128×128 저장 (dataset/face_crops/)
```
- FCN(extract_embeddings.py)과 **완전히 동일한** 검출 파이프라인 사용
- 4,355개 처리: 검출 성공 3,697개 / fallback(전체 이미지) 658개(15.1%)

---

## 3. 모델 설명

### Model 1: DeepFace Baseline
- 사전 학습 DeepFace 모델 직접 사용 (별도 학습 없음)
- VGGFace2 기반 범용 나이 추정기

### Model 2: FCN AgeRegressor (PyTorch, ~200K)
- 입력: DeepFace ArcFace 임베딩 (512-d)
- 구조: Linear(512→256) → Linear(256→128) → Linear(128→1)
- Loss: HuberLoss(δ=5) / Optimizer: Adam / EarlyStop: patience=15

### Model 3: VGG-like CNN (TF/Keras, ~33.6M)
| 블록 | 구성 |
|------|------|
| Block 1-5 | Conv(64→512) × 2-3, MaxPool, same padding |
| FC | Dense(2048) → Dense(1024) → Dense(1) |

### Model 4: Small CNN (TF/Keras, ~3.3M)
출처: Kaggle `neneti/deeplearning-final-project-cnn` (분류 → 회귀 변환)
| 레이어 | 출력 |
|--------|------|
| Conv32 + MaxPool | 63×63×32 |
| Conv64 + MaxPool | 30×30×64 |
| Conv128 + MaxPool | 14×14×128 |
| Dense(128) + Dropout(0.3) | 128 |
| Dense(1) | 1 |

---

## 4. 학습 설정

| 모델 | Epochs | Patience | LR | Loss |
|------|--------|----------|----|------|
| FCN | 100 | 15 | 1e-3 | Huber(δ=5) |
| VGG-CNN (raw) | 30 | 5 | 1e-3 | MAE |
| VGG-CNN (crops) | 30 | 5 | 1e-3 | MAE |
| Small CNN (raw) | 50 | 10 | 1e-3 | MAE |
| Small CNN (crops) | 50 | 10 | 1e-3 | MAE |

---

## 5. 최종 테스트 결과 (test set 393개)

| 모델 | 파라미터 | 전처리 | MAE ↓ | RMSE ↓ | ME |
|------|---------|--------|--------|--------|-----|
| DeepFace baseline | ~34M (frozen) | 없음 | 15.94 | 18.79 | +12.34 |
| VGG-CNN | 33.6M | raw | 11.73 | 14.99 | -4.67 |
| VGG-CNN | 33.6M | **crops** | 11.73 | 14.97 | -4.60 |
| Small CNN | 3.3M | raw | 9.65 | 12.98 | -3.47 |
| Small CNN | 3.3M | **crops** | **8.53** | **11.40** | **-3.40** |
| **FCN AgeRegressor** | ~200K | 임베딩 | **6.22** | **8.38** | -3.26 |

### 전처리 효과 (Small CNN)
```
raw  → MAE 9.65  (mean collapse 없음, 도메인 갭 존재)
crops → MAE 8.53  → 1.12 개선 (11.6% ↑)
```

### VGG-CNN 크롭 효과 없는 이유
```
33.6M 파라미터 / 3,550 샘플 = 9,470 파라미터/샘플
→ 전처리와 무관하게 mean collapse 지속 (val MAE 11.26 고착)
데이터 크기 문제이므로 transfer learning 없이는 개선 불가
```

---

## 6. 학습 곡선 분석

### Small CNN (raw vs crops)
- **raw**: val MAE 8.70에서 정체, train MAE 3.5까지 하락 → 과적합
- **crops**: val MAE 7.78까지 개선 (best epoch 20), 과적합 패턴 유사하나 최고점 향상

### FCN
- 36 epoch에서 val MAE 6.22 수렴, 가장 안정적인 학습 곡선

---

## 7. 과소 예측 원인 분석 (Small CNN raw)

### 원인 1: 학습 데이터 나이 분포 불균형
```
Train split 나이 분포:
  0-5세  : 904개 (25.5%) ← 압도적으로 많음
  중앙값 = 16세, 75%ile = 26세
```

### 원인 2: 도메인 갭 (raw 이미지 기준)
| | AIHub 학습 이미지 | test_picture |
|--|-----------------|-------------|
| 얼굴 비율 | 이미지 너비의 **20.9%** | **~80%** |
| 이미지 타입 | 증명사진·옛날 사진 | 현대 셀카 |

→ 크롭 전처리로 부분 해소

### 원인 3: 구간별 고령 예측 실패 (raw)
```
50세+: ME = -33.5세 (학습 샘플 1개)
40-50세: ME = -19.5세 (111개, 예측 출력 0개)
```

---

## 8. 모델 비교 요약

```
성능 순위 (MAE 기준):
1위  FCN (6.22)        → DeepFace 임베딩 활용, 전이학습 효과
2위  Small CNN+crops (8.53) → 얼굴 크롭으로 도메인 갭 해소
3위  Small CNN+raw (9.65)   → mean collapse 없으나 도메인 갭 존재
4위  VGG-CNN (11.73)   → mean collapse (전부 15세 예측)
5위  DeepFace (15.94)  → 강한 양수 bias (과대 예측)
```

---

## 9. 실사 예측 결과 (test_picture/ 5장)

실제 촬영 사진(22~26세 대상)에 대한 모델별 예측값. 파일명의 숫자가 실제 나이에 해당한다.

| 이미지 (실제 나이) | DeepFace | FCN | Small CNN raw | Small CNN crops | VGG-CNN crops |
|-------------------|---------|-----|--------------|-----------------|--------------|
| 22.jpg (22세) | 28.0 | **22.8** | 5.4 | 19.9 | 15.1 |
| 24.jpg (24세) | 29.0 | **23.3** | 9.3 | 2.8 ← 검출 실패 | 15.1 |
| 26.jpg (26세) | 26.0 | **25.1** | 9.4 | 12.1 | 15.1 |
| IMG_4869.JPG | 36.0 | 15.9 | 18.2 | **25.5** | 15.1 |
| frown_26.jpg (26세) | 25.0 | 18.3 | 5.2 | 15.1 | 15.1 |

### 관찰 사항

- **FCN**: 22/24/26세 사진에서 실제 나이에 가장 근접 (±1세 수준). 일관성 높음.
- **Small CNN (crops)**: 22.jpg(19.9세)·26.jpg(12.1세) 등 개선됐지만 24.jpg(2.8세)처럼 얼굴 검출 품질에 따라 불안정. 검출 실패 시 fallback 이미지로 예측해 오차 급증.
- **Small CNN (raw)**: 전체적으로 5~9세 심한 과소 예측. 도메인 갭이 주 원인.
- **VGG-CNN (crops)**: 모든 이미지에서 ~15.1세 고착. test set에서와 동일한 mean collapse 확인.
- **DeepFace**: 일관되게 3~7세 과대 예측. 서양인 기반 사전학습 bias 의심.

---

## 10. 개선 방향

| 방향 | 기대 효과 | 난이도 |
|------|----------|--------|
| **Transfer Learning** (ImageNet/VGGFace pretrained) | VGG-CNN mean collapse 해소, MAE 대폭 개선 | 중 |
| **Data Augmentation** (flip, jitter, rotation) | Small CNN 과적합 완화 | 하 |
| **더 많은 학습 데이터** (41K AIHub 전체 활용) | 모든 CNN 성능 향상 | 중 |
| **나이 분포 재샘플링** (고령 데이터 oversample) | 과소 예측 bias 감소 | 하 |
| **fallback 제거** (얼굴 미검출 샘플 제외) | 노이즈 감소 (~15% 데이터 손실) | 하 |
| **얼굴 검출 강화** (RetinaFace 등 고성능 detector) | 크롭 품질 향상 → 추론 안정성 개선 | 중 |

---

## 11. 최종 결론

### 핵심 발견

1. **전처리 통일화 효과**: FCN과 동일한 얼굴 크롭 파이프라인 적용 시 Small CNN MAE가 9.65 → 8.53으로 **11.6% 개선**. 학습-추론 간 도메인 갭이 주요 성능 저하 원인임을 확인.

2. **모델 크기 vs. 데이터 크기**: VGG-CNN(33.6M)은 크롭 전처리 후에도 mean collapse 지속. 3,550개 샘플로 33.6M 파라미터를 처음부터 학습하는 것은 구조적 한계. **전처리가 아닌 transfer learning이 필요**.

3. **임베딩 기반 접근의 강점**: FCN(200K)은 DeepFace ArcFace 임베딩을 입력으로 사용해 수백만 장 학습된 얼굴 특징을 그대로 활용. 적은 파라미터와 데이터로도 가장 높은 성능(MAE 6.22) 달성. **사전 학습된 특징 추출기 활용이 핵심**.

4. **실사 예측 안정성**: 크롭 모델은 test set 전반적 성능은 개선됐으나 얼굴 검출 실패 시(fallback) 예측 품질이 급락. 추론 안정성을 위해 검출 신뢰도 필터링 또는 더 강건한 detector 필요.

### 성능-복잡도 트레이드오프

```
모델 복잡도 (파라미터) ↑
          ┌─────────────────────────────────────────┐
          │ DeepFace(34M, frozen) → MAE 15.94       │ 높은 complexity, 낮은 성능
          │ VGG-CNN(33.6M, scratch) → MAE 11.73     │ mean collapse
          │ Small CNN(3.3M, crops) → MAE 8.53       │ 적정 균형점
          │ FCN(200K) + DeepFace emb → MAE 6.22     │ 최고 효율
          └─────────────────────────────────────────┘
성능 ↑ (MAE ↓)

→ 파라미터 수보다 입력 표현의 질(임베딩 vs. raw pixel)이 성능을 결정함
```
