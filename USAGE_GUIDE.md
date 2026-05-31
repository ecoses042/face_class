# 내 얼굴 사진으로 나이 예측하기

학습된 두 모델(DeepFace Baseline, FCN Regressor)로 본인 사진의 나이를 예측하는 방법을 설명합니다.

---

## 사전 조건

| 항목 | 확인 방법 |
|------|-----------|
| 가상환경 | `.venv/` 폴더 존재 |
| FCN 체크포인트 | `results/fcn_regressor/checkpoints/best_model.pt` 존재 |

```bash
# 둘 다 있는지 한번에 확인
ls .venv/ results/fcn_regressor/checkpoints/best_model.pt
```

---

## 단계별 실행

### 1단계 — 프로젝트 폴더로 이동

```bash
cd /home/ecoses042/26-1/deeplearning/face_age
```

### 2단계 — 가상환경 활성화

```bash
source .venv/bin/activate
```

프롬프트 앞에 `(.venv)` 가 붙으면 성공입니다.

### 3단계 — 사진 준비

얼굴이 잘 보이는 JPG 또는 PNG 파일을 준비합니다.

**권장 사항:**
- 정면 얼굴이 찍힌 사진
- 얼굴이 이미지의 상당 부분을 차지하는 사진
- 밝고 선명한 사진 (조명이 어두운 사진은 검출 정확도 저하)

**지원 형식:** `.jpg`, `.jpeg`, `.png`

### 4단계 — 예측 실행

```bash
python src/predict.py --image path/to/your_photo.jpg
```

**예시:**

```bash
# 홈 디렉터리의 사진
python src/predict.py --image ~/my_photo.jpg

# 현재 폴더에 복사해 둔 사진
python src/predict.py --image face.jpg

# 절대 경로
python src/predict.py --image /home/ecoses042/pictures/selfie.png
```

---

## 출력 결과

```
이미지: my_photo.jpg
========================================
  DeepFace Baseline : 27세
  FCN Regressor     : 24.3세
```

| 모델 | 설명 | 테스트 MAE |
|------|------|-----------|
| DeepFace Baseline | DeepFace 내장 나이 추정 (추가 학습 없음) | 15.9세 |
| FCN Regressor | ArcFace 임베딩 + FCN 회귀 (AIHub 데이터 학습) | 6.2세 |

FCN Regressor가 학습된 모델이므로 더 정확한 결과를 기대할 수 있습니다.

---

## 내부 동작 과정

```
사진 입력
  └─> 이미지 리사이즈 (긴 쪽 최대 1280px)
        └─> OpenCV 얼굴 검출 + 20% 패딩 크롭
              └─> ArcFace 512차원 임베딩 추출
                    └─> FCN Regressor → 나이 예측 (세)
```

얼굴 검출에 실패하면 이미지 전체를 ArcFace에 입력합니다(fallback).

---

## 문제 해결

### "체크포인트 없음" 오류

FCN 모델이 아직 학습되지 않은 경우입니다. 먼저 학습을 진행하세요:

```bash
python src/train_fcn.py
```

### "얼굴 검출 불가" 또는 부정확한 결과

- 얼굴이 더 크게 나온 사진으로 교체
- 정면 사진으로 교체
- 해상도를 높여서 재시도

### ModuleNotFoundError

가상환경이 활성화되지 않은 경우입니다:

```bash
source .venv/bin/activate
```

### 처음 실행 시 느린 경우

ArcFace 모델 가중치를 `~/.deepface/` 에 자동 다운로드하기 때문입니다. 이후 실행부터는 빠릅니다.
