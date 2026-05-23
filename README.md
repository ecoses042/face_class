# 한국인 얼굴 나이 예측

AI-Hub **안면 인식 에이징 이미지 데이터**(데이터셋 키: 71415)를 이용한 나이 예측 프로젝트.

## 모델 요약

| 모델 | 방식 | Test MAE |
|------|------|---------|
| DeepFace Baseline | `DeepFace.analyze()` 직접 사용, 학습 없음 | 15.94세 |
| **FCN Regressor** | ArcFace 임베딩 고정 + FCN 학습 | **9.27세** |

## 디렉토리 구조

```
face_age/
├── src/
│   ├── config.py                  # 경로·하이퍼파라미터 중앙 관리
│   ├── preprocess.py              # zip 해제 + metadata.csv 생성
│   ├── extract_embeddings.py      # ArcFace 임베딩 추출 (bbox 크롭 포함)
│   ├── models.py                  # FCN AgeRegressor 아키텍처
│   ├── train_fcn.py               # FCN 학습
│   ├── run_deepface_baseline.py   # DeepFace baseline 평가
│   └── evaluate.py                # 지표 계산 및 시각화
├── docs/
│   ├── pipeline.md                # 전체 파이프라인 상세 설명
│   └── ...                        # 스크립트별 문서
├── dataset/
│   ├── raw/                       # AI-Hub 원본 zip (gitignore)
│   ├── processed/
│   │   ├── metadata.csv           # 전처리된 메타데이터
│   │   └── images/                # 압축 해제된 이미지 (gitignore)
│   └── embeddings/                # 추출된 npy 파일 (gitignore)
├── results/
│   ├── baseline_deepface/         # baseline 결과 및 시각화
│   ├── fcn_regressor/             # FCN 결과 및 체크포인트
│   └── comparison/                # 모델 비교 CSV
├── requirements.txt
└── DATA_DOWNLOAD_GUIDE.md         # AI-Hub 데이터 다운로드 방법
```

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU 사용 시:** TensorFlow + CUDA 조합에서 XLA JIT 에러가 발생할 수 있다.  
> 임베딩 추출은 `CUDA_VISIBLE_DEVICES=-1`(CPU 모드)로 실행하는 것을 권장한다.

## 실행 순서

### 1. 데이터 다운로드

AI-Hub 계정이 필요하며 내국인만 신청 가능하다. 자세한 방법은 `DATA_DOWNLOAD_GUIDE.md` 참고.

### 2. 전처리

```bash
python src/preprocess.py
```

zip 압축 해제 후 `dataset/processed/metadata.csv` 생성. 약 4만여 개 레코드.

### 3. 임베딩 추출

```bash
CUDA_VISIBLE_DEVICES=-1 python src/extract_embeddings.py --model-name ArcFace
```

얼굴 bbox 크롭 → ArcFace 512차원 임베딩 추출 → train/valid/test 분할 저장.  
person_id 기준 분할로 데이터 누수 방지.

### 4a. DeepFace Baseline 실행

```bash
python src/run_deepface_baseline.py
```

### 4b. FCN 학습

```bash
python src/train_fcn.py
```

### 5. 비교 리포트

```bash
python src/evaluate.py --compare
```

## 주요 설계 결정

- **person_id 기준 분할**: 동일 인물의 사진이 train/test에 동시 등장하는 데이터 누수 방지
- **bbox 크롭**: 원본 이미지에서 얼굴이 1~4%에 불과해 bbox로 크롭 후 임베딩 추출 (미적용 시 MAE +1.2세)
- **Huber Loss**: 나이 예측의 이상치에 강건한 학습을 위해 MSE 대신 사용
- **ArcFace embedding 고정**: 대규모 얼굴 인식 데이터로 사전학습된 표현을 그대로 활용

자세한 내용은 `docs/pipeline.md` 참고.
