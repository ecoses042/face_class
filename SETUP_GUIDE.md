# 프로젝트 셋업 가이드

git clone 또는 git pull 후 환경을 구성하고 모델을 사용하는 방법을 설명합니다.

---

## 1. 저장소 클론

```bash
git clone https://github.com/ecoses042/face_class.git
cd face_class
```

---

## 2. 가상환경 생성 및 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> GPU 환경에서 PyTorch를 사용한다면 [pytorch.org](https://pytorch.org/get-started/locally/)에서 CUDA 버전에 맞는 명령어로 별도 설치를 권장합니다.

---

## 3. 모델 다운로드 (Hugging Face Hub)

학습된 모델 5개가 Hugging Face Hub에 업로드되어 있습니다.

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

REPO_ID = "ecoses042/face-age-estimation"

models = [
    ("fcn_regressor/best_model.pt",              "results/fcn_regressor/checkpoints/best_model.pt"),
    ("small_cnn_aihub/best_model.keras",         "results/small_cnn_aihub/checkpoints/best_model.keras"),
    ("small_cnn_aihub_crops/best_model.keras",   "results/small_cnn_aihub_crops/checkpoints/best_model.keras"),
    ("cnn_aihub/best_model.keras",               "results/cnn_aihub/checkpoints/best_model.keras"),
    ("cnn_aihub_crops/best_model.keras",         "results/cnn_aihub_crops/checkpoints/best_model.keras"),
]

for repo_path, local_path in models:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {repo_path}...")
    downloaded = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
    os.rename(downloaded, local_path)
    print(f"  -> {local_path}")

print("Done.")
EOF
```

| 모델 파일 | 크기 | 설명 |
|-----------|------|------|
| `fcn_regressor/best_model.pt` | 688 KB | **최고 성능** FCN (MAE 6.22) |
| `small_cnn_aihub/best_model.keras` | 38 MB | Small CNN, raw 이미지 학습 |
| `small_cnn_aihub_crops/best_model.keras` | 38 MB | Small CNN, 얼굴 크롭 학습 (MAE 8.53) |
| `cnn_aihub/best_model.keras` | 385 MB | VGG-like CNN, raw 이미지 학습 |
| `cnn_aihub_crops/best_model.keras` | 385 MB | VGG-like CNN, 얼굴 크롭 학습 |

---

## 4. 추론 실행

### 단일 이미지 예측 (FCN + DeepFace 비교)

```bash
source .venv/bin/activate
python src/predict.py --image path/to/photo.jpg
```

출력 예시:
```
이미지: photo.jpg
========================================
  DeepFace Baseline : 27세
  FCN Regressor     : 24.3세
```

### 여러 이미지 일괄 예측 (전체 모델 비교)

`test_picture/` 폴더에 이미지를 넣고 실행합니다.

```bash
python src/predict_all.py
```

결과는 `results/test_predictions.png`로 저장됩니다.

---

## 5. 모델별 성능 요약

| 모델 | MAE (AIHub test 393개) | 비고 |
|------|------------------------|------|
| FCN AgeRegressor | **6.22** | ArcFace 임베딩 입력, 최고 성능 |
| Small CNN + crops | 8.53 | 얼굴 크롭 전처리 |
| Small CNN + raw | 9.65 | 원본 이미지 입력 |
| VGG-like CNN | 11.73 | mean collapse (데이터 부족) |
| DeepFace baseline | 15.94 | 추가 학습 없음 |

---

## 6. (선택) 모델 재학습

데이터셋(AI-Hub 한국인 안면 나이 데이터)이 있다면 직접 학습할 수 있습니다.

```bash
# FCN (임베딩 기반, 권장)
python src/train_fcn.py

# Small CNN (얼굴 크롭)
python src/train_small_cnn_aihub.py --use-crops

# Small CNN (원본 이미지)
python src/train_small_cnn_aihub.py
```

> 학습 전 libdevice 심볼릭 링크 확인:
> ```bash
> ls libdevice.10.bc  # 없으면 아래 명령 실행
> ln -sf /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc ./libdevice.10.bc
> ```
> 데이터 전처리는 `docs/preprocess.md` 참고.

---

## 문제 해결

| 증상 | 해결 방법 |
|------|-----------|
| `ModuleNotFoundError` | `source .venv/bin/activate` 확인 |
| 모델 파일 없음 | 3단계 모델 다운로드 재실행 |
| 얼굴 검출 실패 | 정면·고해상도 사진으로 교체 |
| TF XLA 오류 | `export TF_ENABLE_ONEDNN_OPTS=0` 설정 |
| 처음 실행 느림 | ArcFace 가중치 자동 다운로드 중 (~/.deepface/), 이후 빠름 |
