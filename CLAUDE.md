# face_age 프로젝트 컨텍스트

## 환경
- Python venv: `source .venv/bin/activate` (TF 2.21, PyTorch, DeepFace 포함)
- GPU: NVIDIA GeForce RTX 3060 Ti (WSL2)
- 학습 실행 시 항상 `TF_ENABLE_ONEDNN_OPTS=0` 설정

## GPU/XLA 필수 설정
- TF 학습 전 프로젝트 루트에 libdevice 심볼릭 링크 필요 (없으면 MAE/Huber loss 컴파일 실패):
  `ln -sf /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc ./libdevice.10.bc`
- 이미 `libdevice.10.bc` 심볼릭 링크가 루트에 존재함

## Keras 모델 로드
- `tf.keras.models.load_model()` 은 `batch_shape` 역직렬화 오류 발생
- 대신: 모델 아키텍처 재구성 후 .keras zip에서 weights.h5 추출하여 `model.load_weights()` 사용
  (`src/predict_all.py`의 `_predict_keras()` 참고)

## 데이터 경로
- 원본 처리 이미지: `dataset/processed/images/`
- ArcFace 임베딩: `dataset/embeddings/{train,valid,test}_embeddings.npy`
- 분할 메타: `dataset/embeddings/split_meta.csv` (train 3,550 / valid 412 / test 393)
- 얼굴 크롭 이미지: `dataset/face_crops/` + `dataset/embeddings/split_meta_crops.csv`

## 주요 스크립트
- `src/pipeline_aihub.py` — CNN 학습 공통 파이프라인 (load_splits, make_dataset, run_training)
- `src/train_fcn.py` — PyTorch FCN (임베딩 입력)
- `src/train_small_cnn_aihub.py --use-crops` — Small CNN (얼굴 크롭)
- `src/predict_all.py` — test_picture/ 전체 예측 및 시각화

## 실험 결과 요약 (AIHub test set 393개)
| 모델 | MAE | 비고 |
|------|-----|------|
| FCN AgeRegressor (~200K) | 6.22 | 최고 성능, 임베딩 활용 |
| Small CNN 3.3M + crops | 8.53 | 전처리 통일 후 best CNN |
| Small CNN 3.3M + raw | 9.65 | 도메인 갭 존재 |
| VGG-CNN 33.6M | 11.73 | mean collapse (모두 15세 예측) |
| DeepFace baseline | 15.94 | 과대 예측 bias |

## 알려진 이슈
- VGG-CNN(33.6M): 3,550개 샘플로는 mean collapse 불가피 → transfer learning 필요
- Small CNN crops 추론: 얼굴 검출 실패(fallback) 시 예측 품질 급락
- fallback 비율: 전체의 15.1% (658/4,355개)
