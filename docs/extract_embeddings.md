# extract_embeddings.py

## 목적

Validation 이미지에서 얼굴 bbox를 크롭한 뒤 ArcFace 임베딩을 추출하고,  
person_id 기준으로 train/valid/test 분할 후 저장한다.

## 실행

```bash
# 권장: CPU 모드 (GPU XLA JIT 에러 방지)
CUDA_VISIBLE_DEVICES=-1 python src/extract_embeddings.py --model-name ArcFace

# 다른 임베딩 모델 사용
CUDA_VISIBLE_DEVICES=-1 python src/extract_embeddings.py --model-name Facenet512

# 정면 얼굴만 필터링
CUDA_VISIBLE_DEVICES=-1 python src/extract_embeddings.py --frontal-only
```

> **GPU 사용 시 주의:** TF + CUDA 환경에서 `libdevice.10.bc` 관련 XLA JIT 에러가 발생할 수 있다.
> `CUDA_VISIBLE_DEVICES=-1`로 CPU 강제 실행을 권장한다.

## 얼굴 크롭 (핵심)

원본 이미지는 전신/반신 사진으로 얼굴이 전체 픽셀의 1~4%에 불과하다.  
`metadata.csv`의 bbox 정보로 얼굴 영역을 크롭한 후 임베딩을 추출한다.

```python
# bbox에 20% 패딩 추가 후 크롭
pad_x, pad_y = bbox_w * 0.2, bbox_h * 0.2
face_crop = image.crop((bbox_x - pad_x, bbox_y - pad_y,
                         bbox_x + bbox_w + pad_x, bbox_y + bbox_h + pad_y))
```

크롭 없이 전체 이미지를 사용하면 MAE가 약 1.2세 악화된다.

## 분할 기준

person_id 기준으로 train 80% / valid 10% / test 10% 분리 (`RANDOM_SEED=42`).  
같은 인물이 train과 test에 동시에 들어가지 않도록 보장.

## 출력

```
dataset/embeddings/
  train_embeddings.npy   # (3550, 512) float32
  train_labels.npy       # (3550,)     float32  ← photo_age
  valid_embeddings.npy   # (412,  512)
  valid_labels.npy       # (412,)
  test_embeddings.npy    # (393,  512)
  test_labels.npy        # (393,)
  embedding_dim.txt      # "512"
  split_meta.csv         # filename, fcn_split
```
