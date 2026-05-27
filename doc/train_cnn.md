# train_cnn.py

Adience fold_4 데이터셋으로 VGG-like CNN age classifier를 학습하는 스크립트.

## 실행

```bash
python src/train_cnn.py --data-dir /path/to/AdienceGender [옵션]
```

## 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data-dir` | `/kaggle/input/.../AdienceGender` | Adience 데이터셋 루트 |
| `--img-size` | 128 | 입력 이미지 크기 |
| `--batch-size` | 32 | 배치 크기 |
| `--epochs` | 30 | 최대 에폭 |
| `--patience` | 5 | EarlyStopping patience |
| `--lr` | 1e-3 | Adam 학습률 |

## 처리 흐름

1. `fold_4_data.txt` 로딩 → age 문자열 파싱 → 8개 클래스 필터 → 클래스 균형(min_val 기준)
2. `user_id` 기반 train/valid/test 분리 (비율 80/10/10, identity leakage 방지)
3. `tf.data.Dataset` 파이프라인으로 이미지 로딩 (decode_jpeg → resize → /255.0)
4. `build_cnn_model()` → Adam + sparse_categorical_crossentropy
5. ModelCheckpoint + EarlyStopping 학습
6. 결과 저장: `results/cnn_adience/`

## 출력 파일

| 파일 | 내용 |
|------|------|
| `checkpoints/best_model.keras` | val_accuracy 최고 체크포인트 |
| `metrics.json` | test_accuracy, test_loss, split별 샘플 수 |
| `accuracy_curve.png` | Train/Val accuracy 곡선 |

## 주의

- `user_id` 단위 split으로 동일 인물이 여러 split에 걸치지 않도록 보장
- `fold_4_data.txt`만 사용 (fold 0~3 미사용)
- scikit-learn 의존성 없음
