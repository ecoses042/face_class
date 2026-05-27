# models_cnn.py

VGG-like CNN 모델 빌더 모듈.

## 목적

Adience 연령대 분류(8-class)용 TensorFlow/Keras CNN 모델을 정의한다.
`build_cnn_model()` 함수 하나만 export하며, TensorFlow import는 함수 내부에서 수행한다.

## 함수

### `build_cnn_model(img_size=128, num_classes=8) -> tf.keras.Model`

VGG16 구조를 참고한 5-block CNN을 반환한다.

| 항목 | 값 |
|------|----|
| 입력 | `(img_size, img_size, 3)` |
| Conv 블록 | 5개 (64/128/256/512/512 filters, padding='same') |
| Flatten 후 입력 차원 | `(img_size/32)² × 512` |
| FC 레이어 | Dense(2048, relu) → Dropout(0.5) → Dense(1024, relu) → Dropout(0.5) → Dense(num_classes, softmax) |
| 총 파라미터 (기본값) | ~33.6M |

**제약:** `img_size >= 32` (5 MaxPool로 최소 1×1 유지), `num_classes >= 2`
