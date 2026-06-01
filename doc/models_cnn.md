# models_cnn.py

TensorFlow/Keras CNN 모델 빌더 모듈.

## 목적

Adience 연령대 분류 모델과 AI-Hub 한국인 얼굴 나이 회귀 모델을 정의한다.
TensorFlow import는 각 함수 내부에서 수행한다.

## 함수

### `build_cnn_model(img_size=128, num_classes=8) -> tf.keras.Model`

Adience 연령대 분류(8-class)용 VGG-like CNN을 반환한다.

| 항목 | 값 |
|------|----|
| 입력 | `(img_size, img_size, 3)` |
| 출력 | `Dense(num_classes, softmax)` |
| Conv 블록 | 5개 (64/128/256/512/512 filters, padding='same') |
| FC 레이어 | Dense(2048, relu) → Dropout(0.5) → Dense(1024, relu) → Dropout(0.5) |
| 총 파라미터 (기본값) | ~33.6M |

**제약:** `img_size >= 32`, `num_classes >= 2`

### `build_cnn_regressor(img_size=128) -> tf.keras.Model`

AI-Hub 얼굴 crop 기반 연속 나이 회귀용 VGG-like CNN을 반환한다.
`build_cnn_model()`과 같은 5-block VGG 구조를 쓰되 마지막 출력은 activation 없는 `Dense(1)`이다.

| 항목 | 값 |
|------|----|
| 입력 | `(img_size, img_size, 3)` |
| 출력 | `Dense(1)` 연속 나이 예측 |
| Conv 블록 | 5개 (64/128/256/512/512 filters, padding='same') |
| FC 레이어 | Dense(2048, relu) → Dropout(0.5) → Dense(1024, relu) → Dropout(0.5) |
| 총 파라미터 (기본값) | ~33.6M |

**제약:** `img_size >= 32`

### `build_small_cnn_regressor(img_size=128) -> tf.keras.Model`

AI-Hub 얼굴 crop 기반 연속 나이 회귀용 소형 CNN을 반환한다.
3개 Conv/MaxPool 블록과 작은 Dense head를 사용해 VGG-like 회귀 모델보다 파라미터 수를 줄인다.

| 항목 | 값 |
|------|----|
| 입력 | `(img_size, img_size, 3)` |
| 출력 | `Dense(1)` 연속 나이 예측 |
| Conv 블록 | Conv32 → MaxPool → Conv64 → MaxPool → Conv128 → MaxPool |
| FC 레이어 | Dense(128, relu) → Dropout(0.3) |
| 총 파라미터 (기본값) | ~3.3M |

**제약:** `img_size >= 16`
