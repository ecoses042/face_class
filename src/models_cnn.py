"""
models_cnn.py
Adience age class 분류용 VGG-like CNN 모델을 정의한다.

사용법:
    from models_cnn import build_cnn_model            # VGG-like 분류 (8 클래스)
    from models_cnn import build_cnn_regressor        # VGG-like 회귀 (연속 나이)
    from models_cnn import build_small_cnn_regressor  # 소형 CNN 회귀 (연속 나이, ~3.3M)
"""

_VGG_BLOCKS = [
    # (filters, n_convs)
    (64,  2),
    (128, 2),
    (256, 3),
    (512, 3),
    (512, 3),
]


def _vgg_conv_blocks(layers):
    """5개 VGG conv+pool 블록을 반환한다."""
    result = []
    for filters, n_convs in _VGG_BLOCKS:
        for _ in range(n_convs):
            result.append(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        result.append(layers.MaxPooling2D())
    return result


def build_cnn_model(img_size: int = 128, num_classes: int = 8) -> "tf.keras.Model":
    from tensorflow.keras import layers, models

    if img_size < 32:
        raise ValueError("img_size must be at least 32 because the model uses 5 pooling blocks.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    return models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(2048, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="vgg_like_age_classifier")


def build_cnn_regressor(img_size: int = 128) -> "tf.keras.Model":
    """VGG-like CNN for continuous age regression (AIHub dataset).

    출력: Dense(1) — 예측 나이 (스칼라, activation 없음)
    """
    from tensorflow.keras import layers, models

    if img_size < 32:
        raise ValueError("img_size must be at least 32 because the model uses 5 pooling blocks.")

    return models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        *_vgg_conv_blocks(layers),
        layers.Flatten(),
        layers.Dense(2048, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1),
    ], name="vgg_like_age_regressor")


def build_small_cnn_regressor(img_size: int = 128) -> "tf.keras.Model":
    """소형 CNN for continuous age regression (AIHub dataset).

    Kaggle 노트북(neneti/deeplearning-final-project-cnn)의 분류 모델을
    회귀 출력으로 변환한 버전.
    Conv32→Conv64→Conv128 (valid padding, 3블록), Dense(128), Dense(1)
    파라미터: ~3.3M
    """
    from tensorflow.keras import layers, models

    if img_size < 16:
        raise ValueError("img_size must be at least 16.")

    return models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1),
    ], name="small_cnn_age_regressor")
