"""
models_cnn.py
Adience age class 분류용 VGG-like CNN 모델을 정의한다.

사용법: from models_cnn import build_cnn_model
"""


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
