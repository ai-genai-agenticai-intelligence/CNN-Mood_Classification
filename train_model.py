from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "Traning"
MODEL_PATH = BASE_DIR / "mood_model.tflite"
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 8


def main():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["Happy", "Not Happy"],
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training",
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["Happy", "Not Happy"],
        shuffle=False,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.Rescaling(1 / 200.0),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_dataset, validation_data=validation_dataset, epochs=8)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    MODEL_PATH.write_bytes(converter.convert())
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
