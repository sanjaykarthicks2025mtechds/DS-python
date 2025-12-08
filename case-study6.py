# case6_deepfake_detection_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models

img_size = (128, 128)
batch_size = 32

train_dir = "dataset_root/train"
val_dir = "dataset_root/val"

# --------------------------------
# 1. Load image data
# --------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --------------------------------
# 2. Simple CNN for detection
# --------------------------------
model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=img_size + (3,)),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # 0 = real, 1 = fake (for example)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------
# 3. Train and evaluate
# --------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# --------------------------------
# 4. Example prediction
# --------------------------------
import numpy as np

def predict_image(path):
    img = tf.keras.utils.load_img(path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # batch dimension
    prob_fake = model.predict(arr)[0][0]
    print(f"Predicted 'fake' probability: {prob_fake:.3f}")
    if prob_fake > 0.5:
        print("Model thinks: FAKE")
    else:
        print("Model thinks: REAL")

# Example:
# predict_image("some_test_image.jpg")
