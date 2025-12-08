# case5_conditional_gan_mnist.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------
# 1. Load MNIST
# --------------------------------
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 127.5) - 1.0  # [-1,1]
x_train = np.expand_dims(x_train, axis=-1)

num_classes = 10
latent_dim = 64

# --------------------------------
# 2. Build conditional generator and discriminator
# --------------------------------
def build_cgen():
    label_input = layers.Input(shape=(), dtype="int32")
    z_input = layers.Input(shape=(latent_dim,))

    # Embed label to a dense vector
    label_embedding = layers.Embedding(num_classes, 16)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    x = layers.Concatenate()([z_input, label_embedding])
    x = layers.Dense(7 * 7 * 128, use_bias=False)(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                               padding="same", use_bias=False)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                               padding="same", activation="tanh")
    model = tf.keras.Model([z_input, label_input], x, name="cGenerator")
    return model

def build_cdisc():
    img_input = layers.Input(shape=(28, 28, 1))
    label_input = layers.Input(shape=(), dtype="int32")

    # Embed label and replicate spatially to concatenate
    label_embedding = layers.Embedding(num_classes, 28 * 28)(label_input)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

    x = layers.Concatenate(axis=-1)([img_input, label_embedding])
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([img_input, label_input], x, name="cDiscriminator")
    return model

generator = build_cgen()
discriminator = build_cdisc()
cross_entropy = tf.keras.losses.BinaryCrossentropy()
g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

BATCH_SIZE = 128

# --------------------------------
# 3. Single training step
# --------------------------------
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([tf.shape(images)[0], latent_dim])
    random_labels = tf.random.uniform(
        shape=(tf.shape(images)[0],), minval=0, maxval=num_classes, dtype=tf.int32
    )

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = generator([noise, random_labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([fake_images, random_labels], training=True)

        d_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        d_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)

    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss

# --------------------------------
# 4. Training loop (short, for demo)
# --------------------------------
def train(epochs=10):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(1024).batch(BATCH_SIZE)

    for epoch in range(epochs):
        d_losses, g_losses = [], []
        for images, labels in dataset:
            d_loss, g_loss = train_step(images, labels)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
        print(f"Epoch {epoch+1}: D_loss={np.mean(d_losses):.4f}, "
              f"G_loss={np.mean(g_losses):.4f}")

train(epochs=10)

# --------------------------------
# 5. Generate "styled" digits for a chosen label
# --------------------------------
import matplotlib.pyplot as plt

def show_generated(label=7, n=16):
    noise = tf.random.normal([n, latent_dim])
    labels = tf.constant([label] * n, dtype=tf.int32)
    generated = generator([noise, labels], training=False).numpy()
    generated = (generated + 1.0) / 2.0  # back to [0,1]

    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Generated digits with 'style' label = {label}")
    plt.show()

show_generated(label=3)
