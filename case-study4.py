# case4_simple_gan_iot.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------
# 1. Create synthetic "real" IoT sensor data
#    Example: sinusoidal with noise between (0, 1)
# --------------------------------
def generate_real_samples(n_points=2000):
    t = np.linspace(0, 20, n_points)
    values = 0.5 + 0.3 * np.sin(t) + 0.1 * np.random.randn(n_points)
    values = values.reshape(-1, 1).astype("float32")
    return values

real_data = generate_real_samples()

# --------------------------------
# 2. Build generator and discriminator
# --------------------------------
noise_dim = 16

def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(noise_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="tanh"),  # output approx in [-1,1]
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
d_optimizer = tf.keras.optimizers.Adam(1e-4)
g_optimizer = tf.keras.optimizers.Adam(1e-4)

# --------------------------------
# 3. Training step
# --------------------------------
BATCH_SIZE = 64

@tf.function
def train_step(real_batch):
    noise = tf.random.normal([tf.shape(real_batch)[0], noise_dim])

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        fake_samples = generator(noise, training=True)

        real_output = discriminator(real_batch, training=True)
        fake_output = discriminator(fake_samples, training=True)

        d_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        d_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    grads_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    grads_g = gen_tape.gradient(g_loss, generator.trainable_variables)

    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss

# --------------------------------
# 4. GAN training loop
# --------------------------------
def train_gan(data, epochs=200):
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1024).batch(BATCH_SIZE)
    for epoch in range(epochs):
        d_losses, g_losses = [], []
        for batch in dataset:
            d_loss, g_loss = train_step(batch)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}: D_loss={np.mean(d_losses):.4f}, "
                  f"G_loss={np.mean(g_losses):.4f}")

train_gan(real_data, epochs=200)

# --------------------------------
# 5. Compare real vs synthetic distributions
# --------------------------------
import matplotlib.pyplot as plt

noise = tf.random.normal([1000, noise_dim])
fake_data = generator(noise, training=False).numpy()

plt.figure()
plt.hist(real_data, bins=40, alpha=0.6, label="Real")
plt.hist(fake_data, bins=40, alpha=0.6, label="Synthetic (GAN)")
plt.legend()
plt.title("Real vs synthetic IoT sensor values")
plt.show()
