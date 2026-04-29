import os
import cv2
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time


# =========================
# CONFIG
# =========================
CACHE_DIR = "cache"
OUTPUT_DIR = "Figures"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CALLBACK TO VISUALIZE PREDICTIONS DURING TRAINING
# =========================
class PredictionVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, interval=2, prefix="train"):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.interval = interval
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return

        idx = np.random.choice(len(self.x_test), 16, replace=False)
        images = self.x_test[idx]
        labels = self.y_test[idx]

        preds = self.model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)

        plt.figure(figsize=(14,14))  # bigger

        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(images[i].squeeze(), cmap='gray')

            correct = labels[i] == pred_labels[i]
            color = "green" if correct else "red"

            title = f"T:{labels[i]} | P:{pred_labels[i]} ({confidences[i]:.2f})"

            plt.title(title, fontsize=10, color=color)  #  bigger font
            plt.axis('off')

        plt.tight_layout(pad=2.0)  #  spacing

        path = os.path.join(
            OUTPUT_DIR,
            f"{self.prefix}_epoch_{epoch+1}.png"
        )

        plt.savefig(path, dpi=300)
        plt.close()

        print(f"[INFO] Saved epoch visualization: {path}")
                  
# =========================
# SAVE / LOAD CACHE
# =========================
def save_cache(name, data):
    path = os.path.join(CACHE_DIR, name)

    np.save(path, data, allow_pickle=True)  

    print(f"[INFO] Saved cache: {path}")
    
def load_cache(name):
    path = os.path.join(CACHE_DIR, name)

    if os.path.exists(path):
        try:
            print(f"[INFO] Loaded cache: {path}")
            data = np.load(path, allow_pickle=True)

            #  convert numpy object → dict
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()

            return data

        except Exception:
            print(f"[WARNING] Corrupted cache → deleting {path}")
            os.remove(path)
            return None

    return None

# =========================
# LOAD MNIST
# =========================
def load_mnist():
    train_data = load_cache("train_data.npy")
    test_data  = load_cache("test_data.npy")

    if train_data is not None and test_data is not None:
        return (
            train_data["x"], train_data["y"],
            test_data["x"], test_data["y"]
        )

    print("[INFO] Downloading MNIST...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    save_cache("train_data.npy", {"x": x_train, "y": y_train})
    save_cache("test_data.npy", {"x": x_test, "y": y_test})

    return x_train, y_train, x_test, y_test

# =========================
# Shuffle dataset 
# =========================
def shuffle_dataset(x, y):
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

# =========================
# Save figure
# =========================
def save_images_grid(images, labels, aug_types=None, filename="grid.png", n=10):    # Convert single image → batch
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        labels = np.array([labels])

    plt.figure(figsize=(10, 2))

    n = min(n, len(images))

    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')

        if aug_types is not None:
            title = f"{labels[i]}\n{aug_types[i]}"
        else:
            title = str(labels[i])

        plt.title(title, fontsize=8)
        plt.axis('off')

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved grid to: {path}")

# =========================
# cache dataset
# =========================
def get_reduced_dataset(x, y, samples_per_digit, cache_name=None):
    if cache_name:
        cached = load_cache(cache_name)

        if cached is not None:
            return cached["x"], cached["y"]

    x_small = []
    y_small = []

    for digit in range(10):
        idx = np.where(y == digit)[0][:samples_per_digit]
        x_small.append(x[idx])
        y_small.append(y[idx])

    x_small = np.concatenate(x_small)
    y_small = np.concatenate(y_small)

    print(f"[INFO] Reduced dataset: {x_small.shape}")

    if cache_name:
        save_cache(cache_name, {"x": x_small, "y": y_small})

    return x_small, y_small

# =========================
# Select 3 best images per digit based on classifier confidence
# =========================
def select_3_per_digit(fake_imgs, classifier):

    preds = classifier.predict(fake_imgs, verbose=0)
    labels = np.argmax(preds, axis=1)
    confidence = np.max(preds, axis=1)

    selected_imgs = []

    for digit in range(10):
        idx = np.where(labels == digit)[0]

        if len(idx) == 0:
            print(f"[WARNING] No samples for digit {digit}")
            continue

        # take top 3 OR less if not available
        best = idx[np.argsort(confidence[idx])[-min(3, len(idx)):]]
        selected_imgs.extend(fake_imgs[best])

    return np.array(selected_imgs)

# =========================
# BUILD GENERATOR MODEL
# =========================
def build_generator(latent_dim=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*256, input_dim=latent_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Reshape((7,7,256)),

        tf.keras.layers.Conv2DTranspose(128, (4,4), strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(64, (4,4), strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(1, (3,3), padding="same", activation="tanh")
    ])
    return model

# =========================
# BUILD DISCRIMINATOR MODEL
# =========================
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), strides=2, padding="same", input_shape=(28,28,1)),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(128, (3,3), strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(256, (3,3), strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return model

# =========================
# GENERATE AND SAVE IMAGES
# ========================
def generate_and_save_images(generator, epoch, latent_dim=100, n=16):
    noise = np.random.normal(0,1,(n, latent_dim))
    generated = generator.predict(noise, verbose=0)

    plt.figure(figsize=(8,8))

    for i in range(n):
        plt.subplot(4,4,i+1)
        plt.imshow(generated[i].squeeze(), cmap='gray')
        plt.axis('off')

    path = os.path.join(OUTPUT_DIR, f"gan_epoch_{epoch}.png")
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved GAN images: {path}")

# =========================
# TRAIN GAN
# ========================
def train_gan(x_train, epochs=100, batch_size=64, latent_dim=100):
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Freeze discriminator BEFORE building GAN
    discriminator.trainable = False

    gan_input = tf.keras.Input(shape=(latent_dim,))
    fake_img = generator(gan_input)
    gan_output = discriminator(fake_img)

    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                loss="binary_crossentropy")
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    
    d_losses = []
    g_losses = []

    for epoch in range(epochs):

        # ---------------------
        # Train Discriminator
        # ---------------------
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx] 

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        # Label smoothing
        real_labels = np.ones((batch_size,1)) * 0.9
        fake_labels = np.zeros((batch_size,1)) 
        
        # Train discriminator LESS aggressively
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)

        # Only sometimes train on fake (stabilization)
        d_loss_fake = 0
        # Train discriminator LESS
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        
        # ---------------------
        # Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        discriminator.trainable = False
        
        for _ in range(2):  # train generator twice
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, real_labels)
        
        # ---------------------
        # Visualization
        # ---------------------
        d_losses.append(d_loss_real)
        g_losses.append(g_loss)
        
        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] D_loss: {d_loss_real} G_loss: {g_loss}")
            generate_and_save_images(generator, epoch)

    plt.plot(d_losses, label="D loss")
    plt.plot(g_losses, label="G loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "gan_loss_curve.png"))
    plt.close()
    
    return generator, discriminator

# =========================
# SHOW BEST FAKE IMAGES (DISCRIMINATOR APPROVED)
# =========================
def show_discriminator_mistakes(generator, discriminator, real_imgs, latent_dim=100, n=100):
    
    # ---------- Fake images ----------
    noise = np.random.normal(0,1,(n, latent_dim))
    fake_imgs = generator.predict(noise, verbose=0)

    fake_scores = discriminator.predict(fake_imgs, verbose=0).reshape(-1)

    # WRONG: fake classified as real (score high)
    fake_wrong_idx = np.where(fake_scores > 0.7)[0]

    # ---------- Real images ----------
    idx = np.random.randint(0, real_imgs.shape[0], n)
    real_sample = real_imgs[idx]

    real_scores = discriminator.predict(real_sample, verbose=0).reshape(-1)

    # WRONG: real classified as fake (score low)
    real_wrong_idx = np.where(real_scores < 0.3)[0]

    # ---------- Plot ----------
    plt.figure(figsize=(12,6))

    # Fake mistakes
    for i, id_ in enumerate(fake_wrong_idx[:8]):
        plt.subplot(2,8,i+1)
        plt.imshow(fake_imgs[id_].squeeze(), cmap='gray')
        plt.title(f"Fake→Real ({fake_scores[id_]:.2f})", fontsize=8, color='green')
        plt.axis('off')

    # Real mistakes
    for i, id_ in enumerate(real_wrong_idx[:8]):
        plt.subplot(2,8,8+i+1)
        plt.imshow(real_sample[id_].squeeze(), cmap='gray')
        plt.title(f"Real→Fake ({real_scores[id_]:.2f})", fontsize=8, color='red')
        plt.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gan_mistakes.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved discriminator mistakes: {path}")
           
# =========================
# MAIN FUNCTION (GAN)
# =========================
def main():

    print("===== GAN PIPELINE START =====")

    # =========================
    # LOAD DATA (cached)
    # =========================
    x_train, y_train, _, _ = load_mnist()
    print(f"[INFO] Train shape: {x_train.shape}")

    # =========================
    # QUICK VISUAL CHECK
    # =========================
    save_images_grid(
        x_train[:10],
        y_train[:10],
        filename="MNIST_samples.png",
        n=10
    )

    # =========================
    # REDUCE DATASET (350 per digit)
    # =========================
    real_n = 350

    x_small, y_small = get_reduced_dataset(
        x_train, y_train,
        real_n,
        cache_name=f"real_{real_n}.npy"
    )

    print(f"[INFO] Reduced dataset: {x_small.shape}")

    # =========================
    # LOAD CLASSIFIER (LeNet)
    # =========================
    print("[INFO] Loading classifier...")
    classifier = tf.keras.models.load_model("lenet_model.keras")  

    # =========================
    # GAN CONFIG
    # =========================
    gen_path = os.path.join(CACHE_DIR, f"gan_generator_{real_n}.h5")
    disc_path = os.path.join(CACHE_DIR, f"gan_discriminator_{real_n}.h5")

    latent_dim = 100
    total_samples = 2000   # IMPORTANT (big pool)

    # =========================
    # TRAIN OR LOAD GAN
    # =========================
    start_time = time.time()

    if os.path.exists(gen_path) and os.path.exists(disc_path):
        print("[INFO] Loading cached GAN models...")

        generator = tf.keras.models.load_model(gen_path)
        discriminator = tf.keras.models.load_model(disc_path)

        discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
            loss="binary_crossentropy"
        )

    else:
        print("[INFO] Training GAN...")

        generator, discriminator = train_gan(
            x_small,
            epochs=2000,
            batch_size=64,
            latent_dim=latent_dim
        )

        generator.save(gen_path)
        discriminator.save(disc_path)

        print(f"[INFO] Saved generator: {gen_path}")
        print(f"[INFO] Saved discriminator: {disc_path}")

    train_time = time.time() - start_time
    print(f"[TIME] GAN Setup time: {train_time:.2f} sec")

    # =========================
    # GENERATE MANY SAMPLES
    # =========================
    gen_start = time.time()

    cached = load_cache("gan_generated.npy")

    if cached is not None:
        generated_images = cached["x"]
    else:
        noise = np.random.normal(0, 1, (total_samples, latent_dim))        
        generated_images = generator.predict(noise, verbose=0)
        save_cache("gan_generated.npy", {"x": generated_images})

    gen_time = time.time() - gen_start
    print(f"[TIME] Generation time: {gen_time:.4f} sec")

    # =========================
    # SELECT BEST 3 PER DIGIT
    # =========================
    print("[INFO] Selecting best samples per digit...")

    selected_images = select_3_per_digit(generated_images, classifier)

    # =========================
    # VISUALIZE SELECTED (3 x 10) WITH LABELS
    # =========================
    plt.figure(figsize=(12,4))

    preds = classifier.predict(selected_images, verbose=0)
    labels = np.argmax(preds, axis=1)

    for i in range(min(30, len(selected_images))):
        plt.subplot(3,10,i+1)
        plt.imshow(selected_images[i].squeeze(), cmap='gray')
        plt.title(str(labels[i]), fontsize=8)   # ← LABEL HERE
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gan_selected_per_digit.png"), dpi=300)
    plt.close()

    print("[INFO] Saved labeled selected GAN digits")
    # =========================
    # DISCRIMINATOR ANALYSIS
    # =========================

    # FIXED CALL 
    show_discriminator_mistakes(generator, discriminator, x_small)

    # =========================
    # VISUALIZE RAW GENERATED
    # =========================
    plt.figure(figsize=(6,6))

    for i in range(16):   # show first only
        plt.subplot(4,4,i+1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.axis('off')
        
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gan_generated_samples.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved GAN generated samples: {path}")

    print("===== DONE =====")
    
# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()