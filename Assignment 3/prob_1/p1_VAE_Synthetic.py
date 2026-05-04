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
 
    # KEEP in [0,1] for VAE
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
            title = f"{labels[i]}\n{aug_types[i][:25]}..."
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
# AUGMENT DATASET (with cache + types)
# =========================
def augment_dataset(x, y, factor=10, cache_name=None):

    if cache_name:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached["x"], cached["y"], cached["aug_type"]

    print("[INFO] Augmenting dataset...")

    x_aug = []
    y_aug = []
    aug_types = []

    for i in range(len(x)):
        img = x[i]
        label = y[i]

        for _ in range(factor):

            aug_desc = []

            # -------------------------
            # ROTATION
            # -------------------------
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((14,14), angle, 1)
            rotated = cv2.warpAffine(img.squeeze(), M, (28,28))
            aug_desc.append(f"rot({angle:.1f})")

            # -------------------------
            # SHIFT
            # -------------------------
            tx = np.random.randint(-2, 3)
            ty = np.random.randint(-2, 3)
            M_shift = np.float32([[1,0,tx],[0,1,ty]])
            shifted = cv2.warpAffine(rotated, M_shift, (28,28))
            aug_desc.append(f"shift({tx},{ty})")

            # -------------------------
            # SCALE
            # -------------------------
            scale = np.random.uniform(0.9, 1.1)
            M_scale = cv2.getRotationMatrix2D((14,14), 0, scale)
            scaled = cv2.warpAffine(shifted, M_scale, (28,28))
            aug_desc.append(f"scale({scale:.2f})")

            # -------------------------
            # NOISE
            # -------------------------
            noise = np.random.normal(0, 0.03, (28,28))
            final = np.clip(scaled + noise, 0, 1)
            aug_desc.append("noise")

            # Save
            x_aug.append(final[..., None])
            y_aug.append(label)
            aug_types.append(" | ".join(aug_desc))

    x_aug = np.array(x_aug)
    y_aug = np.array(y_aug)
    aug_types = np.array(aug_types)

    print(f"[INFO] Augmented dataset: {x_aug.shape}")

    if cache_name:
        save_cache(cache_name, {
            "x": x_aug,
            "y": y_aug,
            "aug_type": aug_types
        })

    return x_aug, y_aug, aug_types

# =========================
# Split generated images by classifier confidence
# =========================
def split_by_confidence(fake_imgs, classifier):

    preds = classifier.predict(fake_imgs, verbose=0)
    conf = np.max(preds, axis=1)

    set_A = fake_imgs
    set_B = fake_imgs[conf >= 0.9]
    set_C = fake_imgs[(conf >= 0.6) & (conf < 0.9)]

    return set_A, set_B, set_C
# =========================
# Build a simple LeNet-like CNN
# =========================
def build_lenet():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28,28,1)),  # fix warning too

        tf.keras.layers.Conv2D(6, kernel_size=5, activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(16, kernel_size=5, activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# Plot Results from CSV
# =========================
def plot_results_from_csv(csv_path):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,6))

    # Unique real dataset sizes
    real_values = sorted(df["Real per digit"].unique())

    for real_n in real_values:
        subset = df[df["Real per digit"] == real_n]

        plt.plot(
            subset["GAN per digit"],
            subset["Accuracy"],
            marker='o',
            label=f"Real={real_n}"
        )

    plt.title("Accuracy vs VAE Generated Data")
    plt.xlabel("Generated Samples per Digit")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "results_plot.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved results plot: {path}")

# =========================
#  Build Encoder (for VAE)
# =========================
def build_encoder(latent_dim=20):
    x_input = tf.keras.Input(shape=(28,28,1))
    y_input = tf.keras.Input(shape=(10,))

    x = tf.keras.layers.Flatten()(x_input)
    x = tf.keras.layers.Concatenate()([x, y_input])

    x = tf.keras.layers.Dense(256, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)

    return tf.keras.Model([x_input, y_input], [z_mean, z_log_var])

# =========================
# Build Sampling Layer (for VAE)
# =========================
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon  

# =========================
# Build Decoder (for VAE)
# =========================
def build_decoder(latent_dim=20):
    z_input = tf.keras.Input(shape=(latent_dim,))
    y_input = tf.keras.Input(shape=(10,))

    x = tf.keras.layers.Concatenate()([z_input, y_input])
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(28*28, activation='sigmoid')(x)

    x = tf.keras.layers.Reshape((28,28,1))(x)

    return tf.keras.Model([z_input, y_input], x)
           
# =========================
# MAIN FUNCTION (GAN)
# =========================
def main():

    print("===== VAE PIPELINE START =====")

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
    # AUGMENT DATA (Step 1)
    # =========================
    x_aug, y_aug, aug_types = augment_dataset(
        x_small,
        y_small,
        factor=10,
        cache_name=f"aug_vae_{real_n}.npy"
    )
    
    # Visualize some augmented samples
    save_images_grid(
        x_aug[:5],
        y_aug[:5],
        aug_types=aug_types[:5],
        filename="vae_aug_samples.png",
        n=10
    )

    # Combine real + augmented
    x_vae = np.concatenate([x_small, x_aug])
    y_vae = np.concatenate([y_small, y_aug])

    print(f"[INFO] VAE training dataset: {x_vae.shape}")


    # =========================
    # TRAIN OR LOAD VAE
    # =========================
    latent_dim = 20

    encoder_path = os.path.join(CACHE_DIR, "vae_encoder.keras")
    decoder_path = os.path.join(CACHE_DIR, "vae_decoder.keras")

    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print("[INFO] Loading cached VAE...")
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)

    else:
        print("[INFO] Training VAE...")

        # One-hot labels
        y_vae_onehot = tf.keras.utils.to_categorical(y_vae, 10)

        encoder = build_encoder(latent_dim)
        decoder = build_decoder(latent_dim)

        z_mean, z_log_var = encoder([x_vae, y_vae_onehot])
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

        reconstructed = decoder([z, y_vae_onehot])

        vae = tf.keras.Model([x_vae, y_vae_onehot], reconstructed)

        vae.compile(optimizer='adam', loss='mse')

        start = time.time()

        vae.fit(
            [x_vae, y_vae_onehot],
            x_vae,
            epochs=30,
            batch_size=128,
            verbose=1
        )

        train_time = time.time() - start
        print(f"[TIME] VAE training time: {train_time:.2f} sec")

        encoder.save(encoder_path)
        decoder.save(decoder_path)

        print("[INFO] VAE models saved")
        
    # =========================
    # GENERATE SYNTHETIC DATA
    # =========================
    samples_per_digit = 1000
    latent_dim = 20

    gen_path = "vae_generated.npy"
    cached = load_cache(gen_path)

    if cached is not None:
        generated_images = cached["x"]
        generated_labels = cached["y"]

    else:
        print("[INFO] Generating synthetic data...")

        generated_images = []
        generated_labels = []

        for digit in range(10):

            z = np.random.normal(0,1,(samples_per_digit, latent_dim))
            labels = np.full(samples_per_digit, digit)
            labels_onehot = tf.keras.utils.to_categorical(labels, 10)

            imgs = decoder.predict([z, labels_onehot], verbose=0)

            generated_images.append(imgs)
            generated_labels.append(labels)

        generated_images = np.concatenate(generated_images)
        generated_labels = np.concatenate(generated_labels)

        save_cache(gen_path, {
            "x": generated_images,
            "y": generated_labels
        })

    print(f"[INFO] Generated dataset: {generated_images.shape}")

    # Visualize some generated samples
    save_images_grid(
        generated_images[:10],
        generated_labels[:10],
        filename="vae_generated_samples.png",
        n=10
    )
    print("===== DONE =====")
    
# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()