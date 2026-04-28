import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SAVE / LOAD CACHE
# =========================
def save_cache(name, data):
    path = os.path.join(CACHE_DIR, name)
    np.save(path, data)
    print(f"[INFO] Saved cache: {path}")

def load_cache(name):
    path = os.path.join(CACHE_DIR, name)

    if os.path.exists(path):
        print(f"[INFO] Loaded cache: {path}")
        return np.load(path, allow_pickle=True)

    return None

# =========================
# LOAD MNIST
# =========================
def load_mnist():
    cached_x_train = load_cache("x_train.npy")
    cached_y_train = load_cache("y_train.npy")
    cached_x_test = load_cache("x_test.npy")
    cached_y_test = load_cache("y_test.npy")

    if cached_x_train is not None:
        return cached_x_train, cached_y_train, cached_x_test, cached_y_test

    print("[INFO] Downloading MNIST...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    save_cache("x_train.npy", x_train)
    save_cache("y_train.npy", y_train)
    save_cache("x_test.npy", x_test)
    save_cache("y_test.npy", y_test)

    return x_train, y_train, x_test, y_test

# =========================
# Save figure
# =========================
def save_images_grid(images, labels, filename="grid.png", n=10):
    # Convert single image → batch
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        labels = np.array([labels])

    plt.figure(figsize=(10, 2))

    n = min(n, len(images))

    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(str(labels[i]))
        plt.axis('off')

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved grid to: {path}")
       
# =========================
# MAIN FUNCTION
# =========================
def main():
    print("===== MNIST PIPELINE START =====")

    # Load dataset (with caching)
    x_train, y_train, x_test, y_test = load_mnist()

    print(f"[INFO] Train shape: {x_train.shape}")
    print(f"[INFO] Test shape: {x_test.shape}")

    # Show one sample
    save_images_grid(
        x_train[:1],   # ✅ shape becomes (1, 28, 28, 1)
        y_train[:1],   # ✅ shape becomes (1,)
        filename="MNIST_SAMPLE.png",
        n=1
    )    
    print("===== DONE =====")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()