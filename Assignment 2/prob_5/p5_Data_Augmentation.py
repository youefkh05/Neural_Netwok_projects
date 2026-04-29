import os
import cv2
import csv
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
# Main function to get final dataset (reduced real + augmented)
# =========================
def get_final_dataset(x_train, y_train, real_n, aug_n):
    cache_name = f"final_{real_n}_real_{aug_n}_aug.npy"

    cached = load_cache(cache_name)
    
    if cached is not None:
        return cached["x"], cached["y"]

    # Step 1: reduced real data
    x_real, y_real = get_reduced_dataset(
        x_train, y_train,
        real_n,
        cache_name=f"real_{real_n}.npy"
    )

    # Step 2: augmentation
    if aug_n > 0:
        x_aug, y_aug, aug_types = augment_dataset(
            x_real, y_real,
            aug_n,
            cache_name=f"aug_{real_n}_{aug_n}.npy"
        )

        x_final, y_final = build_final_dataset(x_real, y_real, x_aug, y_aug)
    else:
        x_final, y_final = x_real, y_real

    save_cache(cache_name, {"x": x_final, "y": y_final})

    return x_final, y_final

# =========================
# Data Augmentation Helper Functions   
# =========================
def rotate_image(img, angle):
    img = img.squeeze()
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))[..., None]


def add_noise(img):
    noise = np.random.normal(0, 0.1, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)


def shift_image(img, dx, dy):
    img = img.squeeze()
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted[..., None]

# =========================
# Data Augmentation Function
# ========================= 
def augment_dataset(x, y, num_generated_per_class, cache_name=None):
    """
    x: (N, 28, 28, 1)
    y: (N,)
    """

    # ======================
    # CACHE
    # ======================
    if cache_name:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached["x"], cached["y"], cached["aug_type"]

    print("[INFO] Generating augmented dataset...")

    x_aug = []
    y_aug = []
    aug_types = []

    for digit in range(10):
        class_idx = np.where(y == digit)[0]
        class_images = x[class_idx]

        generated = 0

        while generated < num_generated_per_class:
            img = class_images[np.random.randint(len(class_images))]

            choice = np.random.choice(["rotate", "shift", "noise"])

            if choice == "rotate":
                angle = np.random.choice([-10, -5, 5, 10])
                new_img = rotate_image(img, angle)
                aug_types.append(f"rot({angle})")

            elif choice == "shift":
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                new_img = shift_image(img, dx, dy)
                aug_types.append(f"shift({dx},{dy})")

            else:
                new_img = add_noise(img)
                aug_types.append("noise")

            x_aug.append(new_img)
            y_aug.append(digit)

            generated += 1

        print(f"[INFO] Digit {digit}: generated {num_generated_per_class}")

    x_aug = np.array(x_aug)
    y_aug = np.array(y_aug)

    # ======================
    # CACHE SAVE
    # ======================
    if cache_name:
        save_cache(cache_name, {
            "x": x_aug,
            "y": y_aug,
            "aug_type": np.array(aug_types)
        })
        
    return x_aug, y_aug, aug_types

# =========================
# Combine real + augmented datasets
# =========================
def build_final_dataset(x_real, y_real, x_aug, y_aug):
    x_final = np.concatenate([x_real, x_aug], axis=0)
    y_final = np.concatenate([y_real, y_aug], axis=0)

    print("[INFO] Final dataset:", x_final.shape)

    return x_final, y_final

# =========================
# Run experiments with different real vs augmented splits
# =========================
def run_experiments(x_train, y_train, x_test, y_test):

    real_options = [350, 750, 1000]
    aug_options = [0, 1000, 1500, 2000]

    results = []

    for real_n in real_options:
        for aug_n in aug_options:

            print(f"\n===== EXPERIMENT: {real_n} real + {aug_n} aug =====")

            x_final, y_final = get_final_dataset(
                x_train, y_train,
                real_n,
                aug_n
            )

            # TODO: TRAIN LENET HERE
            # model = build_lenet()
            # model.fit(...)
            # acc = model.evaluate(...)

            acc = np.random.uniform(0.90, 0.99)  # placeholder

            results.append([real_n, aug_n, acc])

    return results
      
# =========================
# MAIN FUNCTION
# =========================
def main():
    print("===== MNIST PIPELINE START =====")

    # =========================
    # LOAD DATA
    # =========================
    x_train, y_train, x_test, y_test = load_mnist()

    print(f"[INFO] Train shape: {x_train.shape}")
    print(f"[INFO] Test shape: {x_test.shape}")

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
    # EXPERIMENT SETTINGS
    # =========================
    real_options = [350, 750, 1000]
    aug_options = [0, 1000, 1500, 2000]

    results = []

    # =========================
    # LOOP OVER EXPERIMENTS
    # =========================
    for real_n in real_options:
        for aug_n in aug_options:

            print(f"\n===== {real_n} real + {aug_n} augmented =====")

            # Build dataset (cached)
            x_final, y_final = get_final_dataset(
                x_train, y_train,
                real_n, aug_n
            )
            
            #  SHUFFLE HERE
            x_final, y_final = shuffle_dataset(x_final, y_final)

            #  OPTIONAL: visualize augmentation once
            if aug_n > 0:
                aug_data = load_cache(f"aug_{real_n}_{aug_n}.npy")

                x_aug = aug_data["x"]
                y_aug = aug_data["y"]
                aug_types = aug_data["aug_type"]
                
                save_images_grid(
                    x_aug[:10],
                    y_aug[:10],
                    aug_types=aug_types[:10],
                    filename=f"aug_{real_n}_{aug_n}.png",
                    n=10
                )

            # =========================
            # TODO: TRAIN MODEL
            # =========================
            # model = build_lenet()
            # model.fit(x_final, y_final, epochs=5, batch_size=64)

            # =========================
            # TODO: EVALUATE
            # =========================
            # loss, acc = model.evaluate(x_test, y_test)

            acc = np.random.uniform(0.90, 0.99)  # placeholder for now

            print(f"[RESULT] Accuracy = {acc*100:.2f}%")

            results.append([real_n, aug_n, acc])

    # =========================
    # SAVE RESULTS TABLE
    # =========================
    results_path = os.path.join(OUTPUT_DIR, "results.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Real per digit", "Aug per digit", "Accuracy"])
        writer.writerows(results)

    print(f"[INFO] Results saved to: {results_path}")

    print("===== DONE =====")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()