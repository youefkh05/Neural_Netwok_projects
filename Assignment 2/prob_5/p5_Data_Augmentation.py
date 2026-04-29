import os
import cv2
import csv
import pandas as pd
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
# Visualize Model Predictions
# =========================
def visualize_predictions(model, x_test, y_test, filename="predictions.png", n=25):
    idx = np.random.choice(len(x_test), n, replace=False)

    images = x_test[idx]
    labels = y_test[idx]

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    plt.figure(figsize=(14,14))  # bigger

    for i in range(n):
        plt.subplot(5,5,i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')

        title = f"T:{labels[i]} | P:{pred_labels[i]} ({confidences[i]:.2f})"
        color = "green" if labels[i] == pred_labels[i] else "red"

        plt.title(title, color=color, fontsize=9)  # bigger font
        plt.axis('off')

    plt.tight_layout(pad=2.0)  # spacing

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved predictions: {path}")  

# =========================
# Visualize misclassified samples
# =========================
def visualize_misclassified(model, x_test, y_test, filename="misclassified.png", n=25):
    preds = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    wrong_idx = np.where(pred_labels != y_test)[0]

    if len(wrong_idx) == 0:
        print("[INFO] No misclassified samples!")
        return

    idx = np.random.choice(wrong_idx, min(n, len(wrong_idx)), replace=False)

    plt.figure(figsize=(8,8))

    for i, id_ in enumerate(idx):
        plt.subplot(5,5,i+1)
        plt.imshow(x_test[id_].squeeze(), cmap='gray')

        title = f"T:{y_test[id_]} P:{pred_labels[id_]}"
        plt.title(title, color="red", fontsize=8)
        plt.axis('off')

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved misclassified: {path}")

# =========================
# Plot Training Curves
# =========================
def plot_training_curves(history, filename="training_curves.png"):
    hist = history.history

    plt.figure(figsize=(10,4))

    # -----------------
    # Loss
    # -----------------
    plt.plot(hist["loss"], label="Train Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved training curves: {path}") 

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
            subset["Aug per digit"],
            subset["Accuracy"],
            marker='o',
            label=f"Real={real_n}"
        )

    plt.title("Accuracy vs Augmentation")
    plt.xlabel("Generated Samples per Digit")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "results_plot.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved results plot: {path}")
                  
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
    real_options = [350]
    aug_options = [0, 1000, 1500, 2000]
    aug_options = [1000]

    results = []
    visualize_flag = True

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
            # TRAIN MODEL
            # =========================

            model = build_lenet()
            
            if visualize_flag == True:
                visualize_flag = False  # only visualize the first setting to save time 
                callback = PredictionVisualizationCallback(
                    x_test,
                    y_test,
                    interval=5,
                    prefix=f"{real_n}_{aug_n}"
                )
                
                history =model.fit(
                    x_final, y_final,
                    epochs=30,   
                    batch_size=64,
                    verbose=1,
                    callbacks=[callback]
                )
            else:
                history =model.fit(
                    x_final, y_final,
                    epochs=30,   
                    batch_size=64,
                    verbose=1,
                    )
            
            # =========================
            # EVALUATE MODEL
            # =========================
            loss, acc = model.evaluate(x_test, y_test, verbose=0)


            print(f"[RESULT] Accuracy = {acc*100:.2f}%")

            results.append([real_n, aug_n, acc])

    
    # =========================
    # FINAL PREDICTION VISUALIZATION the last trained model (on the last setting)
    #  =========================
    visualize_predictions(
        model,
        x_test,
        y_test,
        filename=f"final_pred_{real_n}_{aug_n}.png"
    )
            
    visualize_misclassified(
        model,
        x_test,
        y_test,
        filename=f"mis_{real_n}_{aug_n}.png"
        )
            
    plot_training_curves(
        history,
        filename=f"curve_{real_n}_{aug_n}.png"
        )
    # =========================
    # SAVE RESULTS TABLE
    # =========================
    results_path = os.path.join(OUTPUT_DIR, "results.csv")      

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Real per digit", "Aug per digit", "Accuracy"])
        writer.writerows(results)
        
    plot_results_from_csv(results_path)       

    print(f"[INFO] Results saved to: {results_path}")

    print("===== DONE =====")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()