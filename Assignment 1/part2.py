# OS path/file operations and directory checks.
import os
# CSV read/write for benchmark and summary tables.
import csv
# JSON export for metadata artifacts.
import json
# Timing utilities for feature extraction and model benchmarking.
import time
# Access command-line arguments and runtime context.
import sys
# Command-line interface parsing.
import argparse
# Lightweight class container for model artifacts.
from dataclasses import dataclass

# OpenCV image I/O, resizing, and DCT transforms.
import cv2
# Core numerical arrays and vectorized computation.
import numpy as np
# Plotting figures for report visuals and diagnostics.
import matplotlib.pyplot as plt
# HOG feature descriptor extraction.
from skimage.feature import hog
# Principal Component Analysis for dimensionality reduction.
from sklearn.decomposition import PCA
# K-means clustering for per-class centroid models.
from sklearn.cluster import KMeans
# Support Vector Machine classifier implementations.
from sklearn.svm import SVC
# Accuracy and confusion-matrix metrics/visualization.
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Steps:
# 1) load ReducedMNIST,
# 2) build features (DCT / PCA / HOG),
# 3) benchmark K-means-per-class and SVM,
# 4) export report-ready tables and figures.


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "part2", "step1")
DEFAULT_INDIAN_DIR = os.path.join(BASE_DIR, "Indian_Digits_Train")
DEFAULT_LABELS_CSV = os.path.join(BASE_DIR, "pipeline3", "full_labels_vector.csv")

DEFAULT_AUTO_CONFIG = {
    "dataset_dir": "",
    "dataset_npz": "",
    "labels_csv": DEFAULT_LABELS_CSV,
    "train_ratio": 0.8,
    "output_dir": os.path.join(BASE_DIR, "part2", "full_run"),
    "skip_classifiers": False,
}


# ==============================
# DATA LOADING
# ==============================

# Load and normalize all images from a single digit folder.
def _load_images_from_class_folder(class_dir, label):
    # Read all image files for one class folder (e.g., train/3 or test/7).
    files = [f for f in os.listdir(class_dir) if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
    files = sorted(files)

    images = []
    labels = []
    for name in files:
        path = os.path.join(class_dir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Keep all data in the same 28x28 resolution expected by the assignment.
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        images.append(img)
        labels.append(label)

    return images, labels


# Load the ReducedMNIST folder structure with train/test and digit subfolders.
def load_dataset_from_dir(dataset_dir):
    """Load ReducedMNIST-style folder dataset with train/test and class folders 0..9."""
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise ValueError(
            "Dataset directory must contain 'train/' and 'test/' folders, each with class folders 0..9"
        )

    x_train, y_train, x_test, y_test = [], [], [], []

    for digit in range(10):
        # For each digit, collect train and test samples separately.
        tr_class = os.path.join(train_dir, str(digit))
        te_class = os.path.join(test_dir, str(digit))

        if not os.path.isdir(tr_class) or not os.path.isdir(te_class):
            raise ValueError(f"Missing class folder for digit {digit} under train/test")

        imgs, lbls = _load_images_from_class_folder(tr_class, digit)
        x_train.extend(imgs)
        y_train.extend(lbls)

        imgs, lbls = _load_images_from_class_folder(te_class, digit)
        x_test.extend(imgs)
        y_test.extend(lbls)

    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.int32)
    x_test = np.array(x_test, dtype=np.uint8)
    y_test = np.array(y_test, dtype=np.int32)

    # Assignment sanity check for official ReducedMNIST sizes.
    # Expected counts are 1000 train and 200 test images per class.
    for digit in range(10):
        tr_count = int(np.sum(y_train == digit))
        te_count = int(np.sum(y_test == digit))
        if tr_count != 1000 or te_count != 200:
            print(
                f"[WARNING] Digit {digit}: expected train/test counts 1000/200, got {tr_count}/{te_count}."
            )

    return x_train, y_train, x_test, y_test


# Load a prepacked NPZ file containing train/test images and labels.
def load_dataset_from_npz(npz_path):
    """Load dataset from pre-packed npz containing x_train/y_train/x_test/y_test."""
    data = np.load(npz_path)

    required_keys = ["x_train", "y_train", "x_test", "y_test"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing key '{key}' in npz file")

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    # Normalize geometry if npz images are not already 28x28.
    if x_train.ndim == 3 and x_train.shape[1:] != (28, 28):
        x_train = np.array([cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) for img in x_train])
    if x_test.ndim == 3 and x_test.shape[1:] != (28, 28):
        x_test = np.array([cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA) for img in x_test])

    return x_train.astype(np.uint8), y_train.astype(np.int32), x_test.astype(np.uint8), y_test.astype(np.int32)


# Load the unlabeled Indian digits folder as a fallback dataset.
def load_dataset_from_indian_digits(indian_dir, train_ratio=0.8):
    """Fallback loader when no labels are available (labels become -1 placeholders)."""
    files = [f for f in os.listdir(indian_dir) if f.lower().endswith(".bmp")]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for name in files:
        path = os.path.join(indian_dir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        images.append(img)

    if not images:
        raise ValueError(f"No images loaded from: {indian_dir}")

    images = np.array(images, dtype=np.uint8)
    # Deterministic split keeps runs reproducible for debugging/reporting.
    split_idx = int(len(images) * train_ratio)
    split_idx = max(1, min(split_idx, len(images) - 1))

    x_train = images[:split_idx]
    x_test = images[split_idx:]

    # Labels are unknown for Indian_Digits_Train in this context.
    y_train = np.full((len(x_train),), -1, dtype=np.int32)
    y_test = np.full((len(x_test),), -1, dtype=np.int32)

    print("[WARNING] Using Indian_Digits_Train fallback with unknown labels (-1).")
    print(f"[WARNING] Deterministic split applied: train={len(x_train)}, test={len(x_test)}")

    return x_train, y_train, x_test, y_test


# Load the recovered label vector that matches the sorted image order.
def load_label_vector_csv(labels_csv):
    """Load recovered 1D label vector (one label per image, filename order)."""
    if not os.path.isfile(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    labels = np.loadtxt(labels_csv, delimiter=",", dtype=np.int32)
    labels = np.atleast_1d(labels).astype(np.int32)

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    return labels


# Load Indian digits together with recovered labels and split them.
def load_dataset_from_indian_digits_with_labels(indian_dir, labels_csv):
    """Load Indian digits + recovered labels, then split into train/test."""
    files = [f for f in os.listdir(indian_dir) if f.lower().endswith(".bmp")]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for name in files:
        path = os.path.join(indian_dir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        images.append(img)

    if not images:
        raise ValueError(f"No images loaded from: {indian_dir}")

    labels = load_label_vector_csv(labels_csv)
    # Labels must be exactly aligned with sorted image filenames.
    if len(labels) != len(images):
        raise ValueError(
            f"Label count mismatch: {len(labels)} labels for {len(images)} images"
        )

    images = np.array(images, dtype=np.uint8)
    labels = labels.astype(np.int32)
    split_idx = int(len(images) * 0.8)
    split_idx = max(1, min(split_idx, len(images) - 1))

    x_train = images[:split_idx]
    y_train = labels[:split_idx]
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    print(f"[INFO] Loaded {len(images)} labeled images from Indian_Digits_Train")
    print(f"[INFO] Using recovered labels from {os.path.relpath(labels_csv, BASE_DIR)}")
    print(f"[INFO] Deterministic split applied: train={len(x_train)}, test={len(x_test)}")

    return x_train, y_train, x_test, y_test


# Choose the best available dataset source and return the four arrays.
def load_reduced_mnist(dataset_dir=None, dataset_npz=None, labels_csv=None, train_ratio=0.8):
    # Priority order:
    # 1) explicit --dataset-npz
    # 2) explicit --dataset-dir
    # 3) workspace defaults (Indian_Digits_Train + recovered labels)
    if dataset_npz:
        return load_dataset_from_npz(dataset_npz)
    if dataset_dir:
        if labels_csv and os.path.isfile(labels_csv):
            return load_dataset_from_indian_digits_with_labels(dataset_dir, labels_csv)
        return load_dataset_from_dir(dataset_dir)

    if os.path.isdir(DEFAULT_INDIAN_DIR):
        if os.path.isfile(DEFAULT_LABELS_CSV):
            return load_dataset_from_indian_digits_with_labels(DEFAULT_INDIAN_DIR, DEFAULT_LABELS_CSV)
        return load_dataset_from_indian_digits(DEFAULT_INDIAN_DIR, train_ratio=train_ratio)

    raise ValueError(
        "Pass either --dataset-dir or --dataset-npz, or ensure Indian_Digits_Train and pipeline3/full_labels_vector.csv exist"
    )


# ==============================
# VISUALIZATION HELPERS
# ==============================

# Save a grid of sample images so the dataset can be inspected visually.
def plot_samples_grid(images, labels, save_path, n=20):
    """Save a quick visual sanity-check grid for training images and labels."""
    n = min(n, len(images))
    cols = 10
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 1.2, rows * 1.4))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        # If labels are unknown (-1), print '?'.
        if int(labels[i]) >= 0:
            title = str(int(labels[i]))
        else:
            title = "?"
        plt.title(title, fontsize=8)
        plt.axis("off")
    plt.suptitle("ReducedMNIST Training Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Show what DCT coefficients and low-frequency reconstruction look like.
def plot_dct_example(image, save_path):
    """Visualize DCT transform intuition: original, DCT map, and low-frequency reconstruction."""
    img_f = image.astype(np.float32) / 255.0
    dct_map = cv2.dct(img_f)

    # Keep top-left 15x15 coefficients => 225 dims
    dct_15 = np.zeros_like(dct_map)
    dct_15[:15, :15] = dct_map[:15, :15]
    recon = cv2.idct(dct_15)

    plt.figure(figsize=(11, 3.6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_f, cmap="gray")
    plt.title("Original 28x28")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.abs(dct_map) + 1e-4), cmap="magma")
    plt.title("log|DCT| (2D)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(recon, 0.0, 1.0), cmap="gray")
    plt.title("Reconstruction from 15x15 DCT")
    plt.axis("off")

    plt.suptitle("DCT Feature Intuition (225 dims)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Plot the PCA variance curve and mark the 95% threshold.
def plot_pca_curve(pca_model, save_path):
    """Plot cumulative explained variance and mark first component count reaching 95%."""
    cum = np.cumsum(pca_model.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)

    plt.figure(figsize=(7, 4.4))
    plt.plot(np.arange(1, len(cum) + 1), cum, linewidth=2)
    plt.axhline(0.95, linestyle="--")
    plt.axvline(k95, linestyle="--")
    plt.scatter([k95], [cum[k95 - 1]], s=40)
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Ratio")
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Visualize the HOG descriptor for one image.
def plot_hog_example(image, save_path):
    """Visualize HOG representation for one sample image."""
    _, hog_img = hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )

    plt.figure(figsize=(8, 3.2))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original 28x28")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_img, cmap="gray")
    plt.title("HOG Visualization")
    plt.axis("off")

    plt.suptitle("HOG Feature Intuition", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==============================
# FEATURE EXTRACTION
# ==============================

# Convert images into 225-dimensional DCT feature vectors.
def dct_features_225(images):
    """Feature A: keep top-left 15x15 2D-DCT coefficients (225 dimensions)."""
    feats = []
    for img in images:
        # Low frequencies (top-left) usually carry most shape information.
        dct_map = cv2.dct(img.astype(np.float32) / 255.0)
        feats.append(dct_map[:15, :15].reshape(-1))
    return np.array(feats, dtype=np.float32)


# Reduce raw pixel vectors with PCA while keeping at least 95% variance.
def pca_features_95(x_train_flat, x_test_flat):
    """Feature B: PCA with enough components to preserve >=95% variance."""
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    z_train = pca.fit_transform(x_train_flat)
    z_test = pca.transform(x_test_flat)
    return z_train.astype(np.float32), z_test.astype(np.float32), pca


# Compute HOG descriptors for each image.
def hog_features(images):
    """Feature C: HOG descriptor capturing local edge orientations."""
    feats = []
    for img in images:
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


# ==============================
# CLASSIFIERS
# ==============================

# Store all centroids from the per-class K-means model.
@dataclass
class KMeansPerClassModel:
    # centroids: stacked centroids for all classes
    # centroid_labels: digit label for each centroid row
    centroids: np.ndarray
    centroid_labels: np.ndarray


# Fit one K-means model per digit class and combine their centroids.
def fit_kmeans_per_class(x_train, y_train, clusters_per_class):
    """Train one K-means model per class, then concatenate all centroids."""
    centroids = []
    centroid_labels = []

    for digit in range(10):
        # Extract all training vectors for this class.
        class_x = x_train[y_train == digit]
        if len(class_x) == 0:
            continue

        # Guard against asking for more clusters than samples.
        k = min(clusters_per_class, len(class_x))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(class_x)

        centroids.append(kmeans.cluster_centers_)
        centroid_labels.extend([digit] * k)

    centroids = np.vstack(centroids).astype(np.float32)
    centroid_labels = np.array(centroid_labels, dtype=np.int32)

    return KMeansPerClassModel(centroids=centroids, centroid_labels=centroid_labels)


# Predict a label by finding the nearest centroid across all classes.
def predict_kmeans_per_class(model, x_test):
    # Squared Euclidean distance to every centroid.
    dists = np.sum((x_test[:, None, :] - model.centroids[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)
    return model.centroid_labels[nearest]


# Train an SVM classifier with the requested kernel.
def fit_svm(x_train, y_train, kernel):
    """Train multiclass SVM (One-vs-One internally in scikit-learn)."""
    if kernel == "linear":
        svm = SVC(kernel="linear", decision_function_shape="ovo")
    elif kernel == "rbf":
        # C=10 gives a slightly stronger margin penalty for this dataset.
        svm = SVC(kernel="rbf", gamma="scale", C=10.0, decision_function_shape="ovo")
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    svm.fit(x_train, y_train)
    return svm


# Save a confusion matrix figure for a set of predictions.
def save_confusion_matrix(y_true, y_pred, title, save_path):
    """Save confusion matrix image for report discussion and error analysis."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# Benchmark every feature set against every classifier configuration.
def evaluate_part2_classifiers(feature_sets, y_train, y_test, output_dir):
    """Run full benchmark matrix across all features and classifier settings."""
    results = []

    best_kmeans = {"acc": -1.0}
    best_svm = {"acc": -1.0}

    for feature_name, x_train, x_test in feature_sets:
        # Benchmark this feature representation against all required classifiers.
        # K-means per class with K in {1,4,16,32}
        for kpc in [1, 4, 16, 32]:
            t0 = time.perf_counter()
            model = fit_kmeans_per_class(x_train, y_train, clusters_per_class=kpc)
            y_pred = predict_kmeans_per_class(model, x_test)
            elapsed = time.perf_counter() - t0
            acc = accuracy_score(y_test, y_pred)

            row = {
                "feature": feature_name,
                "classifier": "kmeans_per_class",
                "spec": f"clusters_per_class={kpc}",
                "accuracy": acc,
                "processing_time_sec": elapsed,
            }
            results.append(row)

            if acc > best_kmeans["acc"]:
                # Keep only the globally best K-means setup for confusion matrix export.
                best_kmeans = {
                    "acc": acc,
                    "feature": feature_name,
                    "spec": f"clusters_per_class={kpc}",
                    "y_pred": y_pred.copy(),
                }

        # SVM linear and SVM rbf
        for kernel in ["linear", "rbf"]:
            t0 = time.perf_counter()
            svm = fit_svm(x_train, y_train, kernel=kernel)
            y_pred = svm.predict(x_test)
            elapsed = time.perf_counter() - t0
            acc = accuracy_score(y_test, y_pred)

            if kernel == "linear":
                spec = "kernel=linear"
            else:
                spec = "kernel=rbf,gamma=scale,C=10.0"

            row = {
                "feature": feature_name,
                "classifier": "svm",
                "spec": spec,
                "accuracy": acc,
                "processing_time_sec": elapsed,
            }
            results.append(row)

            if acc > best_svm["acc"]:
                # Keep only the globally best SVM setup for confusion matrix export.
                best_svm = {
                    "acc": acc,
                    "feature": feature_name,
                    "spec": spec,
                    "y_pred": y_pred.copy(),
                }

    # Save row-wise benchmark table (one row per run configuration).
    benchmark_csv = os.path.join(output_dir, "part2_results_table.csv")
    with open(benchmark_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature", "classifier", "spec", "accuracy", "processing_time_sec"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Assignment asks confusion matrices only for the best result of each classifier family.
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    save_confusion_matrix(
        y_test,
        best_kmeans["y_pred"],
        title=f"Best KMeans | {best_kmeans['feature']} | {best_kmeans['spec']}",
        save_path=os.path.join(fig_dir, "best_kmeans_confusion_matrix.png"),
    )

    save_confusion_matrix(
        y_test,
        best_svm["y_pred"],
        title=f"Best SVM | {best_svm['feature']} | {best_svm['spec']}",
        save_path=os.path.join(fig_dir, "best_svm_confusion_matrix.png"),
    )

    return results, best_kmeans, best_svm


# Convert the long benchmark list into the wide comparison table format.
def build_comparative_matrix_table(results, output_dir):
    # Build assignment-style compact matrix:
    # one row per feature, columns for all classifier variants.
    table = {
        "DCT_225": {},
        "PCA_95": {},
        "HOG": {},
    }

    for row in results:
        feat = row["feature"]
        clf = row["classifier"]
        spec = row["spec"]
        acc = row["accuracy"]
        sec = row["processing_time_sec"]

        if clf == "kmeans_per_class":
            k = spec.split("=")[-1]
            key = f"KMeans_k{k}"
        elif clf == "svm" and "linear" in spec:
            key = "SVM_linear"
        else:
            key = "SVM_rbf"

        table[feat][f"{key}_accuracy"] = acc
        table[feat][f"{key}_time_sec"] = sec

    columns = [
        "feature",
        "KMeans_k1_accuracy", "KMeans_k1_time_sec",
        "KMeans_k4_accuracy", "KMeans_k4_time_sec",
        "KMeans_k16_accuracy", "KMeans_k16_time_sec",
        "KMeans_k32_accuracy", "KMeans_k32_time_sec",
        "SVM_linear_accuracy", "SVM_linear_time_sec",
        "SVM_rbf_accuracy", "SVM_rbf_time_sec",
    ]

    path = os.path.join(output_dir, "part2_comparative_matrix.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for feat in ["DCT_225", "PCA_95", "HOG"]:
            row = {"feature": feat}
            row.update(table[feat])
            writer.writerow(row)

    return path


# Write a short conclusions draft from the benchmark winners.
def write_conclusions(results, best_kmeans, best_svm, output_dir):
    # Auto-generated conclusion notes to speed up report writing.
    top = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    best_overall = top[0]

    lines = [
        "Part 2 Conclusions (Auto Draft)",
        "",
        f"Best overall setup: feature={best_overall['feature']}, classifier={best_overall['classifier']}, spec={best_overall['spec']}, accuracy={best_overall['accuracy']:.4f}.",
        f"Best K-means setup: feature={best_kmeans['feature']}, {best_kmeans['spec']}, accuracy={best_kmeans['acc']:.4f}.",
        f"Best SVM setup: feature={best_svm['feature']}, {best_svm['spec']}, accuracy={best_svm['acc']:.4f}.",
        "",
        "Notes:",
        "- DCT (225 dims), PCA (>=95% variance), and HOG were all benchmarked.",
        "- K-means was evaluated with 1, 4, 16, and 32 clusters per class.",
        "- SVM was evaluated with linear and RBF kernels.",
        "- Confusion matrices were generated only for the best K-means and best SVM setups as required.",
    ]

    path = os.path.join(output_dir, "part2_conclusions.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


# Map feature names to their train/test arrays for later lookup.
def _feature_map(feature_sets):
    # Convenience map: feature name -> (x_train_feature, x_test_feature)
    return {name: (x_train, x_test) for name, x_train, x_test in feature_sets}


# Show example training images nearest to the best K-means centroids.
def save_kmeans_cluster_representatives(best_kmeans, feature_sets, x_train_images, y_train, output_dir):
    """Visualize nearest real image to each centroid for the best K-means configuration."""
    fmap = _feature_map(feature_sets)
    feat_name = best_kmeans["feature"]
    x_train_feat = fmap[feat_name][0]

    kpc = int(best_kmeans["spec"].split("=")[-1])
    model = fit_kmeans_per_class(x_train_feat, y_train, clusters_per_class=kpc)

    # Choose the nearest training image to each centroid as a visual representative.
    representatives = []
    for c_idx, centroid in enumerate(model.centroids):
        # Pick nearest training sample in feature space to represent this centroid visually.
        d = np.sum((x_train_feat - centroid[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d))
        representatives.append((c_idx, idx, int(model.centroid_labels[c_idx])))

    max_show = min(30, len(representatives))
    reps = representatives[:max_show]
    cols = 10
    rows = int(np.ceil(max_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= max_show:
            ax.axis("off")
            continue

        cluster_id, img_idx, digit = reps[i]
        ax.imshow(x_train_images[img_idx], cmap="gray")
        ax.set_title(f"c{cluster_id}|d{digit}", fontsize=7)
        ax.axis("off")

    fig.suptitle(
        f"Best KMeans Representatives ({feat_name}, clusters/class={kpc})",
        fontsize=12,
    )
    fig.tight_layout()
    save_path = os.path.join(output_dir, "figures", "best_kmeans_cluster_representatives.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


# Save support-vector diagnostics and support-vector image galleries.
def save_svm_support_vector_visuals(best_svm, feature_sets, x_train_images, y_train, output_dir):
    """Export support-vector diagnostics for the best SVM setup."""
    fmap = _feature_map(feature_sets)
    feat_name = best_svm["feature"]
    x_train_feat = fmap[feat_name][0]

    kernel = "linear" if "linear" in best_svm["spec"] else "rbf"
    svm = fit_svm(x_train_feat, y_train, kernel=kernel)

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Support vectors per class bar chart.
    # Higher bars usually mean that class is harder to separate.
    cls = svm.classes_.astype(int)
    counts = svm.n_support_

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cls, counts)
    ax.set_title(f"Support Vectors per Class (Best SVM: {feat_name}, {kernel})")
    ax.set_xlabel("Digit class")
    ax.set_ylabel("# Support vectors")
    ax.grid(axis="y", alpha=0.3)
    bar_path = os.path.join(fig_dir, "best_svm_support_vectors_per_class.png")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=300)
    plt.close(fig)

    # 2) Small gallery of support-vector images.
    # These are the boundary-critical examples learned by the SVM.
    sv_idx = svm.support_
    max_show = min(30, len(sv_idx))
    cols = 10
    rows = int(np.ceil(max_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= max_show:
            ax.axis("off")
            continue
        idx = int(sv_idx[i])
        ax.imshow(x_train_images[idx], cmap="gray")
        ax.set_title(str(int(y_train[idx])), fontsize=7)
        ax.axis("off")

    fig.suptitle(f"Best SVM Support Vector Gallery ({feat_name}, {kernel})", fontsize=12)
    fig.tight_layout()
    gallery_path = os.path.join(fig_dir, "best_svm_support_vector_gallery.png")
    fig.savefig(gallery_path, dpi=300)
    plt.close(fig)

    return bar_path, gallery_path


# ==============================
# MAIN
# ==============================

# Run the full Part 2 pipeline and write all outputs to disk.
def main():
    """Entry point: run full Part 2 pipeline and save all report artifacts."""
    parser = argparse.ArgumentParser(description="Part 2 - Step 1: ReducedMNIST feature extraction")
    parser.add_argument("--dataset-dir", type=str, default="", help="Dataset directory with train/test class folders")
    parser.add_argument("--dataset-npz", type=str, default="", help="NPZ path with x_train,y_train,x_test,y_test")
    parser.add_argument("--labels-csv", type=str, default=DEFAULT_LABELS_CSV, help="Recovered labels CSV for Indian_Digits_Train")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio for Indian_Digits_Train fallback")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--skip-classifiers", action="store_true", help="Only run Step 1 feature extraction")

    if len(sys.argv) == 1:
        # No CLI args => use project-local defaults for a reproducible run.
        args = parser.parse_args([])
        args.dataset_dir = DEFAULT_AUTO_CONFIG["dataset_dir"]
        args.dataset_npz = DEFAULT_AUTO_CONFIG["dataset_npz"]
        args.labels_csv = DEFAULT_AUTO_CONFIG["labels_csv"]
        args.train_ratio = DEFAULT_AUTO_CONFIG["train_ratio"]
        args.output_dir = DEFAULT_AUTO_CONFIG["output_dir"]
        args.skip_classifiers = DEFAULT_AUTO_CONFIG["skip_classifiers"]
        print("[INFO] Running with internal default config (no CLI arguments).")
    else:
        args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = load_reduced_mnist(
        dataset_dir=args.dataset_dir if args.dataset_dir else None,
        dataset_npz=args.dataset_npz if args.dataset_npz else None,
        labels_csv=args.labels_csv if args.labels_csv else None,
        train_ratio=args.train_ratio,
    )

    print(f"[INFO] Train images: {x_train.shape}, Train labels: {y_train.shape}")
    print(f"[INFO] Test images : {x_test.shape}, Test labels : {y_test.shape}")

    # ------------------------------
    # Visualizations (for report narrative)
    # ------------------------------
    plot_samples_grid(
        x_train,
        y_train,
        save_path=os.path.join(fig_dir, "reducedmnist_samples_grid.png"),
        n=20,
    )
    plot_dct_example(
        x_train[0],
        save_path=os.path.join(fig_dir, "dct_feature_example.png"),
    )
    plot_hog_example(
        x_train[0],
        save_path=os.path.join(fig_dir, "hog_feature_example.png"),
    )

    # ------------------------------
    # Feature extraction timing
    # ------------------------------
    # Each block records extraction time to satisfy the assignment comparison table.
    summary_rows = []

    t0 = time.perf_counter()
    x_train_dct = dct_features_225(x_train)
    x_test_dct = dct_features_225(x_test)
    dt = time.perf_counter() - t0
    summary_rows.append(["DCT_225", x_train_dct.shape[1], dt])

    x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test_flat = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0
    # PCA expects 2D samples (N x 784), values scaled to [0,1].

    t0 = time.perf_counter()
    x_train_pca, x_test_pca, pca_model = pca_features_95(x_train_flat, x_test_flat)
    dt = time.perf_counter() - t0
    summary_rows.append(["PCA_95", x_train_pca.shape[1], dt])

    plot_pca_curve(
        pca_model,
        save_path=os.path.join(fig_dir, "pca_cumulative_variance.png"),
    )

    t0 = time.perf_counter()
    x_train_hog = hog_features(x_train)
    x_test_hog = hog_features(x_test)
    dt = time.perf_counter() - t0
    summary_rows.append(["HOG", x_train_hog.shape[1], dt])

    # ------------------------------
    # Save extracted feature arrays
    # ------------------------------
    # Keeping .npy files makes reruns fast (skip repeated feature extraction if needed).
    np.save(os.path.join(args.output_dir, "x_train_dct225.npy"), x_train_dct)
    np.save(os.path.join(args.output_dir, "x_test_dct225.npy"), x_test_dct)

    np.save(os.path.join(args.output_dir, "x_train_pca95.npy"), x_train_pca)
    np.save(os.path.join(args.output_dir, "x_test_pca95.npy"), x_test_pca)

    np.save(os.path.join(args.output_dir, "x_train_hog.npy"), x_train_hog)
    np.save(os.path.join(args.output_dir, "x_test_hog.npy"), x_test_hog)

    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

    # ------------------------------
    # Save metadata + timing summary
    # ------------------------------
    metadata = {
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "has_ground_truth_labels": bool(np.all(y_train >= 0) and np.all(y_test >= 0)),
        "dct_dims": int(x_train_dct.shape[1]),
        "pca_dims_for_95": int(x_train_pca.shape[1]),
        "hog_dims": int(x_train_hog.shape[1]),
        "pca_total_variance": float(np.sum(pca_model.explained_variance_ratio_)),
    }

    with open(os.path.join(args.output_dir, "step1_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(args.output_dir, "step1_feature_timing.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "dims", "extract_time_sec"])
        writer.writerows(summary_rows)

    # ------------------------------
    # Classifier benchmarking (requires labels)
    # ------------------------------
    labels_available = bool(np.all(y_train >= 0) and np.all(y_test >= 0))
    if args.skip_classifiers:
        print("[INFO] --skip-classifiers is set. Stopping after Step 1.")
    elif not labels_available:
        print("[WARNING] Labels are not available.")
        print("[WARNING] Classifier accuracy/comparison is skipped until the recovered label vector is provided.")
    else:
        # Benchmark matrix required by the assignment.
        feature_sets = [
            ("DCT_225", x_train_dct, x_test_dct),
            ("PCA_95", x_train_pca, x_test_pca),
            ("HOG", x_train_hog, x_test_hog),
        ]
        results, best_kmeans, best_svm = evaluate_part2_classifiers(
            feature_sets,
            y_train,
            y_test,
            output_dir=args.output_dir,
        )

        matrix_table_path = build_comparative_matrix_table(results, args.output_dir)
        conclusions_path = write_conclusions(results, best_kmeans, best_svm, args.output_dir)
        # Extra visuals that help explain model behavior in discussion.
        kmeans_vis_path = save_kmeans_cluster_representatives(
            best_kmeans,
            feature_sets,
            x_train,
            y_train,
            args.output_dir,
        )
        svm_bar_path, svm_gallery_path = save_svm_support_vector_visuals(
            best_svm,
            feature_sets,
            x_train,
            y_train,
            args.output_dir,
        )

        print("\n===== PART 2 SUMMARY =====")
        print(f"Total benchmark rows: {len(results)}")
        print(
            f"Best KMeans: feature={best_kmeans['feature']} | "
            f"{best_kmeans['spec']} | acc={best_kmeans['acc']:.4f}"
        )
        print(
            f"Best SVM   : feature={best_svm['feature']} | "
            f"{best_svm['spec']} | acc={best_svm['acc']:.4f}"
        )
        print("Results table: part2_results_table.csv")
        print(f"Comparative matrix table: {os.path.basename(matrix_table_path)}")
        print("Confusion matrices: figures/best_kmeans_confusion_matrix.png, figures/best_svm_confusion_matrix.png")
        print(f"Conclusions draft: {os.path.basename(conclusions_path)}")
        print(f"KMeans image visual: {os.path.basename(kmeans_vis_path)}")
        print(f"SVM visuals: {os.path.basename(svm_bar_path)}, {os.path.basename(svm_gallery_path)}")

    print("\n[INFO] Step 1 complete.")
    print(f"[INFO] Features/labels saved in: {args.output_dir}")
    print(f"[INFO] Figures saved in: {fig_dir}")
    print(f"[INFO] PCA dims (>=95% variance): {x_train_pca.shape[1]}")


if __name__ == "__main__":
    main()
