from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import time

import kagglehub
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC


def get_dataset_paths() -> tuple[Path, Path]:
    dataset_root = Path(kagglehub.dataset_download("mohamedgamal07/reduced-mnist"))
    base = dataset_root / "Reduced MNIST Data"
    return base / "Reduced Trainging data", base / "Reduced Testing data"


def collect_samples(split_dir: Path) -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    for digit in range(10):
        class_dir = split_dir / str(digit)
        for p in sorted(class_dir.glob("*.jpg")):
            paths.append(p)
            labels.append(digit)
    return paths, np.asarray(labels, dtype=np.int32)


def load_images(paths: list[Path]) -> np.ndarray:
    images = []
    for p in paths:
        with Image.open(p) as img:
            images.append(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)
    return np.stack(images, axis=0)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _encode_layerwise(
    x: np.ndarray, coefs: list[np.ndarray], intercepts: list[np.ndarray], stop_layer: int
) -> np.ndarray:
    h = x
    for i in range(stop_layer + 1):
        h = _relu(h @ coefs[i] + intercepts[i])
    return h


def extract_autoencoder_features(
    train_images: np.ndarray,
    test_images: np.ndarray,
    hidden: tuple[int, ...] = (128, 64, 32, 64, 128),
    max_iter: int = 120,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    x_train = train_images.reshape(len(train_images), -1)
    x_test = test_images.reshape(len(test_images), -1)

    t0 = time.perf_counter()
    ae = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        batch_size=256,
        random_state=42,
        early_stopping=True,
    )
    ae.fit(x_train, x_train)
    extract_time = time.perf_counter() - t0

    # For hidden=(128,64,32,64,128), bottleneck layer index is 2.
    bottleneck_idx = len(hidden) // 2
    z_train = _encode_layerwise(x_train, ae.coefs_, ae.intercepts_, bottleneck_idx).astype(np.float32)
    z_test = _encode_layerwise(x_test, ae.coefs_, ae.intercepts_, bottleneck_idx).astype(np.float32)

    return z_train, z_test, z_train.shape[1], extract_time


def fit_kmeans_per_class(x_train: np.ndarray, y_train: np.ndarray, clusters_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    centroid_labels = []

    for digit in range(10):
        class_x = x_train[y_train == digit]
        k = min(clusters_per_class, len(class_x))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(class_x)
        centroids.append(km.cluster_centers_)
        centroid_labels.extend([digit] * k)

    return np.vstack(centroids).astype(np.float32), np.asarray(centroid_labels, dtype=np.int32)


def predict_kmeans_per_class(centroids: np.ndarray, centroid_labels: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    dists = np.sum((x_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return centroid_labels[np.argmin(dists, axis=1)]


def benchmark_autoencoder(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for k in [1, 4, 16, 32]:
        t0 = time.perf_counter()
        centroids, centroid_labels = fit_kmeans_per_class(z_train, y_train, k)
        y_pred = predict_kmeans_per_class(centroids, centroid_labels, z_test)
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "feature": "AutoEncoder",
                "classifier": "kmeans_per_class",
                "spec": f"clusters_per_class={k}",
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "processing_time_sec": float(elapsed),
            }
        )

    for kernel, spec in [
        ("linear", "kernel=linear"),
        ("rbf", "kernel=rbf,gamma=scale,C=10.0"),
    ]:
        t0 = time.perf_counter()
        svm = SVC(kernel=kernel, gamma="scale", C=10.0 if kernel == "rbf" else 1.0, decision_function_shape="ovo")
        svm.fit(z_train, y_train)
        y_pred = svm.predict(z_test)
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "feature": "AutoEncoder",
                "classifier": "svm",
                "spec": spec,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "processing_time_sec": float(elapsed),
            }
        )

    return rows


def build_matrix_row(rows: list[dict[str, object]]) -> dict[str, object]:
    row = {
        "feature": "AutoEncoder",
        "KMeans_k1_accuracy": "",
        "KMeans_k1_time_sec": "",
        "KMeans_k4_accuracy": "",
        "KMeans_k4_time_sec": "",
        "KMeans_k16_accuracy": "",
        "KMeans_k16_time_sec": "",
        "KMeans_k32_accuracy": "",
        "KMeans_k32_time_sec": "",
        "SVM_linear_accuracy": "",
        "SVM_linear_time_sec": "",
        "SVM_rbf_accuracy": "",
        "SVM_rbf_time_sec": "",
    }

    for r in rows:
        clf = str(r["classifier"])
        spec = str(r["spec"])
        acc = float(r["accuracy"])
        sec = float(r["processing_time_sec"])

        if clf == "kmeans_per_class":
            k = spec.split("=")[-1]
            row[f"KMeans_k{k}_accuracy"] = acc
            row[f"KMeans_k{k}_time_sec"] = sec
        elif clf == "svm" and "linear" in spec:
            row["SVM_linear_accuracy"] = acc
            row["SVM_linear_time_sec"] = sec
        else:
            row["SVM_rbf_accuracy"] = acc
            row["SVM_rbf_time_sec"] = sec

    return row


def save_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["feature", "classifier", "spec", "accuracy", "processing_time_sec"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_matrix_row_csv(path: Path, row: dict[str, object]) -> None:
    fieldnames = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)


def save_feature_timing_csv(path: Path, dims: int, extract_time_sec: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "dims", "extract_time_sec"])
        w.writerow(["AutoEncoder", dims, extract_time_sec])


def main() -> None:
    parser = ArgumentParser(description="Run Assignment 1-style benchmark using AutoEncoder features on new data")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs_autoencoder"),
        help="Directory to save CSV outputs",
    )
    parser.add_argument("--max-iter-ae", type=int, default=120, help="Autoencoder training max_iter")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = get_dataset_paths()
    train_paths, y_train = collect_samples(train_dir)
    test_paths, y_test = collect_samples(test_dir)

    x_train = load_images(train_paths)
    x_test = load_images(test_paths)

    z_train, z_test, dims, feat_time = extract_autoencoder_features(
        x_train, x_test, max_iter=args.max_iter_ae
    )

    rows = benchmark_autoencoder(z_train, y_train, z_test, y_test)
    matrix_row = build_matrix_row(rows)

    save_rows_csv(out / "autoencoder_assignment1_results.csv", rows)
    save_matrix_row_csv(out / "autoencoder_assignment1_matrix_row.csv", matrix_row)
    save_feature_timing_csv(out / "autoencoder_feature_timing.csv", dims, feat_time)

    print("Saved:")
    print(out / "autoencoder_assignment1_results.csv")
    print(out / "autoencoder_assignment1_matrix_row.csv")
    print(out / "autoencoder_feature_timing.csv")


if __name__ == "__main__":
    main()
