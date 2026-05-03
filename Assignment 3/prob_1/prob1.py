from pathlib import Path
from argparse import ArgumentParser
import time

import kagglehub
import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor


def get_dataset_paths() -> tuple[Path, Path]:
    """Download the dataset once and return train/test folders."""
    dataset_root = Path(kagglehub.dataset_download("mohamedgamal07/reduced-mnist"))
    base = dataset_root / "Reduced MNIST Data"

    # Keep folder names exactly as they are in the dataset (including typo).
    train_dir = base / "Reduced Trainging data"
    test_dir = base / "Reduced Testing data"
    return train_dir, test_dir


def collect_samples(split_dir: Path) -> tuple[list[Path], list[int]]:
    """Read image paths and digit labels from the class folders."""
    paths: list[Path] = []
    labels: list[int] = []

    for digit in range(10):
        class_dir = split_dir / str(digit)
        # Dataset images are .jpg files in this Kaggle version.
        for p in sorted(class_dir.glob("*.jpg")):
            paths.append(p)
            labels.append(digit)
    return paths, labels


def load_images(paths: list[Path]) -> np.ndarray:
    """Load images as float32 in [0, 1], shape: (N, 28, 28)."""
    images = []
    for p in paths:
        with Image.open(p) as img:
            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            images.append(arr)
    return np.stack(images, axis=0)


def extract_dct_features(images: np.ndarray, keep: int = 15) -> np.ndarray:
    """Keep the top-left 15x15 DCT block and flatten it to 225 features."""
    feats = []
    for img in images:
        coeff = dctn(img, type=2, norm="ortho")
        feats.append(coeff[:keep, :keep].reshape(-1))
    return np.asarray(feats, dtype=np.float32)


def extract_pca_features(train_images: np.ndarray, test_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten images, then reduce them with PCA to 95% variance."""
    x_train = train_images.reshape(len(train_images), -1)
    x_test = test_images.reshape(len(test_images), -1)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    return pca.fit_transform(x_train).astype(np.float32), pca.transform(x_test).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _encode_layerwise(x: np.ndarray, coefs: list[np.ndarray], intercepts: list[np.ndarray], stop_layer: int) -> np.ndarray:
    h = x
    for i in range(stop_layer + 1):
        h = h @ coefs[i] + intercepts[i]
        h = _relu(h)
    return h


def extract_autoencoder_features(train_images: np.ndarray, test_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train a small dense autoencoder and return the bottleneck activations."""
    x_train = train_images.reshape(len(train_images), -1)
    x_test = test_images.reshape(len(test_images), -1)

    ae = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 64, 128),
        activation="relu",
        solver="adam",
        max_iter=120,
        batch_size=256,
        random_state=42,
        early_stopping=True,
    )
    ae.fit(x_train, x_train)

    bottleneck = 2
    return (
        _encode_layerwise(x_train, ae.coefs_, ae.intercepts_, bottleneck).astype(np.float32),
        _encode_layerwise(x_test, ae.coefs_, ae.intercepts_, bottleneck).astype(np.float32),
    )


def build_features(feature_name: str, x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the selected feature set and return its extraction time in seconds."""
    t0 = time.perf_counter()
    if feature_name == "dct":
        x_train_f = extract_dct_features(x_train, keep=15)
        x_test_f = extract_dct_features(x_test, keep=15)
    elif feature_name == "pca":
        x_train_f, x_test_f = extract_pca_features(x_train, x_test)
    elif feature_name == "autoencoder":
        x_train_f, x_test_f = extract_autoencoder_features(x_train, x_test)
    else:
        raise ValueError('feature must be one of: dct, pca, autoencoder')
    return x_train_f, x_test_f, time.perf_counter() - t0


def run_mlp(x_train: np.ndarray, y_train: list[int], x_test: np.ndarray, y_test: list[int], hidden_layers: tuple[int, ...]) -> tuple[float, float]:
    """Train one MLP configuration and return accuracy plus total classifier time."""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=120,
        random_state=42,
        early_stopping=True,
    )

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    elapsed = time.perf_counter() - t0
    return accuracy_score(y_test, y_pred) * 100.0, elapsed


def format_table(rows: list[tuple[str, float, float]]) -> str:
    header = f"{'model':<10} {'accuracy':>10} {'time(ms)':>12}"
    lines = [header]
    for name, acc, ms in rows:
        lines.append(f"{name:<10} {acc:>9.1f}% {ms:>11.1f}")
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser(description="Problem 1 MLP benchmark")
    parser.add_argument("feature", nargs="?", default="dct", choices=["dct", "pca", "autoencoder"], help="Feature type to use")
    args = parser.parse_args()

    train_dir, test_dir = get_dataset_paths()
    train_paths, y_train = collect_samples(train_dir)
    test_paths, y_test = collect_samples(test_dir)

    x_train = load_images(train_paths)
    x_test = load_images(test_paths)

    x_train_f, x_test_f, feature_time = build_features(args.feature, x_train, x_test)

    configs: list[tuple[str, tuple[int, ...]]] = [
        ("1-hidden", (512,)),
        ("3-hidden", (512, 256, 128)),
        ("4-hidden", (512, 256, 128, 64)),
    ]

    rows = []
    for name, layers in configs:
        acc, clf_time = run_mlp(x_train_f, y_train, x_test_f, y_test, layers)
        rows.append((name, acc, (feature_time + clf_time) * 1000.0))

    print(format_table(rows))


if __name__ == "__main__":
    main()
