import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "Indian_Digits_Train")
IMAGES_CACHE = "images.npy"
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "pipeline3")
DEFAULT_FULL_LABELS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "full_labels_vector.csv")


# ==============================
# STEP 1 — Load Images
# ==============================
def load_images(path):
    files = [f for f in os.listdir(path) if f.endswith(".bmp")]

    # Sort numerically: 1.bmp -> 10000.bmp
    files = sorted(files, key=lambda x: int(x.split('.')[0]))

    images = []
    for f in files:
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Error loading image: {img_path}")

        images.append(img)

    images = np.array(images)
    print(f"[INFO] Loaded {len(images)} images")
    return images


# ==============================
# SHOW 10 IMAGES PER CLASS
# ==============================
def show_class_samples(images, matrix, num_per_class=10, save_path=None):
    plt.figure(figsize=(num_per_class, 10))

    for digit in range(matrix.shape[0]):
        indices = matrix[digit][:num_per_class]

        for j, idx in enumerate(indices):
            plt.subplot(matrix.shape[0], num_per_class, digit * num_per_class + j + 1)
            plt.imshow(images[idx], cmap='gray')
            plt.axis('off')

            if j == 0:
                plt.ylabel(f"{digit}", rotation=0, labelpad=20, fontsize=12)

    plt.suptitle(f"Samples per Class (up to {num_per_class})", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Seed preview saved to: {save_path}")

    plt.close()


# ==============================
# STEP 2 — Random Seed Generation from Full Labels
# ==============================
def generate_seed_matrix_from_full_labels(
    full_labels_path,
    target_per_class=50,
    num_classes=10,
    random_seed=42,
):
    labels = np.loadtxt(full_labels_path, delimiter=",", dtype=int)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    rng = np.random.default_rng(seed=random_seed)
    matrix = []

    for digit in range(num_classes):
        class_indices = np.where(labels == digit)[0]
        if len(class_indices) < target_per_class:
            raise ValueError(
                f"Not enough samples for digit {digit}: {len(class_indices)} < {target_per_class}"
            )

        picked = rng.choice(class_indices, size=target_per_class, replace=False)
        matrix.append(picked)

    matrix = np.array(matrix, dtype=int)
    print(f"[INFO] Loaded full labels from: {full_labels_path}")
    print("[INFO] Final matrix shape:", matrix.shape)
    return matrix


# ==============================
# STEP 3 — Save Outputs
# ==============================
def save_matrix_outputs(matrix, output_dir, csv_name="label_matrix_500.csv"):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, csv_name)
    npy_path = os.path.join(output_dir, csv_name.replace(".csv", ".npy"))

    np.savetxt(csv_path, matrix, fmt='%d', delimiter=',')
    np.save(npy_path, matrix)

    print(f"[INFO] Matrix saved to CSV: {csv_path}")
    print(f"[INFO] Matrix saved to NPY: {npy_path}")

    return csv_path, npy_path


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Pipeline 3 random seed generation from full label matrix")
    parser.add_argument("--target-per-class", type=int, default=50, help="Number of samples to collect per class")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--full-labels-path",
        type=str,
        default=DEFAULT_FULL_LABELS_PATH,
        help="Path to full labels vector CSV (10000 labels in file order)",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--csv-name", type=str, default="label_matrix_500.csv", help="CSV output file name")
    parser.add_argument("--no-preview", action="store_true", help="Disable saving preview image grid")

    args = parser.parse_args()

    matrix = generate_seed_matrix_from_full_labels(
        full_labels_path=args.full_labels_path,
        target_per_class=args.target_per_class,
        num_classes=args.num_classes,
        random_seed=args.seed,
    )

    save_matrix_outputs(matrix, args.output_dir, csv_name=args.csv_name)

    if not args.no_preview:
        if os.path.exists(IMAGES_CACHE):
            print("[INFO] Loading cached images...")
            images = np.load(IMAGES_CACHE)
        else:
            print("[INFO] Loading images from disk...")
            images = load_images(DATA_PATH)
            np.save(IMAGES_CACHE, images)

        preview_path = os.path.join(args.output_dir, "Human_seed_pipeline3.png")
        show_class_samples(images, matrix, num_per_class=min(10, args.target_per_class), save_path=preview_path)


if __name__ == "__main__":
    main()
