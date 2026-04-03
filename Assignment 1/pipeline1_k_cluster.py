import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from check_accuracy import check_accuracy


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "Indian_Digits_Train")
IMG_SIZE = (32, 32)

IMAGES_CACHE = "images.npy"
FEATURES_CACHE = "features.npy"

# output directory for results 
OUTPUT_DIR = os.path.join(BASE_DIR, "pipeline1")

# ==============================
# Show an image (for testing)
# ==============================
def show_image(img, title="Image"):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# ==============================
# Show images (for testing)
# ==============================   
def show_samples(images, n=10):
    plt.figure(figsize=(12, 3))
    
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"{i+1}.bmp")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==============================
# Show HOG features (for testing)
# ==============================
def show_hog(img):
    feat, hog_img = hog(
        img,
        pixels_per_cell=(4,4),
        cells_per_block=(2,2),
        visualize=True
    )
    
    plt.figure(figsize=(6,3))
    
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(hog_img, cmap='gray')
    plt.title("HOG Features")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# ==============================
# STEP 1 — Load Images
# ==============================
def load_images(path):
    files = [f for f in os.listdir(path) if f.endswith(".bmp")]
    
    # Sort numerically: 1.bmp → 10000.bmp
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
# STEP 2 — Extract HOG Features
# ==============================
def extract_hog_features(images):
    features = []
    
    for img in images:
        # Resize
        img_resized = cv2.resize(img, IMG_SIZE)
        
        # Extract HOG
        hog_feat = hog(
            img_resized,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        
        features.append(hog_feat)
    
    features = np.array(features)
    
    print(f"[INFO] HOG feature shape: {features.shape}")
    
    return features

# ==============================
# STEP 3 — Run K-means Clustering
# ==============================
def run_kmeans(features, K):
    print(f"\n[INFO] Running K-means with K = {K}...")
    
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(features)
    
    print("[INFO] K-means done.")
    
    return cluster_ids

# K debugging
def analyze_clusters(cluster_ids, K):
    print("\n[INFO] Cluster distribution:")
    
    for k in range(K):
        count = np.sum(cluster_ids == k)
        print(f"Cluster {k}: {count} samples")


# ==============================
# STEP 4 — Label Clusters
# ============================== 
def label_clusters(cluster_ids, K):
    print("\n[INFO] Labeling clusters using oracle...")

    N = len(cluster_ids)
    labels = np.zeros(N, dtype=int)

    for k in range(K):
        idx = np.where(cluster_ids == k)[0]

        best_label = 0
        best_acc = 0

        # Try all digits (0–9)
        for digit in range(10):
            temp_labels = labels.copy()
            temp_labels[idx] = digit

            acc, _, _ = check_accuracy(temp_labels)

            if acc > best_acc:
                best_acc = acc
                best_label = digit

        # Assign best label to cluster
        labels[idx] = best_label

        print(f"Cluster {k} → Label {best_label} (acc={best_acc:.4f})")

    return labels

# Cache cluster labels
def get_cluster_path(K):
    import os
    return os.path.join(OUTPUT_DIR, f"clusters_K{K}.npy")

# ==============================
# Show K accuracy plot
# ============================== 
def plot_k_vs_accuracy(K_list, accuracies):
    import matplotlib.pyplot as plt
    import os

    # Use global path
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(K_list, accuracies, marker='o')

    # Add values on points
    for i in range(len(K_list)):
        plt.text(K_list[i], accuracies[i], f"{accuracies[i]:.2f}%", ha='center')

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Accuracy (%)")
    plt.title("K vs Accuracy (Pipeline 1)")
    plt.grid()

    # Save figure
    file_path = os.path.join(OUTPUT_DIR, "k_vs_accuracy.png")
    plt.savefig(file_path, dpi=300)

    print(f"[INFO] Plot saved at: {file_path}")

    plt.show()
    
# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    
    # ======================
    # LOAD / CACHE IMAGES
    # ======================
    if os.path.exists(IMAGES_CACHE):
        print("[INFO] Loading cached images...")
        images = np.load(IMAGES_CACHE)
    else:
        print("[INFO] Loading images from disk...")
        images = load_images(DATA_PATH)
        np.save(IMAGES_CACHE, images)

    # ======================
    # LOAD / CACHE FEATURES
    # ======================
    if os.path.exists(FEATURES_CACHE):
        print("[INFO] Loading cached features...")
        features = np.load(FEATURES_CACHE)
    else:
        print("[INFO] Extracting HOG features...")
        features = extract_hog_features(images)
        np.save(FEATURES_CACHE, features)

    print("[INFO] Ready for clustering.")  
    
    # DEBUG VISUALIZATION
    print("[INFO] Showing sample images...")
    show_samples(images, 10)
    
    print("[INFO] Showing one image...")
    show_image(images[0], "First Image")
    
    print("[INFO] Showing HOG visualization...")
    show_hog(images[0])
    
    # Extract features
    features = extract_hog_features(images)
    
    print("[INFO] Feature extraction complete.")
    
    # ==============================
    # K-means + Labeling + Evaluation
    # ==============================

    import os
    import numpy as np

    K_list = [40, 60, 80]
    accuracies = []
    clusters_dict = {}

    clusters_path = os.path.join(OUTPUT_DIR, "final_clusters.npy")

    # ==============================
    # CASE 1: Load cached clusters
    # ==============================
    if os.path.exists(clusters_path):
        print("[INFO] Loading saved clusters...")
        clusters_dict = np.load(clusters_path, allow_pickle=True).item()

        for K in K_list:
            print(f"\n[INFO] Using cached clusters for K={K}")
            cluster_ids = clusters_dict[K]

            analyze_clusters(cluster_ids, K)

            # Label clusters
            labels = label_clusters(cluster_ids, K)

            # Evaluate
            acc, correct, total = check_accuracy(labels)
            accuracies.append(acc * 100)

            print("\n===== Initial Cluster Labeling Result =====")
            print(f"K = {K}")
            print(f"Accuracy: {acc*100:.2f}%")
            print("==========================================")

    # ==============================
    # CASE 2: Run K-means and save
    # ==============================
    else:
        print("[INFO] No saved clusters. Running K-means...")

        for K in K_list:
            cluster_ids = run_kmeans(features, K)
            analyze_clusters(cluster_ids, K)

            clusters_dict[K] = cluster_ids  # store clusters

            # Label clusters
            labels = label_clusters(cluster_ids, K)

            # Evaluate
            acc, correct, total = check_accuracy(labels)
            accuracies.append(acc * 100)

            print("\n===== Initial Cluster Labeling Result =====")
            print(f"K = {K}")
            print(f"Accuracy: {acc*100:.2f}%")
            print("==========================================")

        # Save clusters AFTER loop
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(clusters_path, clusters_dict)

        print(f"[INFO] Final clusters saved at: {clusters_path}")

    # ==============================
    # Plot results
    # ==============================
    plot_k_vs_accuracy(K_list, accuracies)