import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from check_accuracy import check_accuracy
from sklearn.svm import SVC
import csv



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
# STEP 5 — Train SVM
# ==============================
def train_svm(features, labels, weights=None):
    print("[INFO] Training SVM...")

    svm = SVC(kernel='rbf', decision_function_shape='ovo')

    if weights is not None:
        svm.fit(features, labels, sample_weight=weights)
    else:
        svm.fit(features, labels)

    print("[INFO] SVM training done.")
    return svm

# ==============================
# STEP 6 — Get hard samples 
# ==============================
def get_hard_samples(svm, features, n_samples=30):
    scores = svm.decision_function(features)

    # sort scores
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    # smallest margin = hardest
    idx_sorted = np.argsort(margin)
    hard_samples = idx_sorted[:n_samples]

    return hard_samples, margin

# ==============================
# STEP 7 — Refine Labels
# ==============================
def refine_labels(labels, hard_samples):
    new_labels = labels.copy()

    for idx in hard_samples:
        best_label = labels[idx]
        best_acc = 0

        for digit in range(10):
            temp = new_labels.copy()
            temp[idx] = digit

            acc, _, _ = check_accuracy(temp)

            if acc > best_acc:
                best_acc = acc
                best_label = digit

        new_labels[idx] = best_label

    return new_labels

# create weights for hard samples
def create_weights(N, hard_samples):
    weights = np.ones(N)
    weights[hard_samples] = 100
    return weights

# SVM + Active Learning Loop
def active_learning_loop(features, labels, target_acc=0.99, threshold=0.001, max_iter=10):
    prev_acc = 0

    for i in range(max_iter):
        print(f"\n[INFO] Iteration {i+1}")

        # Train SVM
        svm = train_svm(features, labels)

        # Get hard samples
        hard_samples, margin = get_hard_samples(svm, features, 30)

        # Refine labels
        labels = refine_labels(labels, hard_samples)

        # Create weights
        weights = create_weights(len(labels), hard_samples)

        # Retrain
        svm = train_svm(features, labels, weights)

        # Evaluate
        acc, _, _ = check_accuracy(labels)
        print(f"[INFO] Accuracy: {acc*100:.2f}%")

        # ==============================
        # STOP CONDITIONS
        # ==============================

        # 1️⃣ Target accuracy
        if acc >= target_acc:
            print(f"[INFO] Stopping: reached target at iteration {i+1}")
            return labels, i+1

        # 2️⃣ Convergence
        improvement = acc - prev_acc
        if i > 0 and improvement < threshold:
            print(f"[INFO] Stopping: converged at iteration {i+1}")
            return labels, i+1

        prev_acc = acc

    print(f"[INFO] Stopping: max_iter reached ({max_iter})")
    return labels, max_iter

# Plot SVM bar chart
def plot_svm_bar_chart(K_list, before_acc, after_acc):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.arange(len(K_list))  # positions
    width = 0.35               # bar width

    plt.figure(figsize=(7,5))

    # Bars
    bars1 = plt.bar(x - width/2, before_acc, width, label='Before SVM')
    bars2 = plt.bar(x + width/2, after_acc, width, label='After SVM')

    # Labels
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Accuracy (%)")
    plt.title("K vs Accuracy (Before vs After SVM)")
    plt.xticks(x, K_list)
    plt.legend()
    plt.grid(axis='y')

    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.2f}%", ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.2f}%", ha='center', va='bottom')

    # Save
    file_path = os.path.join(OUTPUT_DIR, "svm_bar_chart.png")
    plt.savefig(file_path, dpi=300)

    print(f"[INFO] Bar chart saved at: {file_path}")

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
    
    # Extract features
    features = extract_hog_features(images)
    
    print("[INFO] Feature extraction complete.")
    
    # ==============================
    # K-means + Labeling + Evaluation
    # ==============================

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
    # MAIN LOOP (Clustering + SVM)
    # ==============================
    
    before_svm_acc = []
    after_svm_acc = []
    iterations_list = []
    
    for K in K_list:
        print(f"\n================ K = {K} ================")

        cluster_ids = clusters_dict[K]

        # Step 2 — Cluster labeling
        labels = label_clusters(cluster_ids, K)

        # Evaluate BEFORE SVM
        acc_before, _, _ = check_accuracy(labels)
        before_svm_acc.append(acc_before * 100)
        print(f"[INFO] Accuracy before SVM: {acc_before*100:.2f}%")

        # ==============================
        #  SVM + Active Learning
        # ==============================
        labels, num_iters = active_learning_loop(
            features, labels,
            target_acc=0.995,
            threshold=0.0009,
            max_iter=50
        )
        iterations_list.append(num_iters)

        # Evaluate AFTER SVM
        acc_after, _, _ = check_accuracy(labels)
        after_svm_acc.append(acc_after * 100)

        print("\n===== FINAL RESULT AFTER SVM =====")
        print(f"K = {K}")
        print(f"Accuracy: {acc_after*100:.2f}%")
        print("=================================")

    # ==============================
    # Plot results
    # ==============================
    print("\n========== K vs Accuracy Table ==========")
    print(f"{'K':<10}{'Before SVM (%)':<20}{'After SVM (%)':<20}")

    for i in range(len(K_list)):
        print(f"{K_list[i]:<10}{before_svm_acc[i]:<20.2f}{after_svm_acc[i]:<20.2f}")

    print("=========================================")
    
    print("\n====== Iterations Summary ======")
    print(f"{'K':<10}{'Iterations':<15}")

    for i in range(len(K_list)):
        print(f"{K_list[i]:<10}{iterations_list[i]:<15}")

    print("================================")
    
    file_path = os.path.join(OUTPUT_DIR, "iterations_summary.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "Iterations"])

        for i in range(len(K_list)):
            writer.writerow([K_list[i], iterations_list[i]])

    print(f"[INFO] Iteration data saved at: {file_path}")
    
    # Bar chart
    plot_svm_bar_chart(K_list, before_svm_acc, after_svm_acc)