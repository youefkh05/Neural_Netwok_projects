import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# pip install numpy opencv-python scikit-image scikit-learn 

# ==============================
# CONFIG
# ==============================
DATA_PATH = "Indian_Digits_Train"   
IMG_SIZE = (32, 32)                # resize for HOG

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
# MAIN
# ==============================
if __name__ == "__main__":
    
    # Load images
    images = load_images(DATA_PATH)
    
    # 🔍 DEBUG VISUALIZATION
    print("[INFO] Showing sample images...")
    show_samples(images, 10)
    
    print("[INFO] Showing one image...")
    show_image(images[0], "First Image")
    
    print("[INFO] Showing HOG visualization...")
    show_hog(images[0])
    
    # Extract features
    features = extract_hog_features(images)
    
    print("[INFO] Feature extraction complete.")