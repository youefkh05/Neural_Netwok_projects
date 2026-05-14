# ==============================================================================
# Assignment 3 - Problem 3 Part B: Spoken Digits Recognition with Spatial Attention
# Comparing CNN models (with and without Spatial Attention) on spoken digit spectrograms.
# ==============================================================================

import os, warnings, tarfile, requests, librosa, cv2, numpy as np, matplotlib.pyplot as plt, tensorflow as tf, keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# 1. ENVIRONMENT & CONFIGURATION
# ==============================================================================
os.environ.update({'TF_CPP_MIN_LOG_LEVEL': '2', 'TF_ENABLE_ONEDNN_OPTS': '0'})
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')

S_RATE, N_FFT, HOP, IMG, CLASSES, BATCH, EPOCHS, SPLIT = 8000, 512, 256, (64, 64), 10, 32, 10, 0.2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_partb")
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA ACQUISITION & PREPROCESSING
# ==============================================================================
def download_fsdd():
    if os.path.exists('free-spoken-digit-dataset-master'): return
    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.tar.gz"
    try:
        with open('fsdd.tar.gz', 'wb') as f: f.write(requests.get(url).content)
        with tarfile.open('fsdd.tar.gz') as tar: tar.extractall()
        os.remove('fsdd.tar.gz')
    except Exception as e: print(f"Error downloading dataset: {e}")

def create_spectrogram(audio, sr=S_RATE):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=IMG[0], n_fft=N_FFT, hop_length=HOP)
    log_S = librosa.power_to_db(S, ref=np.max)
    norm_S = cv2.resize((log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-8), IMG[::-1])
    return norm_S

def load_fsdd_spectrograms():
    X, y, path = [], [], 'free-spoken-digit-dataset-master/recordings'
    if not os.path.exists(path): return np.array([]), np.array([])
    for f in sorted(os.listdir(path)):
        if f.endswith('.wav'):
            audio, _ = librosa.load(os.path.join(path, f), sr=S_RATE)
            X.append(create_spectrogram(audio)); y.append(f.split('_')[0])
    return np.expand_dims(np.array(X), axis=-1), np.array(y)

def prepare_dataset(X, y):
    le = LabelEncoder()
    return list(train_test_split(X, le.fit_transform(y), test_size=SPLIT, random_state=42, stratify=le.transform(y))) + [le]

# ==============================================================================
# 3. MODEL ARCHITECTURES (BASELINE & ATTENTION)
# ==============================================================================
class SpatialAttentionWithMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
    def call(self, inputs):
        mask = self.conv(keras.ops.concatenate([keras.ops.mean(inputs, -1, True), keras.ops.max(inputs, -1, True)], -1))
        return inputs * mask, mask

def create_baseline_model(ishape, ncl):
    return models.Sequential([
        layers.Input(ishape), 
        layers.Conv2D(32,(3,3),activation='relu',padding='same'), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2,2)), 
        layers.Dropout(0.25), 
        layers.Conv2D(64,(3,3),activation='relu',padding='same'), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2,2)), 
        layers.Dropout(0.3), 
        layers.Flatten(), 
        layers.Dense(256,'relu'), 
        layers.Dense(ncl,'softmax')
    ], name='Baseline_CNN')

def create_attention_model(ishape, ncl):
    inputs = layers.Input(ishape)
    x = layers.BatchNormalization()(layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs))
    x, m1 = SpatialAttentionWithMask(name='attn_1')(x)
    x = layers.BatchNormalization()(layers.Conv2D(64,(3,3),activation='relu',padding='same')(layers.MaxPooling2D((2,2))(x)))
    x, m2 = SpatialAttentionWithMask(name='attn_2')(x)
    logits = layers.Dense(ncl, 'softmax')(layers.Dense(256, 'relu')(layers.Flatten()(layers.MaxPooling2D((2,2))(x))))
    return models.Model(inputs, logits, name='Attention_CNN'), models.Model(inputs, [m1, m2], name='Mask_Model')

# ==============================================================================
# 4. VISUALIZATION & EVALUATION
# ==============================================================================
def plot_curves(h, title, fname):
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(['accuracy', 'loss']):
        plt.subplot(1, 2, i+1); plt.plot(h.history[m], 'k', label='Train'); plt.plot(h.history[f'val_{m}'], 'r', label='Test')
        plt.title(f'{title} {m.capitalize()}'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, fname), dpi=300); plt.close()

def print_performance_comparison(h1, h2, m1, m2):
    b, a = h1.history, h2.history
    b_acc, a_acc = b['val_accuracy'][-1], a['val_accuracy'][-1]
    b_loss, a_loss = b['val_loss'][-1], a['val_loss'][-1]
    
    # Calculate generalization gap (Train Acc - Test Acc)
    b_gap = abs(b['accuracy'][-1] - b['val_accuracy'][-1]) * 100
    a_gap = abs(a['accuracy'][-1] - a['val_accuracy'][-1]) * 100
    
    metrics = [
        ("Train Acc (%)", b['accuracy'][-1]*100, a['accuracy'][-1]*100), 
        ("Test Acc (%)", b_acc*100, a_acc*100), 
        ("Train Loss", b['loss'][-1], a['loss'][-1]), 
        ("Test Loss", b_loss, a_loss)
    ]
    
    print(f"\n{'Metric':<35} {'Baseline':<20} {'Attention':<20} {'Diff':<20}\n" + "-"*100)
    for n, bv, av in metrics: print(f"{n:<35} {bv:<20.4f} {av:<20.4f} {av-bv:>+19.4f}")
    
    p_diff = m2.count_params() - m1.count_params()
    print(f"{'Model Params':<35} {m1.count_params():<20} {m2.count_params():<20} {p_diff:>+19}\n" + "="*100)

    # Key Observations section
    print("\nKEY OBSERVATIONS:")
    best_acc = "Attention" if a_acc > b_acc else "Baseline"
    best_loss = "Attention" if a_loss < b_loss else "Baseline"
    best_gen = "Attention" if a_gap < b_gap else "Baseline"
    
    print(f"  ✓ Better Test Accuracy  : {best_acc} ({max(a_acc, b_acc)*100:.2f}%)")
    print(f"  ✓ Lower Test Loss       : {best_loss}")
    print(f"  ✓ Better Generalization : {best_gen} (Gap: {min(a_gap, b_gap):.2f}%)")
    print(f"  ✓ Parameter Overhead    : {p_diff} extra weights\n")

# ==============================================================================
# 5. MAIN EXECUTION FLOW
# ==============================================================================
def main():
    download_fsdd(); X, y = load_fsdd_spectrograms()
    if not len(X): return print("Dataset not found. Please ensure FSDD is downloaded.")
    
    XT, Xt, yT, yt, le = prepare_dataset(X, y)
    
    m_base = create_baseline_model((64,64,1), 10)
    m_attn, m_mask = create_attention_model((64,64,1), 10)
    
    for m in [m_base, m_attn]: 
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Segmented Training Outputs
    print("\n[*] Training Baseline CNN...")
    h_b = m_base.fit(XT, yT, validation_data=(Xt, yt), epochs=EPOCHS, batch_size=BATCH, verbose=1)
    
    print("\n[*] Training Attention CNN...")
    h_a = m_attn.fit(XT, yT, validation_data=(Xt, yt), epochs=EPOCHS, batch_size=BATCH, verbose=1)
    
    # Evaluation & Summary
    plot_curves(h_b, 'Baseline', 'Figure_5.png')
    plot_curves(h_a, 'Attention', 'Figure_6.png')
    print_performance_comparison(h_b, h_a, m_base, m_attn)

# ... (rest of the code remains the same until main)

# ==============================================================================
# 5. MAIN EXECUTION FLOW
# ==============================================================================
def main():
    download_fsdd(); X, y = load_fsdd_spectrograms()
    if not len(X): return print("Dataset not found. Please ensure FSDD is downloaded.")
    
    XT, Xt, yT, yt, le = prepare_dataset(X, y)
    
    m_base = create_baseline_model((64,64,1), 10)
    m_attn, m_mask = create_attention_model((64,64,1), 10)
    
    for m in [m_base, m_attn]: 
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # --- Baseline Model Summary and Training ---
    print("\n" + "="*50)
    print("Baseline Model Summary:")
    m_base.summary()
    print("="*50)
    
    print("\n[*] Training Baseline CNN...")
    h_b = m_base.fit(XT, yT, validation_data=(Xt, yt), epochs=EPOCHS, batch_size=BATCH, verbose=1)
    
    # --- Attention Model Summary and Training ---
    print("\n" + "="*50)
    print("Attention Model Summary:")
    m_attn.summary()
    print("="*50)
    
    print("\n[*] Training Attention CNN...")
    h_a = m_attn.fit(XT, yT, validation_data=(Xt, yt), epochs=EPOCHS, batch_size=BATCH, verbose=1)
    
    # Evaluation & Summary
    plot_curves(h_b, 'Baseline', 'Figure_5.png')
    plot_curves(h_a, 'Attention', 'Figure_6.png')
    print_performance_comparison(h_b, h_a, m_base, m_attn)

if __name__ == "__main__": 
    main()
