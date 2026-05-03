"""Problem 4: Autoencoder for utterance representation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(BASE_DIR, "..", "audio-dataset", "Train")
TEST_DIR    = os.path.join(BASE_DIR, "..", "audio-dataset", "Test")

SAMPLE_RATE = 16_000
FRAME_MS    = 15
N_FFT       = 256
N_MELS      = 40
BOTTLENECK  = 128
AE_EPOCHS   = 100
AE_LR       = 1e-3
BATCH_SIZE  = 64

device = torch.device("cpu")
FRAME_SIZE = int(FRAME_MS * SAMPLE_RATE / 1000)


def print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_subsection(title: str) -> None:
    print(f"\n  -- {title} --")

def extract_frame_features(audio: np.ndarray, sr: int = SAMPLE_RATE,
                            n_mels: int = N_MELS) -> np.ndarray:
    """Extract log-mel frame features from one utterance."""
    if len(audio) == 0:
        return np.zeros((1, n_mels), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=N_FFT,
        win_length=FRAME_SIZE,   # 240-sample window
        hop_length=FRAME_SIZE,   # non-overlapping
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32)


def load_all_utterances(data_dir: str):
    """Load every utterance in a flat folder and parse the digit label."""
    all_frames: list[np.ndarray] = []
    all_labels: list[int]        = []
    all_fnames: list[str]        = []

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    )

    if not files:
        print(f"[WARN] No audio files found in: {data_dir}", file=sys.stderr)

    for fname in files:
        stem  = os.path.splitext(fname)[0]           # e.g. "C03n_5"
        digit = int(stem.split("_")[-1])             # last token = label

        fpath = os.path.join(data_dir, fname)
        try:
            audio, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
            frames    = extract_frame_features(audio, sr=sr)
        except Exception as exc:
            print(f"[WARN] skipping {fname}: {exc}", file=sys.stderr)
            continue

        all_frames.append(frames)
        all_labels.append(digit)
        all_fnames.append(stem)

    print(f"[INFO] Loaded {len(all_labels)} utterances from {data_dir}")
    return all_frames, np.array(all_labels), all_fnames

def compute_average_features(frames_list: list[np.ndarray]) -> np.ndarray:
    """Average frames for each utterance into one fixed-length vector."""
    return np.array([f.mean(axis=0) for f in frames_list], dtype=np.float32)


def pad_and_flatten(frames_list: list[np.ndarray],
                    max_frames: int,
                    n_mels: int = N_MELS) -> np.ndarray:
    """Pad each utterance to max_frames with zeros and flatten it."""
    N      = len(frames_list)
    result = np.zeros((N, max_frames * n_mels), dtype=np.float32)

    for i, frames in enumerate(frames_list):
        n = min(frames.shape[0], max_frames)
        result[i, : n * n_mels] = frames[:n].ravel()

    return result

class UtteranceAutoEncoder(nn.Module):
    """Symmetric dense AE for padded utterance vectors."""

    def __init__(self, input_dim: int, bottleneck: int = BOTTLENECK):
        super().__init__()
        h1 = min(input_dim // 2, 512)
        h2 = min(input_dim // 4, 256)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1,        h2), nn.ReLU(),
            nn.Linear(h2, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2), nn.ReLU(),
            nn.Linear(h2,         h1), nn.ReLU(),
            nn.Linear(h1,  input_dim),            # no final activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ══════════════════════════════════════════════
def train_ae(X_train: np.ndarray,
             epochs: int     = AE_EPOCHS,
             batch_size: int = BATCH_SIZE,
             lr: float       = AE_LR):
    """Train the AE with MSE reconstruction loss."""
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X_norm))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = UtteranceAutoEncoder(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print_section("AUTOENCODER TRAINING")
    print(f"  input_dim={X_train.shape[1]}")
    print(f"  bottleneck={BOTTLENECK}")
    print(f"  epochs={epochs}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}/{epochs}]  Loss: {total_loss / len(loader):.6f}")

    return model, scaler

def extract_ae_features(model: UtteranceAutoEncoder,
                        scaler: StandardScaler,
                        X: np.ndarray,
                        batch_size: int = 256) -> np.ndarray:
    """Normalize inputs and encode them into bottleneck vectors."""
    model.eval()
    X_norm    = scaler.transform(X).astype(np.float32)
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.from_numpy(X_norm[i : i + batch_size]).to(device)
            feat  = model.encode(batch).cpu().numpy()
            all_feats.append(feat)

    return np.vstack(all_feats)


def main() -> None:
    print_section("PROBLEM 4")
    print_subsection("CONFIGURATION")
    print(f"device: {device}")
    print(f"sample_rate: {SAMPLE_RATE}")
    print(f"frame_size: {FRAME_SIZE}")

    print_section("LOADING DATA")
    print("  loading training utterances")
    train_frames, train_labels, train_fnames = load_all_utterances(TRAIN_DIR)
    print("  loading test utterances")
    test_frames, test_labels, test_fnames = load_all_utterances(TEST_DIR)

    if not train_frames:
        sys.exit(f"[ERROR] No training data found in {TRAIN_DIR}")
    if not test_frames:
        sys.exit(f"[ERROR] No test data found in {TEST_DIR}")

    print_section("BASELINE")
    print("  computing average-frame features")
    X_train_avg = compute_average_features(train_frames)
    X_test_avg = compute_average_features(test_frames)
    print(f"  feature shape: {X_train_avg.shape}")

    scaler_avg = StandardScaler()
    X_tr_avg_sc = scaler_avg.fit_transform(X_train_avg)
    X_te_avg_sc = scaler_avg.transform(X_test_avg)

    print("  training SVM")
    t0 = time.perf_counter()
    svm_avg = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm_avg.fit(X_tr_avg_sc, train_labels)
    t_train_avg = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    y_pred_avg = svm_avg.predict(X_te_avg_sc)
    t_test_avg = (time.perf_counter() - t0) * 1000

    acc_avg = accuracy_score(test_labels, y_pred_avg) * 100
    print(f"  accuracy: {acc_avg:.1f}%")
    print(f"  train time: {t_train_avg:.1f} ms")
    print(f"  test time: {t_test_avg:.1f} ms")

    print_section("AUTOENCODER")
    MAX_FRAMES = max(f.shape[0] for f in train_frames)
    INPUT_DIM = MAX_FRAMES * N_MELS
    print(f"  max_frames: {MAX_FRAMES}")
    print(f"  input_dim: {INPUT_DIM}")

    print("  padding and flattening utterances")
    X_train_padded = pad_and_flatten(train_frames, MAX_FRAMES)
    X_test_padded = pad_and_flatten(test_frames, MAX_FRAMES)
    print(f"  padded shape: {X_train_padded.shape}")

    ae_model, ae_scaler = train_ae(X_train_padded)

    print("  extracting bottleneck features")
    X_train_ae = extract_ae_features(ae_model, ae_scaler, X_train_padded)
    X_test_ae = extract_ae_features(ae_model, ae_scaler, X_test_padded)
    print(f"  ae feature shape: {X_train_ae.shape}")

    print("  training SVM on AE features")
    scaler_ae = StandardScaler()
    X_tr_ae_sc = scaler_ae.fit_transform(X_train_ae)
    X_te_ae_sc = scaler_ae.transform(X_test_ae)

    t0 = time.perf_counter()
    svm_ae = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm_ae.fit(X_tr_ae_sc, train_labels)
    t_train_ae = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    y_pred_ae = svm_ae.predict(X_te_ae_sc)
    t_test_ae = (time.perf_counter() - t0) * 1000

    acc_ae = accuracy_score(test_labels, y_pred_ae) * 100
    print(f"  accuracy: {acc_ae:.1f}%")
    print(f"  train time: {t_train_ae:.1f} ms")
    print(f"  test time: {t_test_ae:.1f} ms")

    print_section("RESULTS")
    print(f"{'Method':<30} {'Acc':>6}  {'Train(ms)':>10}  {'Test(ms)':>8}")
    print("-" * 65)
    print(f"{'Baseline (Average Frame)':<30} {acc_avg:>5.1f}%  {t_train_avg:>10.1f}  {t_test_avg:>8.1f}")
    print(f"{'AE (Pad+Encode+SVM)':<30} {acc_ae:>5.1f}%  {t_train_ae:>10.1f}  {t_test_ae:>8.1f}")
    print("=" * 65)
    print("Note: AE training time is excluded because it is a preprocessing step.")

    print_section("VISUALISATIONS")
    DIGIT_NAMES = [str(d) for d in range(10)]
    FIG_DIR = os.path.join(BASE_DIR, "Figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    cm_avg = confusion_matrix(test_labels, y_pred_avg)
    cm_ae = confusion_matrix(test_labels, y_pred_ae)

    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

    for ax, cm, title in [
        (axes[0], cm_avg, f"Baseline (Average Frame)\nAcc = {acc_avg:.1f}%"),
        (axes[1], cm_ae, f"AutoEncoder (Pad+Encode)\nAcc = {acc_ae:.1f}%"),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticks(range(10))
        ax.set_xticklabels(DIGIT_NAMES)
        ax.set_yticks(range(10))
        ax.set_yticklabels(DIGIT_NAMES)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = cm.max() / 2.0
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=120, bbox_inches="tight")
    print("  saved Figures/confusion_matrices.png")
    plt.close()

    per_digit_avg = []
    per_digit_ae = []
    for d in range(10):
        mask = (test_labels == d)
        per_digit_avg.append(accuracy_score(test_labels[mask], y_pred_avg[mask]) * 100)
        per_digit_ae.append(accuracy_score(test_labels[mask], y_pred_ae[mask]) * 100)

    x = np.arange(10)
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bars1 = ax2.bar(x - width / 2, per_digit_avg, width,
                    label=f"Baseline ({acc_avg:.1f}%)", color="steelblue", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, per_digit_ae, width,
                    label=f"AutoEncoder ({acc_ae:.1f}%)", color="coral", alpha=0.85)
    ax2.set_xlabel("Digit", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Per-Digit Accuracy on Test Set", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DIGIT_NAMES, fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(fontsize=10)
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "per_digit_accuracy.png"), dpi=120, bbox_inches="tight")
    print("  saved Figures/per_digit_accuracy.png")
    plt.close()


if __name__ == "__main__":
    main()