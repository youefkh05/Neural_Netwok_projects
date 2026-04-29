from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, List

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import librosa
    import librosa.display
except Exception:
    librosa = None

try:
    from scipy.io import wavfile
    from scipy import signal
except Exception:
    wavfile = None
    signal = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except Exception:
    LogisticRegression = None


def discover_classes(root: Path) -> List[str]:
    classes = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    return classes


def load_image(path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    sfx = path.suffix.lower()
    if sfx == '.npy':
        arr = np.load(path)
    elif sfx in ('.png', '.jpg', '.jpeg'):
        if Image is None:
            raise RuntimeError('Pillow required to read image files. Install pillow.')
        im = Image.open(path).convert('L')
        im = im.resize(target_size[::-1], Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    elif sfx == '.wav':
        arr = audio_to_spectrogram(path, target_size)
    else:
        raise RuntimeError(f'Unsupported file type: {sfx}')
    # Normalize to [0,1]
    if arr.max() > 0:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    else:
        arr = arr.astype(np.float32)
    return arr


def load_dataset(spect_dir: str, target_size=(128, 128), max_per_class=None):
    root = Path(spect_dir)
    if not root.exists():
        raise FileNotFoundError(f'{spect_dir} not found')
    # Support two layouts:
    # 1) class subfolders: root/<class>/*.(npy|png|jpg|jpeg|wav)
    # 2) flat WAV files: root/*.wav where filename contains digit label after last '_'
    entries = list(root.iterdir())
    X = []
    y = []
    if any(p.is_dir() for p in entries):
        classes = discover_classes(root)
        for ci, cls in enumerate(classes):
            cls_dir = root / cls
            files = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in ('.npy', '.png', '.jpg', '.jpeg', '.wav')])
            if max_per_class:
                files = files[:max_per_class]
            for p in files:
                try:
                    arr = load_image(p, target_size)
                except Exception as e:
                    print(f'warning: failed to load {p}: {e}', file=sys.stderr)
                    continue
                X.append(arr)
                y.append(ci)
        X = np.stack(X, axis=0) if X else np.zeros((0, target_size[0], target_size[1]), dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        return X, y, classes
    else:
        # flat directory with wav files
        wavs = sorted([p for p in entries if p.suffix.lower() == '.wav'])
        if not wavs:
            return np.zeros((0, target_size[0], target_size[1]), dtype=np.float32), np.array([], dtype=np.int64), []
        labels = []
        for p in wavs:
            lbl = extract_label_from_filename(p.name)
            labels.append(lbl)
        classes = sorted(list({int(l) for l in labels if l is not None}))
        class_map = {c: i for i, c in enumerate(classes)}
        for p, lbl in zip(wavs, labels):
            if lbl is None:
                print(f'warning: could not parse label for {p.name}', file=sys.stderr)
                continue
            try:
                arr = load_image(p, target_size)
            except Exception as e:
                print(f'warning: failed to load {p}: {e}', file=sys.stderr)
                continue
            X.append(arr)
            y.append(class_map[int(lbl)])
        X = np.stack(X, axis=0) if X else np.zeros((0, target_size[0], target_size[1]), dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        return X, y, [str(c) for c in classes]
    X = np.stack(X, axis=0) if X else np.zeros((0, target_size[0], target_size[1]), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, classes


def baseline_average_vector(X: np.ndarray) -> np.ndarray:
    # X: (N, H, W) treat W as time axis -> average along time to get length H vector
    if X.ndim != 3:
        raise ValueError('X must be (N, H, W)')
    return X.mean(axis=2)  # returns (N, H)


def extract_label_from_filename(name: str):
    # Expect patterns like <speaker>_<digit>.wav or <speaker>n_<digit>.wav
    base = Path(name).stem
    parts = base.split('_')
    if len(parts) >= 2:
        tok = parts[-1]
        if tok.isdigit():
            return tok
    # fallback: use last character if digit
    if base and base[-1].isdigit():
        return base[-1]
    return None


def audio_to_spectrogram(path: Path, target_size=(128, 128), sr=16000, n_fft=512, hop_length=256):
    # load audio
    if librosa is not None:
        y, _ = librosa.load(str(path), sr=sr)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S = librosa.amplitude_to_db(S, ref=np.max)
        # normalize to 0..1
        S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    else:
        if wavfile is None or signal is None:
            raise RuntimeError('librosa or scipy required to process wav files')
        sr_read, y = wavfile.read(str(path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr_read != sr:
            # simple resample
            num = int(round(len(y) * float(sr) / sr_read))
            y = signal.resample(y, num)
        f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        S = np.abs(Zxx)
        S = 20 * np.log10(S + 1e-9)
        S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    # S shape: (freq_bins, time_frames) -> convert to H x W target_size
    H, W = target_size
    # resize using PIL for simplicity
    if Image is None:
        # fallback: simple numpy resize via interpolation
        S_resized = np.stack([np.interp(np.linspace(0, S.shape[1]-1, W), np.arange(S.shape[1]), S[i]) for i in range(S.shape[0])])
        S_resized = np.stack([np.interp(np.linspace(0, S.shape[0]-1, H), np.arange(S.shape[0]), S_resized[:, j]) for j in range(S_resized.shape[1])], axis=1)
    else:
        img = Image.fromarray(np.uint8(np.clip(S * 255.0, 0, 255)))
        img = img.resize((W, H), Image.BILINEAR)
        S_resized = np.asarray(img, dtype=np.float32) / 255.0
    return S_resized


if torch is not None:
    class SpectrogramDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = self.X[idx][None, ...].astype(np.float32)  # add channel
            return torch.from_numpy(x), int(self.y[idx])


    class ConvAutoencoder(nn.Module):
        def __init__(self, latent_dim=64):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc_enc = nn.Linear(32, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, 32)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.enc(x)
            h = h.view(h.size(0), -1)
            z = self.fc_enc(h)
            return z

        def decode(self, z, target_shape=(1, 128, 128)):
            h = self.fc_dec(z).view(z.size(0), 32, 1, 1)
            x = self.dec(h)
            # crop/resize if needed
            if x.shape[2:] != target_shape[1:]:
                x = F.interpolate(x, size=target_shape[1:], mode='bilinear', align_corners=False)
            return x

        def forward(self, x):
            z = self.encode(x)
            xrec = self.decode(z, target_shape=x.shape)
            return xrec


def train_autoencoder(torch_model, train_loader, device='cpu', epochs=5, lr=1e-3):
    if torch is None:
        raise RuntimeError('PyTorch is required for autoencoder training')
    device = torch.device(device)
    model = torch_model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            xr = model(xb)
            loss = loss_fn(xr, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f'epoch {ep}/{epochs} loss {total/len(train_loader.dataset):.6f}')
    return model


def encode_dataset_with_ae(model, X: np.ndarray, device='cpu', batch_size=64):
    if torch is None:
        raise RuntimeError('PyTorch is required')
    ds = SpectrogramDataset(X, np.zeros(len(X), dtype=np.int32))
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    zs = []
    with torch.no_grad():
        for xb, _ in loader:
            z = model.encode(xb.to(next(model.parameters()).device))
            zs.append(z.cpu().numpy())
    return np.vstack(zs)


def train_logistic_regression(X_train, y_train, X_test, y_test):
    if LogisticRegression is None:
        raise RuntimeError('scikit-learn required for classifier training')
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    acc = accuracy_score(y_test, ypred)
    return acc, clf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spect-dir', type=str, required=True, help='spectrogram directory with class subfolders')
    p.add_argument('--mode', choices=('baseline', 'ae'), default='baseline')
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--target-size', type=int, nargs=2, default=(128, 128))
    p.add_argument('--max-per-class', type=int, default=500)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    X, y, classes = load_dataset(args.spect_dir, target_size=tuple(args.target_size), max_per_class=args.max_per_class)
    if len(X) == 0:
        print('no data found; check --spect-dir layout (class subfolders with images/.npy)')
        sys.exit(1)
    # simple split: 80/20
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    if args.mode == 'baseline':
        Vtr = baseline_average_vector(Xtr)
        Vte = baseline_average_vector(Xte)
        acc, clf = train_logistic_regression(Vtr, ytr, Vte, yte)
        print(f'Baseline average-frame vector accuracy: {acc:.4f}')
        np.save('prob4_baseline_train_vectors.npy', Vtr)
        np.save('prob4_baseline_test_vectors.npy', Vte)
    else:
        if torch is None:
            print('PyTorch not available; cannot run AE mode', file=sys.stderr)
            sys.exit(2)
        ds = SpectrogramDataset(Xtr, ytr)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        ae = ConvAutoencoder(latent_dim=args.latent_dim)
        device = args.device if torch.cuda.is_available() else 'cpu'
        ae = train_autoencoder(ae, loader, device=device, epochs=args.epochs)
        Ztr = encode_dataset_with_ae(ae, Xtr, device=device)
        Zte = encode_dataset_with_ae(ae, Xte, device=device)
        acc, clf = train_logistic_regression(Ztr, ytr, Zte, yte)
        print(f'AE ({args.latent_dim}) + Logistic accuracy: {acc:.4f}')
        np.save('prob4_ae_train_latents.npy', Ztr)
        np.save('prob4_ae_test_latents.npy', Zte)


if __name__ == '__main__':
    main()
