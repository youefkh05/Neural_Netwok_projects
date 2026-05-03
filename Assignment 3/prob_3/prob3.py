# -*- coding: utf-8 -*-
import sys, io, os, time, warnings, random, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.text import Text
from rich.align import Align
from rich import box
import rich.traceback

rich.traceback.install()
console = Console()
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(os.path.dirname(SCRIPT_DIR), "audio-dataset")
TRAIN_DIR   = os.path.join(DATA_ROOT, "Train")
TEST_DIR    = os.path.join(DATA_ROOT, "Test")

IMG_HEIGHT = IMG_WIDTH = 64
NUM_CLASSES  = 10
BATCH_SIZE   = 32
NUM_EPOCHS   = 10
LEARNING_RATE = 0.001

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

SPEED_UP_RATE   = 1.03
SPEED_DOWN_RATE = 0.97
NOISE_SNR_DB    = 15.0
PITCH_SHIFT_STEPS = 2.0

SQUEEZE_RATE = 0.88
EXPAND_RATE  = 1.12

SPEC_AUG_F = 12
SPEC_AUG_T = 12

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_banner(suffix, color, extra):
    banner = Text()
    banner.append("  Assignment 2  ", style="bold white on dark_blue")
    banner.append("  Problem 3  ",    style="bold white on blue")
    banner.append(f"  {suffix}  ",    style=f"bold black on {color}")
    console.print(); console.print(Align.center(banner))
    console.print(Panel(extra, title=f"[bold white]Configuration[/bold white]",
                        border_style=color, padding=(0, 2)))


def audio_array_to_spectrogram(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, n_mels=N_MELS)
    db  = librosa.power_to_db(mel, ref=np.max)
    lo, hi = db.min(), db.max()
    norm = (db - lo) / (hi - lo) if hi > lo else np.zeros_like(db)
    return (plt.get_cmap("magma")(norm)[..., :3] * 255).astype(np.uint8)


def audio_file_to_spectrogram(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio_array_to_spectrogram(y, sr)


def preprocess_spectrogram(arr):
    img = Image.fromarray(arr, mode="RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def parse_label(fname):
    try:    return int(os.path.splitext(fname)[0].split("_")[-1])
    except: return None


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = test_transform = transforms.Compose([normalize])

def aug_speed(y, rate):       return librosa.effects.time_stretch(y, rate=rate)
def aug_noise(y, snr=NOISE_SNR_DB):
    rms = np.sqrt(np.mean(y**2))
    if rms == 0: return y.copy()
    rms_n = rms / np.sqrt(10.0 ** (snr / 10.0))
    return (y + np.random.randn(len(y)) * rms_n).astype(np.float32)
def aug_pitch(y, steps=PITCH_SHIFT_STEPS):
    return librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)

def spec_augment(t, fs=SPEC_AUG_F, ts=SPEC_AUG_T):
    _, H, W = t.shape; t = t.clone()
    fs, ts = min(fs, H), min(ts, W)
    if fs > 0: f0 = random.randint(0, max(H - fs, 0)); t[:, f0:f0+fs, :] = 0
    if ts > 0: t0 = random.randint(0, max(W - ts, 0)); t[:, :, t0:t0+ts] = 0
    return t

def add_image_noise(arr, amount=0.02):
    return np.clip(arr + np.random.normal(0, amount, arr.shape).astype(np.float32), 0, 1)

def resize_spectrum(img, rate):
    nw = int(IMG_WIDTH * rate)
    resized = img.resize((nw, IMG_HEIGHT), Image.BILINEAR)
    out = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    if rate < 1.0: out.paste(resized, ((IMG_WIDTH - nw) // 2, 0))
    else:
        left = (nw - IMG_WIDTH) // 2
        out = resized.crop((left, 0, left + IMG_WIDTH, IMG_HEIGHT))
    return out


SPEECH_VARIANTS = [None, "speed_up", "speed_down", "noise", "pitch"]
SPEC_VARIANTS   = [None, "squeeze",  "expand",     "img_noise", "hybrid"]

def apply_speech_aug(y, variant):
    if variant == "speed_up":   return aug_speed(y, SPEED_UP_RATE)
    if variant == "speed_down": return aug_speed(y, SPEED_DOWN_RATE)
    if variant == "noise":      return aug_noise(y)
    if variant == "pitch":      return aug_pitch(y)
    return y

def apply_spec_aug(img, variant):
    if variant == "squeeze":    return resize_spectrum(img, SQUEEZE_RATE)
    if variant == "expand":     return resize_spectrum(img, EXPAND_RATE)
    if variant == "img_noise":
        arr = add_image_noise(np.array(img, dtype=np.float32) / 255.0, 0.08)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if variant == "hybrid":
        img = resize_spectrum(img, random.choice([SQUEEZE_RATE, EXPAND_RATE]))
        arr = add_image_noise(np.array(img, dtype=np.float32) / 255.0, 0.05)
        return Image.fromarray((arr * 255).astype(np.uint8))
    return img

def _scan_files(root):
    files = []
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(".wav"): continue
        lbl = parse_label(fname)
        if lbl is None or not 0 <= lbl <= 9: continue
        files.append((os.path.join(root, fname), lbl))
    return files


class SpeechSpectrogramDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = _scan_files(root)
        self.classes = sorted({l for _, l in self.samples})
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        t = torch.tensor(preprocess_spectrogram(audio_file_to_spectrogram(path))).permute(2,0,1)
        if self.transform: t = self.transform(t)
        return t, lbl

class SpeechAugmentedDataset(Dataset):
    def __init__(self, root, transform=None, aug=True):
        self.transform, self.aug = transform, aug
        self._files = _scan_files(root)
        self.classes = sorted({l for _, l in self._files})
        nv = len(SPEECH_VARIANTS) if aug else 1
        self.samples = [(fi, vi) for fi in range(len(self._files)) for vi in range(nv)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        fi, vi = self.samples[i]
        path, lbl = self._files[fi]
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        y = apply_speech_aug(y, SPEECH_VARIANTS[vi])
        spec = audio_array_to_spectrogram(y, sr)
        t = torch.tensor(preprocess_spectrogram(spec)).permute(2,0,1)
        if self.transform: t = self.transform(t)
        if self.aug: t = spec_augment(t)
        return t, lbl

class SpectrumAugmentedDataset(Dataset):
    def __init__(self, root, transform=None, aug=True):
        self.transform, self.aug = transform, aug
        self._files = _scan_files(root)
        self.classes = sorted({l for _, l in self._files})
        nv = len(SPEC_VARIANTS) if aug else 1
        self.samples = [(fi, vi) for fi in range(len(self._files)) for vi in range(nv)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        fi, vi = self.samples[i]
        path, lbl = self._files[fi]
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        spec = audio_array_to_spectrogram(y, sr)
        img = Image.fromarray(spec).resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        img = apply_spec_aug(img, SPEC_VARIANTS[vi] if self.aug else None)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.tensor(arr).permute(2,0,1)
        if self.transform: t = self.transform(t)
        if self.aug: t = spec_augment(t)
        return t, lbl

class HybridAugmentedDataset(Dataset):
    def __init__(self, root, transform=None, aug=True):
        self.transform, self.aug = transform, aug
        self._files = _scan_files(root)
        self.classes = sorted({l for _, l in self._files})
        if aug:
            self.samples = [(fi, sv, xv) for fi in range(len(self._files))
                            for sv in range(len(SPEECH_VARIANTS)) for xv in range(len(SPEC_VARIANTS))]
        else:
            self.samples = [(fi, 0, 0) for fi in range(len(self._files))]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        fi, sv, xv = self.samples[i]
        path, lbl = self._files[fi]
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        y = apply_speech_aug(y, SPEECH_VARIANTS[sv])
        spec = audio_array_to_spectrogram(y, sr)
        img = Image.fromarray(spec).resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        img = apply_spec_aug(img, SPEC_VARIANTS[xv])
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.tensor(arr).permute(2,0,1)
        if self.transform: t = self.transform(t)
        if self.aug: t = spec_augment(t)
        return t, lbl
class LeNet5Adapted(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.ReLU(inplace=True), nn.AvgPool2d(2,2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True), nn.AvgPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AvgPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128),    nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


def print_model_summary(model, part_label, rule_color, border_style, header_style):
    console.print(); console.rule(f"[bold {rule_color}]  Model Architecture  ", style=rule_color)
    table = Table(box=box.ROUNDED, border_style=border_style, header_style=header_style,
                  show_lines=True,
                  title=f"[bold cyan]LeNet5Adapted (3-Conv, Dropout 0.5) — {part_label}[/bold cyan]")
    for col, sty, w in [("Layer (type)","bold white",30),
                         ("Output Shape","cyan",26),("Param #","bright_green",12)]:
        table.add_column(col, style=sty, min_width=w,
                         justify="center" if col != "Layer (type)" else "left")

    rows, x = [], torch.zeros(1,3,64,64)

    x = model.features[0](x); rows.append(("conv2d_1  (Conv2D)", f"(None,{x.shape[1]},{x.shape[2]},{x.shape[3]})", sum(p.numel() for p in model.features[0].parameters())))
    x = model.features[1](x); x = model.features[2](x); rows.append(("avg_pool2d_1  (AvgPool2D)", f"(None,{x.shape[1]},{x.shape[2]},{x.shape[3]})", 0))
    x = model.features[3](x); rows.append(("conv2d_2  (Conv2D)", f"(None,{x.shape[1]},{x.shape[2]},{x.shape[3]})", sum(p.numel() for p in model.features[3].parameters())))
    x = model.features[4](x); x = model.features[5](x); rows.append(("avg_pool2d_2  (AvgPool2D)", f"(None,{x.shape[1]},{x.shape[2]},{x.shape[3]})", 0))
    x = model.features[6](x); rows.append(("conv2d_3  (Conv2D)", f"(None,{x.shape[1]},{x.shape[2]},{x.shape[3]})", sum(p.numel() for p in model.features[6].parameters())))
    x = model.features[7](x); x = model.features[8](x); rows.append(("avg_pool2d_3  (AvgPool2D)", f"(None,{x.shape[1]},{x.shape[2]},{x.shape[3]})", 0))
    x = x.view(1,-1); rows.append(("flatten  (Flatten)", f"(None,{x.shape[1]})", 0))
    x = model.classifier[1](x); rows.append(("dense_1  (Dense 256)", f"(None,{x.shape[1]})", sum(p.numel() for p in model.classifier[1].parameters())))
    rows.append(("dropout_1  (Dropout 0.5)", f"(None,{x.shape[1]})", 0))
    x = model.classifier[4](x); rows.append(("dense_2  (Dense 128)", f"(None,{x.shape[1]})", sum(p.numel() for p in model.classifier[4].parameters())))
    x = model.classifier[6](x); rows.append(("dense_3  (Dense output)", f"(None,{x.shape[1]})", sum(p.numel() for p in model.classifier[6].parameters())))

    total = sum(r[2] for r in rows)
    for name, shape, params in rows:
        table.add_row(name, shape, f"{params:,}" if params else "[dim]0[/dim]")
    console.print(table)

    stats = Table.grid(padding=(0,3))
    stats.add_column(style="bold cyan"); stats.add_column(style="bold yellow")
    mb = lambda n: n*4/(1024**2)
    stats.add_row("Total params:",         f"{total:,}  ({mb(total):.2f} MB)")
    stats.add_row("Trainable params:",     f"{total:,}  ({mb(total):.2f} MB)")
    stats.add_row("Non-trainable params:", "0  (0.00 B)")
    console.print(Panel(stats, border_style=border_style,
                        title="[bold white]Parameter Summary[/bold white]", padding=(0,2)))
    console.print()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); loss_sum = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(); out = model(X); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*X.size(0); correct += out.max(1)[1].eq(y).sum().item(); total += y.size(0)
    return loss_sum/total, 100.*correct/total


def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum = correct = total = 0; preds, truths = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X); loss = criterion(out, y)
            loss_sum += loss.item()*X.size(0); pred = out.max(1)[1]
            correct += pred.eq(y).sum().item(); total += y.size(0)
            preds.extend(pred.cpu().numpy()); truths.extend(y.cpu().numpy())
    return loss_sum/total, 100.*correct/total, preds, truths


def acc_color(a):
    if a >= 90: return "bold bright_green"
    if a >= 75: return "bold yellow"
    return "bold red"


def run_training(model, train_dl, test_dl, criterion, optimizer, scheduler, device,
                 rule_color, progress_color, header_style):
    console.print(); console.rule(f"[bold {rule_color}]  Training  ", style=rule_color)
    ht = Table(box=box.SIMPLE_HEAVY, border_style=rule_color, header_style=header_style, expand=False)
    for col, sty, w in [("Epoch","bold white",7),("Time (s)","dim white",7),
                         ("ms/step","dim white",7),("Train Loss","cyan",10),
                         ("Train Acc","",10),("Val Loss","magenta",10),("Val Acc","",10)]:
        ht.add_column(col, justify="center", style=sty, min_width=w)

    tl, vl, ta, va = [], [], [], []; t0 = time.time()
    with Progress(SpinnerColumn(spinner_name="dots", style=progress_color),
                  TextColumn(f"[bold {progress_color}]{{task.description}}"),
                  BarColumn(bar_width=30, style=progress_color, complete_style=rule_color),
                  TextColumn("[bold white]{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        task = prog.add_task("Training epochs", total=NUM_EPOCHS)
        for ep in range(1, NUM_EPOCHS+1):
            et0 = time.time()
            tl_, ta_ = train_one_epoch(model, train_dl, criterion, optimizer, device)
            vl_, va_, _, _ = evaluate(model, test_dl, criterion, device)
            scheduler.step(); es = time.time()-et0
            tl.append(tl_); vl.append(vl_); ta.append(ta_); va.append(va_)
            ts_, vs_ = acc_color(ta_), acc_color(va_)
            ht.add_row(f"{ep:02d}/{NUM_EPOCHS}", f"{es:.1f}s", f"{es*1000/max(len(train_dl),1):.0f}ms",
                       f"{tl_:.4f}", f"[{ts_}]{ta_:.2f}%[/{ts_}]",
                       f"{vl_:.4f}", f"[{vs_}]{va_:.2f}%[/{vs_}]")
            prog.advance(task)

    training_ms = (time.time()-t0)*1000
    console.print(ht)
    row = Table.grid(padding=(0,4)); row.add_column(style="bold cyan"); row.add_column(style="bold yellow")
    row.add_row("⏱  Training Time:", f"{training_ms:,.2f} ms  ({training_ms/1000:.1f} s)")
    console.print(Panel(row, border_style=rule_color, padding=(0,2)))
    return tl, vl, ta, va, training_ms


def print_classification_report(labels, preds, names, rule_color, border, title):
    console.print(); console.rule(f"[bold {rule_color}]  Per-Class Report  ", style=rule_color)
    prec, rec, f1, sup = precision_recall_fscore_support(labels, preds,
                          labels=list(range(len(names))), zero_division=0)
    acc = accuracy_score(labels, preds)*100
    tbl = Table(box=box.ROUNDED, border_style=border,
                header_style="bold white on dark_blue", show_lines=True)
    for col, sty, w in [("Class","bold white",8),("Precision","cyan",12),
                         ("Recall","magenta",12),("F1-Score","yellow",12),("Support","dim white",10)]:
        tbl.add_column(col, justify="center", style=sty, min_width=w)
    for i, name in enumerate(names):
        fs = acc_color(f1[i]*100)
        tbl.add_row(name, f"{prec[i]:.4f}", f"{rec[i]:.4f}",
                    f"[{fs}]{f1[i]:.4f}[/{fs}]", str(sup[i]))
    console.print(tbl)
    a_s = acc_color(acc)
    console.print(Panel(f"[bold white]{title}[/bold white]  [{a_s}]{acc:.1f}%[/{a_s}]",
                        border_style=border, padding=(0,2)))
    return acc

def plot_training_curves(tl, vl, ta, va, path, loss_title, acc_title,
                         train_color="royalblue", val_color="tomato"):
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    for a, y1, y2, t, yl in [(ax[0],tl,vl,loss_title,"Loss"),
                               (ax[1],ta,va,acc_title,"Accuracy (%)")]:
        a.plot(y1, label="Train", color=train_color)
        a.plot(y2, label="Val",   color=val_color)
        a.set(title=t, xlabel="Epoch", ylabel=yl); a.legend(); a.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    console.print(f"[dim][Saved] Training curves → {path}[/dim]")


def plot_confusion_matrix(truths, preds, names, path, cmap, title):
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(confusion_matrix(truths, preds), annot=True, fmt="d", cmap=cmap,
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    console.print(f"[dim][Saved] Confusion matrix → {path}[/dim]")


def find_first_wav(directory):
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(".wav"): return os.path.join(directory, f)
    return None


def save_augmented_spectrogram_images(sample_wav, label, out_dir="augmented_spectrograms"):
    """Part B style: 5 individual mel-spectrogram PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    y, sr = librosa.load(sample_wav, sr=SAMPLE_RATE, mono=True)
    variants = [
        ("original",   y),
        ("speed_up",   aug_speed(y, SPEED_UP_RATE)),
        ("speed_down", aug_speed(y, SPEED_DOWN_RATE)),
        ("noise_15db", aug_noise(y, NOISE_SNR_DB)),
        ("pitch_up",   aug_pitch(y, PITCH_SHIFT_STEPS)),
    ]
    titles = {"original":"Original","speed_up":"Speed +3%","speed_down":"Speed −3%",
              "noise_15db":f"Noise {NOISE_SNR_DB} dB SNR","pitch_up":f"Pitch +{PITCH_SHIFT_STEPS} semitones"}
    stem = os.path.splitext(os.path.basename(sample_wav))[0]
    saved = []
    for tag, yv in variants:
        mel = librosa.feature.melspectrogram(y=yv, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mdb = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(6,4))
        img = librosa.display.specshow(mdb, sr=sr, hop_length=HOP_LENGTH,
                                       x_axis="time", y_axis="mel", cmap="magma", ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(f"Digit {label} — {titles[tag]}", fontsize=12)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Mel)")
        plt.tight_layout()
        p = os.path.join(out_dir, f"{stem}_{tag}.png")
        plt.savefig(p, dpi=150); plt.close(); saved.append(p)
        console.print(f"[dim][Saved] {p}[/dim]")
    return saved


def save_spectrum_augmented_images(sample_wav, label, out_dir="augmented_spectrum_examples"):
    """Part C/D style — spectrum augmentation visualisations."""
    os.makedirs(out_dir, exist_ok=True)
    y, sr = librosa.load(sample_wav, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mdb = librosa.power_to_db(mel, ref=np.max)
    norm = (mdb - mdb.min()) / (mdb.max() - mdb.min() + 1e-6)
    base_rgb = (plt.get_cmap("magma")(norm)[..., :3] * 255).astype(np.uint8)
    base_img = Image.fromarray(base_rgb).resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)

    squeezed = resize_spectrum(base_img, SQUEEZE_RATE)
    expanded = resize_spectrum(base_img, EXPAND_RATE)
    noisy_arr = add_image_noise(np.array(base_img, dtype=np.float32)/255.0, 0.08)
    noisy_img = Image.fromarray((noisy_arr*255).astype(np.uint8))
    hybrid_tmp = resize_spectrum(base_img, random.choice([SQUEEZE_RATE, EXPAND_RATE]))
    hybrid_arr = add_image_noise(np.array(hybrid_tmp, dtype=np.float32)/255.0, 0.05)
    hybrid_img = Image.fromarray((hybrid_arr*255).astype(np.uint8))

    variants = [
        ("original",  base_img,  "Original"),
        ("squeeze",   squeezed,  f"Squeeze −{int((1-SQUEEZE_RATE)*100)}%"),
        ("expand",    expanded,  f"Expand +{int((EXPAND_RATE-1)*100)}%"),
        ("img_noise", noisy_img, "Image Noise σ=0.08"),
        ("hybrid",    hybrid_img,"Hybrid (Resize + Noise)"),
    ]

    stem = os.path.splitext(os.path.basename(sample_wav))[0]
    saved = []
    for tag, img, title in variants:
        fig, ax = plt.subplots(figsize=(6, 4))
        arr = np.array(img, dtype=np.float32).mean(axis=-1)
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, len(y)/sr, 0, sr/2/1000])
        fig.colorbar(im, ax=ax, label="Intensity")
        ax.set_title(f"Digit {label} — {title}", fontsize=12)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (kHz)")
        plt.tight_layout()
        p = os.path.join(out_dir, f"{stem}_{tag}.png")
        plt.savefig(p, dpi=150); plt.close(); saved.append(p)
        console.print(f"[dim][Saved] {p}[/dim]")

    fig, axes = plt.subplots(1, len(variants), figsize=(5*len(variants), 4))
    for ax, (tag, img, title) in zip(axes, variants):
        arr = np.array(img, dtype=np.float32).mean(axis=-1)
        ax.imshow(arr, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"Digit {label}\n{title}", fontsize=10)
        ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
    plt.suptitle("Spectrum Augmentation Examples (5 variants)", fontsize=13)
    plt.tight_layout()
    grid_p = os.path.join(out_dir, f"{stem}_spectrum_grid.png")
    plt.savefig(grid_p, dpi=150); plt.close()
    console.print(f"[dim][Saved] {grid_p}[/dim]")
    return saved


def plot_augmentation_comparison(orig_path, aug_variants, label, path="augmentation_examples_partB.png"):
    y_orig, sr = librosa.load(orig_path, sr=SAMPLE_RATE, mono=True)
    all_v = [("Original", y_orig)] + aug_variants
    fig, axes = plt.subplots(1, len(all_v), figsize=(5*len(all_v), 4))
    for ax, (title, y) in zip(np.atleast_1d(axes), all_v):
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        ax.imshow(librosa.power_to_db(mel, ref=np.max), origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"Digit {label}\n{title}", fontsize=11)
        ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
    plt.suptitle("Speech Augmentation Examples (5 variants) — Part B", fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    console.print(f"[dim][Saved] Augmentation examples → {path}[/dim]")

def run_part(label, color, train_ds_fn, test_ds_fn,
             extra_config, pre_train_hook=None, post_eval_hook=None,
             curves_path="training_curves.png", loss_title="Loss", acc_title="Acc",
             cm_path="confusion_matrix.png", cm_cmap="Blues", cm_title="CM",
             weights_path="model.pth", train_color="royalblue", report_title="Accuracy:"):

    print_banner(label, color, extra_config)
    rc = f"bright_{color}" if color != "blue" else "bright_blue"

    console.print(); console.rule(f"[bold {rc}]  Step 1 — Loading Datasets  ", style=rc)
    with console.status(f"[{color}]Scanning dataset directories...[/{color}]", spinner="dots"):
        train_ds = train_ds_fn()
        test_ds  = test_ds_fn()
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if pre_train_hook: pre_train_hook()

    console.print(); console.rule(f"[bold {rc}]  Step 2 — Building Model  ", style=rc)
    model = LeNet5Adapted(NUM_CLASSES).to(DEVICE)
    dark_variant = 'blue' if 'blue' in rc else 'green' if 'green' in rc else 'magenta'
    print_model_summary(model, label, rc, rc, f"bold white on dark_{dark_variant}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    console.rule(f"[bold {rc}]  Step 3 — Training  ", style=rc)
    tl, vl, ta, va, train_ms = run_training(model, train_dl, test_dl, criterion, optimizer,
                                             scheduler, DEVICE, rc, color,
                                             f"bold white on dark_{dark_variant}")

    console.print(); console.rule(f"[bold {rc}]  Step 4 — Evaluation  ", style=rc)
    with console.status(f"[{color}]Running inference on test set...[/{color}]", spinner="dots"):
        t1 = time.time()
        _, acc, preds, labels = evaluate(model, test_dl, criterion, DEVICE)
        test_ms = (time.time()-t1)*1000

    row = Table.grid(padding=(0,4)); row.add_column(style="bold cyan"); row.add_column(style="bold yellow")
    row.add_row("⚡ Testing Time:", f"{test_ms:,.2f} ms")
    console.print(Panel(row, border_style=rc, padding=(0,2)))

    class_names = [str(i) for i in range(NUM_CLASSES)]
    acc = print_classification_report(labels, preds, class_names, rc, rc, report_title)

    console.print(); console.rule(f"[bold {rc}]  Step 5 — Saving Outputs  ", style=rc)
    if post_eval_hook: post_eval_hook()
    plot_training_curves(tl, vl, ta, va, curves_path, loss_title, acc_title, train_color)
    plot_confusion_matrix(labels, preds, class_names, cm_path, cm_cmap, cm_title)
    torch.save(model.state_dict(), weights_path)
    console.print(f"[dim][Saved] Model weights → {weights_path}[/dim]")

    console.print(); console.rule(f"[bold {rc}]  Result Summary — {label}  ", style=rc)
    a_s = acc_color(acc)
    summ = Table.grid(padding=(0,4))
    summ.add_column(style="bold cyan", min_width=28); summ.add_column(style="bold yellow")
    summ.add_row("🎯 Accuracy:",       f"[{a_s}]{acc:.1f}%[/{a_s}]")
    summ.add_row("🏗  Architecture:",   "3-Conv, Dropout(0.5)")
    summ.add_row("📅 Scheduler:",       f"CosineAnnealingLR (T_max={NUM_EPOCHS}, η_min=1e-5)")
    summ.add_row("⏱  Training Time:",  f"{train_ms:,.2f} ms  ({train_ms/1000:.1f} s)")
    summ.add_row("⚡ Testing Time:",    f"{test_ms:,.2f} ms")
    console.print(Panel(summ, border_style="bright_green",
                        title=f"[bold white]✅  {label} Done[/bold white]", padding=(1,3)))
    console.print()

    return {"model": model, "final_acc": acc, "training_time_ms": train_ms,
            "testing_time_ms": test_ms, "train_ds": train_ds, "test_ds": test_ds}

def run_part_a():
    return run_part(
        "Part A", "blue",
        train_ds_fn=lambda: SpeechSpectrogramDataset(TRAIN_DIR, train_transform),
        test_ds_fn =lambda: SpeechSpectrogramDataset(TEST_DIR,  test_transform),
        extra_config=(f"[bold cyan]Device:[/bold cyan] [yellow]{DEVICE}[/yellow]   "
                      f"[bold cyan]Epochs:[/bold cyan] [yellow]{NUM_EPOCHS}[/yellow]   "
                      f"[bold cyan]Batch:[/bold cyan] [yellow]{BATCH_SIZE}[/yellow]   "
                      f"[bold cyan]LR:[/bold cyan] [yellow]{LEARNING_RATE}[/yellow]   "
                      f"[bold cyan]Classes:[/bold cyan] [yellow]{NUM_CLASSES}[/yellow]"),
        curves_path="training_curves_partA.png",
        loss_title="Loss per Epoch (Part A — Baseline)",
        acc_title ="Accuracy per Epoch (Part A — Baseline)",
        cm_path="confusion_matrix_partA.png", cm_cmap="Blues",
        cm_title="Confusion Matrix — Part A (Baseline CNN)",
        weights_path="lenet5_partA_baseline.pth",
        train_color="royalblue",
        report_title="Accuracy before data augmentation:",
    )


def run_part_b():
    def _pre():
        console.print(); console.rule("[bold bright_green]  Step 0 — Augmentation Illustration  ", style="bright_green")
        try:
            wav = find_first_wav(TRAIN_DIR)
            if wav is None: raise StopIteration
            label = int(os.path.splitext(os.path.basename(wav))[0].split("_")[-1])
            y, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
            plot_augmentation_comparison(wav, [
                ("Speed +3%",              aug_speed(y, SPEED_UP_RATE)),
                ("Speed -3%",              aug_speed(y, SPEED_DOWN_RATE)),
                (f"Noise {NOISE_SNR_DB}dB", aug_noise(y, NOISE_SNR_DB)),
                (f"Pitch +{PITCH_SHIFT_STEPS}st", aug_pitch(y, PITCH_SHIFT_STEPS)),
            ], label)
        except StopIteration:
            console.print("[yellow]⚠  No .wav file found.[/yellow]")

    def _post():
        try:
            wav = find_first_wav(TRAIN_DIR)
            if wav: save_augmented_spectrogram_images(
                wav, int(os.path.splitext(os.path.basename(wav))[0].split("_")[-1]))
        except Exception: pass

    res = run_part(
        "Part B", "green",
        train_ds_fn=lambda: SpeechAugmentedDataset(TRAIN_DIR, train_transform, aug=True),
        test_ds_fn =lambda: SpeechAugmentedDataset(TEST_DIR,  test_transform,  aug=False),
        extra_config=(f"[bold cyan]Device:[/bold cyan] [yellow]{DEVICE}[/yellow]   "
                      f"[bold cyan]Epochs:[/bold cyan] [yellow]{NUM_EPOCHS}[/yellow]   "
                      f"[bold cyan]Batch:[/bold cyan] [yellow]{BATCH_SIZE}[/yellow]   "
                      f"[bold cyan]Speed±:[/bold cyan] [yellow]±3%[/yellow]   "
                      f"[bold cyan]Noise SNR:[/bold cyan] [yellow]{NOISE_SNR_DB} dB[/yellow]   "
                      f"[bold cyan]Pitch:[/bold cyan] [yellow]+{PITCH_SHIFT_STEPS} st[/yellow]"),
        pre_train_hook=_pre, post_eval_hook=_post,
        curves_path="training_curves_partB.png",
        loss_title="Loss per Epoch (Part B — Speech Aug.)",
        acc_title ="Accuracy per Epoch (Part B — Speech Aug.)",
        cm_path="confusion_matrix_partB.png", cm_cmap="Greens",
        cm_title="Confusion Matrix — Part B (Speech Augmentation)",
        weights_path="lenet5_partB_speech_aug.pth",
        train_color="seagreen",
        report_title="Accuracy after speech augmentation:",
    )
    res["train_samples"] = len(res["train_ds"])
    return res

def run_part_c():
    def _post():
        try:
            wav = find_first_wav(TRAIN_DIR)
            if wav: save_spectrum_augmented_images(
                wav, int(os.path.splitext(os.path.basename(wav))[0].split("_")[-1]))
        except Exception: pass

    return run_part(
        "Part C", "magenta",
        train_ds_fn=lambda: SpectrumAugmentedDataset(TRAIN_DIR, train_transform, aug=True),
        test_ds_fn =lambda: SpectrumAugmentedDataset(TEST_DIR,  test_transform,  aug=False),
        extra_config=(f"[bold cyan]Device:[/bold cyan] [yellow]{DEVICE}[/yellow]   "
                      f"[bold cyan]Epochs:[/bold cyan] [yellow]{NUM_EPOCHS}[/yellow]   "
                      f"[bold cyan]Batch:[/bold cyan] [yellow]{BATCH_SIZE}[/yellow]   "
                      f"[bold cyan]Squeeze:[/bold cyan] [yellow]{int((1-SQUEEZE_RATE)*100)}%[/yellow]   "
                      f"[bold cyan]Expand:[/bold cyan] [yellow]{int((EXPAND_RATE-1)*100)}%[/yellow]   "
                      f"[bold cyan]SpecAugment:[/bold cyan] [yellow]F={SPEC_AUG_F} T={SPEC_AUG_T}[/yellow]"),
        post_eval_hook=_post,
        curves_path="training_curves_partC.png",
        loss_title="Loss per Epoch (Part C — Spectrum Aug.)",
        acc_title ="Accuracy per Epoch (Part C — Spectrum Aug.)",
        cm_path="confusion_matrix_partC.png", cm_cmap="Purples",
        cm_title="Confusion Matrix — Part C (Spectrum Augmentation)",
        weights_path="lenet5_partC_spectrum_aug.pth",
        train_color="purple",
        report_title="Accuracy with spectrum augmentation:",
    )


def run_part_d():
    def _post():
        try:
            wav = find_first_wav(TRAIN_DIR)
            if wav: save_spectrum_augmented_images(
                wav, int(os.path.splitext(os.path.basename(wav))[0].split("_")[-1]),
                out_dir="augmented_spectrum_examples_partD")
        except Exception: pass

    return run_part(
        "Part D", "blue",
        train_ds_fn=lambda: HybridAugmentedDataset(TRAIN_DIR, train_transform, aug=True),
        test_ds_fn =lambda: HybridAugmentedDataset(TEST_DIR,  test_transform,  aug=False),
        extra_config=(f"[bold cyan]Device:[/bold cyan] [yellow]{DEVICE}[/yellow]   "
                      f"[bold cyan]Epochs:[/bold cyan] [yellow]{NUM_EPOCHS}[/yellow]   "
                      f"[bold cyan]Batch:[/bold cyan] [yellow]{BATCH_SIZE}[/yellow]   "
                      f"[bold cyan]Hybrid Aug:[/bold cyan] [yellow]Speech × Spectrum (25 variants)[/yellow]   "
                      f"[bold cyan]SpecAugment:[/bold cyan] [yellow]F={SPEC_AUG_F} T={SPEC_AUG_T}[/yellow]"),
        post_eval_hook=_post,
        curves_path="training_curves_partD.png",
        loss_title="Loss per Epoch (Part D — Hybrid Aug.)",
        acc_title ="Accuracy per Epoch (Part D — Hybrid Aug.)",
        cm_path="confusion_matrix_partD.png", cm_cmap="Blues",
        cm_title="Confusion Matrix — Part D (Hybrid Augmentation)",
        weights_path="lenet5_partD_hybrid.pth",
        train_color="royalblue",
        report_title="Accuracy with hybrid (speech + spectrum) augmentation:",
    )

def validate_dirs():
    for d in [TRAIN_DIR, TEST_DIR]:
        if not os.path.isdir(d):
            console.print(f"[bold red]✗  Directory not found:[/bold red] {d}"); sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A","B","C","D","all"], default="all")
    args = parser.parse_args(); validate_dirs()

    results = {}
    part_map = {"A": run_part_a, "B": run_part_b, "C": run_part_c, "D": run_part_d}
    parts = ["A","B","C","D"] if args.part == "all" else [args.part]
    for p in parts:
        results[p] = part_map[p]()

    if args.part == "all":
        console.print(); console.rule("[bold bright_magenta]  Combined Summary — All Parts  ", style="bright_magenta")
        cmp = Table(box=box.ROUNDED, border_style="bright_magenta",
                    header_style="bold white on dark_magenta", show_lines=True)
        for col, sty, w in [("Part","bold white",8),("Augmentation","cyan",36),
                              ("Accuracy","yellow",12),("Training Time","cyan",18),
                              ("Testing Time","magenta",14)]:
            cmp.add_column(col, style=sty, min_width=w, justify="center")
        aug_desc = {
            "A": "None (baseline)",
            "B": "Speech (speed ± noise + pitch) + SpecAug",
            "C": "Spectrum (squeeze ±12% + noise) + SpecAug",
            "D": "Hybrid (speech × spectrum, 25 variants) + SpecAug",
        }
        for p, r in results.items():
            a_s = acc_color(r["final_acc"])
            cmp.add_row(f"Part {p}", aug_desc[p],
                        f"[{a_s}]{r['final_acc']:.1f}%[/{a_s}]",
                        f"{r['training_time_ms']:,.1f} ms",
                        f"{r['testing_time_ms']:,.1f} ms")
        console.print(cmp); console.print()


if __name__ == "__main__":
    main()