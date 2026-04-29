from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import kagglehub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_dataset_paths() -> tuple[Path, Path]:
    dataset_root = Path(kagglehub.dataset_download("mohamedgamal07/reduced-mnist"))
    base = dataset_root / "Reduced MNIST Data"
    return base / "Reduced Trainging data", base / "Reduced Testing data"


class LeNet5(nn.Module):
    """LeNet-5 adapted for 28x28 grayscale input (Option B: no padding).

    Spatial flow:
        28x28 -> conv1(5x5) -> 24x24x6  -> pool -> 12x12x6
              -> conv2(5x5) -> 8x8x16   -> pool -> 4x4x16
              -> flatten(256) -> fc1 -> 120 -> fc2 -> 84 -> fc3 -> 10
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)         # 28->24
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)  # halves spatial dims
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # 12->8
        # After two pools: 4x4x16 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 28->24->12
        x = self.pool(F.relu(self.conv2(x)))  # 12->8->4
        x = x.view(x.size(0), -1)             # flatten: 4x4x16=256
        x = F.relu(self.fc1(x))               # 256->120
        x = F.relu(self.fc2(x))               # 120->84
        x = self.fc3(x)                        # 84->10 (logits, no softmax)
        return x


def make_dataloaders(
    train_dir: Path, test_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_ds = ImageFolder(str(train_dir), transform=transform)
    test_ds  = ImageFolder(str(test_dir),  transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
) -> float:
    model.train()
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - t0


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total   += y.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="LeNet-5 baseline on ReducedMNIST")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs_lenet"))
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build model/dataloaders only, then exit")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = get_dataset_paths()
    train_loader, test_loader = make_dataloaders(train_dir, test_dir, args.batch_size)

    device = torch.device("cpu")
    model = LeNet5().to(device)

    if args.dry_run:
        print("Built LeNet5 model and dataloaders")
        print(model)
        print(f"Train samples: {len(train_loader.dataset)}, "
              f"Test samples: {len(test_loader.dataset)}")
        return

    # ← Fixed: Adam instead of SGD
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0.0
    train_acc = 0.0
    test_acc = 0.0
    test_time = 0.0

    for epoch in range(1, args.epochs + 1):
        train_time = train_one_epoch(model, train_loader, device, optimizer, criterion, epoch)
        total_train_time += train_time

        train_acc, _        = evaluate(model, train_loader, device)
        test_acc, test_time = evaluate(model, test_loader,  device)

        print(f"Epoch {epoch:02d}: "
              f"train_acc={train_acc:.1f}%, "
              f"test_acc={test_acc:.1f}%, "
              f"epoch_train_time={train_time:.1f}s, "
              f"test_time={test_time:.1f}s")

    metrics_path = out / "lenet_baseline_results.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("model,epochs,train_accuracy_percent,test_accuracy_percent,train_time_sec,test_time_sec\n")
        f.write(
            f"LeNet5,{args.epochs},{train_acc:.1f},{test_acc:.1f},"
            f"{total_train_time:.1f},{test_time:.1f}\n"
        )

if __name__ == "__main__":
    main()