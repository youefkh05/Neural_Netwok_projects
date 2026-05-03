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


def build_model(variant: str) -> tuple[nn.Module, str]:
    """Factory returning (model, spec) for a given variant name.

    Variants:
    - baseline: original LeNet-5 adapted for 28x28
    - wide: increase filter counts in conv layers
    - deep: add an extra small conv layer before flattening
    - dropout: baseline + dropout before FC layers
    """
    if variant == "baseline":
        return LeNet5(), "LeNet5 baseline (6,16 filters)"

    if variant == "wide":
        class LeNetWide(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
                self.pool = nn.AvgPool2d(2, 2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
                self.fc1 = nn.Linear(32 * 4 * 4, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        return LeNetWide(), "wide: conv filters (16,32)"

    if variant == "deep":
        class LeNetDeep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
                self.pool = nn.AvgPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
                # extra conv layer (keeps spatial dims via padding=1)
                self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(32 * 4 * 4, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool(F.relu(self.conv1(x)))
                x = F.relu(self.conv2(x))
                x = self.pool(F.relu(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        return LeNetDeep(), "deep: added conv3 (3x3,pad=1) -> 32 maps"

    if variant == "sigmoid":
        class LeNetSigmoid(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
                self.pool = nn.AvgPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
                self.fc1 = nn.Linear(16 * 4 * 4, 120)
                # use Sigmoid activation (no dropout)
                self.act = nn.Sigmoid()
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool(self.act(self.conv1(x)))
                x = self.pool(self.act(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.act(self.fc1(x))
                x = self.act(self.fc2(x))
                return self.fc3(x)

        return LeNetSigmoid(), "sigmoid: Sigmoid activations, no dropout"

    raise ValueError(f"Unknown variant: {variant}")


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
    parser.add_argument("--variant", choices=["baseline","wide","deep","sigmoid","all"], default="all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build model/dataloaders only, then exit")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = get_dataset_paths()
    train_loader, test_loader = make_dataloaders(train_dir, test_dir, args.batch_size)

    device = torch.device("cpu")
    # If requested, run one or all variants and save comparative CSV
    variants = [args.variant] if args.variant != "all" else ["baseline","wide","deep","sigmoid"]
    results = []

    for var in variants:
        model, spec = build_model(var)
        model = model.to(device)
        if args.dry_run:
            print(f"Built model variant={var}: {spec}")
            continue

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        total_train_time = 0.0
        train_acc = test_acc = 0.0
        test_time = 0.0

        for epoch in range(1, args.epochs + 1):
            train_time = train_one_epoch(model, train_loader, device, optimizer, criterion, epoch)
            total_train_time += train_time

            train_acc, _        = evaluate(model, train_loader, device)
            test_acc, test_time = evaluate(model, test_loader,  device)

        print(f"Variant {var} epochs={args.epochs}: "
            f"train_acc={train_acc:.1f}%, "
            f"test_acc={test_acc:.1f}%, "
            f"total_train_time={total_train_time:.1f}s, "
            f"test_time={test_time:.1f}s")

        results.append({
            "variant": var,
            "spec": spec,
            "train_accuracy_percent": f"{train_acc:.1f}",
            "test_accuracy_percent": f"{test_acc:.1f}",
            "train_time_sec": f"{total_train_time:.1f}",
            "test_time_sec": f"{test_time:.1f}",
        })

    # save comparative CSV
    metrics_path = out / "lenet_variants_results.csv"
    import csv
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","spec","train_accuracy_percent","test_accuracy_percent","train_time_sec","test_time_sec"])
        w.writeheader()
        w.writerows(results)

if __name__ == "__main__":
    main()