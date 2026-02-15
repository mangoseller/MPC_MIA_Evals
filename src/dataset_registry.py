"""
Dataset registry for MIA experiments.

Each dataset is described by a DatasetSpec that includes metadata (num_classes,
input shape), transform pipelines, and a load function. The runner iterates
over requested datasets and uses the spec to configure the experiment.

Supported datasets:
    cifar10   — CIFAR-10  (10 classes, 32×32 RGB)
    cifar100  — CIFAR-100 (100 classes, 32×32 RGB)
    pathmnist — PathMNIST (9 classes, 28×28→32×32 RGB, colon histopathology)
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable, Any
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ═══════════════════════════════════════════════════════════════════════════
# Spec
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    input_channels: int
    input_size: int                         # spatial dim (square assumed)
    train_transform: transforms.Compose
    test_transform: transforms.Compose
    load_fn: Callable[..., tuple[Dataset, Dataset]]  # → (train, test)
    applicable_archs: list[str] = field(default_factory=lambda: ["CNN", "MLP", "LeNet"])


# ═══════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════

def _load_cifar10(data_dir: str, train_transform, test_transform):
    train = datasets.CIFAR10(root=data_dir, train=True,  download=True,  transform=train_transform)
    test  = datasets.CIFAR10(root=data_dir, train=False, download=True,  transform=test_transform)
    return train, test


def _load_cifar100(data_dir: str, train_transform, test_transform):
    train = datasets.CIFAR100(root=data_dir, train=True,  download=True,  transform=train_transform)
    test  = datasets.CIFAR100(root=data_dir, train=False, download=True,  transform=test_transform)
    return train, test


class _MedMNISTWrapper(Dataset):
    """Thin wrapper so labels are scalar ints (MedMNIST returns shape [1])."""

    def __init__(self, medmnist_ds):
        self.ds = medmnist_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, int(label.squeeze())


def _load_pathmnist(data_dir: str, train_transform, test_transform):
    try:
        from medmnist import PathMNIST         # type: ignore
    except ImportError:
        raise RuntimeError(
            "PathMNIST requires the `medmnist` package.\n"
            "Install with:  pip install medmnist"
        )
    raw_train = PathMNIST(split="train", download=True, transform=train_transform, root=data_dir)
    raw_test  = PathMNIST(split="test",  download=True, transform=test_transform,  root=data_dir)
    return _MedMNISTWrapper(raw_train), _MedMNISTWrapper(raw_test)


# ═══════════════════════════════════════════════════════════════════════════
# Transform factories
# ═══════════════════════════════════════════════════════════════════════════

def _cifar10_transforms():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, test


def _cifar100_transforms():
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, test


def _pathmnist_transforms():
    mean, std = (0.5,), (0.5,)  # generic; works well for MedMNIST
    train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean * 3, std * 3),  # 3-channel
    ])
    test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean * 3, std * 3),
    ])
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

def _build_specs() -> dict[str, DatasetSpec]:
    c10_tr, c10_te = _cifar10_transforms()
    c100_tr, c100_te = _cifar100_transforms()
    pm_tr, pm_te = _pathmnist_transforms()

    return {
        "cifar10": DatasetSpec(
            name="cifar10", num_classes=10,
            input_channels=3, input_size=32,
            train_transform=c10_tr, test_transform=c10_te,
            load_fn=_load_cifar10,
        ),
        "cifar100": DatasetSpec(
            name="cifar100", num_classes=100,
            input_channels=3, input_size=32,
            train_transform=c100_tr, test_transform=c100_te,
            load_fn=_load_cifar100,
        ),
        "pathmnist": DatasetSpec(
            name="pathmnist", num_classes=9,
            input_channels=3, input_size=32,
            train_transform=pm_tr, test_transform=pm_te,
            load_fn=_load_pathmnist,
        ),
    }

DATASETS: dict[str, DatasetSpec] = _build_specs()

# Aliases for forgiving CLI parsing
_ALIASES = {
    "cifar-10":  "cifar10",
    "cifar_10":  "cifar10",
    "cifar-100": "cifar100",
    "cifar_100": "cifar100",
    "path":      "pathmnist",
}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASETS[key]


def list_datasets() -> list[str]:
    return sorted(DATASETS.keys())
