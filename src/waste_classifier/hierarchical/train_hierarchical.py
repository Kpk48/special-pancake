"""Training coordinator for the Hierarchical CNN Waste Classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from .hierarchy import get_stage1_label, get_stage2_label
from .stage1_model import Stage1Model
from .stage2_model import Stage2Model
from .stage3_model import Stage3Model
from .loss import FocalLoss
from ..augment import TargetedAugmentedDataset

logger = logging.getLogger("train_hierarchical")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class HierarchicalDataset(Dataset):
    """Wrapper that maps Stage 3 ImageFolder targets to Stage 1 and Stage 2 labels."""

    def __init__(self, base_dataset: Dataset, classes: list[str]) -> None:
        self.base_dataset = base_dataset
        self.classes = classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        img, target3 = self.base_dataset[idx]
        class_name = self.classes[target3]
        target1 = get_stage1_label(class_name)
        target2 = get_stage2_label(class_name)
        return img, target1, target2, target3


def compute_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Computes inverse-frequency weights for class imbalance."""
    targets = []
    for i in range(len(dataset)):
        # Retrieve target of base dataset
        _, target = dataset[i]
        targets.append(int(target))
    
    counts = torch.zeros(num_classes)
    for t in targets:
        counts[t] += 1
    
    # Calculate inverse frequency
    total = len(targets)
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    return weights


def get_loss_fn(
    loss_type: str,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    """Instantiates the specified loss function."""
    if loss_type == "focal_loss":
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        return nn.CrossEntropyLoss(weight=alpha)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    stage: int,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets1, targets2, targets3 in loader:
        images = images.to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)

        optimizer.zero_grad()

        # Forward pass based on stage
        if stage == 1:
            outputs = model(images)
            loss = criterion(outputs, targets1)
        elif stage == 2:
            # Conditioned on Stage 1 ground truth (teacher forcing during train)
            outputs = model(images, targets1)
            loss = criterion(outputs, targets2)
        elif stage == 3:
            # Conditioned on Stage 2 ground truth
            outputs = model(images, targets2)
            loss = criterion(outputs, targets3)
        else:
            raise ValueError(f"Invalid stage: {stage}")

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage: int,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, targets1, targets2, targets3 in loader:
            images = images.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)

            if stage == 1:
                outputs = model(images)
                loss = criterion(outputs, targets1)
                preds = outputs.argmax(dim=-1)
                correct += preds.eq(targets1).sum().item()
            elif stage == 2:
                outputs = model(images, targets1)
                loss = criterion(outputs, targets2)
                preds = outputs.argmax(dim=-1)
                correct += preds.eq(targets2).sum().item()
            elif stage == 3:
                outputs = model(images, targets2)
                loss = criterion(outputs, targets3)
                preds = outputs.argmax(dim=-1)
                correct += preds.eq(targets3).sum().item()
            else:
                raise ValueError(f"Invalid stage: {stage}")

            running_loss += loss.item() * images.size(0)

    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hierarchical CNN Waste Classifier.")
    parser.add_argument("--data", default="data/final", help="Path to preprocessed dataset root.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per stage.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--loss-type", default="cross_entropy", choices=["cross_entropy", "focal_loss"])
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter.")
    parser.add_argument("--use-proto", action="store_true", help="Use prototypical metric learning head for Stage 3.")
    parser.add_argument("--augment-factor", type=float, default=2.0, help="Targeted augmentation multiplier.")
    parser.add_argument("--max-copies", type=int, default=8, help="Max copies per minority sample.")
    parser.add_argument("--model-dir", default="artifacts/hierarchical", help="Directory to save model checkpoints.")
    parser.add_argument("--max-samples-per-class", type=int, default=100, help="Maximum training samples per class.")
    parser.add_argument("--device", default="auto", help="Device to use: mps, cuda, cpu, auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)
    logger.info(f"Using training device: {device}")

    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Transforms (input dimensions: 128x128)
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # Do not use ToTensor here since TargetedAugmentedDataset handles conversion
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_root = Path(args.data)
    train_root = data_root / "train"
    val_root = data_root / "val"

    if not train_root.exists() or not val_root.exists():
        logger.error(f"Dataset splits not found at {args.data}. Make sure preprocess_pipeline.py has run.")
        return

    # Load raw datasets
    raw_train = ImageFolder(root=train_root, transform=image_transform, allow_empty=True)
    raw_val = ImageFolder(root=val_root, transform=val_test_transform, allow_empty=True)

    classes = raw_train.classes
    logger.info(f"Loaded {len(classes)} classes: {classes}")

    # Apply class subset filtering if configured
    if args.max_samples_per_class > 0:
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(raw_train.samples):
            class_indices[target].append(idx)
        
        subset_indices = []
        for target, indices in class_indices.items():
            subset_indices.extend(indices[:args.max_samples_per_class])
        
        # Override raw_train with a Subset wrapper that mimics ImageFolder attributes
        raw_train_subset = torch.utils.data.Subset(raw_train, subset_indices)
        raw_train_subset.classes = classes
        raw_train_subset.targets = [raw_train.targets[i] for i in subset_indices]
        raw_train_subset.samples = [raw_train.samples[i] for i in subset_indices]
        raw_train = raw_train_subset

        # Filter raw_val to speed up validation loops during training
        class_indices_val = defaultdict(list)
        for idx, (_, target) in enumerate(raw_val.samples):
            class_indices_val[target].append(idx)
        
        subset_indices_val = []
        for target, indices in class_indices_val.items():
            subset_indices_val.extend(indices[:args.max_samples_per_class])
        
        raw_val_subset = torch.utils.data.Subset(raw_val, subset_indices_val)
        raw_val_subset.classes = classes
        raw_val_subset.targets = [raw_val.targets[i] for i in subset_indices_val]
        raw_val_subset.samples = [raw_val.samples[i] for i in subset_indices_val]
        raw_val = raw_val_subset

    # Targeted Augmentations on Train
    logger.info("Initializing targeted minority-class data augmentations...")
    augmented_train = TargetedAugmentedDataset(
        base_dataset=raw_train,
        base_factor=args.augment_factor,
        max_copies=args.max_copies,
    )
    logger.info(f"Train dataset size before augmentation: {len(raw_train)}, after: {len(augmented_train)}")

    # Hierarchical wrapper
    train_dataset = HierarchicalDataset(augmented_train, classes=classes)
    val_dataset = HierarchicalDataset(raw_val, classes=classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize models
    stage1 = Stage1Model().to(device)
    stage2 = Stage2Model().to(device)
    stage3 = Stage3Model(use_prototypical=args.use_proto).to(device)

    # Calculate class weights for Stage 2 and Stage 3 based on frequencies in raw train set
    logger.info("Computing inverse-frequency class weights for loss functions...")
    # Map target index list to compute weights
    stage1_targets = [get_stage1_label(classes[t]) for t in raw_train.targets]
    stage2_targets = [get_stage2_label(classes[t]) for t in raw_train.targets]
    stage3_targets = raw_train.targets

    def calc_weights(targets: list[int], num_cls: int) -> torch.Tensor:
        counts = torch.zeros(num_cls)
        for t in targets:
            counts[t] += 1
        return len(targets) / (num_cls * torch.clamp(counts, min=1.0))

    s1_weights = calc_weights(stage1_targets, 2).to(device)
    s2_weights = calc_weights(stage2_targets, 6).to(device)
    s3_weights = calc_weights(stage3_targets, 11).to(device)

    # Train Stage 1
    logger.info("=== Training Stage 1 Model (Biodegradable/Non-biodegradable) ===")
    criterion1 = get_loss_fn(args.loss_type, alpha=s1_weights, gamma=args.gamma)
    optimizer1 = optim.Adam(stage1.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(stage1, train_loader, optimizer1, criterion1, device, stage=1)
        val_loss, val_acc = validate(stage1, val_loader, criterion1, device, stage=1)
        logger.info(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    torch.save(stage1.state_dict(), model_dir / "stage1.pt")
    logger.info("Stage 1 training complete. Model saved.")

    # Train Stage 2
    logger.info("=== Training Stage 2 Model (6 Coarse Categories) ===")
    criterion2 = get_loss_fn(args.loss_type, alpha=s2_weights, gamma=args.gamma)
    optimizer2 = optim.Adam(stage2.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_epoch(stage2, train_loader, optimizer2, criterion2, device, stage=2)
        val_loss, val_acc = validate(stage2, val_loader, criterion2, device, stage=2)
        logger.info(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(stage2.state_dict(), model_dir / "stage2.pt")
    logger.info("Stage 2 training complete. Model saved.")

    # Train Stage 3
    logger.info("=== Training Stage 3 Model (11 Fine-grained Classes) ===")
    criterion3 = get_loss_fn(args.loss_type, alpha=s3_weights, gamma=args.gamma)
    optimizer3 = optim.Adam(stage3.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_epoch(stage3, train_loader, optimizer3, criterion3, device, stage=3)
        val_loss, val_acc = validate(stage3, val_loader, criterion3, device, stage=3)
        logger.info(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(stage3.state_dict(), model_dir / "stage3.pt")
    logger.info("Stage 3 training complete. Model saved.")


if __name__ == "__main__":
    main()
