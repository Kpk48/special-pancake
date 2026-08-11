"""Training coordinator for the Hierarchical CNN Waste Classifier.

Improvements over v1:
  1. MobileNetV3-Small pretrained backbone (via backbone.py update)
  2. 224×224 input resolution
  3. Best-checkpoint saving (saves on lowest val_loss, not final epoch)
  4. WeightedRandomSampler — inverse-frequency sampling per class
  5. CosineAnnealingLR — lr decays from initial to lr/100 over all epochs
  6. MixUp augmentation (alpha=0.4) applied in training loop
  7. Focal loss gamma raised to 3.0 for Stage 3 (from 2.0)
  8. 25 epochs (from 15)
  9. Gradual backbone unfreeze: all backbone layers unfrozen after epoch 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalDataset(Dataset):
    """Wrapper that maps Stage 3 ImageFolder targets to Stage 1 and Stage 2 labels.

    Caches all tensors in RAM on construction to avoid repeated disk I/O.
    """

    def __init__(self, base_dataset: Dataset, classes: list[str]) -> None:
        self.classes = classes
        self.cached_samples: list[tuple[torch.Tensor, int, int, int]] = []

        for i in range(len(base_dataset)):
            img, target3 = base_dataset[i]
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            class_name = self.classes[target3]
            filepath = None
            if hasattr(base_dataset, "base_dataset"):
                orig_idx = base_dataset.indices_map[i][0]
                filepath = base_dataset.base_dataset.samples[orig_idx][0]
            elif hasattr(base_dataset, "samples"):
                filepath = base_dataset.samples[i][0]
            target1 = get_stage1_label(class_name, filepath)
            target2 = get_stage2_label(class_name)
            self.cached_samples.append((img, target1, target2, target3))

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        return self.cached_samples[idx]


def build_weighted_sampler(targets3: list[int], classes: list[str]) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler so each class is sampled proportionally."""
    from collections import Counter
    counts = Counter(targets3)
    n = len(targets3)
    # Weight per sample = 1 / count_of_its_class
    weights = [1.0 / counts[t] for t in targets3]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=n,
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MixUp
# ──────────────────────────────────────────────────────────────────────────────

def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp to a batch. Returns (mixed_images, targets_a, targets_b, lam)."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, targets, targets[idx], lam


def mixup_criterion(
    criterion: nn.Module,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)


# ──────────────────────────────────────────────────────────────────────────────
# Loss factory
# ──────────────────────────────────────────────────────────────────────────────

def get_loss_fn(
    loss_type: str,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    if loss_type == "focal_loss":
        return FocalLoss(alpha=alpha, gamma=gamma)
    return nn.CrossEntropyLoss(weight=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    stage: int,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets1, targets2, targets3 in loader:
        images   = images.to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)

        optimizer.zero_grad()

        if stage == 1:
            if use_mixup:
                mixed, ta, tb, lam = mixup_batch(images, targets1, mixup_alpha)
                outputs = model(mixed)
                loss = mixup_criterion(criterion, outputs, ta, tb, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets1)

        elif stage == 2:
            if use_mixup:
                mixed, ta, tb, lam = mixup_batch(images, targets2, mixup_alpha)
                outputs = model(mixed, targets1)      # use GT stage1 for conditioning
                loss = mixup_criterion(criterion, outputs, ta, tb, lam)
            else:
                outputs = model(images, targets1)
                loss = criterion(outputs, targets2)

        elif stage == 3:
            if use_mixup:
                mixed, ta, tb, lam = mixup_batch(images, targets3, mixup_alpha)
                outputs = model(mixed, targets2)
                loss = mixup_criterion(criterion, outputs, ta, tb, lam)
            else:
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
            images   = images.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)

            if stage == 1:
                outputs = model(images)
                loss    = criterion(outputs, targets1)
                preds   = outputs.argmax(dim=-1)
                correct += preds.eq(targets1).sum().item()
            elif stage == 2:
                outputs = model(images, targets1)
                loss    = criterion(outputs, targets2)
                preds   = outputs.argmax(dim=-1)
                correct += preds.eq(targets2).sum().item()
            elif stage == 3:
                outputs = model(images, targets2)
                loss    = criterion(outputs, targets3)
                preds   = outputs.argmax(dim=-1)
                correct += preds.eq(targets3).sum().item()
            else:
                raise ValueError(f"Invalid stage: {stage}")

            running_loss += loss.item() * images.size(0)

    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters (called after warm-up epochs)."""
    unfrozen = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen += 1
    if unfrozen:
        logger.info("Unfroze %d backbone parameters for full fine-tuning.", unfrozen)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hierarchical CNN Waste Classifier.")
    parser.add_argument("--data",        default="data/final",            help="Preprocessed dataset root.")
    parser.add_argument("--epochs",      type=int,   default=25,          help="Epochs per stage.")
    parser.add_argument("--batch-size",  type=int,   default=64,          help="Batch size.")
    parser.add_argument("--lr",          type=float, default=0.001,       help="Initial learning rate.")
    parser.add_argument("--loss-type",   default="focal_loss",            choices=["cross_entropy", "focal_loss"])
    parser.add_argument("--gamma",       type=float, default=2.0,         help="Focal loss gamma (Stage 1/2).")
    parser.add_argument("--gamma-s3",    type=float, default=3.0,         help="Focal loss gamma (Stage 3, stronger).")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,         help="MixUp alpha (0 to disable).")
    parser.add_argument("--unfreeze-epoch", type=int, default=10,         help="Epoch at which backbone is fully unfrozen.")
    parser.add_argument("--use-proto",   action="store_true",             help="Use prototypical head for Stage 3.")
    parser.add_argument("--augment-factor", type=float, default=0.0,      help="Targeted augmentation multiplier.")
    parser.add_argument("--max-copies",  type=int,   default=0,           help="Max augmented copies per minority sample.")
    parser.add_argument("--model-dir",   default="artifacts/hierarchical", help="Directory to save checkpoints.")
    parser.add_argument("--device",      default="auto",                  help="Device: cuda, cpu, auto.")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)
    logger.info("Using training device: %s", device)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Transforms (224×224 for MobileNetV3) ──────────────────────────────
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        # No ToTensor here — TargetedAugmentedDataset handles it
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    data_root = Path(args.data)
    train_root = data_root / "train"
    val_root   = data_root / "val"

    if not train_root.exists() or not val_root.exists():
        logger.error("Dataset splits not found at %s. Run preprocess_pipeline.py first.", args.data)
        return

    # ── Load raw datasets ─────────────────────────────────────────────────
    raw_train = ImageFolder(root=str(train_root), transform=train_transform, allow_empty=True)
    raw_val   = ImageFolder(root=str(val_root),   transform=val_transform,   allow_empty=True)

    classes = raw_train.classes
    logger.info("Loaded %d classes: %s", len(classes), classes)

    # ── Targeted augmentation on minority classes ─────────────────────────
    logger.info("Initializing targeted minority-class data augmentations...")
    augmented_train = TargetedAugmentedDataset(
        base_dataset=raw_train,
        base_factor=args.augment_factor,
        max_copies=args.max_copies,
    )
    logger.info("Train dataset size before augmentation: %d, after: %d",
                len(raw_train), len(augmented_train))

    # ── HierarchicalDataset (caches all tensors in RAM) ──────────────────
    logger.info("Caching training set into RAM (this may take a moment)...")
    train_dataset = HierarchicalDataset(augmented_train, classes=classes)
    logger.info("Caching validation set into RAM ...")
    val_dataset   = HierarchicalDataset(raw_val, classes=classes)

    # ── WeightedRandomSampler ─────────────────────────────────────────────
    targets3_list = [s[3] for s in train_dataset.cached_samples]
    sampler = build_weighted_sampler(targets3_list, classes)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Normalizer to apply after MixUp (must be on-device) ──────────────
    # We apply Normalize inside the model forward or as a separate module
    # to allow MixUp to operate on unnormalized float tensors.
    normalizer = transforms.Normalize(mean=imagenet_mean, std=imagenet_std).to(device) \
        if hasattr(transforms.Normalize, "to") else None

    # ── Models ────────────────────────────────────────────────────────────
    stage1 = Stage1Model().to(device)
    stage2 = Stage2Model().to(device)
    stage3 = Stage3Model(use_prototypical=args.use_proto, num_classes=len(classes)).to(device)

    # ── Class weights ─────────────────────────────────────────────────────
    logger.info("Computing inverse-frequency class weights ...")

    def calc_weights(targets: list[int], num_cls: int) -> torch.Tensor:
        counts = torch.zeros(num_cls)
        for t in targets:
            counts[t] += 1
        return (len(targets) / (num_cls * torch.clamp(counts, min=1.0))).to(device)

    s1_targets = [get_stage1_label(classes[raw_train.targets[idx]], raw_train.samples[idx][0]) for idx in range(len(raw_train))]
    s2_targets = [get_stage2_label(classes[t]) for t in raw_train.targets]
    s3_targets = list(raw_train.targets)

    s1_weights = calc_weights(s1_targets, 2)
    s2_weights = calc_weights(s2_targets, 6)
    s3_weights = calc_weights(s3_targets, len(classes))

    # ── Training loop helper ──────────────────────────────────────────────
    def train_stage(
        stage_id: int,
        model: nn.Module,
        criterion: nn.Module,
        label: str,
        gamma_val: float,
    ) -> None:
        logger.info("=== Training %s ===", label)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr / 100
        )

        best_val_loss = float("inf")
        best_ckpt = model_dir / f"stage{stage_id}.pt"

        for epoch in range(1, args.epochs + 1):
            # Gradual unfreeze after warm-up
            if epoch == args.unfreeze_epoch:
                unfreeze_backbone(model)
                # Re-init optimizer to include newly unfrozen params
                optimizer = optim.AdamW(
                    model.parameters(), lr=args.lr / 10, weight_decay=1e-4
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - epoch,
                    eta_min=args.lr / 1000,
                )
                logger.info("Epoch %02d: Backbone fully unfrozen, lr reset to %.6f", epoch, args.lr / 10)

            t_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                stage=stage_id,
                use_mixup=(args.mixup_alpha > 0),
                mixup_alpha=args.mixup_alpha,
            )
            v_loss, v_acc = validate(model, val_loader, criterion, device, stage=stage_id)
            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | LR: %.6f",
                epoch, args.epochs, t_loss, v_loss, v_acc, lr_now,
            )

            # Save best checkpoint
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), best_ckpt)
                logger.info("  -> New best checkpoint saved (val_loss=%.4f)", best_val_loss)

        logger.info("%s training complete. Best val_loss=%.4f. Checkpoint: %s", label, best_val_loss, best_ckpt)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    criterion1 = get_loss_fn(args.loss_type, alpha=s1_weights, gamma=args.gamma)
    train_stage(1, stage1, criterion1, "Stage 1 Model (Biodegradable/Non-biodegradable)", args.gamma)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    criterion2 = get_loss_fn(args.loss_type, alpha=s2_weights, gamma=args.gamma)
    train_stage(2, stage2, criterion2, "Stage 2 Model (6 Coarse Categories)", args.gamma)

    # ── Stage 3 (stronger gamma) ──────────────────────────────────────────
    criterion3 = get_loss_fn(args.loss_type, alpha=s3_weights, gamma=args.gamma_s3)
    train_stage(3, stage3, criterion3, f"Stage 3 Model ({len(classes)} Fine-grained Classes)", args.gamma_s3)


if __name__ == "__main__":
    main()
