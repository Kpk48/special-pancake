"""Fine-tuning coordinator with layer-freezing capability for downstream adaptation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.train_hierarchical import HierarchicalDataset, get_loss_fn

logger = logging.getLogger("finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained Stage 3 Waste Classifier.")
    parser.add_argument("--data", default="data/demo_waste", help="Supplementary dataset folder (with train/val subdirs).")
    parser.add_argument("--pretrained-model", default="artifacts/hierarchical/stage3.pt", help="Path to pretrained stage3 checkpoint.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Fine-tuning learning rate (should be small).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--unfreeze-last-n", type=int, default=4, help="Number of parameter tensors to unfreeze from the end of the network.")
    parser.add_argument("--use-proto", action="store_true", help="Use prototypical head.")
    parser.add_argument("--output-model", default="artifacts/hierarchical/stage3_finetuned.pt", help="Path to save fine-tuned checkpoint.")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"Using fine-tuning device: {device}")

    # 1. Initialize and Load Pretrained Model
    model = Stage3Model(use_prototypical=args.use_proto).to(device)
    
    pretrained_path = Path(args.pretrained_model)
    if pretrained_path.exists():
        logger.info(f"Loading pretrained weights from {args.pretrained_model}...")
        # Map location to CPU if needed
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    else:
        logger.warning(f"Pretrained weights not found at {args.pretrained_model}. Training from scratch.")

    # 2. Freeze layers
    params = list(model.named_parameters())
    # Freeze everything first
    for name, param in params:
        param.requires_grad = False

    # Unfreeze last N layers
    unfreeze_n = args.unfreeze_last_n
    logger.info(f"Freezing all layers except the last {unfreeze_n} parameter tensors:")
    for name, param in params[-unfreeze_n:]:
        param.requires_grad = True
        logger.info(f"  Unfrozen: {name} | shape: {list(param.shape)}")

    # 3. Load supplementary dataset
    data_path = Path(args.data)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    # Fallback to direct directory if splits are not present
    if not train_dir.exists():
        train_dir = data_path
        val_dir = data_path
        logger.info(f"Splits not found, loading directly from dataset root: {data_path}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    try:
        raw_train = ImageFolder(root=train_dir, transform=transform, allow_empty=True)
        raw_val = ImageFolder(root=val_dir, transform=transform, allow_empty=True)
        classes = raw_train.classes
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    train_dataset = HierarchicalDataset(raw_train, classes=classes)
    val_dataset = HierarchicalDataset(raw_val, classes=classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Optimizer & Loss
    # Filter only parameters requiring gradients to save memory and avoid issues
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, _, targets2, targets3 in train_loader:
            images = images.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)

            optimizer.zero_grad()
            outputs = model(images, targets2)
            loss = criterion(outputs, targets3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, _, targets2, targets3 in val_loader:
                images = images.to(device)
                targets2 = targets2.to(device)
                targets3 = targets3.to(device)
                outputs = model(images, targets2)
                preds = outputs.argmax(dim=-1)
                correct += preds.eq(targets3).sum().item()
        val_acc = correct / len(val_dataset)

        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 5. Save Model
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Fine-tuned model checkpoint saved to {args.output_model}")


if __name__ == "__main__":
    main()
