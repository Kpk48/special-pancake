import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

from waste_classifier.hierarchical.hierarchy import get_stage1_label, STAGE1_CLASSES
from waste_classifier.hierarchical.loss import FocalLoss

class DSConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)

class DSConv2DBackbone(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.branch11x11 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch9x9 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dsconv1 = DSConvBlock(160, 128)
        self.dsconv2 = DSConvBlock(128, 64)
        self.dsconv3 = DSConvBlock(64, 32)
        self.dsconv4 = DSConvBlock(32, 16)
        self.flatten_dim = 16 * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b11 = self.branch11x11(x)
        b9 = self.branch9x9(x)
        b7 = self.branch7x7(x)
        b5 = self.branch5x5(x)
        b3 = self.branch3x3(x)
        out = torch.cat([b11, b9, b7, b5, b3], dim=1)
        out = self.pool(out)
        out = self.pool(self.dsconv1(out))
        out = self.pool(self.dsconv2(out))
        out = self.pool(self.dsconv3(out))
        out = self.pool(self.dsconv4(out))
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return self.relu(out)

class OldStage1Model(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

class Stage1DatasetWrapper(Dataset):
    def __init__(self, base_dataset, classes):
        self.base_dataset = base_dataset
        self.classes = classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target3 = self.base_dataset[idx]
        class_name = self.classes[target3]
        filepath = self.base_dataset.samples[idx][0]
        target1 = get_stage1_label(class_name, filepath)
        return img, target1

def get_source_tag(filepath):
    filename = os.path.basename(filepath)
    if filename.startswith("gc12_"):
        return "gc12_"
    elif filename.startswith("gcv2_"):
        return "gcv2_"
    elif filename.startswith("taco_"):
        return "taco_"
    else:
        return "unknown"

def calc_weights(targets, num_cls, device):
    counts = torch.zeros(num_cls)
    for t in targets:
        counts[t] += 1
    return (len(targets) / (num_cls * torch.clamp(counts, min=1.0))).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    raw_train = ImageFolder(root="data/final/train", transform=train_transform)
    raw_val = ImageFolder(root="data/final/val", transform=val_test_transform)
    raw_test = ImageFolder(root="data/final/test", transform=val_test_transform)
    
    classes = raw_train.classes
    
    train_dataset = Stage1DatasetWrapper(raw_train, classes)
    val_dataset = Stage1DatasetWrapper(raw_val, classes)
    test_dataset = Stage1DatasetWrapper(raw_test, classes)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    train_targets1 = [get_stage1_label(classes[raw_train.targets[idx]], raw_train.samples[idx][0]) for idx in range(len(raw_train))]
    s1_weights = calc_weights(train_targets1, 2, device)
    
    model = OldStage1Model()
    model.to(device)
    
    criterion = FocalLoss(alpha=s1_weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_model_path = "artifacts/hierarchical/stage1_v2_relabeled.pt"
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets1 in train_loader:
            images = images.to(device)
            targets1 = targets1.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets1)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, targets1 in val_loader:
                images = images.to(device)
                targets1 = targets1.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets1)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=-1)
                val_correct += preds.eq(targets1).sum().item()
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_correct = 0
    all_preds = []
    all_targets = []
    all_paths = [path for path, _ in raw_test.samples]
    all_t3_classes = [classes[t3] for _, t3 in raw_test.samples]
    
    with torch.no_grad():
        for images, targets1 in test_loader:
            images = images.to(device)
            targets1 = targets1.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=-1)
            test_correct += preds.eq(targets1).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets1.cpu().numpy())
            
    test_acc = test_correct / len(test_dataset)
    print(f"Custom Relabeled Test Accuracy: {test_acc * 100:.2f}%")
    
    class_stats = {}
    for t3 in classes:
        class_stats[t3] = {"total": 0, "errors": 0}
        
    source_stats = {"gc12_": {"total": 0, "errors": 0}, "gcv2_": {"total": 0, "errors": 0}, "taco_": {"total": 0, "errors": 0}, "unknown": {"total": 0, "errors": 0}}
    
    for idx in range(len(raw_test)):
        t3_cls = all_t3_classes[idx]
        file_path = all_paths[idx]
        true_s1 = all_targets[idx]
        pred_s1 = all_preds[idx]
        src_tag = get_source_tag(file_path)
        
        is_error = (true_s1 != pred_s1)
        
        class_stats[t3_cls]["total"] += 1
        if is_error:
            class_stats[t3_cls]["errors"] += 1
            
        source_stats[src_tag]["total"] += 1
        if is_error:
            source_stats[src_tag]["errors"] += 1
            
    with open("results/stage1_custom_relabeled_errors_by_class.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "total_images", "error_count", "error_rate"])
        for c_name in sorted(classes):
            stats = class_stats[c_name]
            rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            writer.writerow([c_name, stats["total"], stats["errors"], f"{rate:.6f}"])
            
    with open("results/stage1_custom_relabeled_errors_by_source.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_tag", "total_images", "error_count", "error_rate"])
        for s_tag in ["gc12_", "gcv2_", "taco_"]:
            stats = source_stats[s_tag]
            rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            writer.writerow([s_tag, stats["total"], stats["errors"], f"{rate:.6f}"])
            
if __name__ == "__main__":
    main()
