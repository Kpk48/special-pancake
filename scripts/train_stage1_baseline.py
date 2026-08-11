import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from waste_classifier.hierarchical.hierarchy import get_stage1_label, STAGE1_CLASSES
from waste_classifier.hierarchical.loss import FocalLoss

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
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
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
    
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    
    criterion = FocalLoss(alpha=s1_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_model_path = "artifacts/resnet18_stage1_v2_relabeled.pt"
    os.makedirs("artifacts", exist_ok=True)
    
    epochs = 12
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
    print(f"ResNet18 Relabeled Test Accuracy: {test_acc * 100:.2f}%")
    
    pred_biodegradable_count = sum(1 for p in all_preds if p == 0)
    pred_non_biodegradable_count = sum(1 for p in all_preds if p == 1)
    print(f"Predictions - Biodegradable: {pred_biodegradable_count}, Non-Biodegradable: {pred_non_biodegradable_count}")
    
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
            
    os.makedirs("results", exist_ok=True)
    
    with open("results/stage1_resnet18_relabeled_errors_by_class.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "total_images", "error_count", "error_rate"])
        for c_name in sorted(classes):
            stats = class_stats[c_name]
            rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            writer.writerow([c_name, stats["total"], stats["errors"], f"{rate:.6f}"])
            
    with open("results/stage1_resnet18_relabeled_errors_by_source.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_tag", "total_images", "error_count", "error_rate"])
        for s_tag in ["gc12_", "gcv2_", "taco_"]:
            stats = source_stats[s_tag]
            rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            writer.writerow([s_tag, stats["total"], stats["errors"], f"{rate:.6f}"])
            
if __name__ == "__main__":
    main()
