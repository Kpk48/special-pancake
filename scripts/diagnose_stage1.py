import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from waste_classifier.hierarchical.hierarchy import get_stage1_label, STAGE1_CLASSES

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OldStage1Model()
    model.load_state_dict(torch.load("artifacts/hierarchical/stage1.pt", map_location=device))
    model.to(device)
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    test_dataset = ImageFolder(root="data/final/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_paths = [path for path, _ in test_dataset.samples]
    all_targets = []
    all_preds = []
    all_confs = []
    
    with torch.no_grad():
        for images, targets3 in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            
            for t3 in targets3.cpu().numpy():
                class_name = test_dataset.classes[t3]
                target1 = get_stage1_label(class_name)
                all_targets.append(target1)
                
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    
    acc = np.mean(all_targets == all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    cm = confusion_matrix(all_targets, all_preds)
    
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Stage 1 Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(STAGE1_CLASSES))
    plt.xticks(tick_marks, STAGE1_CLASSES, rotation=45)
    plt.yticks(tick_marks, STAGE1_CLASSES)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("results/stage1_confusion_matrix.png", dpi=300)
    plt.close()
    
    wrong_indices = np.where(all_targets != all_preds)[0]
    wrong_confs = all_confs[wrong_indices]
    
    sorted_wrong_indices_of_wrong = np.argsort(wrong_confs)[::-1]
    sorted_wrong_indices = wrong_indices[sorted_wrong_indices_of_wrong]
    
    top_50_indices = sorted_wrong_indices[:50]
    
    with open("results/stage1_hard_errors.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "true_label", "predicted_label", "confidence"])
        for idx in top_50_indices:
            file_path = all_paths[idx]
            true_lbl = STAGE1_CLASSES[all_targets[idx]]
            pred_lbl = STAGE1_CLASSES[all_preds[idx]]
            conf = all_confs[idx]
            writer.writerow([file_path, true_lbl, pred_lbl, f"{conf:.6f}"])
            
    bio_total = np.sum(all_targets == 0)
    non_total = np.sum(all_targets == 1)
    
    bio_to_non = cm[0, 1]
    non_to_bio = cm[1, 0]
    
    print(f"Total Biodegradable: {bio_total}")
    print(f"Total Non-Biodegradable: {non_total}")
    print(f"Errors Biodegradable -> Non-Biodegradable: {bio_to_non} (Error rate: {bio_to_non / bio_total * 100:.2f}%)")
    print(f"Errors Non-Biodegradable -> Biodegradable: {non_to_bio} (Error rate: {non_to_bio / non_total * 100:.2f}%)")
    
if __name__ == "__main__":
    main()
