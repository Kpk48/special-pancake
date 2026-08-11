import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE2_CLASSES,
    STAGE3_CLASSES,
    get_stage1_label,
    get_stage2_label,
)

def plot_cm(cm, classes, title, filepath, cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
            
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    raw_test = ImageFolder(root="data/final/test", transform=transform)
    test_loader = DataLoader(raw_test, batch_size=64, shuffle=False, num_workers=0)
    classes = raw_test.classes
    
    s1_model = Stage1Model().to(device)
    s2_model = Stage2Model().to(device)
    s3_model = Stage3Model().to(device)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_model.load_state_dict(torch.load(os.path.join(root_dir, "artifacts", "hierarchical", "stage1_v2_relabeled.pt"), map_location=device))
    s2_model.load_state_dict(torch.load(os.path.join(root_dir, "artifacts", "hierarchical", "stage2.pt"), map_location=device))
    s3_model.load_state_dict(torch.load(os.path.join(root_dir, "artifacts", "hierarchical", "stage3.pt"), map_location=device))
    
    s1_model.eval()
    s2_model.eval()
    s3_model.eval()
    
    gt2_list = []
    gt3_list = []
    pred2_list = []
    pred3_list = []
    
    with torch.no_grad():
        for idx, (images, targets3) in enumerate(test_loader):
            images = images.to(device)
            out1 = s1_model(images)
            probs1 = torch.softmax(out1, dim=-1)[:, 1]
            pred1 = (probs1 >= 0.55).long()
            
            out2 = s2_model(images, pred1)
            pred2 = out2.argmax(dim=-1)
            
            out3 = s3_model(images, pred2)
            pred3 = out3.argmax(dim=-1)
            
            pred2_list.extend(pred2.cpu().numpy())
            pred3_list.extend(pred3.cpu().numpy())
            
            start_idx = idx * test_loader.batch_size
            end_idx = min(start_idx + test_loader.batch_size, len(raw_test))
            for i in range(start_idx, end_idx):
                c_name = classes[raw_test.targets[i]]
                gt2 = get_stage2_label(c_name)
                gt3 = STAGE3_CLASSES.index(c_name)
                gt2_list.append(gt2)
                gt3_list.append(gt3)
                
    gt2_arr = np.array(gt2_list)
    pred2_arr = np.array(pred2_list)
    gt3_arr = np.array(gt3_list)
    pred3_arr = np.array(pred3_list)
    
    cm2 = confusion_matrix(gt2_arr, pred2_arr, labels=list(range(6)))
    cm3 = confusion_matrix(gt3_arr, pred3_arr, labels=list(range(8)))
    
    cm2_path = os.path.join(root_dir, "results", "stage2_confusion_matrix.png")
    plot_cm(cm2, STAGE2_CLASSES, "Stage 2 Confusion Matrix (6 Coarse Categories)", cm2_path, cmap=plt.cm.Blues)
    
    cm3_path = os.path.join(root_dir, "results", "stage3_confusion_matrix.png")
    plot_cm(cm3, STAGE3_CLASSES, "Stage 3 Confusion Matrix (8 Fine-grained Classes)", cm3_path, cmap=plt.cm.Greens)
    
    p2, r2, f2, s2 = precision_recall_fscore_support(gt2_arr, pred2_arr, labels=list(range(6)), zero_division=0)
    p3, r3, f3, s3 = precision_recall_fscore_support(gt3_arr, pred3_arr, labels=list(range(8)), zero_division=0)
    
    stage2_metrics = []
    for idx, name in enumerate(STAGE2_CLASSES):
        stage2_metrics.append({
            "class": name, "precision": float(p2[idx]), "recall": float(r2[idx]), "f1": float(f2[idx]), "support": int(s2[idx])
        })
        
    stage3_metrics = []
    for idx, name in enumerate(STAGE3_CLASSES):
        stage3_metrics.append({
            "class": name, "precision": float(p3[idx]), "recall": float(r3[idx]), "f1": float(f3[idx]), "support": int(s3[idx])
        })
        
    weakest_s2 = sorted(stage2_metrics, key=lambda x: x["f1"])[:3]
    weakest_s3 = sorted(stage3_metrics, key=lambda x: x["f1"])[:3]
    
    md_lines = [
        "# Per-Class Metrics and Confusion Statistics",
        "",
        "This report breaks down model evaluation metrics for Stage 2 (6 coarse categories) and Stage 3 (8 fine-grained classes) evaluated on the held-out test split of 2,568 real images at calibrated Stage 1 threshold $t=0.55$.",
        "",
        "## 1. Stage 2 (Coarse Categories) Per-Class Metrics",
        "",
        "| Class Name | Precision | Recall | F1-Score | Support (Test Images) | Weakest Flag |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    
    weakest_s2_names = [x["class"] for x in weakest_s2]
    for row in stage2_metrics:
        flag = "⚠️ **Weakest**" if row["class"] in weakest_s2_names else "Normal"
        md_lines.append(f"| `{row['class']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} | {flag} |")
        
    md_lines.extend([
        "",
        "### Stage 2 Confusion Matrix",
        "![Stage 2 Confusion Matrix](file:///" + cm2_path.replace("\\", "/") + ")",
        "",
        "---",
        "",
        "## 2. Stage 3 (Fine-grained Target Classes) Per-Class Metrics",
        "",
        "| Class Name | Precision | Recall | F1-Score | Support (Test Images) | Weakest Flag |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    weakest_s3_names = [x["class"] for x in weakest_s3]
    for row in stage3_metrics:
        flag = "⚠️ **Weakest**" if row["class"] in weakest_s3_names else "Normal"
        md_lines.append(f"| `{row['class']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} | {flag} |")
        
    md_lines.extend([
        "",
        "### Stage 3 Confusion Matrix",
        "![Stage 3 Confusion Matrix](file:///" + cm3_path.replace("\\", "/") + ")",
        "",
        "---",
        "",
        "## 3. Weakest Performing Classes Analysis",
        "",
        "### Stage 2 Weakest Classes",
    ])
    
    for item in weakest_s2:
        md_lines.append(f"- **`{item['class']}`**: F1-Score = **{item['f1']:.4f}** (Precision = {item['precision']:.4f}, Recall = {item['recall']:.4f}, Support = {item['support']})")
        
    md_lines.extend([
        "",
        "### Stage 3 Weakest Classes",
    ])
    
    for item in weakest_s3:
        md_lines.append(f"- **`{item['class']}`**: F1-Score = **{item['f1']:.4f}** (Precision = {item['precision']:.4f}, Recall = {item['recall']:.4f}, Support = {item['support']})")
        
    out_md_path = os.path.join(root_dir, "results", "per_class_metrics.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print("\n".join(md_lines))

if __name__ == "__main__":
    main()
