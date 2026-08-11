import sys
import os
import time
import psutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES,
    STAGE3_TO_STAGE1, STAGE3_TO_STAGE2,
    get_stage1_label, get_stage2_label
)
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correct Transform: NO ImageNet normalization for custom DSConv models (trained on [0, 1] inputs)
dsconv_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ImageNet normalization transform for ResNet18 transfer baseline
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

raw_test_dsconv = ImageFolder(root="data/final/test", transform=dsconv_transform)
raw_test_resnet = ImageFolder(root="data/final/test", transform=resnet_transform)
classes = raw_test_dsconv.classes  # ['battery', 'cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile']

class CorrectedDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.classes = base_dataset.classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target3 = self.base_dataset[idx]
        class_name = self.classes[target3]
        filepath = self.base_dataset.samples[idx][0]
        target1_orig = get_stage1_label(class_name, None)
        target1_rel  = get_stage1_label(class_name, filepath)
        target2      = get_stage2_label(class_name)
        target3_idx  = STAGE3_CLASSES.index(class_name)
        return img, target1_orig, target1_rel, target2, target3_idx, filepath

test_ds_dsconv = CorrectedDatasetWrapper(raw_test_dsconv)
test_ds_resnet = CorrectedDatasetWrapper(raw_test_resnet)

test_loader_dsconv = DataLoader(test_ds_dsconv, batch_size=64, shuffle=False, num_workers=0)
test_loader_resnet = DataLoader(test_ds_resnet, batch_size=64, shuffle=False, num_workers=0)

# Load Models
s1_orig = Stage1Model().to(device)
s1_orig.load_state_dict(torch.load("artifacts/hierarchical/stage1.pt", map_location=device))
s1_orig.eval()

s1_rel = Stage1Model().to(device)
s1_rel.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
s1_rel.eval()

s2_model = Stage2Model().to(device)
s2_model.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
s2_model.eval()

s3_model = Stage3Model().to(device)
s3_model.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
s3_model.eval()

resnet_s1 = resnet18(weights=None)
resnet_s1.fc = nn.Linear(resnet_s1.fc.in_features, 2)
resnet_s1.load_state_dict(torch.load("artifacts/resnet18_stage1_v2_relabeled.pt", map_location=device))
resnet_s1.to(device)
resnet_s1.eval()

print("Loaded all models with corrected data pipelines.")

def get_source_tag(filepath):
    fname = os.path.basename(filepath)
    if fname.startswith("gc12_"):
        return "gc12_"
    elif fname.startswith("gcv2_"):
        return "gcv2_"
    elif fname.startswith("taco_"):
        return "taco_"
    else:
        return "other"

all_t1_orig, all_t1_rel, all_t2, all_t3, all_paths = [], [], [], [], []
p1_a_preds, p2_a_preds, p3_a_preds = [], [], []
p1_b_preds, p2_b_preds, p3_b_preds = [], [], []
p1_b_probs = []
all_resnet_preds = []

t_calibrated = 0.55

with torch.no_grad():
    for imgs, t1_o, t1_r, t2, t3, paths in test_loader_dsconv:
        imgs = imgs.to(device)
        
        # Pipeline A (Canonical Pre-Relabeled Baseline: stage1.pt, argmax t=0.50)
        out1_a = s1_orig(imgs)
        pred1_a = out1_a.argmax(dim=-1)
        out2_a = s2_model(imgs, pred1_a)
        pred2_a = out2_a.argmax(dim=-1)
        out3_a = s3_model(imgs, pred2_a)
        pred3_a = out3_a.argmax(dim=-1)
        
        # Pipeline B (Phase 4 Relabeled Baseline: stage1_v2_relabeled.pt, t=0.55)
        out1_b = s1_rel(imgs)
        probs1_b = torch.softmax(out1_b, dim=-1)
        p1_b_nonbio = probs1_b[:, 1]
        pred1_b = (p1_b_nonbio >= t_calibrated).long()
        out2_b = s2_model(imgs, pred1_b)
        pred2_b = out2_b.argmax(dim=-1)
        out3_b = s3_model(imgs, pred2_b)
        pred3_b = out3_b.argmax(dim=-1)
        
        all_t1_orig.extend(t1_o.numpy())
        all_t1_rel.extend(t1_r.numpy())
        all_t2.extend(t2.numpy())
        all_t3.extend(t3.numpy())
        all_paths.extend(paths)
        
        p1_a_preds.extend(pred1_a.cpu().numpy())
        p2_a_preds.extend(pred2_a.cpu().numpy())
        p3_a_preds.extend(pred3_a.cpu().numpy())
        
        p1_b_preds.extend(pred1_b.cpu().numpy())
        p2_b_preds.extend(pred2_b.cpu().numpy())
        p3_b_preds.extend(pred3_b.cpu().numpy())
        p1_b_probs.extend(p1_b_nonbio.cpu().numpy())
        
    for imgs, t1_o, t1_r, t2, t3, paths in test_loader_resnet:
        imgs = imgs.to(device)
        out_r = resnet_s1(imgs)
        pred_r = out_r.argmax(dim=-1)
        all_resnet_preds.extend(pred_r.cpu().numpy())

all_t1_orig = np.array(all_t1_orig)
all_t1_rel  = np.array(all_t1_rel)
all_t2      = np.array(all_t2)
all_t3      = np.array(all_t3)

p1_a_preds = np.array(p1_a_preds)
p2_a_preds = np.array(p2_a_preds)
p3_a_preds = np.array(p3_a_preds)

p1_b_preds = np.array(p1_b_preds)
p2_b_preds = np.array(p2_b_preds)
p3_b_preds = np.array(p3_b_preds)
p1_b_probs = np.array(p1_b_probs)

all_resnet_preds = np.array(all_resnet_preds)

print("\n=================================================================")
print("PIPELINE A: CANONICAL PAPER TABLE II BASELINE")
print("  - Checkpoint: stage1.pt + stage2.pt + stage3.pt")
print("  - GT: Original hierarchy (pre-relabeled)")
print("  - Stage 1 Decision: default argmax (t=0.50)")
print("=================================================================")

source_tags = np.array([get_source_tag(p) for p in all_paths])
unique_sources = sorted(list(set(source_tags)))

print("\n--- Pipeline A Domain-Shift Breakdown ---")
for src in unique_sources:
    mask = (source_tags == src)
    cnt = mask.sum()
    s1_acc = accuracy_score(all_t1_orig[mask], p1_a_preds[mask])
    s2_acc = accuracy_score(all_t2[mask], p2_a_preds[mask])
    s3_acc = accuracy_score(all_t3[mask], p3_a_preds[mask])
    print(f"Source: {src:<8} | Count: {cnt:<5} | Stage 1: {s1_acc*100:.2f}% | Stage 2: {s2_acc*100:.2f}% | Stage 3: {s3_acc*100:.2f}%")

pooled_s1_a = accuracy_score(all_t1_orig, p1_a_preds)
pooled_s2_a = accuracy_score(all_t2, p2_a_preds)
pooled_s3_a = accuracy_score(all_t3, p3_a_preds)
print(f"Pooled Total | Count: {len(all_t1_orig):<5} | Stage 1: {pooled_s1_a*100:.2f}% (Target: 79.52%) | Stage 2: {pooled_s2_a*100:.2f}% (Target: 66.98%) | Stage 3: {pooled_s3_a*100:.2f}% (Target: 63.40%)")

print("\n--- Pipeline A Stage 2 Confusion Matrix (6x6) ---")
s2_classes = ["paper_cardboard", "organic", "glass", "metal", "plastic", "textile_battery"]
cm_s2_a = confusion_matrix(all_t2, p2_a_preds)
print(cm_s2_a)

p_s2_a, r_s2_a, f1_s2_a, sup_s2_a = precision_recall_fscore_support(all_t2, p2_a_preds)
print("\nStage 2 Per-Class Metrics (Pipeline A):")
for idx, cname in enumerate(s2_classes):
    print(f"  {cname:<18} | Precision: {p_s2_a[idx]:.4f} | Recall: {r_s2_a[idx]:.4f} | F1: {f1_s2_a[idx]:.4f} | Support: {sup_s2_a[idx]}")

print("\n--- Pipeline A Stage 3 Confusion Matrix (8x8) ---")
cm_s3_a = confusion_matrix(all_t3, p3_a_preds)
print(cm_s3_a)

p_s3_a, r_s3_a, f1_s3_a, sup_s3_a = precision_recall_fscore_support(all_t3, p3_a_preds)
print("\nStage 3 Per-Class Metrics (Pipeline A):")
for idx, cname in enumerate(STAGE3_CLASSES):
    print(f"  {cname:<12} | Precision: {p_s3_a[idx]:.4f} | Recall: {r_s3_a[idx]:.4f} | F1: {f1_s3_a[idx]:.4f} | Support: {sup_s3_a[idx]}")

off_diag_s3_a = []
for i in range(8):
    for j in range(8):
        if i != j and cm_s3_a[i, j] > 0:
            off_diag_s3_a.append((cm_s3_a[i, j], STAGE3_CLASSES[i], STAGE3_CLASSES[j]))
off_diag_s3_a.sort(reverse=True)
print("\nTop 10 Stage 3 Misclassifications (Pipeline A):")
for cnt, true_c, pred_c in off_diag_s3_a[:10]:
    print(f"  True: {true_c:<10} -> Pred: {pred_c:<10} | Count: {cnt} ({cnt/len(all_t3)*100:.2f}%)")


print("\n=================================================================")
print("PIPELINE B: PHASE 4 POST-RELABELED BASELINE")
print("  - Checkpoint: stage1_v2_relabeled.pt + stage2.pt + stage3.pt")
print("  - GT: Post-relabeled hierarchy (with Phase 4 overrides)")
print("  - Stage 1 Decision: calibrated threshold t=0.55")
print("=================================================================")

print("\n--- Pipeline B Domain-Shift Breakdown ---")
for src in unique_sources:
    mask = (source_tags == src)
    cnt = mask.sum()
    s1_acc = accuracy_score(all_t1_rel[mask], p1_b_preds[mask])
    s1_resnet_acc = accuracy_score(all_t1_rel[mask], all_resnet_preds[mask])
    s2_acc = accuracy_score(all_t2[mask], p2_b_preds[mask])
    s3_acc = accuracy_score(all_t3[mask], p3_b_preds[mask])
    print(f"Source: {src:<8} | Count: {cnt:<5} | S1 (DSConv t=0.55): {s1_acc*100:.2f}% | S1 (ResNet18): {s1_resnet_acc*100:.2f}% | Stage 2: {s2_acc*100:.2f}% | Stage 3: {s3_acc*100:.2f}%")

pooled_s1_b = accuracy_score(all_t1_rel, p1_b_preds)
pooled_resnet = accuracy_score(all_t1_rel, all_resnet_preds)
pooled_s2_b = accuracy_score(all_t2, p2_b_preds)
pooled_s3_b = accuracy_score(all_t3, p3_b_preds)
print(f"Pooled Total | Count: {len(all_t1_rel):<5} | S1 (DSConv t=0.55): {pooled_s1_b*100:.2f}% (Target: 88.20%) | S1 (ResNet18): {pooled_resnet*100:.2f}% | Stage 2: {pooled_s2_b*100:.2f}% | Stage 3: {pooled_s3_b*100:.2f}%")

print("\n--- Pipeline B Stage 2 Confusion Matrix (6x6) ---")
cm_s2_b = confusion_matrix(all_t2, p2_b_preds)
print(cm_s2_b)

p_s2_b, r_s2_b, f1_s2_b, sup_s2_b = precision_recall_fscore_support(all_t2, p2_b_preds)
print("\nStage 2 Per-Class Metrics (Pipeline B):")
for idx, cname in enumerate(s2_classes):
    print(f"  {cname:<18} | Precision: {p_s2_b[idx]:.4f} | Recall: {r_s2_b[idx]:.4f} | F1: {f1_s2_b[idx]:.4f} | Support: {sup_s2_b[idx]}")

print("\n--- Pipeline B Stage 3 Confusion Matrix (8x8) ---")
cm_s3_b = confusion_matrix(all_t3, p3_b_preds)
print(cm_s3_b)

p_s3_b, r_s3_b, f1_s3_b, sup_s3_b = precision_recall_fscore_support(all_t3, p3_b_preds)
print("\nStage 3 Per-Class Metrics (Pipeline B):")
for idx, cname in enumerate(STAGE3_CLASSES):
    print(f"  {cname:<12} | Precision: {p_s3_b[idx]:.4f} | Recall: {r_s3_b[idx]:.4f} | F1: {f1_s3_b[idx]:.4f} | Support: {sup_s3_b[idx]}")

off_diag_s3_b = []
for i in range(8):
    for j in range(8):
        if i != j and cm_s3_b[i, j] > 0:
            off_diag_s3_b.append((cm_s3_b[i, j], STAGE3_CLASSES[i], STAGE3_CLASSES[j]))
off_diag_s3_b.sort(reverse=True)
print("\nTop 10 Stage 3 Misclassifications (Pipeline B):")
for cnt, true_c, pred_c in off_diag_s3_b[:10]:
    print(f"  True: {true_c:<10} -> Pred: {pred_c:<10} | Count: {cnt} ({cnt/len(all_t3)*100:.2f}%)")
