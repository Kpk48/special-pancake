import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES,
    STAGE3_TO_STAGE1, STAGE3_TO_STAGE2,
    get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

no_norm_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

raw_test = ImageFolder(root="data/final/test", transform=no_norm_tf)
classes = raw_test.classes

loader = DataLoader(raw_test, batch_size=128, shuffle=False, num_workers=0)

s1_orig = Stage1Model().to(device)
s1_orig.load_state_dict(torch.load("artifacts/hierarchical/stage1.pt", map_location=device))
s1_orig.eval()

s1_rel = Stage1Model().to(device)
s1_rel.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
s1_rel.eval()

s2 = Stage2Model().to(device)
s2.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
s2.eval()

s3 = Stage3Model().to(device)
s3.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
s3.eval()

# Ground truth lists
gt1_orig = []
gt1_rel = []
gt2 = []
gt3 = []
paths = []

for idx in range(len(raw_test)):
    _, t3_idx = raw_test.samples[idx]
    cname = classes[t3_idx]
    fp = raw_test.samples[idx][0]
    gt1_orig.append(get_stage1_label(cname, None))
    gt1_rel.append(get_stage1_label(cname, fp))
    gt2.append(get_stage2_label(cname))
    gt3.append(STAGE3_CLASSES.index(cname))
    paths.append(fp)

gt1_orig = np.array(gt1_orig)
gt1_rel = np.array(gt1_rel)
gt2 = np.array(gt2)
gt3 = np.array(gt3)

# Evaluate Pipeline A (Canonical Pre-Relabeled Baseline: stage1.pt, no overrides, argmax t=0.50)
p1_a, p2_a, p3_a = [], [], []

with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        o1 = s1_orig(imgs)
        pred1 = o1.argmax(dim=-1)
        
        o2 = s2(imgs, pred1)
        pred2 = o2.argmax(dim=-1)
        
        o3 = s3(imgs, pred2)
        pred3 = o3.argmax(dim=-1)
        
        p1_a.extend(pred1.cpu().numpy())
        p2_a.extend(pred2.cpu().numpy())
        p3_a.extend(pred3.cpu().numpy())

p1_a = np.array(p1_a)
p2_a = np.array(p2_a)
p3_a = np.array(p3_a)

a1_a = accuracy_score(gt1_orig, p1_a)
a2_a = accuracy_score(gt2, p2_a)
a3_a = accuracy_score(gt3, p3_a)

# Evaluate Pipeline B (Phase 4 Relabeled Baseline: stage1_v2_relabeled.pt, overrides, t=0.55)
p1_b, p2_b, p3_b = [], [], []

with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        o1 = s1_rel(imgs)
        probs1 = torch.softmax(o1, dim=-1)
        pred1 = (probs1[:, 1] >= 0.55).long()
        
        o2 = s2(imgs, pred1)
        pred2 = o2.argmax(dim=-1)
        
        o3 = s3(imgs, pred2)
        pred3 = o3.argmax(dim=-1)
        
        p1_b.extend(pred1.cpu().numpy())
        p2_b.extend(pred2.cpu().numpy())
        p3_b.extend(pred3.cpu().numpy())

p1_b = np.array(p1_b)
p2_b = np.array(p2_b)
p3_b = np.array(p3_b)

a1_b = accuracy_score(gt1_rel, p1_b)
a2_b = accuracy_score(gt2, p2_b)
a3_b = accuracy_score(gt3, p3_b)

print("=== PIPELINE RECONCILIATION SUMMARY ===")
print("PIPELINE A (Canonical Pre-Relabeled Baseline - README / Table II):")
print(f"  Stage 1 Accuracy: {a1_a*100:.2f}% (Target: 79.52%)")
print(f"  Stage 2 Accuracy: {a2_a*100:.2f}% (Target: 66.98%)")
print(f"  Stage 3 Accuracy: {a3_a*100:.2f}% (Target: 63.40%)\n")

print("PIPELINE B (Phase 4 Relabeled Baseline - Manuscript Table I / Ablation Baseline):")
print(f"  Stage 1 Accuracy: {a1_b*100:.2f}% (Target: 88.20%)")
print(f"  Stage 2 Accuracy: {a2_b*100:.2f}% (Target: 70.95%)")
print(f"  Stage 3 Accuracy: {a3_b*100:.2f}% (Target: 67.72%)\n")

# Print Stage 2/3 confusion matrices and per-class metrics for PIPELINE A (Canonical Table II)
s2_classes = ["paper_cardboard", "organic", "glass", "metal", "plastic", "textile_battery"]

print("=== STAGE 2 CONFUSION MATRIX (Pipeline A - Canonical 66.98%) ===")
cm_s2_a = confusion_matrix(gt2, p2_a)
print(cm_s2_a)
p2_prec, p2_rec, p2_f1, p2_sup = precision_recall_fscore_support(gt2, p2_a)
print("\nStage 2 Per-Class Metrics (Pipeline A):")
for idx, cname in enumerate(s2_classes):
    print(f"  {cname:<18} | Precision: {p2_prec[idx]:.4f} | Recall: {p2_rec[idx]:.4f} | F1: {p2_f1[idx]:.4f} | Support: {p2_sup[idx]}")

print("\n=== STAGE 3 CONFUSION MATRIX (Pipeline A - Canonical 63.40%) ===")
cm_s3_a = confusion_matrix(gt3, p3_a)
print(cm_s3_a)
p3_prec, p3_rec, p3_f1, p3_sup = precision_recall_fscore_support(gt3, p3_a)
print("\nStage 3 Per-Class Metrics (Pipeline A):")
for idx, cname in enumerate(STAGE3_CLASSES):
    print(f"  {cname:<12} | Precision: {p3_prec[idx]:.4f} | Recall: {p3_rec[idx]:.4f} | F1: {p3_f1[idx]:.4f} | Support: {p3_sup[idx]}")

# Print Stage 2/3 confusion matrices and per-class metrics for PIPELINE B (Phase 4 Relabeled 70.95% / 67.72%)
print("\n=== STAGE 2 CONFUSION MATRIX (Pipeline B - Phase 4 Relabeled 70.95%) ===")
cm_s2_b = confusion_matrix(gt2, p2_b)
print(cm_s2_b)

print("\n=== STAGE 3 CONFUSION MATRIX (Pipeline B - Phase 4 Relabeled 67.72%) ===")
cm_s3_b = confusion_matrix(gt3, p3_b)
print(cm_s3_b)
