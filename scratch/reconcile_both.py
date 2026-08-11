import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
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

class ReconcileDS(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds
        self.classes = base_ds.classes
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, t3_idx = self.base_ds[idx]
        cname = self.classes[t3_idx]
        fp = self.base_ds.samples[idx][0]
        gt1_orig = get_stage1_label(cname, None)
        gt1_rel = get_stage1_label(cname, fp)
        gt2 = get_stage2_label(cname)
        gt3 = STAGE3_CLASSES.index(cname)
        return img, gt1_orig, gt1_rel, gt2, gt3, fp

test_loader = DataLoader(ReconcileDS(raw_test), batch_size=128, shuffle=False, num_workers=0)

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

# Run Pipeline A (Canonical Baseline: stage1.pt, no overrides, argmax t=0.50)
g1_a, g2_a, g3_a = [], [], []
p1_a, p2_a, p3_a = [], [], []

with torch.no_grad():
    for imgs, gt1_orig, gt1_rel, gt2, gt3, fps in test_loader:
        imgs = imgs.to(device)
        o1 = s1_orig(imgs)
        pred1 = o1.argmax(dim=-1)
        o2 = s2(imgs, pred1)
        pred2 = o2.argmax(dim=-1)
        o3 = s3(imgs, pred2)
        pred3 = o3.argmax(dim=-1)
        
        g1_a.extend(gt1_orig.numpy())
        g2_a.extend(gt2.numpy())
        g3_a.extend(gt3.numpy())
        
        p1_a.extend(pred1.cpu().numpy())
        p2_a.extend(pred2.cpu().numpy())
        p3_a.extend(pred3.cpu().numpy())

a1_a = accuracy_score(g1_a, p1_a)
a2_a = accuracy_score(g2_a, p2_a)
a3_a = accuracy_score(g3_a, p3_a)

# Run Pipeline B (Phase 4 Relabeled Baseline: stage1_v2_relabeled.pt, overrides, t=0.55)
g1_b, g2_b, g3_b = [], [], []
p1_b, p2_b, p3_b = [], [], []

with torch.no_grad():
    for imgs, gt1_orig, gt1_rel, gt2, gt3, fps in test_loader:
        imgs = imgs.to(device)
        o1 = s1_rel(imgs)
        probs1 = torch.softmax(o1, dim=-1)
        pred1 = (probs1[:, 1] >= 0.55).long()
        o2 = s2(imgs, pred1)
        pred2 = o2.argmax(dim=-1)
        o3 = s3(imgs, pred2)
        pred3 = o3.argmax(dim=-1)
        
        g1_b.extend(gt1_rel.numpy())
        g2_b.extend(gt2.numpy())
        g3_b.extend(gt3.numpy())
        
        p1_b.extend(pred1.cpu().numpy())
        p2_b.extend(pred2.cpu().numpy())
        p3_b.extend(pred3.cpu().numpy())

a1_b = accuracy_score(g1_b, p1_b)
a2_b = accuracy_score(g2_b, p2_b)
a3_b = accuracy_score(g3_b, p3_b)

print("==================================================")
print("PIPELINE RECONCILIATION RESULTS")
print("==================================================")
print(f"PIPELINE A (Canonical Pre-Relabeled Baseline: stage1.pt, no overrides, t=0.50):")
print(f"  Stage 1 Accuracy: {a1_a*100:.4f}% (Paper Table II Target: 79.52%)")
print(f"  Stage 2 Accuracy: {a2_a*100:.4f}% (Paper Table II Target: 66.98%)")
print(f"  Stage 3 Accuracy: {a3_a*100:.4f}% (Paper Table II Target: 63.40%)\n")

print(f"PIPELINE B (Phase 4 Post-Relabeled Baseline: stage1_v2_relabeled.pt, overrides, t=0.55):")
print(f"  Stage 1 Accuracy: {a1_b*100:.4f}% (Paper Phase 4 Target: 88.20%)")
print(f"  Stage 2 Accuracy: {a2_b*100:.4f}% (Paper Phase 4 Target: 70.95%)")
print(f"  Stage 3 Accuracy: {a3_b*100:.4f}% (Paper Phase 4 Target: 67.72%)\n")

print("Stage 2 Confusion Matrix (Pipeline A - Canonical Table II):\n", confusion_matrix(g2_a, p2_a))
print("Stage 3 Confusion Matrix (Pipeline A - Canonical Table II):\n", confusion_matrix(g3_a, p3_a))
