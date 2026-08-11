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

loader = DataLoader(raw_test, batch_size=256, shuffle=False, num_workers=0)

all_imgs = []
for imgs, _ in loader:
    all_imgs.append(imgs)
all_imgs = torch.cat(all_imgs, dim=0).to(device)

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

# PIPELINE A: Canonical Baseline (stage1.pt, no overrides, argmax t=0.50)
with torch.no_grad():
    o1_a = s1_orig(all_imgs)
    pred1_a = o1_a.argmax(dim=-1)
    
    o2_a = s2(all_imgs, pred1_a)
    pred2_a = o2_a.argmax(dim=-1)
    
    o3_a = s3(all_imgs, pred2_a)
    pred3_a = o3_a.argmax(dim=-1)

p1_a = pred1_a.cpu().numpy()
p2_a = pred2_a.cpu().numpy()
p3_a = pred3_a.cpu().numpy()

acc1_a = accuracy_score(gt1_orig, p1_a)
acc2_a = accuracy_score(gt2, p2_a)
acc3_a = accuracy_score(gt3, p3_a)

# PIPELINE B: Phase 4 Relabeled Baseline (stage1_v2_relabeled.pt, overrides, t=0.55)
with torch.no_grad():
    o1_b = s1_rel(all_imgs)
    probs1_b = torch.softmax(o1_b, dim=-1)
    pred1_b = (probs1_b[:, 1] >= 0.55).long()
    
    o2_b = s2(all_imgs, pred1_b)
    pred2_b = o2_b.argmax(dim=-1)
    
    o3_b = s3(all_imgs, pred2_b)
    pred3_b = o3_b.argmax(dim=-1)

p1_b = pred1_b.cpu().numpy()
p2_b = pred2_b.cpu().numpy()
p3_b = pred3_b.cpu().numpy()

acc1_b = accuracy_score(gt1_rel, p1_b)
acc2_b = accuracy_score(gt2, p2_b)
acc3_b = accuracy_score(gt3, p3_b)

print("==================================================")
print("EXACT RECONCILIATION SUMMARY")
print("==================================================")
print("PIPELINE A (Canonical Pre-Relabeled Table II Setup):")
print(f"  Stage 1 Accuracy: {acc1_a*100:.4f}% (Paper Target: 79.52%)")
print(f"  Stage 2 Accuracy: {acc2_a*100:.4f}% (Paper Target: 66.98%)")
print(f"  Stage 3 Accuracy: {acc3_a*100:.4f}% (Paper Target: 63.40%)\n")

print("PIPELINE B (Phase 4 Post-Relabeled Setup):")
print(f"  Stage 1 Accuracy: {acc1_b*100:.4f}% (Paper Target: 88.20%)")
print(f"  Stage 2 Accuracy: {acc2_b*100:.4f}% (Paper Target: 70.95%)")
print(f"  Stage 3 Accuracy: {acc3_b*100:.4f}% (Paper Target: 67.72%)\n")

print("STAGE 2 CONFUSION MATRIX (Pipeline A - Canonical Table II):\n", confusion_matrix(gt2, p2_a))
print("\nSTAGE 3 CONFUSION MATRIX (Pipeline A - Canonical Table II):\n", confusion_matrix(gt3, p3_a))
