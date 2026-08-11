import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES,
    get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dsconv_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

raw_test = ImageFolder(root="data/final/test", transform=dsconv_transform)
classes = raw_test.classes

class TestDS(Dataset):
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

loader = DataLoader(TestDS(raw_test), batch_size=128, shuffle=False, num_workers=0)

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

print("==================================================")
print("PIPELINE A (Original stage1.pt, argmax t=0.50, Pre-Relabeled GT)")
print("==================================================")

gt1_orig_list, gt2_list, gt3_list = [], [], []
p1_a_list = []

# Stage 2 predictions: oracle vs cascade
s2_oracle_a, s2_cascade_a = [], []

# Stage 3 predictions: oracle (gt2) vs partial-cascade (gt1->s2->s3) vs full-cascade (s1->s2->s3)
s3_oracle_a, s3_cascade_a = [], []

with torch.no_grad():
    for imgs, gt1_o, gt1_r, gt2, gt3, fps in loader:
        imgs = imgs.to(device)
        gt1_dev = gt1_o.to(device)
        gt2_dev = gt2.to(device)
        
        # Stage 1 prediction
        out1 = s1_orig(imgs)
        pred1 = out1.argmax(dim=-1)
        
        # Stage 2 Oracle (ground-truth gt1 fed into s2)
        out2_oracle = s2(imgs, gt1_dev)
        pred2_oracle = out2_oracle.argmax(dim=-1)
        
        # Stage 2 Cascade (predicted pred1 fed into s2)
        out2_cascade = s2(imgs, pred1)
        pred2_cascade = out2_cascade.argmax(dim=-1)
        
        # Stage 3 Oracle (ground-truth gt2 fed into s3)
        out3_oracle = s3(imgs, gt2_dev)
        pred3_oracle = out3_oracle.argmax(dim=-1)
        
        # Stage 3 Cascade (predicted pred2_cascade fed into s3)
        out3_cascade = s3(imgs, pred2_cascade)
        pred3_cascade = out3_cascade.argmax(dim=-1)
        
        gt1_orig_list.extend(gt1_o.numpy())
        gt2_list.extend(gt2.numpy())
        gt3_list.extend(gt3.numpy())
        
        p1_a_list.extend(pred1.cpu().numpy())
        s2_oracle_a.extend(pred2_oracle.cpu().numpy())
        s2_cascade_a.extend(pred2_cascade.cpu().numpy())
        s3_oracle_a.extend(pred3_oracle.cpu().numpy())
        s3_cascade_a.extend(pred3_cascade.cpu().numpy())

a1_a = accuracy_score(gt1_orig_list, p1_a_list)
a2_oracle_a = accuracy_score(gt2_list, s2_oracle_a)
a2_cascade_a = accuracy_score(gt2_list, s2_cascade_a)
a3_oracle_a = accuracy_score(gt3_list, s3_oracle_a)
a3_cascade_a = accuracy_score(gt3_list, s3_cascade_a)

print(f"Stage 1 Acc:                     {a1_a*100:.2f}% (Target Table II: 79.52% or 88.20%)")
print(f"Stage 2 Top-1 (Oracle gt1):      {a2_oracle_a*100:.2f}% (Target Table II: 66.98%)")
print(f"Stage 2 Joint (Cascade pred1):   {a2_cascade_a*100:.2f}% (Target Table II: 64.12%)")
print(f"Stage 3 Top-1 (Oracle gt2):      {a3_oracle_a*100:.2f}% (Target Table II: 63.40%)")
print(f"Stage 3 Joint (Cascade pred2):   {a3_cascade_a*100:.2f}% (Target Table II: 59.81%)")


print("\n==================================================")
print("PIPELINE B (Calibrated stage1_v2_relabeled.pt, t=0.55, Post-Relabeled GT)")
print("==================================================")

gt1_rel_list = []
p1_b_list = []

s2_oracle_b, s2_cascade_b = [], []
s3_oracle_b, s3_cascade_b = [], []

with torch.no_grad():
    for imgs, gt1_o, gt1_r, gt2, gt3, fps in loader:
        imgs = imgs.to(device)
        gt1_rel_dev = gt1_r.to(device)
        gt2_dev = gt2.to(device)
        
        # Stage 1 prediction (t=0.55)
        out1 = s1_rel(imgs)
        probs1 = torch.softmax(out1, dim=-1)
        pred1 = (probs1[:, 1] >= 0.55).long()
        
        # Stage 2 Oracle (ground-truth gt1_rel fed into s2)
        out2_oracle = s2(imgs, gt1_rel_dev)
        pred2_oracle = out2_oracle.argmax(dim=-1)
        
        # Stage 2 Cascade (predicted pred1 fed into s2)
        out2_cascade = s2(imgs, pred1)
        pred2_cascade = out2_cascade.argmax(dim=-1)
        
        # Stage 3 Oracle (ground-truth gt2 fed into s3)
        out3_oracle = s3(imgs, gt2_dev)
        pred3_oracle = out3_oracle.argmax(dim=-1)
        
        # Stage 3 Cascade (predicted pred2_cascade fed into s3)
        out3_cascade = s3(imgs, pred2_cascade)
        pred3_cascade = out3_cascade.argmax(dim=-1)
        
        gt1_rel_list.extend(gt1_r.numpy())
        
        p1_b_list.extend(pred1.cpu().numpy())
        s2_oracle_b.extend(pred2_oracle.cpu().numpy())
        s2_cascade_b.extend(pred2_cascade.cpu().numpy())
        s3_oracle_b.extend(pred3_oracle.cpu().numpy())
        s3_cascade_b.extend(pred3_cascade.cpu().numpy())

a1_b = accuracy_score(gt1_rel_list, p1_b_list)
a2_oracle_b = accuracy_score(gt2_list, s2_oracle_b)
a2_cascade_b = accuracy_score(gt2_list, s2_cascade_b)
a3_oracle_b = accuracy_score(gt3_list, s3_oracle_b)
a3_cascade_b = accuracy_score(gt3_list, s3_cascade_b)

print(f"Stage 1 Acc (t=0.55):            {a1_b*100:.2f}% (Target: 88.20%)")
print(f"Stage 2 Top-1 (Oracle gt1):      {a2_oracle_b*100:.2f}%")
print(f"Stage 2 Joint (Cascade pred1):   {a2_cascade_b*100:.2f}%")
print(f"Stage 3 Top-1 (Oracle gt2):      {a3_oracle_b*100:.2f}%")
print(f"Stage 3 Joint (Cascade pred2):   {a3_cascade_b*100:.2f}%")
