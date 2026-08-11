import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np

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

loader = DataLoader(ReconcileDS(raw_test), batch_size=128, shuffle=False, num_workers=0)

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

# Compute Pipeline A
joint_correct_a = 0
s1_correct_a = 0
s2_correct_a = 0
s3_correct_a = 0
total = 0

with torch.no_grad():
    for imgs, gt1_o, gt1_r, gt2, gt3, fps in loader:
        imgs = imgs.to(device)
        
        # Pipeline A
        o1_a = s1_orig(imgs)
        p1_a = o1_a.argmax(dim=-1)
        o2_a = s2(imgs, p1_a)
        p2_a = o2_a.argmax(dim=-1)
        o3_a = s3(imgs, p2_a)
        p3_a = o3_a.argmax(dim=-1)
        
        g1_a = gt1_o.to(device)
        g2_a = gt2.to(device)
        g3_a = gt3.to(device)
        
        c1_a = (p1_a == g1_a)
        c2_a = (p2_a == g2_a)
        c3_a = (p3_a == g3_a)
        
        s1_correct_a += c1_a.sum().item()
        s2_correct_a += c2_a.sum().item()
        s3_correct_a += c3_a.sum().item()
        joint_correct_a += (c1_a & c2_a & c3_a).sum().item()
        
        total += len(imgs)

joint_acc_a = joint_correct_a / total
print(f"Pipeline A: Stage 1 = {s1_correct_a/total*100:.2f}%, Stage 2 = {s2_correct_a/total*100:.2f}%, Stage 3 = {s3_correct_a/total*100:.2f}%, Joint Acc = {joint_acc_a*100:.2f}% ({joint_correct_a}/{total})")

# Compute Pipeline B
joint_correct_b = 0
s1_correct_b = 0
s2_correct_b = 0
s3_correct_b = 0

with torch.no_grad():
    for imgs, gt1_o, gt1_r, gt2, gt3, fps in loader:
        imgs = imgs.to(device)
        
        # Pipeline B
        o1_b = s1_rel(imgs)
        probs1_b = torch.softmax(o1_b, dim=-1)
        p1_b = (probs1_b[:, 1] >= 0.55).long()
        o2_b = s2(imgs, p1_b)
        p2_b = o2_b.argmax(dim=-1)
        o3_b = s3(imgs, p2_b)
        p3_b = o3_b.argmax(dim=-1)
        
        g1_b = gt1_r.to(device)
        g2_b = gt2.to(device)
        g3_b = gt3.to(device)
        
        c1_b = (p1_b == g1_b)
        c2_b = (p2_b == g2_b)
        c3_b = (p3_b == g3_b)
        
        s1_correct_b += c1_b.sum().item()
        s2_correct_b += c2_b.sum().item()
        s3_correct_b += c3_b.sum().item()
        joint_correct_b += (c1_b & c2_b & c3_b).sum().item()

joint_acc_b = joint_correct_b / total
print(f"Pipeline B: Stage 1 = {s1_correct_b/total*100:.2f}%, Stage 2 = {s2_correct_b/total*100:.2f}%, Stage 3 = {s3_correct_b/total*100:.2f}%, Joint Acc = {joint_acc_b*100:.2f}% ({joint_correct_b}/{total})")
