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
    STAGE3_TO_STAGE1, STAGE3_TO_STAGE2,
    get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dsconv_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

raw_test = ImageFolder(root="data/final/test", transform=dsconv_transform)
classes = raw_test.classes

class DiagnosticDS(Dataset):
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
        return img, gt1_orig, gt1_rel, gt2, gt3, fp, cname

loader = DataLoader(DiagnosticDS(raw_test), batch_size=64, shuffle=False, num_workers=0)

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
print("DIAGNOSTIC 1: PIPELINE A (stage1.pt, argmax t=0.50, pre-relabeled GT)")
print("==================================================")

c1_list, c2_list, c3_list = [], [], []
pred1_list, pred2_list, pred3_list = [], [], []
gt1_list, gt2_list, gt3_list = [], [], []
class_names = []

with torch.no_grad():
    for imgs, g1_o, g1_r, g2, g3, fps, cnames in loader:
        imgs = imgs.to(device)
        out1 = s1_orig(imgs)
        pred1 = out1.argmax(dim=-1)
        
        out2 = s2(imgs, pred1)
        pred2 = out2.argmax(dim=-1)
        
        out3 = s3(imgs, pred2)
        pred3 = out3.argmax(dim=-1)
        
        pred1_list.extend(pred1.cpu().numpy())
        pred2_list.extend(pred2.cpu().numpy())
        pred3_list.extend(pred3.cpu().numpy())
        
        gt1_list.extend(g1_o.numpy())
        gt2_list.extend(g2.numpy())
        gt3_list.extend(g3.numpy())
        class_names.extend(cnames)

pred1_arr = np.array(pred1_list)
pred2_arr = np.array(pred2_list)
pred3_arr = np.array(pred3_list)
gt1_arr = np.array(gt1_list)
gt2_arr = np.array(gt2_list)
gt3_arr = np.array(gt3_list)

c1 = (pred1_arr == gt1_arr)
c2 = (pred2_arr == gt2_arr)
c3 = (pred3_arr == gt3_arr)

print(f"Total samples: {len(gt1_arr)}")
print(f"c1 (Stage 1 correct): {c1.sum()} / {len(c1)} ({c1.mean()*100:.2f}%)")
print(f"c2 (Stage 2 correct): {c2.sum()} / {len(c2)} ({c2.mean()*100:.2f}%)")
print(f"c3 (Stage 3 correct): {c3.sum()} / {len(c3)} ({c3.mean()*100:.2f}%)")
print(f"c1 & c2 correct:       {(c1 & c2).sum()} ({ (c1 & c2).mean()*100:.2f}%)")
print(f"c1 & c3 correct:       {(c1 & c3).sum()} ({ (c1 & c3).mean()*100:.2f}%)")
print(f"c2 & c3 correct:       {(c2 & c3).sum()} ({ (c2 & c3).mean()*100:.2f}%)")
print(f"c1 & c2 & c3 correct:  {(c1 & c2 & c3).sum()} ({ (c1 & c2 & c3).mean()*100:.2f}%)")

# Check if there are any samples where c3 is True but c1 or c2 is False
c3_true_c1_false = (c3 & ~c1).sum()
c3_true_c2_false = (c3 & ~c2).sum()
print(f"Samples where Stage 3 is correct BUT Stage 1 is wrong: {c3_true_c1_false}")
print(f"Samples where Stage 3 is correct BUT Stage 2 is wrong: {c3_true_c2_false}")

# Now let's check hierarchy mapping of predictions:
# If pred3 is predicted fine-grained class, what is STAGE3_TO_STAGE1[STAGE3_CLASSES[pred3]] vs pred1?
# What is STAGE3_TO_STAGE2[STAGE3_CLASSES[pred3]] vs pred2?
pred3_implied_s1 = np.array([STAGE3_TO_STAGE1[STAGE3_CLASSES[p]] for p in pred3_arr])
pred3_implied_s2 = np.array([STAGE3_TO_STAGE2[STAGE3_CLASSES[p]] for p in pred3_arr])

print(f"Does pred3 match implied s1? {(pred3_implied_s1 == pred1_arr).sum()} / {len(pred1_arr)}")
print(f"Does pred3 match implied s2? {(pred3_implied_s2 == pred2_arr).sum()} / {len(pred2_arr)}")

# Check hierarchical consistency:
# Is an image considered jointly correct ONLY if pred3_implied_s1 == gt1 and pred3_implied_s2 == gt2 and pred3 == gt3?
# OR is joint correct defined as: pred1 == gt1 AND pred2 == gt2 AND pred3 == gt3?

print("\n==================================================")
print("DIAGNOSTIC 2: PIPELINE B (stage1_v2_relabeled.pt, t=0.55, post-relabeled GT)")
print("==================================================")

pred1_b_list, pred2_b_list, pred3_b_list = [], [], []
gt1_b_list = []

with torch.no_grad():
    for imgs, g1_o, g1_r, g2, g3, fps, cnames in loader:
        imgs = imgs.to(device)
        out1 = s1_rel(imgs)
        probs1 = torch.softmax(out1, dim=-1)
        pred1 = (probs1[:, 1] >= 0.55).long()
        
        out2 = s2(imgs, pred1)
        pred2 = out2.argmax(dim=-1)
        
        out3 = s3(imgs, pred2)
        pred3 = out3.argmax(dim=-1)
        
        pred1_b_list.extend(pred1.cpu().numpy())
        pred2_b_list.extend(pred2.cpu().numpy())
        pred3_b_list.extend(pred3.cpu().numpy())
        gt1_b_list.extend(g1_r.numpy())

pred1_b_arr = np.array(pred1_b_list)
pred2_b_arr = np.array(pred2_b_list)
pred3_b_arr = np.array(pred3_b_list)
gt1_b_arr = np.array(gt1_b_list)

c1_b = (pred1_b_arr == gt1_b_arr)
c2_b = (pred2_b_arr == gt2_arr)
c3_b = (pred3_b_arr == gt3_arr)

print(f"c1_b (Stage 1 correct): {c1_b.sum()} / {len(c1_b)} ({c1_b.mean()*100:.2f}%)")
print(f"c2_b (Stage 2 correct): {c2_b.sum()} / {len(c2_b)} ({c2_b.mean()*100:.2f}%)")
print(f"c3_b (Stage 3 correct): {c3_b.sum()} / {len(c3_b)} ({c3_b.mean()*100:.2f}%)")
print(f"c1_b & c2_b correct:       {(c1_b & c2_b).sum()} ({ (c1_b & c2_b).mean()*100:.2f}%)")
print(f"c1_b & c3_b correct:       {(c1_b & c3_b).sum()} ({ (c1_b & c3_b).mean()*100:.2f}%)")
print(f"c2_b & c3_b correct:       {(c2_b & c3_b).sum()} ({ (c2_b & c3_b).mean()*100:.2f}%)")
print(f"c1_b & c2_b & c3_b correct:{(c1_b & c2_b & c3_b).sum()} ({ (c1_b & c2_b & c3_b).mean()*100:.2f}%)")

c3_b_true_c1_b_false = (c3_b & ~c1_b).sum()
c3_b_true_c2_b_false = (c3_b & ~c2_b).sum()
print(f"Pipeline B: Samples where Stage 3 is correct BUT Stage 1 is wrong: {c3_b_true_c1_b_false}")
print(f"Pipeline B: Samples where Stage 3 is correct BUT Stage 2 is wrong: {c3_b_true_c2_b_false}")
