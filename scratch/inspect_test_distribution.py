import sys
import os
import torch
import numpy as np
from collections import Counter
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES, STAGE3_CLASSES, get_stage1_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

raw_test = ImageFolder(root="data/final/test", transform=transform)
loader = DataLoader(raw_test, batch_size=64, shuffle=False, num_workers=0)
classes = raw_test.classes

print("=================================================================")
print("TASK 2: LITERAL CLASS DISTRIBUTION OF data/final/test")
print("=================================================================")
print(f"Total test samples in filesystem: {len(raw_test)}")

# 1. Raw target distribution directly from ImageFolder fine-grained classes
raw_s3_counts = Counter([raw_test.samples[i][1] for i in range(len(raw_test))])
print("\nRaw ImageFolder Stage 3 Class Counts (directory structure):")
for idx, cname in enumerate(classes):
    print(f"  [{idx}] {cname:<12}: {raw_s3_counts[idx]}")

# 2. Stage 1 distribution WITHOUT filepath overrides (get_stage1_label(cname, None))
gts1_no_override = []
for i in range(len(raw_test)):
    cname = classes[raw_test.samples[i][1]]
    gts1_no_override.append(get_stage1_label(cname, None))

counts_no_override = Counter(gts1_no_override)
print("\nStage 1 Class Distribution WITHOUT Overrides (get_stage1_label(cname, None)):")
print(f"  0 (biodegradable):     {counts_no_override[0]}")
print(f"  1 (non_biodegradable): {counts_no_override[1]}")

# 3. Stage 1 distribution WITH filepath overrides (get_stage1_label(cname, filepath))
gts1_with_override = []
overridden_files = []
for i in range(len(raw_test)):
    fp, t3 = raw_test.samples[i]
    cname = classes[t3]
    lbl = get_stage1_label(cname, fp)
    gts1_with_override.append(lbl)
    lbl_no = get_stage1_label(cname, None)
    if lbl != lbl_no:
        overridden_files.append((fp, cname, lbl_no, lbl))

counts_with_override = Counter(gts1_with_override)
print("\nStage 1 Class Distribution WITH Overrides (get_stage1_label(cname, filepath)):")
print(f"  0 (biodegradable):     {counts_with_override[0]}")
print(f"  1 (non_biodegradable): {counts_with_override[1]}")

print(f"\nNumber of files in data/final/test with active Stage 1 overrides: {len(overridden_files)}")

# 4. Evaluate Stage 1 model predictions in batches
s1_v2 = Stage1Model().to(device)
s1_v2.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
s1_v2.eval()

preds_t055 = []
preds_t050 = []

with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        logits = s1_v2(imgs)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        p_t055 = (probs >= 0.55).long()
        p_t050 = (probs >= 0.50).long()
        preds_t055.extend(p_t055.cpu().numpy())
        preds_t050.extend(p_t050.cpu().numpy())

preds_t055 = np.array(preds_t055)
preds_t050 = np.array(preds_t050)

print("\n=================================================================")
print("TASK 3: VERBATIM CONFUSION MATRICES (sklearn.metrics.confusion_matrix)")
print("=================================================================")

print("\n1. Stage 1 v2_relabeled Confusion Matrix (643 Bio / 1925 Non-Bio) @ t=0.55:")
cm_v2_t055 = confusion_matrix(gts1_with_override, preds_t055)
print(repr(cm_v2_t055))
print("Confusion Matrix array (0=Bio, 1=Non-Bio):")
print(cm_v2_t055)

print("\n2. Stage 1 v2_relabeled Confusion Matrix (643 Bio / 1925 Non-Bio) @ t=0.50:")
cm_v2_t050 = confusion_matrix(gts1_with_override, preds_t050)
print(repr(cm_v2_t050))
print("Confusion Matrix array (0=Bio, 1=Non-Bio):")
print(cm_v2_t050)

# Check original stage1.pt baseline model
s1_orig = Stage1Model().to(device)
s1_orig.load_state_dict(torch.load("artifacts/hierarchical/stage1.pt", map_location=device))
s1_orig.eval()

preds_orig_t050 = []
with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        logits = s1_orig(imgs)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        p_t050 = (probs >= 0.50).long()
        preds_orig_t050.extend(p_t050.cpu().numpy())

preds_orig_t050 = np.array(preds_orig_t050)

print("\n3. Original stage1.pt Baseline Confusion Matrix (643 Bio / 1925 Non-Bio) @ t=0.50:")
cm_orig_t050 = confusion_matrix(gts1_no_override, preds_orig_t050)
print(repr(cm_orig_t050))
print("Confusion Matrix array (0=Bio, 1=Non-Bio):")
print(cm_orig_t050)
