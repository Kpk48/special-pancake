import sys
import os
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
    STAGE3_TO_STAGE1, STAGE3_TO_STAGE2, get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

raw_test = ImageFolder(root="data/final/test", transform=val_test_transform)
classes = raw_test.classes

print("==================================================")
print("STEP 1: CLASS INDEX AUDIT")
print("==================================================")
print(f"ImageFolder raw_test.classes: {raw_test.classes}")
print(f"hierarchy.py STAGE3_CLASSES:  {STAGE3_CLASSES}")
print(f"hierarchy.py STAGE1_CLASSES:  {STAGE1_CLASSES}")
print(f"hierarchy.py STAGE2_CLASSES:  {STAGE2_CLASSES}")

class Stage1CalibDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.classes = base_dataset.classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target3 = self.base_dataset[idx]
        class_name = self.classes[target3]
        filepath = self.base_dataset.samples[idx][0]
        target1 = get_stage1_label(class_name, filepath)
        target2 = get_stage2_label(class_name)
        target3_idx = STAGE3_CLASSES.index(class_name)
        return img, target1, target2, target3_idx, filepath

test_dataset = Stage1CalibDataset(raw_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Load Stage 1 model
s1_model = Stage1Model().to(device)
s1_path = "artifacts/hierarchical/stage1_v2_relabeled.pt"
if not os.path.exists(s1_path):
    s1_path = "artifacts/hierarchical/stage1.pt"
s1_model.load_state_dict(torch.load(s1_path, map_location=device))
s1_model.eval()

# Load Stage 2 model
s2_model = Stage2Model().to(device)
s2_model.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
s2_model.eval()

# Load Stage 3 model
s3_model = Stage3Model().to(device)
s3_model.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
s3_model.eval()

# Collect outputs
all_imgs = []
all_t1 = []
all_t2 = []
all_t3 = []
all_paths = []
all_s1_probs = []

with torch.no_grad():
    for imgs, t1, t2, t3, paths in test_loader:
        imgs = imgs.to(device)
        logits1 = s1_model(imgs)
        probs1 = torch.softmax(logits1, dim=-1)
        
        all_s1_probs.append(probs1.cpu().numpy())
        all_t1.extend(t1.numpy())
        all_t2.extend(t2.numpy())
        all_t3.extend(t3.numpy())
        all_paths.extend(paths)

all_s1_probs = np.concatenate(all_s1_probs, axis=0)
all_t1 = np.array(all_t1)
all_t2 = np.array(all_t2)
all_t3 = np.array(all_t3)

# In get_stage1_label:
# 0 = biodegradable (cardboard, organic, paper)
# 1 = non_biodegradable (battery, glass, metal, plastic, textile)
# In model prediction: prob[:, 1] is probability of non_biodegradable!
probs_nonbio = all_s1_probs[:, 1]

print("\n==================================================")
print("STEP 2: THRESHOLD CONSISTENCY CHECK")
print("==================================================")
t_calibrated = 0.55
print(f"Applied threshold: t = {t_calibrated}")
print("Sample of 10 predictions (probs_nonbio, ground_truth_t1, predicted_t1):")
for i in range(10):
    p_nonbio = probs_nonbio[i]
    gt = all_t1[i]
    pred = 1 if p_nonbio >= t_calibrated else 0
    fname = os.path.basename(all_paths[i])
    print(f"  [{i}] File: {fname:<30} | Non-Bio Prob: {p_nonbio:.4f} | GT S1: {gt} | Pred S1: {pred}")

print("\n==================================================")
print("STEP 3: REPRODUCE PAPER BASELINE FIRST")
print("==================================================")
s1_preds_t055 = (probs_nonbio >= t_calibrated).astype(int)
s1_preds_t050 = (probs_nonbio >= 0.50).astype(int)

acc_s1_t055 = accuracy_score(all_t1, s1_preds_t055)
acc_s1_t050 = accuracy_score(all_t1, s1_preds_t050)

print(f"Stage 1 Test Accuracy (t=0.55): {acc_s1_t055*100:.2f}% (Paper Target: 88.20%)")
print(f"Stage 1 Test Accuracy (t=0.50): {acc_s1_t050*100:.2f}% (Paper Target: 87.58%)")

print("\n==================================================")
print("STEP 5: SANITY CHECK STAGE 2 AND STAGE 3 POOLED ACCURACY")
print("==================================================")

# For Stage 2 and Stage 3, let's run inference end-to-end with Stage 1 predictions feeding into Stage 2/3
all_s2_preds = []
all_s3_preds = []

with torch.no_grad():
    for imgs, t1, t2, t3, paths in test_loader:
        imgs = imgs.to(device)
        logits1 = s1_model(imgs)
        probs1 = torch.softmax(logits1, dim=-1)
        pred1 = (probs1[:, 1] >= t_calibrated).long()
        
        logits2 = s2_model(imgs, pred1)
        pred2 = logits2.argmax(dim=-1)
        
        logits3 = s3_model(imgs, pred2)
        pred3 = logits3.argmax(dim=-1)
        
        all_s2_preds.extend(pred2.cpu().numpy())
        all_s3_preds.extend(pred3.cpu().numpy())

all_s2_preds = np.array(all_s2_preds)
all_s3_preds = np.array(all_s3_preds)

acc_s2 = accuracy_score(all_t2, all_s2_preds)
acc_s3 = accuracy_score(all_t3, all_s3_preds)

print(f"Stage 2 Test Accuracy (Conditioned on S1 t=0.55): {acc_s2*100:.2f}% (Paper Target: 66.98%)")
print(f"Stage 3 Test Accuracy (Conditioned on S2):         {acc_s3*100:.2f}% (Paper Target: 63.40%)")

