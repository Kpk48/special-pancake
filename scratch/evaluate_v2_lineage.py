import sys
import os
import time
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, classification_report
)
from torchvision.models import resnet18

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

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

raw_test_dsconv = ImageFolder(root="data/final/test", transform=dsconv_transform)
raw_test_resnet = ImageFolder(root="data/final/test", transform=resnet_transform)
classes = raw_test_dsconv.classes

class V2DatasetWrapper(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds
        self.classes = base_ds.classes
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, t3_idx = self.base_ds[idx]
        cname = self.classes[t3_idx]
        fp = self.base_ds.samples[idx][0]
        gt1_rel = get_stage1_label(cname, fp)
        gt2 = get_stage2_label(cname)
        gt3 = STAGE3_CLASSES.index(cname)
        return img, gt1_rel, gt2, gt3, fp

loader_dsconv = DataLoader(V2DatasetWrapper(raw_test_dsconv), batch_size=64, shuffle=False, num_workers=0)
loader_resnet = DataLoader(V2DatasetWrapper(raw_test_resnet), batch_size=64, shuffle=False, num_workers=0)

# Load v2_relabeled models
s1_v2 = Stage1Model().to(device)
s1_v2.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
s1_v2.eval()

s2_model = Stage2Model().to(device)
s2_model.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
s2_model.eval()

s3_model = Stage3Model().to(device)
s3_model.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
s3_model.eval()

resnet_v2 = resnet18(weights=None)
resnet_v2.fc = nn.Linear(resnet_v2.fc.in_features, 2)
resnet_v2.load_state_dict(torch.load("artifacts/resnet18_stage1_v2_relabeled.pt", map_location=device))
resnet_v2.to(device)
resnet_v2.eval()

# Also load original resnet18 baseline for check
resnet_base = resnet18(weights=None)
resnet_base.fc = nn.Linear(resnet_base.fc.in_features, 2)
resnet_base.load_state_dict(torch.load("artifacts/resnet18_stage1_baseline.pt", map_location=device))
resnet_base.to(device)
resnet_base.eval()

# ---------------------------------------------------------
# Run DSConv v2_relabeled Cascade Evaluation (t=0.55)
# ---------------------------------------------------------
gts1, gts2, gts3 = [], [], []
preds1, preds2, preds3 = [], [], []
probs1_list, probs2_list, probs3_list = [], [], []

with torch.no_grad():
    for imgs, g1, g2, g3, fps in loader_dsconv:
        imgs = imgs.to(device)
        out1 = s1_v2(imgs)
        probs1 = torch.softmax(out1, dim=-1)
        p1_nonbio = probs1[:, 1]
        pred1 = (p1_nonbio >= 0.55).long()
        
        out2 = s2_model(imgs, pred1)
        probs2 = torch.softmax(out2, dim=-1)
        pred2 = out2.argmax(dim=-1)
        
        out3 = s3_model(imgs, pred2)
        probs3 = torch.softmax(out3, dim=-1)
        pred3 = out3.argmax(dim=-1)
        
        gts1.extend(g1.numpy())
        gts2.extend(g2.numpy())
        gts3.extend(g3.numpy())
        
        preds1.extend(pred1.cpu().numpy())
        preds2.extend(pred2.cpu().numpy())
        preds3.extend(pred3.cpu().numpy())
        
        probs1_list.append(probs1.cpu().numpy())
        probs2_list.append(probs2.cpu().numpy())
        probs3_list.append(probs3.cpu().numpy())

gts1 = np.array(gts1)
gts2 = np.array(gts2)
gts3 = np.array(gts3)

preds1 = np.array(preds1)
preds2 = np.array(preds2)
preds3 = np.array(preds3)

probs1_arr = np.concatenate(probs1_list, axis=0)
probs2_arr = np.concatenate(probs2_list, axis=0)
probs3_arr = np.concatenate(probs3_list, axis=0)

# Compute Joint Correctness (Sample-level 3-way AND)
joint_mask = (preds1 == gts1) & (preds2 == gts2) & (preds3 == gts3)
joint_acc = joint_mask.sum() / len(joint_mask)

# ---------------------------------------------------------
# Run ResNet18 Evaluations on Same Decontaminated Test Set
# ---------------------------------------------------------
r_preds_v2, r_probs_v2 = [], []
r_preds_base, r_probs_base = [], []

with torch.no_grad():
    for imgs, g1, g2, g3, fps in loader_resnet:
        imgs = imgs.to(device)
        out_v2 = resnet_v2(imgs)
        pr_v2 = torch.softmax(out_v2, dim=-1)
        r_preds_v2.extend(out_v2.argmax(dim=-1).cpu().numpy())
        r_probs_v2.append(pr_v2.cpu().numpy())
        
        out_base = resnet_base(imgs)
        pr_base = torch.softmax(out_base, dim=-1)
        r_preds_base.extend(out_base.argmax(dim=-1).cpu().numpy())
        r_probs_base.append(pr_base.cpu().numpy())

r_preds_v2 = np.array(r_preds_v2)
r_probs_v2 = np.concatenate(r_probs_v2, axis=0)
r_preds_base = np.array(r_preds_base)
r_probs_base = np.concatenate(r_probs_base, axis=0)

# Function to compute full metric dict for CSV
def get_metrics_dict(gts, preds, probs, num_classes):
    acc = accuracy_score(gts, preds)
    bacc = balanced_accuracy_score(gts, preds)
    mcc = matthews_corrcoef(gts, preds)
    kappa = cohen_kappa_score(gts, preds)
    pm, rm, f1m, _ = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(gts, preds, average="weighted", zero_division=0)
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(gts, probs[:, 1])
            pr_auc = average_precision_score(gts, probs[:, 1])
        else:
            auc = roc_auc_score(gts, probs, multi_class="ovr")
            pr_auc = 0.0
    except Exception:
        auc = 0.5
        pr_auc = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "prec_macro": pm,
        "rec_macro": rm,
        "f1_macro": f1m,
        "prec_weighted": pw,
        "rec_weighted": rw,
        "f1_weighted": f1w,
        "roc_auc": auc,
        "pr_auc": pr_auc,
    }

m1_v2 = get_metrics_dict(gts1, preds1, probs1_arr, 2)
m2_v2 = get_metrics_dict(gts2, preds2, probs2_arr, 6)
m3_v2 = get_metrics_dict(gts3, preds3, probs3_arr, 8)

res_v2_metrics = get_metrics_dict(gts1, r_preds_v2, r_probs_v2, 2)
res_base_metrics = get_metrics_dict(gts1, r_preds_base, r_probs_base, 2)

# Build CSV entries for final_metrics_v2.csv
rows = []
for stage_num, stage_name, m_dict in [
    (1, "Stage 1", m1_v2),
    (2, "Stage 2", m2_v2),
    (3, "Stage 3", m3_v2),
]:
    for k, v in m_dict.items():
        rows.append({"model": "Hierarchical CNN v2_relabeled", "stage": stage_name, "metric": k, "value": f"{v:.6f}"})

for k, v in res_v2_metrics.items():
    rows.append({"model": "ResNet18 v2_relabeled", "stage": "Stage 1", "metric": k, "value": f"{v:.6f}"})

df_v2 = pd.DataFrame(rows)
df_v2.to_csv("results/final_metrics_v2.csv", index=False)
print("Saved results/final_metrics_v2.csv successfully.")

# Print exact summary outputs for user task items
print("\n=================================================================")
print("ITEM 1 & 2: V2_RELABELED LINEAGE EVALUATION (t=0.55)")
print("=================================================================")
print(f"Stage 1 Top-1 Acc:  {m1_v2['accuracy']*100:.4f}% ({preds1.tolist().count(1)} predicted non-bio)")
print(f"Stage 1 Macro F1:   {m1_v2['f1_macro']:.6f}")
print(f"Stage 1 BAcc:       {m1_v2['balanced_accuracy']*100:.4f}%\n")

print(f"Stage 2 Cascaded Top-1 Acc: {m2_v2['accuracy']*100:.4f}%")
print(f"Stage 2 Macro F1:          {m2_v2['f1_macro']:.6f}\n")

print(f"Stage 3 Cascaded Top-1 Acc: {m3_v2['accuracy']*100:.4f}%")
print(f"Stage 3 Macro F1:          {m3_v2['f1_macro']:.6f}\n")

print(f"Joint 3-Way Cascaded Acc:  {joint_acc*100:.4f}% ({joint_mask.sum()}/{len(joint_mask)})\n")

print("=================================================================")
print("ITEM 3: STAGE 1 CLASSIFICATION REPORT (sklearn exact output)")
print("=================================================================")
print(classification_report(gts1, preds1, target_names=["biodegradable", "non_biodegradable"], digits=6))

print("=================================================================")
print("ITEM 4: RESNET18 COMPARISON ON DECONTAMINATED TEST SET")
print("=================================================================")
print(f"ResNet18 v2_relabeled Checkpoint:")
print(f"  Accuracy:          {res_v2_metrics['accuracy']*100:.4f}% ({sum(r_preds_v2==gts1)}/{len(gts1)})")
print(f"  Balanced Accuracy: {res_v2_metrics['balanced_accuracy']*100:.4f}%")
print(f"  Macro F1:          {res_v2_metrics['f1_macro']:.6f}\n")

print(f"ResNet18 baseline Checkpoint (evaluated on decontaminated test set):")
print(f"  Accuracy:          {res_base_metrics['accuracy']*100:.4f}% ({sum(r_preds_base==gts1)}/{len(gts1)})")
print(f"  Balanced Accuracy: {res_base_metrics['balanced_accuracy']*100:.4f}%")
print(f"  Macro F1:          {res_base_metrics['f1_macro']:.6f}\n")

