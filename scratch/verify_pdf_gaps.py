import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torchvision.models import resnet18

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

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

raw_test_dsconv = ImageFolder(root="data/final/test", transform=dsconv_transform)
raw_test_resnet = ImageFolder(root="data/final/test", transform=resnet_transform)
classes = raw_test_dsconv.classes

# Test dataset wrapper
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

loader_dsconv = DataLoader(TestDS(raw_test_dsconv), batch_size=64, shuffle=False, num_workers=0)
loader_resnet = DataLoader(TestDS(raw_test_resnet), batch_size=64, shuffle=False, num_workers=0)

# Load Models
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

# 1. ResNet18 Accuracy
resnet_preds, resnet_gts = [], []
with torch.no_grad():
    for imgs, g1_o, g1_r, g2, g3, fps in loader_resnet:
        imgs = imgs.to(device)
        out = resnet_s1(imgs)
        pred = out.argmax(dim=-1)
        resnet_preds.extend(pred.cpu().numpy())
        resnet_gts.extend(g1_r.numpy())

resnet_acc = accuracy_score(resnet_gts, resnet_preds)
print(f"ResNet18 Stage 1 Accuracy (on post-relabeled test set): {resnet_acc*100:.2f}% ({sum(np.array(resnet_preds)==np.array(resnet_gts))}/{len(resnet_gts)})")

# 2. Stage 1 DSConv Confusion Matrix at t=0.55
s1_preds, s1_gts = [], []
with torch.no_grad():
    for imgs, g1_o, g1_r, g2, g3, fps in loader_dsconv:
        imgs = imgs.to(device)
        out = s1_rel(imgs)
        probs = torch.softmax(out, dim=-1)[:, 1]
        pred = (probs >= 0.55).long()
        s1_preds.extend(pred.cpu().numpy())
        s1_gts.extend(g1_r.numpy())

cm_s1 = confusion_matrix(s1_gts, s1_preds)
print("Stage 1 Confusion Matrix at t=0.55 (0=Bio, 1=Non-Bio):")
print(cm_s1)
tn, fp, fn, tp = cm_s1.ravel()
print(f"  TN (Bio predicted Bio): {tn}")
print(f"  FP (Bio predicted Non-Bio): {fp}")
print(f"  FN (Non-Bio predicted Bio): {fn}")
print(f"  TP (Non-Bio predicted Non-Bio): {tp}")

# 3. Stage 2 and Stage 3 metrics (Pipeline B)
s2_preds, s3_preds = [], []
gt2_gts, gt3_gts = [], []
with torch.no_grad():
    for imgs, g1_o, g1_r, g2, g3, fps in loader_dsconv:
        imgs = imgs.to(device)
        out1 = s1_rel(imgs)
        probs1 = torch.softmax(out1, dim=-1)[:, 1]
        pred1 = (probs1 >= 0.55).long()
        
        out2 = s2_model(imgs, pred1)
        pred2 = out2.argmax(dim=-1)
        
        out3 = s3_model(imgs, pred2)
        pred3 = out3.argmax(dim=-1)
        
        s2_preds.extend(pred2.cpu().numpy())
        s3_preds.extend(pred3.cpu().numpy())
        gt2_gts.extend(g2.numpy())
        gt3_gts.extend(g3.numpy())

s2_acc = accuracy_score(gt2_gts, s2_preds)
s3_acc = accuracy_score(gt3_gts, s3_preds)
_, _, s2_f1, _ = precision_recall_fscore_support(gt2_gts, s2_preds, average="macro", zero_division=0)
_, _, s3_f1, _ = precision_recall_fscore_support(gt3_gts, s3_preds, average="macro", zero_division=0)

print(f"Pipeline B Stage 2 Acc: {s2_acc*100:.2f}%, Macro F1: {s2_f1:.4f}")
print(f"Pipeline B Stage 3 Acc: {s3_acc*100:.2f}%, Macro F1: {s3_f1:.4f}")
