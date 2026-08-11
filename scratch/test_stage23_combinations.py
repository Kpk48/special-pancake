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
    STAGE3_TO_STAGE1, STAGE3_TO_STAGE2,
    get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Combo A: generate_audited_report.py setup (224x224, Normalize, stage1.pt, get_stage1_label without filepath)
tf_224_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Combo B: 128x128 no norm
tf_128_nonorm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Combo C: 128x128 with norm
tf_128_norm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def evaluate_pipeline(tf, s1_ckpt, use_filepath_overrides=False, s1_thresh=None):
    raw_test = ImageFolder(root="data/final/test", transform=tf)
    classes = raw_test.classes
    
    s1 = Stage1Model().to(device)
    s1.load_state_dict(torch.load(os.path.join("artifacts/hierarchical", s1_ckpt), map_location=device))
    s1.eval()

    s2 = Stage2Model().to(device)
    s2.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
    s2.eval()

    s3 = Stage3Model().to(device)
    s3.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
    s3.eval()

    gt1_list, gt2_list, gt3_list = [], [], []
    pred1_list, pred2_list, pred3_list = [], [], []

    for idx in range(len(raw_test)):
        img_tensor, target3_idx = raw_test[idx]
        class_name = classes[target3_idx]
        filepath = raw_test.samples[idx][0]
        
        gt1 = get_stage1_label(class_name, filepath if use_filepath_overrides else None)
        gt2 = get_stage2_label(class_name)
        gt3 = STAGE3_CLASSES.index(class_name)
        
        img_dev = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            out1 = s1(img_dev)
            if s1_thresh is not None:
                prob1_nonbio = torch.softmax(out1, dim=-1)[0, 1].item()
                pred1 = 1 if prob1_nonbio >= s1_thresh else 0
            else:
                pred1 = int(out1.argmax(dim=-1).item())
                
            p1t = torch.tensor([pred1], device=device)
            out2 = s2(img_dev, p1t)
            pred2 = int(out2.argmax(dim=-1).item())
            
            p2t = torch.tensor([pred2], device=device)
            out3 = s3(img_dev, p2t)
            pred3 = int(out3.argmax(dim=-1).item())
            
        gt1_list.append(gt1)
        gt2_list.append(gt2)
        gt3_list.append(gt3)
        pred1_list.append(pred1)
        pred2_list.append(pred2)
        pred3_list.append(pred3)

    acc1 = accuracy_score(gt1_list, pred1_list)
    acc2 = accuracy_score(gt2_list, pred2_list)
    acc3 = accuracy_score(gt3_list, pred3_list)
    return acc1, acc2, acc3

experiments = [
    ("1. generate_audited_report exact setup (224x224 norm, stage1.pt, no overrides, argmax)", tf_224_norm, "stage1.pt", False, None),
    ("2. 224x224 norm, stage1.pt, with overrides, argmax", tf_224_norm, "stage1.pt", True, None),
    ("3. 224x224 norm, stage1_v2_relabeled.pt, no overrides, argmax", tf_224_norm, "stage1_v2_relabeled.pt", False, None),
    ("4. 224x224 norm, stage1_v2_relabeled.pt, with overrides, t=0.55", tf_224_norm, "stage1_v2_relabeled.pt", True, 0.55),
    ("5. 128x128 no-norm, stage1.pt, no overrides, argmax", tf_128_nonorm, "stage1.pt", False, None),
    ("6. 128x128 no-norm, stage1.pt, with overrides, argmax", tf_128_nonorm, "stage1.pt", True, None),
    ("7. 128x128 no-norm, stage1_v2_relabeled.pt, with overrides, t=0.55", tf_128_nonorm, "stage1_v2_relabeled.pt", True, 0.55),
]

print("Target paper baseline accuracy: Stage 1 = 79.52% (or 88.20%), Stage 2 = 66.98%, Stage 3 = 63.40%\n")
for desc, tf, ckpt, overrides, thresh in experiments:
    a1, a2, a3 = evaluate_pipeline(tf, ckpt, overrides, thresh)
    print(f"{desc}:")
    print(f"   Stage 1: {a1*100:.2f}% | Stage 2: {a2*100:.2f}% | Stage 3: {a3*100:.2f}%\n")
