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

tf_128_nonorm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

raw_test = ImageFolder(root="data/final/test", transform=tf_128_nonorm)
classes = raw_test.classes

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

# TTA transforms (5-view)
tta_transforms = [
    lambda t: t,
    lambda t: torch.flip(t, dims=[2]),
    lambda t: torch.flip(t, dims=[1]),
    lambda t: torch.clamp(t + 0.15, 0.0, 1.0),
    lambda t: torch.clamp(t - 0.15, 0.0, 1.0),
]

def eval_with_options(s1_model, use_tta=False, use_overrides=False, s1_thresh=None):
    gt1_list, gt2_list, gt3_list = [], [], []
    pred1_list, pred2_list, pred3_list = [], [], []
    
    for idx in range(len(raw_test)):
        img_tensor, target3_idx = raw_test[idx]
        class_name = classes[target3_idx]
        filepath = raw_test.samples[idx][0]
        
        gt1 = get_stage1_label(class_name, filepath if use_overrides else None)
        gt2 = get_stage2_label(class_name)
        gt3 = STAGE3_CLASSES.index(class_name)
        
        if use_tta:
            p1_acc = np.zeros(2)
            p2_acc = np.zeros(6)
            p3_acc = np.zeros(len(STAGE3_CLASSES))
            
            for aug in tta_transforms:
                aug_img = aug(img_tensor).unsqueeze(0).to(device)
                with torch.no_grad():
                    o1 = s1_model(aug_img)
                    prob1 = torch.softmax(o1, dim=-1).cpu().squeeze(0).numpy()
                    pred1_aug = int(o1.argmax(dim=-1).item())
                    
                    p1t = torch.tensor([pred1_aug], device=device)
                    o2 = s2(aug_img, p1t)
                    prob2 = torch.softmax(o2, dim=-1).cpu().squeeze(0).numpy()
                    pred2_aug = int(o2.argmax(dim=-1).item())
                    
                    p2t = torch.tensor([pred2_aug], device=device)
                    o3 = s3(aug_img, p2t)
                    prob3 = torch.softmax(o3, dim=-1).cpu().squeeze(0).numpy()
                    
                p1_acc += prob1
                p2_acc += prob2
                p3_acc += prob3
                
            p1_avg = p1_acc / len(tta_transforms)
            p2_avg = p2_acc / len(tta_transforms)
            p3_avg = p3_acc / len(tta_transforms)
            
            if s1_thresh is not None:
                pred1 = 1 if p1_avg[1] >= s1_thresh else 0
            else:
                pred1 = int(np.argmax(p1_avg))
            pred2 = int(np.argmax(p2_avg))
            pred3 = int(np.argmax(p3_avg))
        else:
            img_dev = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                o1 = s1_model(img_dev)
                prob1 = torch.softmax(o1, dim=-1).cpu().squeeze(0).numpy()
                if s1_thresh is not None:
                    pred1 = 1 if prob1[1] >= s1_thresh else 0
                else:
                    pred1 = int(o1.argmax(dim=-1).item())
                    
                p1t = torch.tensor([pred1], device=device)
                o2 = s2(img_dev, p1t)
                pred2 = int(o2.argmax(dim=-1).item())
                
                p2t = torch.tensor([pred2], device=device)
                o3 = s3(img_dev, p2t)
                pred3 = int(o3.argmax(dim=-1).item())
                
        gt1_list.append(gt1)
        gt2_list.append(gt2)
        gt3_list.append(gt3)
        pred1_list.append(pred1)
        pred2_list.append(pred2)
        pred3_list.append(pred3)

    a1 = accuracy_score(gt1_list, pred1_list)
    a2 = accuracy_score(gt2_list, pred2_list)
    a3 = accuracy_score(gt3_list, pred3_list)
    return a1, a2, a3

print("Running test suite...")
print("1. Single view, stage1.pt, no overrides, argmax:")
a1, a2, a3 = eval_with_options(s1_orig, use_tta=False, use_overrides=False, s1_thresh=None)
print(f"   S1: {a1*100:.2f}% | S2: {a2*100:.2f}% | S3: {a3*100:.2f}%")

print("2. 5-view TTA, stage1.pt, no overrides, argmax:")
a1, a2, a3 = eval_with_options(s1_orig, use_tta=True, use_overrides=False, s1_thresh=None)
print(f"   S1: {a1*100:.2f}% | S2: {a2*100:.2f}% | S3: {a3*100:.2f}%")

print("3. Single view, stage1_v2_relabeled.pt, with overrides, t=0.55:")
a1, a2, a3 = eval_with_options(s1_rel, use_tta=False, use_overrides=True, s1_thresh=0.55)
print(f"   S1: {a1*100:.2f}% | S2: {a2*100:.2f}% | S3: {a3*100:.2f}%")

print("4. 5-view TTA, stage1_v2_relabeled.pt, with overrides, t=0.55:")
a1, a2, a3 = eval_with_options(s1_rel, use_tta=True, use_overrides=True, s1_thresh=0.55)
print(f"   S1: {a1*100:.2f}% | S2: {a2*100:.2f}% | S3: {a3*100:.2f}%")
