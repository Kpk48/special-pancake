import sys
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import STAGE3_CLASSES, get_stage1_label, get_stage2_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
raw_test = ImageFolder(root="data/final/test", transform=tf)
loader = DataLoader(raw_test, batch_size=128, shuffle=False, num_workers=0)

s1_rel = Stage1Model().to(device)
s1_rel.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
s1_rel.eval()

s2 = Stage2Model().to(device)
s2.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
s2.eval()

s3 = Stage3Model().to(device)
s3.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
s3.eval()

gt1_decontam = []
gt2_list = []
gt3_list = []

for idx in range(len(raw_test)):
    fp, t3_idx = raw_test.samples[idx]
    cname = raw_test.classes[t3_idx]
    gt1_decontam.append(get_stage1_label(cname, None))
    gt2_list.append(get_stage2_label(cname))
    gt3_list.append(STAGE3_CLASSES.index(cname))

gt1_decontam = np.array(gt1_decontam)
gt2_list = np.array(gt2_list)
gt3_list = np.array(gt3_list)

p1_list = []
p2_list = []
p3_list = []

print("Starting inference...", flush=True)
with torch.no_grad():
    for b_idx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        o1 = s1_rel(imgs)
        probs1 = torch.softmax(o1, dim=-1)
        pred1 = (probs1[:, 1] >= 0.55).long()
        
        o2 = s2(imgs, pred1)
        pred2 = o2.argmax(dim=-1)
        
        o3 = s3(imgs, pred2)
        pred3 = o3.argmax(dim=-1)
        
        p1_list.append(pred1.cpu())
        p2_list.append(pred2.cpu())
        p3_list.append(pred3.cpu())
        if (b_idx + 1) % 5 == 0:
            print(f"Processed batch {b_idx + 1}/{len(loader)}", flush=True)

p1_arr = torch.cat(p1_list, dim=0).numpy()
p2_arr = torch.cat(p2_list, dim=0).numpy()
p3_arr = torch.cat(p3_list, dim=0).numpy()

c1 = (p1_arr == gt1_decontam)
c2 = (p2_arr == gt2_list)
c3 = (p3_arr == gt3_list)

joint = (c1 & c2 & c3)

print("=== DECONTAMINATED TEST RESULT ===", flush=True)
print(f"Total Test Samples:   {len(gt1_decontam)}", flush=True)
print(f"Stage 1 Accuracy:     {c1.sum()} / {len(c1)} = {c1.mean()*100:.4f}% ({c1.mean()*100:.2f}%)", flush=True)
print(f"Stage 2 Accuracy:     {c2.sum()} / {len(c2)} = {c2.mean()*100:.4f}% ({c2.mean()*100:.2f}%)", flush=True)
print(f"Stage 3 Accuracy:     {c3.sum()} / {len(c3)} = {c3.mean()*100:.4f}% ({c3.mean()*100:.2f}%)", flush=True)
print(f"Joint 3-Way Accuracy: {joint.sum()} / {len(joint)} = {joint.mean()*100:.4f}% ({joint.mean()*100:.2f}%)", flush=True)
