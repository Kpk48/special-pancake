import sys
import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from waste_classifier.hierarchical.backbone import DSConv2DBackbone, PlainConv2DBackbone
from waste_classifier.hierarchical.hierarchy import get_stage1_label, get_stage2_label
from waste_classifier.hierarchical.loss import FocalLoss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HierarchicalDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.classes = base_dataset.classes
        self.targets3 = [s[1] for s in base_dataset.samples]
        self.targets1 = [get_stage1_label(self.classes[t3], s[0]) for s, t3 in zip(base_dataset.samples, self.targets3)]
        self.targets2 = [get_stage2_label(self.classes[t3]) for t3 in self.targets3]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target3 = self.base_dataset[idx]
        target1 = self.targets1[idx]
        target2 = self.targets2[idx]
        return img, target1, target2, target3

def calc_weights(targets_list, num_cls, device):
    counts = torch.zeros(num_cls)
    for t in targets_list:
        counts[t] += 1
    return (len(targets_list) / (num_cls * torch.clamp(counts, min=1.0))).to(device)

def eval_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, bacc, float(f1)

# --- DSCONV ABLATION MODELS ---

class Stage1VariantModel(nn.Module):
    def __init__(self, use_plain=False, feature_dim=128):
        super().__init__()
        self.backbone = PlainConv2DBackbone(feature_dim=feature_dim) if use_plain else DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)
    def forward(self, x):
        return self.classifier(self.backbone(x))

class Stage2VariantModel(nn.Module):
    def __init__(self, use_plain=False, feature_dim=128, embedding_dim=16):
        super().__init__()
        self.backbone = PlainConv2DBackbone(feature_dim=feature_dim) if use_plain else DSConv2DBackbone(feature_dim=feature_dim)
        self.stage1_embedding = nn.Embedding(2, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
    def forward(self, x, stage1_class):
        features = self.backbone(x)
        cond_emb = self.stage1_embedding(stage1_class)
        return self.classifier(torch.cat([features, cond_emb], dim=-1))

class Stage3VariantModel(nn.Module):
    def __init__(self, use_plain=False, feature_dim=128, embedding_dim=16, num_classes=8):
        super().__init__()
        self.backbone = PlainConv2DBackbone(feature_dim=feature_dim) if use_plain else DSConv2DBackbone(feature_dim=feature_dim)
        self.stage2_embedding = nn.Embedding(6, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
    def forward(self, x, stage2_class):
        features = self.backbone(x)
        cond_emb = self.stage2_embedding(stage2_class)
        return self.classifier(torch.cat([features, cond_emb], dim=-1))

# --- CONDITIONING ABLATION MODELS ---

class Stage2FlatModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class Stage3FlatModel(nn.Module):
    def __init__(self, feature_dim=128, num_classes=8):
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

# --- EXPERIMENT RUNNERS ---

def run_dsconv_single_variant(use_plain, seed, train_loader, val_loader, test_loader, device, epochs=15, s1_threshold=0.55):
    set_seed(seed)
    
    # Stage 1
    s1_model = Stage1VariantModel(use_plain=use_plain).to(device)
    weights1 = calc_weights(train_loader.dataset.targets1, 2, device)
    criterion1 = FocalLoss(alpha=weights1, gamma=2.0)
    optimizer1 = optim.Adam(s1_model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_s1_w = None
    for epoch in range(epochs):
        s1_model.train()
        for imgs, t1, _, _ in train_loader:
            imgs, t1 = imgs.to(device), t1.to(device)
            optimizer1.zero_grad()
            out = s1_model(imgs)
            loss = criterion1(out, t1)
            loss.backward()
            optimizer1.step()
            
        s1_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, t1, _, _ in val_loader:
                imgs, t1 = imgs.to(device), t1.to(device)
                val_loss += criterion1(s1_model(imgs), t1).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_s1_w = {k: v.cpu() for k, v in s1_model.state_dict().items()}
            
    s1_model.load_state_dict(best_s1_w)
    s1_model.to(device).eval()
    
    s1_preds, s1_targets = [], []
    with torch.no_grad():
        for imgs, t1, _, _ in test_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
            preds = (probs >= s1_threshold).long().cpu().numpy()
            s1_preds.extend(preds)
            s1_targets.extend(t1.numpy())
    s1_acc, s1_bacc, s1_f1 = eval_metrics(s1_targets, s1_preds)

    # Stage 2
    s2_model = Stage2VariantModel(use_plain=use_plain).to(device)
    weights2 = calc_weights(train_loader.dataset.targets2, 6, device)
    criterion2 = FocalLoss(alpha=weights2, gamma=2.0)
    optimizer2 = optim.Adam(s2_model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_s2_w = None
    for epoch in range(epochs):
        s2_model.train()
        for imgs, _, t2, _ in train_loader:
            imgs, t2 = imgs.to(device), t2.to(device)
            with torch.no_grad():
                probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                pred1 = (probs1 >= s1_threshold).long()
            optimizer2.zero_grad()
            out = s2_model(imgs, pred1)
            loss = criterion2(out, t2)
            loss.backward()
            optimizer2.step()
            
        s2_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _, t2, _ in val_loader:
                imgs, t2 = imgs.to(device), t2.to(device)
                probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                pred1 = (probs1 >= s1_threshold).long()
                val_loss += criterion2(s2_model(imgs, pred1), t2).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_s2_w = {k: v.cpu() for k, v in s2_model.state_dict().items()}
            
    s2_model.load_state_dict(best_s2_w)
    s2_model.to(device).eval()
    
    s2_preds, s2_targets = [], []
    with torch.no_grad():
        for imgs, _, t2, _ in test_loader:
            imgs = imgs.to(device)
            probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
            pred1 = (probs1 >= s1_threshold).long()
            out = s2_model(imgs, pred1)
            preds = out.argmax(dim=-1).cpu().numpy()
            s2_preds.extend(preds)
            s2_targets.extend(t2.numpy())
    s2_acc, s2_bacc, s2_f1 = eval_metrics(s2_targets, s2_preds)

    # Stage 3
    s3_model = Stage3VariantModel(use_plain=use_plain).to(device)
    weights3 = calc_weights(train_loader.dataset.targets3, 8, device)
    criterion3 = FocalLoss(alpha=weights3, gamma=2.0)
    optimizer3 = optim.Adam(s3_model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_s3_w = None
    for epoch in range(epochs):
        s3_model.train()
        for imgs, _, _, t3 in train_loader:
            imgs, t3 = imgs.to(device), t3.to(device)
            with torch.no_grad():
                probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                pred1 = (probs1 >= s1_threshold).long()
                pred2 = s2_model(imgs, pred1).argmax(dim=-1)
            optimizer3.zero_grad()
            out = s3_model(imgs, pred2)
            loss = criterion3(out, t3)
            loss.backward()
            optimizer3.step()
            
        s3_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _, _, t3 in val_loader:
                imgs, t3 = imgs.to(device), t3.to(device)
                probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                pred1 = (probs1 >= s1_threshold).long()
                pred2 = s2_model(imgs, pred1).argmax(dim=-1)
                val_loss += criterion3(s3_model(imgs, pred2), t3).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_s3_w = {k: v.cpu() for k, v in s3_model.state_dict().items()}
            
    s3_model.load_state_dict(best_s3_w)
    s3_model.to(device).eval()
    
    s3_preds, s3_targets = [], []
    with torch.no_grad():
        for imgs, _, _, t3 in test_loader:
            imgs = imgs.to(device)
            probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
            pred1 = (probs1 >= s1_threshold).long()
            pred2 = s2_model(imgs, pred1).argmax(dim=-1)
            out = s3_model(imgs, pred2)
            preds = out.argmax(dim=-1).cpu().numpy()
            s3_preds.extend(preds)
            s3_targets.extend(t3.numpy())
    s3_acc, s3_bacc, s3_f1 = eval_metrics(s3_targets, s3_preds)

    return {
        "s1_acc": s1_acc, "s1_bacc": s1_bacc, "s1_f1": s1_f1,
        "s2_acc": s2_acc, "s2_bacc": s2_bacc, "s2_f1": s2_f1,
        "s3_acc": s3_acc, "s3_bacc": s3_bacc, "s3_f1": s3_f1,
    }

def run_conditioning_single_variant(use_conditioned, seed, train_loader, val_loader, test_loader, device, epochs=15, s1_threshold=0.55):
    set_seed(seed)
    
    # Base Stage 1 DSConv Model
    s1_model = Stage1VariantModel(use_plain=False).to(device)
    weights1 = calc_weights(train_loader.dataset.targets1, 2, device)
    criterion1 = FocalLoss(alpha=weights1, gamma=2.0)
    optimizer1 = optim.Adam(s1_model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_s1_w = None
    for epoch in range(epochs):
        s1_model.train()
        for imgs, t1, _, _ in train_loader:
            imgs, t1 = imgs.to(device), t1.to(device)
            optimizer1.zero_grad()
            out = s1_model(imgs)
            loss = criterion1(out, t1)
            loss.backward()
            optimizer1.step()
            
        s1_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, t1, _, _ in val_loader:
                imgs, t1 = imgs.to(device), t1.to(device)
                val_loss += criterion1(s1_model(imgs), t1).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_s1_w = {k: v.cpu() for k, v in s1_model.state_dict().items()}
            
    s1_model.load_state_dict(best_s1_w)
    s1_model.to(device).eval()
    
    s1_preds, s1_targets = [], []
    with torch.no_grad():
        for imgs, t1, _, _ in test_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
            preds = (probs >= s1_threshold).long().cpu().numpy()
            s1_preds.extend(preds)
            s1_targets.extend(t1.numpy())
    s1_acc, s1_bacc, s1_f1 = eval_metrics(s1_targets, s1_preds)

    # Stage 2
    if use_conditioned:
        s2_model = Stage2VariantModel(use_plain=False).to(device)
    else:
        s2_model = Stage2FlatModel().to(device)
        
    weights2 = calc_weights(train_loader.dataset.targets2, 6, device)
    criterion2 = FocalLoss(alpha=weights2, gamma=2.0)
    optimizer2 = optim.Adam(s2_model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_s2_w = None
    for epoch in range(epochs):
        s2_model.train()
        for imgs, _, t2, _ in train_loader:
            imgs, t2 = imgs.to(device), t2.to(device)
            if use_conditioned:
                with torch.no_grad():
                    probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                    pred1 = (probs1 >= s1_threshold).long()
                out = s2_model(imgs, pred1)
            else:
                out = s2_model(imgs)
            optimizer2.zero_grad()
            loss = criterion2(out, t2)
            loss.backward()
            optimizer2.step()
            
        s2_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _, t2, _ in val_loader:
                imgs, t2 = imgs.to(device), t2.to(device)
                if use_conditioned:
                    probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                    pred1 = (probs1 >= s1_threshold).long()
                    out = s2_model(imgs, pred1)
                else:
                    out = s2_model(imgs)
                val_loss += criterion2(out, t2).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_s2_w = {k: v.cpu() for k, v in s2_model.state_dict().items()}
            
    s2_model.load_state_dict(best_s2_w)
    s2_model.to(device).eval()
    
    s2_preds, s2_targets = [], []
    with torch.no_grad():
        for imgs, _, t2, _ in test_loader:
            imgs = images.to(device) if False else imgs.to(device)
            if use_conditioned:
                probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                pred1 = (probs1 >= s1_threshold).long()
                out = s2_model(imgs, pred1)
            else:
                out = s2_model(imgs)
            preds = out.argmax(dim=-1).cpu().numpy()
            s2_preds.extend(preds)
            s2_targets.extend(t2.numpy())
    s2_acc, s2_bacc, s2_f1 = eval_metrics(s2_targets, s2_preds)

    # Stage 3
    if use_conditioned:
        s3_model = Stage3VariantModel(use_plain=False).to(device)
    else:
        s3_model = Stage3FlatModel().to(device)
        
    weights3 = calc_weights(train_loader.dataset.targets3, 8, device)
    criterion3 = FocalLoss(alpha=weights3, gamma=2.0)
    optimizer3 = optim.Adam(s3_model.parameters(), lr=0.001)
    
    best_val_loss = float("inf")
    best_s3_w = None
    for epoch in range(epochs):
        s3_model.train()
        for imgs, _, t2, t3 in train_loader:
            imgs, t2, t3 = imgs.to(device), t2.to(device), t3.to(device)
            if use_conditioned:
                with torch.no_grad():
                    probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                    pred1 = (probs1 >= s1_threshold).long()
                    pred2 = s2_model(imgs, pred1).argmax(dim=-1)
                out = s3_model(imgs, pred2)
            else:
                out = s3_model(imgs)
            optimizer3.zero_grad()
            loss = criterion3(out, t3)
            loss.backward()
            optimizer3.step()
            
        s3_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _, t2, t3 in val_loader:
                imgs, t2, t3 = imgs.to(device), t2.to(device), t3.to(device)
                if use_conditioned:
                    probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                    pred1 = (probs1 >= s1_threshold).long()
                    pred2 = s2_model(imgs, pred1).argmax(dim=-1)
                    out = s3_model(imgs, pred2)
                else:
                    out = s3_model(imgs)
                val_loss += criterion3(out, t3).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_s3_w = {k: v.cpu() for k, v in s3_model.state_dict().items()}
            
    s3_model.load_state_dict(best_s3_w)
    s3_model.to(device).eval()
    
    s3_preds, s3_targets = [], []
    with torch.no_grad():
        for imgs, _, _, t3 in test_loader:
            imgs = imgs.to(device)
            if use_conditioned:
                probs1 = torch.softmax(s1_model(imgs), dim=-1)[:, 1]
                pred1 = (probs1 >= s1_threshold).long()
                pred2 = s2_model(imgs, pred1).argmax(dim=-1)
                out = s3_model(imgs, pred2)
            else:
                out = s3_model(imgs)
            preds = out.argmax(dim=-1).cpu().numpy()
            s3_preds.extend(preds)
            s3_targets.extend(t3.numpy())
    s3_acc, s3_bacc, s3_f1 = eval_metrics(s3_targets, s3_preds)

    return {
        "s1_acc": s1_acc, "s1_bacc": s1_bacc, "s1_f1": s1_f1,
        "s2_acc": s2_acc, "s2_bacc": s2_bacc, "s2_f1": s2_f1,
        "s3_acc": s3_acc, "s3_bacc": s3_bacc, "s3_f1": s3_f1,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing optimized 5-seed sweep on device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    raw_train = ImageFolder(root="data/final/train", transform=transform)
    raw_val = ImageFolder(root="data/final/val", transform=transform)
    raw_test = ImageFolder(root="data/final/test", transform=transform)
    
    train_dataset = HierarchicalDatasetWrapper(raw_train)
    val_dataset = HierarchicalDatasetWrapper(raw_val)
    test_dataset = HierarchicalDatasetWrapper(raw_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    seeds = [0, 1, 2, 3, 4]
    
    # --- Part 1: DSConv 5-Seed Sweep ---
    print("\n=======================================================")
    print("  PART 1: DSCONV ABLATION SWEEP (Seeds 0, 1, 2, 3, 4)")
    print("=======================================================")
    dsconv_results = {"full": [], "plain": []}
    for s in seeds:
        print(f"Seed {s}: Training Full DSConv Backbone...")
        res_full = run_dsconv_single_variant(False, s, train_loader, val_loader, test_loader, device)
        print(f"  -> S1 Acc: {res_full['s1_acc']*100:.2f}%, S2 Acc: {res_full['s2_acc']*100:.2f}%, S3 Acc: {res_full['s3_acc']*100:.2f}%")
        
        print(f"Seed {s}: Training Plain Conv Stack Variant...")
        res_plain = run_dsconv_single_variant(True, s, train_loader, val_loader, test_loader, device)
        print(f"  -> S1 Acc: {res_plain['s1_acc']*100:.2f}%, S2 Acc: {res_plain['s2_acc']*100:.2f}%, S3 Acc: {res_plain['s3_acc']*100:.2f}%")
        
        dsconv_results["full"].append(res_full)
        dsconv_results["plain"].append(res_plain)

    # --- Part 2: Conditioning 5-Seed Sweep ---
    print("\n=======================================================")
    print("  PART 2: CONDITIONING ABLATION SWEEP (Seeds 0, 1, 2, 3, 4)")
    print("=======================================================")
    cond_results = {"conditioned": [], "flat": []}
    for s in seeds:
        print(f"Seed {s}: Training Conditioned Heads...")
        res_cond = run_conditioning_single_variant(True, s, train_loader, val_loader, test_loader, device)
        print(f"  -> S2 Acc: {res_cond['s2_acc']*100:.2f}%, S3 Acc: {res_cond['s3_acc']*100:.2f}%")
        
        print(f"Seed {s}: Training Flat Heads...")
        res_flat = run_conditioning_single_variant(False, s, train_loader, val_loader, test_loader, device)
        print(f"  -> S2 Acc: {res_flat['s2_acc']*100:.2f}%, S3 Acc: {res_flat['s3_acc']*100:.2f}%")
        
        cond_results["conditioned"].append(res_cond)
        cond_results["flat"].append(res_flat)

    full_payload = {
        "seeds": seeds,
        "dsconv": dsconv_results,
        "conditioning": cond_results
    }
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_file = os.path.join(root_dir, "results", "ablation_5seed_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
        
    print(f"\nSuccessfully finished all 5-seed ablations! Saved results to {out_file}")

if __name__ == "__main__":
    main()
