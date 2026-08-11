import sys
import os
import random
import json
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

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target3 = self.base_dataset[idx]
        class_name = self.classes[target3]
        filepath = self.base_dataset.samples[idx][0]
        target1 = get_stage1_label(class_name, filepath)
        target2 = get_stage2_label(class_name)
        return img, target1, target2, target3

def calc_weights(targets, num_cls, device):
    counts = torch.zeros(num_cls)
    for t in targets:
        counts[t] += 1
    return (len(targets) / (num_cls * torch.clamp(counts, min=1.0))).to(device)

def evaluate_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, bacc, float(f1)

# Models
class DSConvStage1(nn.Module):
    def __init__(self, use_plain=False, feature_dim=128):
        super().__init__()
        self.backbone = PlainConv2DBackbone(feature_dim=feature_dim) if use_plain else DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)
    def forward(self, x): return self.classifier(self.backbone(x))

class DSConvStage2(nn.Module):
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

class DSConvStage3(nn.Module):
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

class FlatStage2(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
    def forward(self, x): return self.classifier(self.backbone(x))

class FlatStage3(nn.Module):
    def __init__(self, feature_dim=128, num_classes=8):
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.classifier(self.backbone(x))

def train_eval_dsconv(seed, train_loader, val_loader, test_loader, device, epochs=10):
    res = {}
    for is_plain, name in [(False, "dsconv"), (True, "plain")]:
        set_seed(seed + (100 if is_plain else 0))
        # S1
        s1 = DSConvStage1(use_plain=is_plain).to(device)
        w1 = calc_weights([s[1] for s in train_loader.dataset], 2, device)
        opt1 = optim.Adam(s1.parameters(), lr=0.001)
        crit1 = FocalLoss(alpha=w1, gamma=2.0)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s1.train()
            for imgs, t1, _, _ in train_loader:
                imgs, t1 = imgs.to(device), t1.to(device)
                opt1.zero_grad(); loss = crit1(s1(imgs), t1); loss.backward(); opt1.step()
            s1.eval()
            vl = sum(crit1(s1(imgs.to(device)), t1.to(device)).item() * imgs.size(0) for imgs, t1, _, _ in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s1.state_dict().items()}
        s1.load_state_dict(best_w); s1.to(device).eval()
        
        preds1, targets1 = [], []
        with torch.no_grad():
            for imgs, t1, _, _ in test_loader:
                probs = torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1]
                preds1.extend((probs >= 0.55).int().cpu().numpy()); targets1.extend(t1.numpy())
        s1_acc, s1_bacc, s1_f1 = evaluate_metrics(targets1, preds1)

        # S2
        s2 = DSConvStage2(use_plain=is_plain).to(device)
        w2 = calc_weights([s[2] for s in train_loader.dataset], 6, device)
        opt2 = optim.Adam(s2.parameters(), lr=0.001)
        crit2 = FocalLoss(alpha=w2, gamma=2.0)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s2.train()
            for imgs, _, t2, _ in train_loader:
                imgs, t2 = imgs.to(device), t2.to(device)
                with torch.no_grad(): p1 = (torch.softmax(s1(imgs), dim=-1)[:, 1] >= 0.55).long()
                opt2.zero_grad(); loss = crit2(s2(imgs, p1), t2); loss.backward(); opt2.step()
            s2.eval()
            vl = sum(crit2(s2(imgs.to(device), (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()), t2.to(device)).item() * imgs.size(0) for imgs, _, t2, _ in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s2.state_dict().items()}
        s2.load_state_dict(best_w); s2.to(device).eval()

        preds2, targets2 = [], []
        with torch.no_grad():
            for imgs, _, t2, _ in test_loader:
                p1 = (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
                preds2.extend(s2(imgs.to(device), p1).argmax(dim=-1).cpu().numpy()); targets2.extend(t2.numpy())
        s2_acc, s2_bacc, s2_f1 = evaluate_metrics(targets2, preds2)

        # S3
        s3 = DSConvStage3(use_plain=is_plain).to(device)
        w3 = calc_weights([s[3] for s in train_loader.dataset], 8, device)
        opt3 = optim.Adam(s3.parameters(), lr=0.001)
        crit3 = FocalLoss(alpha=w3, gamma=2.0)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s3.train()
            for imgs, _, _, t3 in train_loader:
                imgs, t3 = imgs.to(device), t3.to(device)
                with torch.no_grad():
                    p1 = (torch.softmax(s1(imgs), dim=-1)[:, 1] >= 0.55).long()
                    p2 = s2(imgs, p1).argmax(dim=-1)
                opt3.zero_grad(); loss = crit3(s3(imgs, p2), t3); loss.backward(); opt3.step()
            s3.eval()
            vl = sum(crit3(s3(imgs.to(device), s2(imgs.to(device), (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()).argmax(dim=-1)), t3.to(device)).item() * imgs.size(0) for imgs, _, _, t3 in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s3.state_dict().items()}
        s3.load_state_dict(best_w); s3.to(device).eval()

        preds3, targets3 = [], []
        with torch.no_grad():
            for imgs, _, _, t3 in test_loader:
                p1 = (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
                p2 = s2(imgs.to(device), p1).argmax(dim=-1)
                preds3.extend(s3(imgs.to(device), p2).argmax(dim=-1).cpu().numpy()); targets3.extend(t3.numpy())
        s3_acc, s3_bacc, s3_f1 = evaluate_metrics(targets3, preds3)

        res[name] = {
            "s1_acc": s1_acc, "s1_bacc": s1_bacc, "s1_f1": s1_f1,
            "s2_acc": s2_acc, "s2_bacc": s2_bacc, "s2_f1": s2_f1,
            "s3_acc": s3_acc, "s3_bacc": s3_bacc, "s3_f1": s3_f1,
        }
    return res

def train_eval_conditioning(seed, train_loader, val_loader, test_loader, device, epochs=10):
    # Conditioned S2 & S3
    set_seed(seed)
    w1 = calc_weights([s[1] for s in train_loader.dataset], 2, device)
    s1 = DSConvStage1(use_plain=False).to(device)
    opt1 = optim.Adam(s1.parameters(), lr=0.001)
    crit1 = FocalLoss(alpha=w1, gamma=2.0)
    best_loss, best_w = float("inf"), None
    for _ in range(epochs):
        s1.train()
        for imgs, t1, _, _ in train_loader:
            imgs, t1 = imgs.to(device), t1.to(device)
            opt1.zero_grad(); loss = crit1(s1(imgs), t1); loss.backward(); opt1.step()
        s1.eval()
        vl = sum(crit1(s1(imgs.to(device)), t1.to(device)).item() * imgs.size(0) for imgs, t1, _, _ in val_loader) / len(val_loader.dataset)
        if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s1.state_dict().items()}
    s1.load_state_dict(best_w); s1.to(device).eval()

    s2_cond = DSConvStage2(use_plain=False).to(device)
    w2 = calc_weights([s[2] for s in train_loader.dataset], 6, device)
    opt2 = optim.Adam(s2_cond.parameters(), lr=0.001)
    crit2 = FocalLoss(alpha=w2, gamma=2.0)
    best_loss, best_w = float("inf"), None
    for _ in range(epochs):
        s2_cond.train()
        for imgs, _, t2, _ in train_loader:
            imgs, t2 = imgs.to(device), t2.to(device)
            with torch.no_grad(): p1 = (torch.softmax(s1(imgs), dim=-1)[:, 1] >= 0.55).long()
            opt2.zero_grad(); loss = crit2(s2_cond(imgs, p1), t2); loss.backward(); opt2.step()
        s2_cond.eval()
        vl = sum(crit2(s2_cond(imgs.to(device), (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()), t2.to(device)).item() * imgs.size(0) for imgs, _, t2, _ in val_loader) / len(val_loader.dataset)
        if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s2_cond.state_dict().items()}
    s2_cond.load_state_dict(best_w); s2_cond.to(device).eval()

    preds2, targets2 = [], []
    with torch.no_grad():
        for imgs, _, t2, _ in test_loader:
            p1 = (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
            preds2.extend(s2_cond(imgs.to(device), p1).argmax(dim=-1).cpu().numpy()); targets2.extend(t2.numpy())
    s2_cond_acc, s2_cond_bacc, s2_cond_f1 = evaluate_metrics(targets2, preds2)

    s3_cond = DSConvStage3(use_plain=False).to(device)
    w3 = calc_weights([s[3] for s in train_loader.dataset], 8, device)
    opt3 = optim.Adam(s3_cond.parameters(), lr=0.001)
    crit3 = FocalLoss(alpha=w3, gamma=2.0)
    best_loss, best_w = float("inf"), None
    for _ in range(epochs):
        s3_cond.train()
        for imgs, _, _, t3 in train_loader:
            imgs, t3 = imgs.to(device), t3.to(device)
            with torch.no_grad():
                p1 = (torch.softmax(s1(imgs), dim=-1)[:, 1] >= 0.55).long()
                p2 = s2_cond(imgs, p1).argmax(dim=-1)
            opt3.zero_grad(); loss = crit3(s3_cond(imgs, p2), t3); loss.backward(); opt3.step()
        s3_cond.eval()
        vl = sum(crit3(s3_cond(imgs.to(device), s2_cond(imgs.to(device), (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()).argmax(dim=-1)), t3.to(device)).item() * imgs.size(0) for imgs, _, _, t3 in val_loader) / len(val_loader.dataset)
        if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s3_cond.state_dict().items()}
    s3_cond.load_state_dict(best_w); s3_cond.to(device).eval()

    preds3, targets3 = [], []
    with torch.no_grad():
        for imgs, _, _, t3 in test_loader:
            p1 = (torch.softmax(s1(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
            p2 = s2_cond(imgs.to(device), p1).argmax(dim=-1)
            preds3.extend(s3_cond(imgs.to(device), p2).argmax(dim=-1).cpu().numpy()); targets3.extend(t3.numpy())
    s3_cond_acc, s3_cond_bacc, s3_cond_f1 = evaluate_metrics(targets3, preds3)

    # Flat S2 & S3
    set_seed(seed + 200)
    s2_flat = FlatStage2().to(device)
    opt2_f = optim.Adam(s2_flat.parameters(), lr=0.001)
    best_loss, best_w = float("inf"), None
    for _ in range(epochs):
        s2_flat.train()
        for imgs, _, t2, _ in train_loader:
            imgs, t2 = imgs.to(device), t2.to(device)
            opt2_f.zero_grad(); loss = crit2(s2_flat(imgs), t2); loss.backward(); opt2_f.step()
        s2_flat.eval()
        vl = sum(crit2(s2_flat(imgs.to(device)), t2.to(device)).item() * imgs.size(0) for imgs, _, t2, _ in val_loader) / len(val_loader.dataset)
        if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s2_flat.state_dict().items()}
    s2_flat.load_state_dict(best_w); s2_flat.to(device).eval()

    preds2, targets2 = [], []
    with torch.no_grad():
        for imgs, _, t2, _ in test_loader:
            preds2.extend(s2_flat(imgs.to(device)).argmax(dim=-1).cpu().numpy()); targets2.extend(t2.numpy())
    s2_flat_acc, s2_flat_bacc, s2_flat_f1 = evaluate_metrics(targets2, preds2)

    s3_flat = FlatStage3().to(device)
    opt3_f = optim.Adam(s3_flat.parameters(), lr=0.001)
    best_loss, best_w = float("inf"), None
    for _ in range(epochs):
        s3_flat.train()
        for imgs, _, _, t3 in train_loader:
            imgs, t3 = imgs.to(device), t3.to(device)
            opt3_f.zero_grad(); loss = crit3(s3_flat(imgs), t3); loss.backward(); opt3_f.step()
        s3_flat.eval()
        vl = sum(crit3(s3_flat(imgs.to(device)), t3.to(device)).item() * imgs.size(0) for imgs, _, _, t3 in val_loader) / len(val_loader.dataset)
        if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s3_flat.state_dict().items()}
    s3_flat.load_state_dict(best_w); s3_flat.to(device).eval()

    preds3, targets3 = [], []
    with torch.no_grad():
        for imgs, _, _, t3 in test_loader:
            preds3.extend(s3_flat(imgs.to(device)).argmax(dim=-1).cpu().numpy()); targets3.extend(t3.numpy())
    s3_flat_acc, s3_flat_bacc, s3_flat_f1 = evaluate_metrics(targets3, preds3)

    return {
        "conditioned": {
            "s2_acc": s2_cond_acc, "s2_bacc": s2_cond_bacc, "s2_f1": s2_cond_f1,
            "s3_acc": s3_cond_acc, "s3_bacc": s3_cond_bacc, "s3_f1": s3_cond_f1,
        },
        "flat": {
            "s2_acc": s2_flat_acc, "s2_bacc": s2_flat_bacc, "s2_f1": s2_flat_f1,
            "s3_acc": s3_flat_acc, "s3_bacc": s3_flat_bacc, "s3_f1": s3_flat_f1,
        }
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing 5-seed sweep on {device}...")
    
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    raw_train = ImageFolder(root="data/final/train", transform=transform)
    raw_val = ImageFolder(root="data/final/val", transform=transform)
    raw_test = ImageFolder(root="data/final/test", transform=transform)
    
    train_loader = DataLoader(HierarchicalDatasetWrapper(raw_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(HierarchicalDatasetWrapper(raw_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(HierarchicalDatasetWrapper(raw_test), batch_size=64, shuffle=False)
    
    seeds = [0, 1, 2, 3, 4]
    
    dsconv_results = []
    for s in seeds:
        print(f"Running DSConv Seed {s}...")
        r = train_eval_dsconv(s, train_loader, val_loader, test_loader, device, epochs=10)
        dsconv_results.append(r)
        print(f"  Seed {s} DSConv S1={r['dsconv']['s1_acc']*100:.2f}%, Plain S1={r['plain']['s1_acc']*100:.2f}%")

    cond_results = []
    for s in seeds:
        print(f"Running Conditioning Seed {s}...")
        r = train_eval_conditioning(s, train_loader, val_loader, test_loader, device, epochs=10)
        cond_results.append(r)
        print(f"  Seed {s} Cond S2={r['conditioned']['s2_acc']*100:.2f}%, Flat S2={r['flat']['s2_acc']*100:.2f}%")

    payload = {"dsconv": dsconv_results, "conditioning": cond_results}
    
    out_file = os.path.abspath("results/ablation_5seed_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSUCCESS! Saved results to {out_file}")

if __name__ == "__main__":
    main()
