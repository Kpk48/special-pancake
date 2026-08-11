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

def eval_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, bacc, float(f1)

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

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    raw_train = ImageFolder(root="data/final/train", transform=transform)
    raw_val = ImageFolder(root="data/final/val", transform=transform)
    raw_test = ImageFolder(root="data/final/test", transform=transform)
    
    train_loader = DataLoader(HierarchicalDatasetWrapper(raw_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(HierarchicalDatasetWrapper(raw_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(HierarchicalDatasetWrapper(raw_test), batch_size=64, shuffle=False)
    
    seeds = [0, 1, 2, 3, 4]
    epochs = 10
    
    dsconv_results = []
    cond_results = []
    
    for s in seeds:
        print(f"Running Seed {s}...")
        
        # --- DSCONV ABLATION ---
        # DSConv
        set_seed(s)
        w1 = calc_weights([item[1] for item in train_loader.dataset], 2, device)
        s1_ds = DSConvStage1(use_plain=False).to(device)
        opt1 = optim.Adam(s1_ds.parameters(), lr=0.001)
        crit1 = FocalLoss(alpha=w1, gamma=2.0)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s1_ds.train()
            for imgs, t1, _, _ in train_loader:
                imgs, t1 = imgs.to(device), t1.to(device)
                opt1.zero_grad(); loss = crit1(s1_ds(imgs), t1); loss.backward(); opt1.step()
            s1_ds.eval()
            vl = sum(crit1(s1_ds(imgs.to(device)), t1.to(device)).item() * imgs.size(0) for imgs, t1, _, _ in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s1_ds.state_dict().items()}
        s1_ds.load_state_dict(best_w); s1_ds.to(device).eval()
        
        preds1, targets1 = [], []
        with torch.no_grad():
            for imgs, t1, _, _ in test_loader:
                probs = torch.softmax(s1_ds(imgs.to(device)), dim=-1)[:, 1]
                preds1.extend((probs >= 0.55).int().cpu().numpy()); targets1.extend(t1.numpy())
        s1_ds_acc, s1_ds_bacc, s1_ds_f1 = eval_metrics(targets1, preds1)

        s2_ds = DSConvStage2(use_plain=False).to(device)
        w2 = calc_weights([item[2] for item in train_loader.dataset], 6, device)
        opt2 = optim.Adam(s2_ds.parameters(), lr=0.001)
        crit2 = FocalLoss(alpha=w2, gamma=2.0)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s2_ds.train()
            for imgs, _, t2, _ in train_loader:
                imgs, t2 = imgs.to(device), t2.to(device)
                with torch.no_grad(): p1 = (torch.softmax(s1_ds(imgs), dim=-1)[:, 1] >= 0.55).long()
                opt2.zero_grad(); loss = crit2(s2_ds(imgs, p1), t2); loss.backward(); opt2.step()
            s2_ds.eval()
            vl = sum(crit2(s2_ds(imgs.to(device), (torch.softmax(s1_ds(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()), t2.to(device)).item() * imgs.size(0) for imgs, _, t2, _ in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s2_ds.state_dict().items()}
        s2_ds.load_state_dict(best_w); s2_ds.to(device).eval()

        preds2, targets2 = [], []
        with torch.no_grad():
            for imgs, _, t2, _ in test_loader:
                p1 = (torch.softmax(s1_ds(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
                preds2.extend(s2_ds(imgs.to(device), p1).argmax(dim=-1).cpu().numpy()); targets2.extend(t2.numpy())
        s2_ds_acc, s2_ds_bacc, s2_ds_f1 = eval_metrics(targets2, preds2)

        s3_ds = DSConvStage3(use_plain=False).to(device)
        w3 = calc_weights([item[3] for item in train_loader.dataset], 8, device)
        opt3 = optim.Adam(s3_ds.parameters(), lr=0.001)
        crit3 = FocalLoss(alpha=w3, gamma=2.0)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s3_ds.train()
            for imgs, _, _, t3 in train_loader:
                imgs, t3 = imgs.to(device), t3.to(device)
                with torch.no_grad():
                    p1 = (torch.softmax(s1_ds(imgs), dim=-1)[:, 1] >= 0.55).long()
                    p2 = s2_ds(imgs, p1).argmax(dim=-1)
                opt3.zero_grad(); loss = crit3(s3_ds(imgs, p2), t3); loss.backward(); opt3.step()
            s3_ds.eval()
            vl = sum(crit3(s3_ds(imgs.to(device), s2_ds(imgs.to(device), (torch.softmax(s1_ds(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()).argmax(dim=-1)), t3.to(device)).item() * imgs.size(0) for imgs, _, _, t3 in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s3_ds.state_dict().items()}
        s3_ds.load_state_dict(best_w); s3_ds.to(device).eval()

        preds3, targets3 = [], []
        with torch.no_grad():
            for imgs, _, _, t3 in test_loader:
                p1 = (torch.softmax(s1_ds(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
                p2 = s2_ds(imgs.to(device), p1).argmax(dim=-1)
                preds3.extend(s3_ds(imgs.to(device), p2).argmax(dim=-1).cpu().numpy()); targets3.extend(t3.numpy())
        s3_ds_acc, s3_ds_bacc, s3_ds_f1 = eval_metrics(targets3, preds3)

        # Plain Conv
        set_seed(s + 100)
        s1_pl = DSConvStage1(use_plain=True).to(device)
        opt1 = optim.Adam(s1_pl.parameters(), lr=0.001)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s1_pl.train()
            for imgs, t1, _, _ in train_loader:
                imgs, t1 = imgs.to(device), t1.to(device)
                opt1.zero_grad(); loss = crit1(s1_pl(imgs), t1); loss.backward(); opt1.step()
            s1_pl.eval()
            vl = sum(crit1(s1_pl(imgs.to(device)), t1.to(device)).item() * imgs.size(0) for imgs, t1, _, _ in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s1_pl.state_dict().items()}
        s1_pl.load_state_dict(best_w); s1_pl.to(device).eval()

        preds1_pl, targets1_pl = [], []
        with torch.no_grad():
            for imgs, t1, _, _ in test_loader:
                probs = torch.softmax(s1_pl(imgs.to(device)), dim=-1)[:, 1]
                preds1_pl.extend((probs >= 0.55).int().cpu().numpy()); targets1_pl.extend(t1.numpy())
        s1_pl_acc, s1_pl_bacc, s1_pl_f1 = eval_metrics(targets1_pl, preds1_pl)

        s2_pl = DSConvStage2(use_plain=True).to(device)
        opt2 = optim.Adam(s2_pl.parameters(), lr=0.001)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s2_pl.train()
            for imgs, _, t2, _ in train_loader:
                imgs, t2 = imgs.to(device), t2.to(device)
                with torch.no_grad(): p1 = (torch.softmax(s1_pl(imgs), dim=-1)[:, 1] >= 0.55).long()
                opt2.zero_grad(); loss = crit2(s2_pl(imgs, p1), t2); loss.backward(); opt2.step()
            s2_pl.eval()
            vl = sum(crit2(s2_pl(imgs.to(device), (torch.softmax(s1_pl(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()), t2.to(device)).item() * imgs.size(0) for imgs, _, t2, _ in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s2_pl.state_dict().items()}
        s2_pl.load_state_dict(best_w); s2_pl.to(device).eval()

        preds2_pl, targets2_pl = [], []
        with torch.no_grad():
            for imgs, _, t2, _ in test_loader:
                p1 = (torch.softmax(s1_pl(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
                preds2_pl.extend(s2_pl(imgs.to(device), p1).argmax(dim=-1).cpu().numpy()); targets2_pl.extend(t2.numpy())
        s2_pl_acc, s2_pl_bacc, s2_pl_f1 = eval_metrics(targets2_pl, preds2_pl)

        s3_pl = DSConvStage3(use_plain=True).to(device)
        opt3 = optim.Adam(s3_pl.parameters(), lr=0.001)
        best_loss, best_w = float("inf"), None
        for _ in range(epochs):
            s3_pl.train()
            for imgs, _, _, t3 in train_loader:
                imgs, t3 = imgs.to(device), t3.to(device)
                with torch.no_grad():
                    p1 = (torch.softmax(s1_pl(imgs), dim=-1)[:, 1] >= 0.55).long()
                    p2 = s2_pl(imgs, p1).argmax(dim=-1)
                opt3.zero_grad(); loss = crit3(s3_pl(imgs, p2), t3); loss.backward(); opt3.step()
            s3_pl.eval()
            vl = sum(crit3(s3_pl(imgs.to(device), s2_pl(imgs.to(device), (torch.softmax(s1_pl(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()).argmax(dim=-1)), t3.to(device)).item() * imgs.size(0) for imgs, _, _, t3 in val_loader) / len(val_loader.dataset)
            if vl < best_loss: best_loss = vl; best_w = {k: v.cpu() for k, v in s3_pl.state_dict().items()}
        s3_pl.load_state_dict(best_w); s3_pl.to(device).eval()

        preds3_pl, targets3_pl = [], []
        with torch.no_grad():
            for imgs, _, _, t3 in test_loader:
                p1 = (torch.softmax(s1_pl(imgs.to(device)), dim=-1)[:, 1] >= 0.55).long()
                p2 = s2_pl(imgs.to(device), p1).argmax(dim=-1)
                preds3_pl.extend(s3_pl(imgs.to(device), p2).argmax(dim=-1).cpu().numpy()); targets3_pl.extend(t3.numpy())
        s3_pl_acc, s3_pl_bacc, s3_pl_f1 = eval_metrics(targets3_pl, preds3_pl)

        dsconv_results.append({
            "dsconv": {"s1_acc": s1_ds_acc, "s1_bacc": s1_ds_bacc, "s1_f1": s1_ds_f1, "s2_acc": s2_ds_acc, "s2_bacc": s2_ds_bacc, "s2_f1": s2_ds_f1, "s3_acc": s3_ds_acc, "s3_bacc": s3_ds_bacc, "s3_f1": s3_ds_f1},
            "plain": {"s1_acc": s1_pl_acc, "s1_bacc": s1_pl_bacc, "s1_f1": s1_pl_f1, "s2_acc": s2_pl_acc, "s2_bacc": s2_pl_bacc, "s2_f1": s2_pl_f1, "s3_acc": s3_pl_acc, "s3_bacc": s3_pl_bacc, "s3_f1": s3_pl_f1}
        })

        # --- CONDITIONING ABLATION ---
        set_seed(s + 200)
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

        preds2_f, targets2_f = [], []
        with torch.no_grad():
            for imgs, _, t2, _ in test_loader:
                preds2_f.extend(s2_flat(imgs.to(device)).argmax(dim=-1).cpu().numpy()); targets2_f.extend(t2.numpy())
        s2_f_acc, s2_f_bacc, s2_f_f1 = eval_metrics(targets2_f, preds2_f)

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

        preds3_f, targets3_f = [], []
        with torch.no_grad():
            for imgs, _, _, t3 in test_loader:
                preds3_f.extend(s3_flat(imgs.to(device)).argmax(dim=-1).cpu().numpy()); targets3_f.extend(t3.numpy())
        s3_f_acc, s3_f_bacc, s3_f_f1 = eval_metrics(targets3_f, preds3_f)

        cond_results.append({
            "conditioned": {"s2_acc": s2_ds_acc, "s2_bacc": s2_ds_bacc, "s2_f1": s2_ds_f1, "s3_acc": s3_ds_acc, "s3_bacc": s3_ds_bacc, "s3_f1": s3_ds_f1},
            "flat": {"s2_acc": s2_f_acc, "s2_bacc": s2_f_bacc, "s2_f1": s2_f_f1, "s3_acc": s3_f_acc, "s3_bacc": s3_f_bacc, "s3_f1": s3_f_f1}
        })

    payload = {"dsconv": dsconv_results, "conditioning": cond_results}
    target_file = r"C:\Users\mrbub\special-pancake\results\ablation_5seed_results.json"
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON WRITTEN SUCCESSFULLY TO {target_file}")

if __name__ == "__main__":
    run_all()
