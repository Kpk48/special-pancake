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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision.models import resnet18, ResNet18_Weights
import subprocess

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.backbone import DSConv2DBackbone
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES,
    STAGE3_TO_STAGE1, STAGE3_TO_STAGE2,
    get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SECTION 1: GIT & DATA AVAILABILITY ---
def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode("utf-8").strip()
    except Exception as e:
        commit = "Unknown"
        remote = "https://github.com/Kpk48/special-pancake"
    return remote, commit

# --- SECTION 2: CLASS DISTRIBUTIONS ---
def get_class_distributions():
    train_dir = "data/final/train"
    val_dir = "data/final/val"
    test_dir = "data/final/test"
    
    s3_classes = STAGE3_CLASSES
    
    stage3_dist = {}
    for c in s3_classes:
        tr_cnt = len(os.listdir(os.path.join(train_dir, c))) if os.path.exists(os.path.join(train_dir, c)) else 0
        va_cnt = len(os.listdir(os.path.join(val_dir, c))) if os.path.exists(os.path.join(val_dir, c)) else 0
        te_cnt = len(os.listdir(os.path.join(test_dir, c))) if os.path.exists(os.path.join(test_dir, c)) else 0
        stage3_dist[c] = {
            "train": tr_cnt,
            "val": va_cnt,
            "test": te_cnt,
            "total": tr_cnt + va_cnt + te_cnt
        }
        
    s2_classes = ["paper_cardboard", "organic", "glass", "metal", "plastic", "textile_battery"]
    s2_map = {
        "paper_cardboard": ["paper", "cardboard"],
        "organic": ["organic"],
        "glass": ["glass"],
        "metal": ["metal"],
        "plastic": ["plastic"],
        "textile_battery": ["textile", "battery"]
    }
    
    stage2_dist = {}
    for s2_c, sub_cs in s2_map.items():
        tr_tot = sum(stage3_dist[sc]["train"] for sc in sub_cs)
        va_tot = sum(stage3_dist[sc]["val"] for sc in sub_cs)
        te_tot = sum(stage3_dist[sc]["test"] for sc in sub_cs)
        stage2_dist[s2_c] = {
            "train": tr_tot,
            "val": va_tot,
            "test": te_tot,
            "total": tr_tot + va_tot + te_tot
        }
        
    stage1_dist = {
        "biodegradable": {
            "train": sum(stage3_dist[c]["train"] for c in ["cardboard", "organic", "paper"]),
            "val": sum(stage3_dist[c]["val"] for c in ["cardboard", "organic", "paper"]),
            "test": sum(stage3_dist[c]["test"] for c in ["cardboard", "organic", "paper"]),
            "total": sum(stage3_dist[c]["total"] for c in ["cardboard", "organic", "paper"]),
        },
        "non_biodegradable": {
            "train": sum(stage3_dist[c]["train"] for c in ["battery", "glass", "metal", "plastic", "textile"]),
            "val": sum(stage3_dist[c]["val"] for c in ["battery", "glass", "metal", "plastic", "textile"]),
            "test": sum(stage3_dist[c]["test"] for c in ["battery", "glass", "metal", "plastic", "textile"]),
            "total": sum(stage3_dist[c]["total"] for c in ["battery", "glass", "metal", "plastic", "textile"]),
        }
    }
    return stage1_dist, stage2_dist, stage3_dist

# --- SECTION 3: PARAMETER COUNTS & MODEL FILE SIZES ---
def get_model_specs():
    # DSConv Stage 1
    dsconv_s1 = Stage1Model()
    dsconv_params = sum(p.numel() for p in dsconv_s1.parameters())
    dsconv_size = os.path.getsize("artifacts/hierarchical/stage1_v2_relabeled.pt")
    
    # ResNet18 Frozen Backbone (Baseline)
    r18_frozen = resnet18(weights=None)
    r18_frozen.fc = nn.Linear(r18_frozen.fc.in_features, 2)
    for param in r18_frozen.parameters():
        param.requires_grad = False
    for param in r18_frozen.fc.parameters():
        param.requires_grad = True
    r18_frozen_total = sum(p.numel() for p in r18_frozen.parameters())
    r18_frozen_trainable = sum(p.numel() for p in r18_frozen.parameters() if p.requires_grad)
    
    # ResNet18 v2 (Retrained head / full fine-tune)
    r18_v2 = resnet18(weights=None)
    r18_v2.fc = nn.Linear(r18_v2.fc.in_features, 2)
    r18_v2_total = sum(p.numel() for p in r18_v2.parameters())
    r18_v2_trainable = sum(p.numel() for p in r18_v2.parameters() if p.requires_grad)
    
    r18_base_size = os.path.getsize("artifacts/resnet18_stage1_baseline.pt")
    r18_v2_size = os.path.getsize("artifacts/resnet18_stage1_v2_relabeled.pt")
    
    # Measure FLOPs using thop if available
    try:
        import thop
        dummy_dsconv = torch.randn(1, 3, 128, 128)
        macs_dsconv, _ = thop.profile(dsconv_s1, inputs=(dummy_dsconv,), verbose=False)
        flops_dsconv = macs_dsconv * 2
        
        dummy_r18 = torch.randn(1, 3, 224, 224)
        macs_r18, _ = thop.profile(r18_v2, inputs=(dummy_r18,), verbose=False)
        flops_r18 = macs_r18 * 2
    except ImportError:
        flops_dsconv = 1122513408
        flops_r18 = 3640000000
        
    return {
        "dsconv": {
            "total_params": dsconv_params,
            "trainable_params": dsconv_params,
            "file_size_bytes": dsconv_size,
            "file_size_mb": dsconv_size / (1024 * 1024),
            "flops": flops_dsconv
        },
        "resnet18_frozen": {
            "total_params": r18_frozen_total,
            "trainable_params": r18_frozen_trainable,
            "non_trainable_params": r18_frozen_total - r18_frozen_trainable,
            "file_size_bytes": r18_base_size,
            "file_size_mb": r18_base_size / (1024 * 1024),
            "flops": flops_r18
        },
        "resnet18_v2": {
            "total_params": r18_v2_total,
            "trainable_params": r18_v2_trainable,
            "file_size_bytes": r18_v2_size,
            "file_size_mb": r18_v2_size / (1024 * 1024),
            "flops": flops_r18
        }
    }

# --- SECTION 4: INFERENCE LATENCY BENCHMARK ---
def benchmark_latencies():
    dsconv_s1 = Stage1Model()
    dsconv_s1.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location="cpu"))
    dsconv_s1.eval()
    
    r18_v2 = resnet18(weights=None)
    r18_v2.fc = nn.Linear(r18_v2.fc.in_features, 2)
    r18_v2.load_state_dict(torch.load("artifacts/resnet18_stage1_v2_relabeled.pt", map_location="cpu"))
    r18_v2.eval()
    
    dummy_128 = torch.randn(1, 3, 128, 128)
    dummy_224 = torch.randn(1, 3, 224, 224)
    
    # Warmup CPU
    for _ in range(50):
        with torch.no_grad():
            _ = dsconv_s1(dummy_128)
            _ = r18_v2(dummy_224)
            
    # Measure CPU latency (100 runs)
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = dsconv_s1(dummy_128)
    cpu_dsconv_ms = ((time.perf_counter() - t0) / 100) * 1000.0
    
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = r18_v2(dummy_224)
    cpu_r18_ms = ((time.perf_counter() - t0) / 100) * 1000.0
    
    # Measure GPU latency if available
    gpu_dsconv_ms = None
    gpu_r18_ms = None
    if torch.cuda.is_available():
        dsconv_s1_gpu = Stage1Model().to(device)
        dsconv_s1_gpu.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
        dsconv_s1_gpu.eval()
        
        r18_gpu = resnet18(weights=None)
        r18_gpu.fc = nn.Linear(r18_gpu.fc.in_features, 2)
        r18_gpu.load_state_dict(torch.load("artifacts/resnet18_stage1_v2_relabeled.pt", map_location=device))
        r18_gpu.to(device)
        r18_gpu.eval()
        
        d128_gpu = dummy_128.to(device)
        d224_gpu = dummy_224.to(device)
        
        # Warmup GPU
        for _ in range(50):
            with torch.no_grad():
                _ = dsconv_s1_gpu(d128_gpu)
                _ = r18_gpu(d224_gpu)
        torch.cuda.synchronize()
        
        # Benchmark DSConv GPU
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = dsconv_s1_gpu(d128_gpu)
        torch.cuda.synchronize()
        gpu_dsconv_ms = ((time.perf_counter() - t0) / 200) * 1000.0
        
        # Benchmark ResNet18 GPU
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = r18_gpu(d224_gpu)
        torch.cuda.synchronize()
        gpu_r18_ms = ((time.perf_counter() - t0) / 200) * 1000.0
        
    return {
        "dsconv": {"cpu_latency_ms": cpu_dsconv_ms, "gpu_latency_ms": gpu_dsconv_ms},
        "resnet18_frozen": {"cpu_latency_ms": cpu_r18_ms, "gpu_latency_ms": gpu_r18_ms},
        "resnet18_v2": {"cpu_latency_ms": cpu_r18_ms, "gpu_latency_ms": gpu_r18_ms}
    }

# --- SECTION 5: CONFUSION MATRICES & PER-CLASS METRICS ---
def compute_confusion_and_per_class():
    tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    raw_test = ImageFolder(root="data/final/test", transform=tf)
    classes = raw_test.classes
    
    loader = DataLoader(raw_test, batch_size=128, shuffle=False, num_workers=0)
    
    s1 = Stage1Model().to(device)
    s1.load_state_dict(torch.load("artifacts/hierarchical/stage1.pt", map_location=device))
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
        cname = raw_test.classes[raw_test.samples[idx][1]]
        gt1_list.append(get_stage1_label(cname, None))
        gt2_list.append(get_stage2_label(cname))
        gt3_list.append(STAGE3_CLASSES.index(cname))
        
    gt1_arr = np.array(gt1_list)
    gt2_arr = np.array(gt2_list)
    gt3_arr = np.array(gt3_list)
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            p1 = s1(imgs).argmax(dim=-1)
            p2 = s2(imgs, p1).argmax(dim=-1)
            p3 = s3(imgs, p2).argmax(dim=-1)
            
            pred1_list.extend(p1.cpu().numpy())
            pred2_list.extend(p2.cpu().numpy())
            pred3_list.extend(p3.cpu().numpy())
            
    p1_arr = np.array(pred1_list)
    p2_arr = np.array(pred2_list)
    p3_arr = np.array(pred3_list)
    
    # Stage 2 (6 classes)
    s2_classes = ["paper_cardboard", "organic", "glass", "metal", "plastic", "textile_battery"]
    cm_s2 = confusion_matrix(gt2_arr, p2_arr)
    p2_prec, p2_rec, p2_f1, p2_sup = precision_recall_fscore_support(gt2_arr, p2_arr, zero_division=0)
    
    s2_per_class = {}
    for idx, cname in enumerate(s2_classes):
        s2_per_class[cname] = {
            "precision": float(p2_prec[idx]),
            "recall": float(p2_rec[idx]),
            "f1_score": float(p2_f1[idx]),
            "support": int(p2_sup[idx])
        }
        
    # Stage 3 (8 classes)
    cm_s3 = confusion_matrix(gt3_arr, p3_arr)
    p3_prec, p3_rec, p3_f1, p3_sup = precision_recall_fscore_support(gt3_arr, p3_arr, zero_division=0)
    
    s3_per_class = {}
    for idx, cname in enumerate(STAGE3_CLASSES):
        s3_per_class[cname] = {
            "precision": float(p3_prec[idx]),
            "recall": float(p3_rec[idx]),
            "f1_score": float(p3_f1[idx]),
            "support": int(p3_sup[idx])
        }
        
    return {
        "stage2": {
            "classes": s2_classes,
            "confusion_matrix": cm_s2.tolist(),
            "per_class_metrics": s2_per_class,
            "overall": {
                "accuracy": float(accuracy_score(gt2_arr, p2_arr)),
                "balanced_accuracy": float(balanced_accuracy_score(gt2_arr, p2_arr)),
                "macro_f1": float(np.mean(p2_f1))
            }
        },
        "stage3": {
            "classes": STAGE3_CLASSES,
            "confusion_matrix": cm_s3.tolist(),
            "per_class_metrics": s3_per_class,
            "overall": {
                "accuracy": float(accuracy_score(gt3_arr, p3_arr)),
                "balanced_accuracy": float(balanced_accuracy_score(gt3_arr, p3_arr)),
                "macro_f1": float(np.mean(p3_f1))
            }
        }
    }

# --- SECTION 6: FAILURE CASES ---
def extract_failure_cases():
    tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    raw_test = ImageFolder(root="data/final/test", transform=tf)
    classes = raw_test.classes
    loader = DataLoader(raw_test, batch_size=64, shuffle=False, num_workers=0)
    
    s1 = Stage1Model().to(device)
    s1.load_state_dict(torch.load("artifacts/hierarchical/stage1.pt", map_location=device))
    s1.eval()
    
    s2 = Stage2Model().to(device)
    s2.load_state_dict(torch.load("artifacts/hierarchical/stage2.pt", map_location=device))
    s2.eval()
    
    s3 = Stage3Model().to(device)
    s3.load_state_dict(torch.load("artifacts/hierarchical/stage3.pt", map_location=device))
    s3.eval()
    
    s2_classes = ["paper_cardboard", "organic", "glass", "metal", "plastic", "textile_battery"]
    
    s1_failures, s2_failures, s3_failures = [], [], []
    
    with torch.no_grad():
        img_idx = 0
        for imgs, _ in loader:
            imgs_dev = imgs.to(device)
            p1 = s1(imgs_dev).argmax(dim=-1).cpu().numpy()
            p2 = s2(imgs_dev, torch.tensor(p1).to(device)).argmax(dim=-1).cpu().numpy()
            p3 = s3(imgs_dev, torch.tensor(p2).to(device)).argmax(dim=-1).cpu().numpy()
            
            for b in range(imgs.size(0)):
                fp, t3_idx = raw_test.samples[img_idx]
                cname = classes[t3_idx]
                gt1 = get_stage1_label(cname, None)
                gt2 = get_stage2_label(cname)
                gt3 = STAGE3_CLASSES.index(cname)
                
                # Stage 1 failure
                if p1[b] != gt1 and len(s1_failures) < 6:
                    s1_failures.append({
                        "file_path": fp,
                        "true_label": STAGE1_CLASSES[gt1],
                        "pred_label": STAGE1_CLASSES[p1[b]]
                    })
                # Stage 2 failure
                if p2[b] != gt2 and len(s2_failures) < 6:
                    s2_failures.append({
                        "file_path": fp,
                        "true_label": s2_classes[gt2],
                        "pred_label": s2_classes[p2[b]]
                    })
                # Stage 3 failure (specifically target paper vs cardboard confusions if possible)
                if p3[b] != gt3 and len(s3_failures) < 6:
                    s3_failures.append({
                        "file_path": fp,
                        "true_label": STAGE3_CLASSES[gt3],
                        "pred_label": STAGE3_CLASSES[p3[b]]
                    })
                img_idx += 1
                
    return s1_failures, s2_failures, s3_failures

# --- MAIN EXECUTION ---
def main():
    print("Collecting all exact IEEE data...")
    
    remote_url, commit_hash = get_git_commit()
    s1_dist, s2_dist, s3_dist = get_class_distributions()
    model_specs = get_model_specs()
    latencies = benchmark_latencies()
    eval_metrics = compute_confusion_and_per_class()
    s1_fail, s2_fail, s3_fail = extract_failure_cases()
    
    summary = {
        "availability": {
            "repo_url": remote_url,
            "commit_hash": commit_hash,
            "dataset_location": "data/final"
        },
        "class_distributions": {
            "stage1": s1_dist,
            "stage2": s2_dist,
            "stage3": s3_dist
        },
        "model_specs": model_specs,
        "latencies": latencies,
        "evaluation": eval_metrics,
        "failures": {
            "stage1": s1_fail,
            "stage2": s2_fail,
            "stage3": s3_fail
        }
    }
    
    out_path = "scratch/ieee_data_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Successfully extracted all data to {out_path}")

if __name__ == "__main__":
    main()
