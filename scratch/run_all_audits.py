import sys
import os
import time
import psutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE3_CLASSES, STAGE3_TO_STAGE1, STAGE3_TO_STAGE2,
    get_stage1_label, get_stage2_label
)
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

raw_test = ImageFolder(root="data/final/test", transform=val_test_transform)
raw_test_resnet = ImageFolder(root="data/final/test", transform=resnet_transform)
classes = raw_test.classes  # 8 fine-grained classes

# Load models
s1_model = Stage1Model().to(device)
s1_model.load_state_dict(torch.load("artifacts/hierarchical/stage1_v2_relabeled.pt", map_location=device))
s1_model.eval()

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

print("Loaded all models successfully.")

# Collect per-sample predictions
samples = raw_test.samples
n_samples = len(samples)

def get_source_tag(filepath):
    fname = os.path.basename(filepath)
    if fname.startswith("gc12_"):
        return "gc12_"
    elif fname.startswith("gcv2_"):
        return "gcv2_"
    elif fname.startswith("taco_"):
        return "taco_"
    else:
        return "other"

test_loader = DataLoader(raw_test, batch_size=64, shuffle=False, num_workers=0)
test_loader_resnet = DataLoader(raw_test_resnet, batch_size=64, shuffle=False, num_workers=0)

all_s1_logits = []
all_s2_logits = []
all_s3_logits = []
all_resnet_logits = []

with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        out1 = s1_model(imgs)
        pred1 = out1.argmax(dim=-1)
        out2 = s2_model(imgs, pred1)
        pred2 = out2.argmax(dim=-1)
        out3 = s3_model(imgs, pred2)
        all_s1_logits.append(out1.cpu())
        all_s2_logits.append(out2.cpu())
        all_s3_logits.append(out3.cpu())
    
    for imgs, _ in test_loader_resnet:
        imgs = imgs.to(device)
        out_r = resnet_s1(imgs)
        all_resnet_logits.append(out_r.cpu())

all_s1_logits = torch.cat(all_s1_logits, dim=0)
all_s2_logits = torch.cat(all_s2_logits, dim=0)
all_s3_logits = torch.cat(all_s3_logits, dim=0)
all_resnet_logits = torch.cat(all_resnet_logits, dim=0)

# Decision probabilities / predictions
# Calibrated threshold for S1 is t=0.55 or t=0.50
s1_probs = torch.softmax(all_s1_logits, dim=-1)[:, 1]
s1_preds_default = all_s1_logits.argmax(dim=-1).numpy()
s1_preds_t055 = (s1_probs >= 0.55).long().numpy()

s2_preds = all_s2_logits.argmax(dim=-1).numpy()
s3_preds = all_s3_logits.argmax(dim=-1).numpy()
resnet_preds = all_resnet_logits.argmax(dim=-1).numpy()

# Ground truth
gt_s3_names = [classes[t3] for _, t3 in samples]
gt_s3_indices = np.array([STAGE3_CLASSES.index(c) for c in gt_s3_names])
gt_s1_indices = np.array([STAGE3_TO_STAGE1[c] for c in gt_s3_names])
gt_s2_indices = np.array([STAGE3_TO_STAGE2[c] for c in gt_s3_names])

# TASK 1: DOMAIN-SHIFT / FILENAME ARTIFACT AUDIT
print("\n=== TASK 1: DOMAIN-SHIFT / FILENAME ARTIFACT AUDIT ===")
source_tags = [get_source_tag(p) for p, _ in samples]
unique_sources = sorted(list(set(source_tags)))

for src in unique_sources:
    mask = np.array([st == src for st in source_tags])
    cnt = mask.sum()
    s1_acc = accuracy_score(gt_s1_indices[mask], s1_preds_t055[mask])
    s1_resnet_acc = accuracy_score(gt_s1_indices[mask], resnet_preds[mask])
    s2_acc = accuracy_score(gt_s2_indices[mask], s2_preds[mask])
    s3_acc = accuracy_score(gt_s3_indices[mask], s3_preds[mask])
    print(f"Source: {src:<8} | Count: {cnt:<5} | S1 (DSConv t=0.55): {s1_acc*100:.2f}% | S1 (ResNet18): {s1_resnet_acc*100:.2f}% | S2: {s2_acc*100:.2f}% | S3: {s3_acc*100:.2f}%")

pooled_s1 = accuracy_score(gt_s1_indices, s1_preds_t055)
pooled_resnet = accuracy_score(gt_s1_indices, resnet_preds)
pooled_s2 = accuracy_score(gt_s2_indices, s2_preds)
pooled_s3 = accuracy_score(gt_s3_indices, s3_preds)
print(f"Pooled Total | Count: {n_samples:<5} | S1 (DSConv t=0.55): {pooled_s1*100:.2f}% | S1 (ResNet18): {pooled_resnet*100:.2f}% | S2: {pooled_s2*100:.2f}% | S3: {pooled_s3*100:.2f}%")

# TASK 2: STAGE 1 ERROR ASYMMETRY
print("\n=== TASK 2: STAGE 1 ERROR ASYMMETRY ===")
# S1 confusion matrix for default threshold (arg max) vs t=0.55
cm_s1_def = confusion_matrix(gt_s1_indices, s1_preds_default)
print("S1 Confusion Matrix (Default Argmax, 0=biodegradable, 1=non_biodegradable):\n", cm_s1_def)

cm_s1_t055 = confusion_matrix(gt_s1_indices, s1_preds_t055)
print("S1 Confusion Matrix (t=0.55):\n", cm_s1_t055)

# TN, FP, FN, TP for default argmax:
# 0 = bio, 1 = non-bio
# TN = pred=1, gt=1; FP = pred=0, gt=1 (non-bio predicted bio); FN = pred=1, gt=0 (bio predicted non-bio); TP = pred=0, gt=0
# Let's inspect confusion matrix where 0=bio, 1=non-bio:
# cm[0,0] = Bio predicted Bio (TP_bio)
# cm[0,1] = Bio predicted Non-Bio (FN_bio)
# cm[1,0] = Non-Bio predicted Bio (FP_bio)
# cm[1,1] = Non-Bio predicted Non-Bio (TN_bio)

print("Default Threshold breakdown:")
print(f"  True Bio predicted Bio (TP): {cm_s1_def[0,0]}")
print(f"  True Bio predicted Non-Bio (FN): {cm_s1_def[0,1]}")
print(f"  True Non-Bio predicted Bio (FP): {cm_s1_def[1,0]}")
print(f"  True Non-Bio predicted Non-Bio (TN): {cm_s1_def[1,1]}")

# FP breakdown by fine class and source:
fp_mask = (gt_s1_indices == 1) & (s1_preds_default == 0)
fn_mask = (gt_s1_indices == 0) & (s1_preds_default == 1)

print("\nFP (Non-Biodegradable -> Predicted Biodegradable) breakdown by Fine-Grained Class:")
for idx, cls_name in enumerate(STAGE3_CLASSES):
    if STAGE3_TO_STAGE1[cls_name] == 1: # Non-bio class
        cls_mask = np.array([c == cls_name for c in gt_s3_names])
        fp_cls_count = (fp_mask & cls_mask).sum()
        total_cls = cls_mask.sum()
        print(f"  {cls_name:<10}: {fp_cls_count}/{total_cls} ({fp_cls_count/total_cls*100:.2f}%)")

print("\nFN (Biodegradable -> Predicted Non-Biodegradable) breakdown by Fine-Grained Class:")
for idx, cls_name in enumerate(STAGE3_CLASSES):
    if STAGE3_TO_STAGE1[cls_name] == 0: # Bio class
        cls_mask = np.array([c == cls_name for c in gt_s3_names])
        fn_cls_count = (fn_mask & cls_mask).sum()
        total_cls = cls_mask.sum()
        print(f"  {cls_name:<10}: {fn_cls_count}/{total_cls} ({fn_cls_count/total_cls*100:.2f}%)")

print("\nFP breakdown by Source Dataset:")
for src in unique_sources:
    src_mask = np.array([st == src for st in source_tags])
    fp_src_count = (fp_mask & src_mask).sum()
    total_src_nonbio = ((gt_s1_indices == 1) & src_mask).sum()
    print(f"  {src:<8}: {fp_src_count}/{total_src_nonbio} non-bio misclassified as bio ({fp_src_count/total_src_nonbio*100:.2f}%)")

# TASK 3: STAGE 2 AND STAGE 3 CONFUSION MATRICES
print("\n=== TASK 3: STAGE 2 AND STAGE 3 CONFUSION MATRICES ===")
s2_classes = ["paper_cardboard", "organic", "glass", "metal", "plastic", "textile_battery"]
cm_s2 = confusion_matrix(gt_s2_indices, s2_preds)
print("Stage 2 Confusion Matrix (6x6):\n", cm_s2)

p_s2, r_s2, f1_s2, sup_s2 = precision_recall_fscore_support(gt_s2_indices, s2_preds)
print("\nStage 2 Per-Class Metrics:")
for idx, cname in enumerate(s2_classes):
    print(f"  {cname:<18} | Precision: {p_s2[idx]:.4f} | Recall: {r_s2[idx]:.4f} | F1: {f1_s2[idx]:.4f} | Support: {sup_s2[idx]}")

cm_s3 = confusion_matrix(gt_s3_indices, s3_preds)
print("\nStage 3 Confusion Matrix (8x8):\n", cm_s3)

p_s3, r_s3, f1_s3, sup_s3 = precision_recall_fscore_support(gt_s3_indices, s3_preds)
print("\nStage 3 Per-Class Metrics:")
for idx, cname in enumerate(STAGE3_CLASSES):
    print(f"  {cname:<12} | Precision: {p_s3[idx]:.4f} | Recall: {r_s3[idx]:.4f} | F1: {f1_s3[idx]:.4f} | Support: {sup_s3[idx]}")

# Top misclassified class pairs in Stage 3
off_diag_s3 = []
for i in range(8):
    for j in range(8):
        if i != j and cm_s3[i, j] > 0:
            off_diag_s3.append((cm_s3[i, j], STAGE3_CLASSES[i], STAGE3_CLASSES[j]))
off_diag_s3.sort(reverse=True)
print("\nTop 10 Stage 3 Misclassification Class Pairs (True -> Predicted):")
for cnt, true_c, pred_c in off_diag_s3[:10]:
    print(f"  True: {true_c:<10} -> Pred: {pred_c:<10} | Count: {cnt} ({cnt/n_samples*100:.2f}% of test set)")

# TASK 6: INFERENCE BENCHMARKING
print("\n=== TASK 6: INFERENCE BENCHMARKS (DEV MACHINE) ===")

def benchmark_model(model_tuple, input_size, run_device, n_runs=100, is_pipeline=False):
    if run_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    dummy_input = torch.randn(1, 3, input_size, input_size).to(run_device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            if is_pipeline:
                m1, m2, m3 = model_tuple
                o1 = m1(dummy_input)
                p1 = o1.argmax(dim=-1)
                o2 = m2(dummy_input, p1)
                p2 = o2.argmax(dim=-1)
                o3 = m3(dummy_input, p2)
            else:
                m = model_tuple
                o = m(dummy_input)
            if run_device == "cuda":
                torch.cuda.synchronize()
                
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            if is_pipeline:
                m1, m2, m3 = model_tuple
                o1 = m1(dummy_input)
                p1 = o1.argmax(dim=-1)
                o2 = m2(dummy_input, p1)
                p2 = o2.argmax(dim=-1)
                o3 = m3(dummy_input, p2)
            else:
                m = model_tuple
                o = m(dummy_input)
            if run_device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat
    
    peak_mem = 0.0
    if run_device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) # MB
    else:
        process = psutil.Process()
        peak_mem = process.memory_info().rss / (1024.0 * 1024.0) # MB
        
    return mean_lat, std_lat, fps, peak_mem

# GPU Benchmarks
s1_gpu = s1_model.to("cuda")
s2_gpu = s2_model.to("cuda")
s3_gpu = s3_model.to("cuda")
resnet_gpu = resnet_s1.to("cuda")

pipe_gpu_lat, pipe_gpu_std, pipe_gpu_fps, pipe_gpu_vram = benchmark_model((s1_gpu, s2_gpu, s3_gpu), 128, "cuda", n_runs=200, is_pipeline=True)
res_gpu_lat, res_gpu_std, res_gpu_fps, res_gpu_vram = benchmark_model(resnet_gpu, 224, "cuda", n_runs=200, is_pipeline=False)

# CPU Benchmarks
s1_cpu = s1_model.to("cpu")
s2_cpu = s2_model.to("cpu")
s3_cpu = s3_model.to("cpu")
resnet_cpu = resnet_s1.to("cpu")

pipe_cpu_lat, pipe_cpu_std, pipe_cpu_fps, pipe_cpu_ram = benchmark_model((s1_cpu, s2_cpu, s3_cpu), 128, "cpu", n_runs=200, is_pipeline=True)
res_cpu_lat, res_cpu_std, res_cpu_fps, res_cpu_ram = benchmark_model(resnet_cpu, 224, "cpu", n_runs=200, is_pipeline=False)

print(f"DSConv2D 3-Stage Pipeline (GPU): Latency = {pipe_gpu_lat:.3f} ± {pipe_gpu_std:.3f} ms | FPS = {pipe_gpu_fps:.2f} | Peak VRAM = {pipe_gpu_vram:.2f} MB")
print(f"DSConv2D 3-Stage Pipeline (CPU): Latency = {pipe_cpu_lat:.3f} ± {pipe_cpu_std:.3f} ms | FPS = {pipe_cpu_fps:.2f} | Peak RAM = {pipe_cpu_ram:.2f} MB")
print(f"ResNet18 Reference (GPU):        Latency = {res_gpu_lat:.3f} ± {res_gpu_std:.3f} ms | FPS = {res_gpu_fps:.2f} | Peak VRAM = {res_gpu_vram:.2f} MB")
print(f"ResNet18 Reference (CPU):        Latency = {res_cpu_lat:.3f} ± {res_cpu_std:.3f} ms | FPS = {res_cpu_fps:.2f} | Peak RAM = {res_cpu_ram:.2f} MB")
