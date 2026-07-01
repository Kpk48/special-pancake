"""Evaluation harness for comparing KNN baseline and Hierarchical CNN models."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

# Import baseline model
from waste_classifier.model import load_model as load_knn_model
from waste_classifier.features import extract_features_from_image
from waste_classifier.image_io import load_ppm

# Import hierarchical modules
from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES,
    STAGE2_CLASSES,
    STAGE3_CLASSES,
    STAGE3_TO_STAGE1,
    STAGE3_TO_STAGE2,
    get_stage1_label,
    get_stage2_label,
)


def get_file_size_mb(path: Path) -> float:
    """Returns size of a file in Megabytes."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024.0 * 1024.0)


def compute_metrics(
    targets: list[int],
    preds: list[int],
    probs: np.ndarray,
    num_classes: int,
) -> tuple[float, float, float, float, float]:
    """Computes standard evaluation metrics."""
    # Handle edge case where only one class is present in split
    targets_arr = np.array(targets)
    unique_targets = np.unique(targets_arr)
    
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )

    # Compute AUC
    try:
        if len(unique_targets) <= 1:
            auc = 0.5  # Undefined
        elif num_classes == 2:
            # Binary AUC expects probability of the positive class
            auc = roc_auc_score(targets, probs[:, 1])
        else:
            # Multiclass AUC
            # Check if all classes have at least one sample, filter out classes without positive samples
            # to prevent sklearn value error
            present_classes = sorted(list(unique_targets))
            if len(present_classes) < num_classes:
                # subset prediction probabilities to only present classes and re-normalize
                sub_probs = probs[:, present_classes]
                sub_probs_sum = sub_probs.sum(axis=1, keepdims=True)
                sub_probs = np.divide(sub_probs, sub_probs_sum, out=np.zeros_like(sub_probs), where=sub_probs_sum!=0)
                auc = roc_auc_score(targets, sub_probs, multi_class="ovr", labels=present_classes)
            else:
                auc = roc_auc_score(targets, probs, multi_class="ovr")
    except Exception:
        auc = 0.5

    return precision, recall, f1, accuracy, auc


def evaluate_models(
    data_dir: Path,
    knn_model_path: Path,
    cnn_model_dir: Path,
    use_proto: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs evaluation on the test split for both models."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    
    # 1. Load KNN Baseline
    logger_path = Path("artifacts/waste_model.json")
    knn = None
    if knn_model_path.exists():
        try:
            knn = load_knn_model(knn_model_path)
        except Exception as e:
            print(f"Warning: Failed to load KNN model: {e}")
    else:
        print(f"Warning: KNN model not found at {knn_model_path}")

    # 2. Load Hierarchical CNN Models
    s1 = Stage1Model().to(device)
    s2 = Stage2Model().to(device)
    s3 = Stage3Model(use_prototypical=use_proto).to(device)

    s1.load_state_dict(torch.load(cnn_model_dir / "stage1.pt", map_location=device))
    s2.load_state_dict(torch.load(cnn_model_dir / "stage2.pt", map_location=device))
    s3.load_state_dict(torch.load(cnn_model_dir / "stage3.pt", map_location=device))

    s1.eval()
    s2.eval()
    s3.eval()

    # Model parameters/sizes
    cnn_params = {
        "stage1": sum(p.numel() for p in s1.parameters()),
        "stage2": sum(p.numel() for p in s2.parameters()),
        "stage3": sum(p.numel() for p in s3.parameters()),
    }
    cnn_sizes = {
        "stage1": get_file_size_mb(cnn_model_dir / "stage1.pt"),
        "stage2": get_file_size_mb(cnn_model_dir / "stage2.pt"),
        "stage3": get_file_size_mb(cnn_model_dir / "stage3.pt"),
    }

    knn_size = get_file_size_mb(knn_model_path)
    knn_params_count = len(knn.vectors) * len(knn.vectors[0]) if knn else 0

    # 3. Load Test Data
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    test_dataset = ImageFolder(root=data_dir / "test", transform=test_transform, allow_empty=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    classes = test_dataset.classes

    # Verification counts
    print(f"Loaded {len(test_dataset)} test images.")

    # Storage for predictions & ground truth
    # Hierarchical CNN
    cnn_gt = {1: [], 2: [], 3: []}
    cnn_pred = {1: [], 2: [], 3: []}
    cnn_prob = {1: [], 2: [], 3: []}

    # KNN Baseline
    knn_gt = {1: [], 2: [], 3: []}
    knn_pred = {1: [], 2: [], 3: []}
    knn_prob = {1: [], 2: [], 3: []}

    # Timing
    cnn_times = []
    knn_times = []

    # Run Inference loop
    for idx in range(len(test_dataset)):
        # Load sample
        img_tensor, target3_idx = test_dataset[idx]
        class_name = classes[target3_idx]
        
        # Ground truths
        gt1 = get_stage1_label(class_name)
        gt2 = get_stage2_label(class_name)
        gt3 = STAGE3_CLASSES.index(class_name) if class_name in STAGE3_CLASSES else target3_idx

        # --- CNN INFERENCE ---
        t0 = time.perf_counter()
        img_device = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Stage 1 prediction
            out1 = s1(img_device)
            prob1 = torch.softmax(out1, dim=-1).cpu().squeeze(0).numpy()
            pred1 = int(out1.argmax(dim=-1).item())

            # Stage 2 prediction (conditioned on Stage 1 predicted class)
            pred1_tensor = torch.tensor([pred1], device=device)
            out2 = s2(img_device, pred1_tensor)
            prob2 = torch.softmax(out2, dim=-1).cpu().squeeze(0).numpy()
            pred2 = int(out2.argmax(dim=-1).item())

            # Stage 3 prediction (conditioned on Stage 2 predicted class)
            pred2_tensor = torch.tensor([pred2], device=device)
            out3 = s3(img_device, pred2_tensor)
            prob3 = torch.softmax(out3, dim=-1).cpu().squeeze(0).numpy()
            pred3 = int(out3.argmax(dim=-1).item())

        cnn_times.append(time.perf_counter() - t0)

        cnn_gt[1].append(gt1)
        cnn_pred[1].append(pred1)
        cnn_prob[1].append(prob1)

        cnn_gt[2].append(gt2)
        cnn_pred[2].append(pred2)
        cnn_prob[2].append(prob2)

        cnn_gt[3].append(gt3)
        cnn_pred[3].append(pred3)
        cnn_prob[3].append(prob3)

        # --- KNN INFERENCE ---
        if knn is not None:
            t0 = time.perf_counter()
            # Extract handcrafted features using legacy pipeline
            # Load original file directly to keep feature extraction identical to baseline
            img_file_path, _ = test_dataset.samples[idx]
            try:
                # extract features
                features = extract_features_from_image(load_ppm(img_file_path))
                
                # KNN predict and predict_proba
                pred_label = knn.predict(features)
                probs_dict = knn.predict_proba(features)
                
                # Map prediction to Stage 3 idx
                knn_pred3 = STAGE3_CLASSES.index(pred_label) if pred_label in STAGE3_CLASSES else 0
                
                # Construct stage 3 probabilities
                prob3 = np.zeros(11)
                for label, score in probs_dict.items():
                    if label in STAGE3_CLASSES:
                        prob3[STAGE3_CLASSES.index(label)] = score
            except Exception:
                # Fallback on failure
                knn_pred3 = 0
                prob3 = np.zeros(11)
                prob3[0] = 1.0

            knn_times.append(time.perf_counter() - t0)

            # Map Stage 3 KNN outputs back to Stage 1 and Stage 2
            pred_label_name = STAGE3_CLASSES[knn_pred3]
            knn_pred1 = STAGE3_TO_STAGE1[pred_label_name]
            knn_pred2 = STAGE3_TO_STAGE2[pred_label_name]

            # Reconstruct probabilities for hierarchical stages
            prob1 = np.zeros(2)
            prob2 = np.zeros(6)
            for c_name, c_idx in STAGE3_TO_STAGE1.items():
                prob1[c_idx] += prob3[STAGE3_CLASSES.index(c_name)]
            for c_name, c_idx in STAGE3_TO_STAGE2.items():
                prob2[c_idx] += prob3[STAGE3_CLASSES.index(c_name)]

            knn_gt[1].append(gt1)
            knn_pred[1].append(knn_pred1)
            knn_prob[1].append(prob1)

            knn_gt[2].append(gt2)
            knn_pred[2].append(knn_pred2)
            knn_prob[2].append(prob2)

            knn_gt[3].append(gt3)
            knn_pred[3].append(knn_pred3)
            knn_prob[3].append(prob3)

    # 4. Metric aggregation
    results_cnn = {}
    results_knn = {}

    for stage in [1, 2, 3]:
        n_classes = 2 if stage == 1 else (6 if stage == 2 else 11)
        
        # CNN Metrics
        p, r, f, acc, auc = compute_metrics(
            cnn_gt[stage], cnn_pred[stage], np.array(cnn_prob[stage]), n_classes
        )
        results_cnn[stage] = {
            "precision": p, "recall": r, "f1": f, "accuracy": acc, "auc": auc,
            "params": cnn_params[f"stage{stage}"],
            "size_mb": cnn_sizes[f"stage{stage}"],
            "time": np.mean(cnn_times),
        }

        # KNN Metrics
        if knn is not None:
            p, r, f, acc, auc = compute_metrics(
                knn_gt[stage], knn_pred[stage], np.array(knn_prob[stage]), n_classes
            )
            results_knn[stage] = {
                "precision": p, "recall": r, "f1": f, "accuracy": acc, "auc": auc,
                "params": knn_params_count,
                "size_mb": knn_size,
                "time": np.mean(knn_times),
            }

    return results_knn, results_cnn


def generate_reports(
    knn_results: dict[str, Any],
    cnn_results: dict[str, Any],
    output_md_path: Path,
    output_png_path: Path,
) -> None:
    """Generates comparison Markdown table and matplotlib graph."""
    # 1. Generate Markdown Report
    md_lines = [
        "# Model Evaluation and Paper Comparison Report",
        "",
        "This report evaluates the performance of the **Hierarchical CNN** versus the **KNN Baseline** model. Results are structured by classification stages to mirror the metrics in Table 9 and Table 10 of Nahiduzzaman et al. (2025).",
        "",
        "| Model | Stage | Classes | Precision (macro) | Recall (macro) | F1-Score | Accuracy | AUC | Params | Size (MB) | Inference Time (s/img) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for stage in [1, 2, 3]:
        n_classes = 2 if stage == 1 else (6 if stage == 2 else 11)
        
        # KNN Row
        if stage in knn_results:
            kr = knn_results[stage]
            md_lines.append(
                f"| KNN Baseline | Stage {stage} | {n_classes} | {kr['precision']:.4f} | {kr['recall']:.4f} | {kr['f1']:.4f} | {kr['accuracy']:.4f} | {kr['auc']:.4f} | {kr['params']} | {kr['size_mb']:.4f} | {kr['time']:.5f} |"
            )
            
        # CNN Row
        if stage in cnn_results:
            cr = cnn_results[stage]
            md_lines.append(
                f"| Hierarchical CNN | Stage {stage} | {n_classes} | {cr['precision']:.4f} | {cr['recall']:.4f} | {cr['f1']:.4f} | {cr['accuracy']:.4f} | {cr['auc']:.4f} | {cr['params']} | {cr['size_mb']:.4f} | {cr['time']:.5f} |"
            )

    md_lines.append("")
    
    # Write file
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown report generated at {output_md_path}")

    # 2. Generate Matplotlib Chart
    stages = ["Stage 1\n(2 classes)", "Stage 2\n(6 classes)", "Stage 3\n(11 classes)"]
    x = np.arange(len(stages))
    width = 0.35

    knn_accs = [knn_results[s]["accuracy"] for s in [1, 2, 3]] if knn_results else [0, 0, 0]
    cnn_accs = [cnn_results[s]["accuracy"] for s in [1, 2, 3]]

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, knn_accs, width, label="KNN Baseline", color="#a1b5a0")
    rects2 = ax.bar(x + width/2, cnn_accs, width, label="Hierarchical CNN", color="#206a3b")

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison by Classification Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    print(f"Comparison chart saved at {output_png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical CNN vs KNN.")
    parser.add_argument("--data", default="data/final", help="Path to preprocessed dataset root.")
    parser.add_argument("--knn-model", default="artifacts/waste_model.json", help="Path to KNN model json.")
    parser.add_argument("--cnn-model-dir", default="artifacts/hierarchical", help="Directory containing stage checkpoints.")
    parser.add_argument("--use-proto", action="store_true", help="Use prototypical metric learning head for Stage 3.")
    parser.add_argument("--out-md", default="results/comparison.md", help="Markdown comparison output path.")
    parser.add_argument("--out-png", default="results/comparison.png", help="PNG chart output path.")
    args = parser.parse_args()

    knn_res, cnn_res = evaluate_models(
        Path(args.data),
        Path(args.knn_model),
        Path(args.cnn_model_dir),
        args.use_proto,
    )

    generate_reports(
        knn_res,
        cnn_res,
        Path(args.out_md),
        Path(args.out_png),
    )


if __name__ == "__main__":
    # Fix potential argument parsing bug for hyphens
    import sys
    # Replace '--knn-model' with '--knn_model' to match argparse conversion
    sys.argv = [arg.replace("--knn-model", "--knn_model") for arg in sys.argv]
    main()
