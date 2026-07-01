"""Grad-CAM visualization overlay and prediction helper script."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import STAGE3_CLASSES
from waste_classifier.xai.gradcam_lite import GradCAMLite

logger = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def overlay_heatmap(image_path: Path, heatmap: torch.Tensor, save_path: Path) -> None:
    """Overlays the 2D heatmap on the source image and saves the result."""
    orig_img = Image.open(image_path).convert("RGB")
    w, h = orig_img.size

    # Resize heatmap to match original image dimensions
    heatmap_np = heatmap.numpy()
    heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Apply colormap (jet)
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Drop alpha channel

    # Blend original and colored heatmap
    alpha = 0.4
    blended = alpha * colored_heatmap + (1.0 - alpha) * (np.array(orig_img) / 255.0)
    blended = np.clip(blended, 0.0, 1.0)

    # Save the overlay
    save_img = Image.fromarray((blended * 255).astype(np.uint8))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_img.save(save_path)
    logger.info(f"Grad-CAM overlay saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grad-CAM on a waste image.")
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--model-dir", default="artifacts/hierarchical", help="Checkpoints directory.")
    parser.add_argument("--output", default="results/gradcam_overlay.png", help="Path to save output visualization.")
    parser.add_argument("--use-proto", action="store_true", help="Use prototypical metric learning head for Stage 3.")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model_dir = Path(args.model_dir)

    # 1. Load Hierarchical Stage Models
    logger.info("Loading hierarchical models...")
    s1 = Stage1Model().to(device)
    s2 = Stage2Model().to(device)
    s3 = Stage3Model(use_prototypical=args.use_proto).to(device)

    s1.load_state_dict(torch.load(model_dir / "stage1.pt", map_location=device))
    s2.load_state_dict(torch.load(model_dir / "stage2.pt", map_location=device))
    s3.load_state_dict(torch.load(model_dir / "stage3.pt", map_location=device))

    s1.eval()
    s2.eval()
    s3.eval()

    # 2. Preprocess Input Image
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error(f"Image not found at {args.image}")
        return

    # Keep PPM load compatible or load via PIL
    try:
        # standard PIL loader handles PNG/JPG/PPM automatically
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(pil_img).to(device)

    # 3. Perform Hierarchical Inference to get conditioning labels
    with torch.no_grad():
        # Stage 1 Inference
        s1_out = s1(img_tensor.unsqueeze(0))
        s1_pred = s1_out.argmax(dim=-1)

        # Stage 2 Inference
        s2_out = s2(img_tensor.unsqueeze(0), s1_pred)
        s2_pred = s2_out.argmax(dim=-1)

    # 4. Generate Grad-CAM for Stage 3 conditioned on Stage 2
    logger.info("Executing Grad-CAM...")
    gcam = GradCAMLite(s3)
    try:
        heatmap, pred_idx, latency = gcam.generate_heatmap(img_tensor, s2_pred)
        predicted_label = STAGE3_CLASSES[pred_idx]
        
        logger.info(f"Prediction: {predicted_label} (index {pred_idx})")
        logger.info(f"Grad-CAM CPU Latency: {latency:.2f}ms")
        
        if latency <= 50.0:
            logger.info("✓ CPU Explainability latency constraint satisfied (< 50ms).")
        else:
            logger.warning("⚠️ CPU Explainability latency exceeded 50ms limit.")

        # 5. Overlay and Save
        overlay_heatmap(img_path, heatmap, Path(args.output))
    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM: {e}")


if __name__ == "__main__":
    main()
