#!/usr/bin/env python3
"""AI Waste Classification Dataset Engineering Pipeline.

Acquires, crops, merges, deduplicates, validates, balances, and splits
Garbage Classification V2, Garbage Classification (12 classes), and TACO.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import os
import random
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import imagehash
from tqdm import tqdm

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("pipeline")


# TARGET CLASSES
TARGET_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic",
    "textile",
    "battery",
    "wood",
    "ceramic",
    "nylon",
]

# Source Category Mappings
GCV2_MAP = {
    "metal": "metal",
    "glass": "glass",
    "biological": "organic",
    "paper": "paper",
    "battery": "battery",
    "cardboard": "cardboard",
    "shoes": "textile",
    "clothes": "textile",
    "plastic": "plastic",
    "trash": None,  # DISCARD
}

GC12_MAP = {
    "paper": "paper",
    "cardboard": "cardboard",
    "biological": "organic",
    "metal": "metal",
    "plastic": "plastic",
    "green-glass": "glass",
    "brown-glass": "glass",
    "white-glass": "glass",
    "clothes": "textile",
    "shoes": "textile",
    "batteries": "battery",
    "trash": None,  # DISCARD
}

TACO_MAP = {
    "Aluminium foil": "metal",
    "Aluminium blister pack": "metal",
    "Metal bottle cap": "metal",
    "Drink can": "metal",
    "Food can": "metal",
    "Aerosol": "metal",
    "Scrap metal": "metal",
    
    "Glass bottle": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Broken glass": "glass",
    
    "Plastic bottle": "plastic",
    "Plastic bag": "plastic",
    "Plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Plastic straw": "plastic",
    "Plastic film": "plastic",
    "Six pack rings": "plastic",
    "Styrofoam piece": "plastic",
    "Other plastic": "plastic",
    
    "Corrugated carton": "cardboard",
    "Egg carton": "cardboard",
    "Meal carton": "cardboard",
    "Pizza box": "cardboard",
    
    "Normal paper": "paper",
    "Paper bag": "paper",
    "Magazine paper": "paper",
    "Wrapping paper": "paper",
    "Paper cup": "paper",
    "Paper straw": "paper",
    "Toilet tube": "paper",
    
    "Food waste": "organic",
    
    "Rope": "textile",
    "Shoe": "textile",
    "Clothes": "textile",
    
    "Battery": "battery",
    
    "Cigarette": None,
    "Unlabelled litter": None,
    "Other litter": None
}


def setup_logger(log_dir: Path) -> None:
    """Setup logging to file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)


def generate_mock_datasets(raw_dir: Path, mock_size: int) -> None:
    """Generates small mock datasets for pipeline verification."""
    logger.info("Generating mock raw datasets for pipeline validation...")
    
    # 1. Mock Garbage Classification V2
    gcv2_dir = raw_dir / "gcv2"
    gcv2_classes = ["Metal", "Glass", "Biological", "Paper", "Battery", "Cardboard", "Shoes", "Clothes", "Plastic", "Trash"]
    for cls in gcv2_classes:
        cls_dir = gcv2_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(mock_size):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(cls_dir / f"img_{i:04d}.jpg")

    # 2. Mock Garbage Classification (12 classes)
    gc12_dir = raw_dir / "gc12"
    gc12_classes = ["paper", "cardboard", "biological", "metal", "plastic", "green-glass", "brown-glass", "white-glass", "clothes", "shoes", "batteries", "trash"]
    for cls in gc12_classes:
        cls_dir = gc12_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(mock_size):
            img = Image.fromarray(np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8))
            img.save(cls_dir / f"img_{i:04d}.jpg")

    # 3. Mock TACO
    taco_dir = raw_dir / "taco"
    taco_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images
    images_info = []
    annotations = []
    for i in range(mock_size):
        img_filename = f"img_{i:04d}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        img.save(taco_dir / img_filename)
        
        images_info.append({
            "id": i,
            "file_name": img_filename,
            "width": 200,
            "height": 200
        })
        
        # Add 1-2 bounding box annotations per image
        for ann_id in range(random.randint(1, 2)):
            cat_id = random.randint(0, len(TACO_MAP) - 1)
            # bbox format: [x, y, width, height]
            x = random.randint(10, 100)
            y = random.randint(10, 100)
            w = random.randint(40, 90)
            h = random.randint(40, 90)
            annotations.append({
                "id": len(annotations),
                "image_id": i,
                "category_id": cat_id,
                "bbox": [x, y, w, h]
            })

    categories_info = [
        {"id": idx, "name": name, "supercategory": name}
        for idx, name in enumerate(TACO_MAP.keys())
    ]
    
    coco_data = {
        "images": images_info,
        "annotations": annotations,
        "categories": categories_info
    }
    
    with open(taco_dir / "annotations.json", "w") as f:
        json.dump(coco_data, f, indent=2)

    logger.info("Mock dataset generation completed successfully.")


def download_datasets(raw_dir: Path) -> None:
    """Acquires the datasets via Kaggle API / GitHub URL lists."""
    logger.info("Starting Dataset Download stage...")
    
    # Setup directories
    gcv2_dir = raw_dir / "gcv2"
    gc12_dir = raw_dir / "gc12"
    taco_dir = raw_dir / "taco"
    
    gcv2_dir.mkdir(parents=True, exist_ok=True)
    gc12_dir.mkdir(parents=True, exist_ok=True)
    taco_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        logger.error("Kaggle API authentication failed. Make sure kaggle.json is present in ~/.kaggle/.")
        logger.error("Instructions: Sign up on Kaggle, go to your Account page, click 'Create New API Token', and save the file to ~/.kaggle/kaggle.json")
        raise e

    # 1. Download Garbage Classification V2
    if not any(gcv2_dir.iterdir()):
        logger.info("Downloading Garbage Classification V2 dataset from Kaggle...")
        kaggle.api.dataset_download_files("sumn2u/garbage-classification-v2", path=gcv2_dir, unzip=True)
    else:
        logger.info("Garbage Classification V2 already downloaded. Skipping.")

    # 2. Download Garbage Classification (12 classes)
    if not any(gc12_dir.iterdir()):
        logger.info("Downloading Garbage Classification (12 classes) dataset from Kaggle...")
        kaggle.api.dataset_download_files("mostafaabla/garbage-classification", path=gc12_dir, unzip=True)
    else:
        logger.info("Garbage Classification (12 classes) already downloaded. Skipping.")

    # 3. Download TACO
    if not (taco_dir / "annotations.json").exists() or not any(taco_dir.glob("batch_*/*.jpg")):
        import requests
        if not (taco_dir / "annotations.json").exists():
            logger.info("Downloading TACO annotations file...")
            ann_url = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
            res = requests.get(ann_url)
            with open(taco_dir / "annotations.json", "wb") as f:
                f.write(res.content)
            
        # Download images based on annotations
        logger.info("Downloading TACO images from URLs list in parallel...")
        with open(taco_dir / "annotations.json") as f:
            coco = json.load(f)

        from concurrent.futures import ThreadPoolExecutor

        def download_single_taco_image(img_info):
            flickr_url = img_info.get("flickr_url")
            file_name = img_info["file_name"]
            img_path = taco_dir / file_name
            img_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not img_path.exists() and flickr_url:
                try:
                    img_res = requests.get(flickr_url, timeout=15)
                    if img_res.status_code == 200:
                        img_path.write_bytes(img_res.content)
                        return True
                except Exception:
                    pass
            return False

        # Use ThreadPoolExecutor with 32 workers for parallel downloads
        with ThreadPoolExecutor(max_workers=32) as executor:
            list(tqdm(executor.map(download_single_taco_image, coco["images"]), total=len(coco["images"]), desc="Downloading TACO images"))
    else:
        logger.info("TACO dataset already initialized. Skipping.")


def crop_taco_objects(taco_dir: Path) -> None:
    """Parses annotations.json, crops annotated bounding boxes with padding,

    and saves cropped files.
    """
    logger.info("Starting TACO bounding box cropping...")
    ann_path = taco_dir / "annotations.json"
    if not ann_path.exists():
        logger.error(f"TACO annotations file not found at {ann_path}")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    # Index images and categories
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat for cat in coco["categories"]}

    cropped_dir = taco_dir / "cropped"
    if cropped_dir.exists():
        shutil.rmtree(cropped_dir)
    cropped_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for ann in tqdm(coco["annotations"], desc="Cropping TACO annotations"):
        img_info = images.get(ann["image_id"])
        if not img_info:
            continue

        img_path = taco_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        category = categories.get(ann["category_id"])
        if not category:
            continue

        supercategory = category["supercategory"]

        try:
            with Image.open(img_path) as img:
                w_img, h_img = img.size
                bbox = ann["bbox"]  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 10px padding, clamped
                x_min = max(0, int(x - 10))
                y_min = max(0, int(y - 10))
                x_max = min(w_img, int(x + w + 10))
                y_max = min(h_img, int(y + h + 10))
                
                # Skip invalid bboxes
                if x_max <= x_min or y_max <= y_min:
                    continue

                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                # Create category directory
                dest_dir = cropped_dir / supercategory
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Save cropped image
                cropped_img.save(dest_dir / f"crop_{ann['id']}.jpg", "JPEG")
                success_count += 1
        except Exception as e:
            logger.warning(f"Error cropping annotation {ann['id']}: {e}")

    logger.info(f"TACO Cropping completed. Successfully cropped {success_count} object samples.")


def remap_and_copy(raw_dir: Path, merged_dir: Path) -> dict[str, int]:
    """Remaps source folders and copies images into unified class dirs."""
    logger.info("Remapping classes and merging datasets...")
    
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGET_CLASSES:
        (merged_dir / target).mkdir(parents=True, exist_ok=True)

    counts_before = defaultdict(int)
    counts_after = defaultdict(int)

    # Helper helper to copy files
    def copy_file(src_path: Path, target_class: str, source_id: str, original_class: str):
        if not target_class:
            return
        
        counts_before[source_id] += 1
        
        target_class_clean = target_class.lower()
        if target_class_clean not in TARGET_CLASSES:
            return

        unique_id = uuid.uuid4().hex[:6]
        dest_filename = f"{source_id}_{original_class.replace(' ', '_')}_{unique_id}.jpg"
        dest_path = merged_dir / target_class_clean / dest_filename
        
        try:
            # Open, convert to RGB, and save as JPG to standardise format
            with Image.open(src_path) as img:
                img.convert("RGB").save(dest_path, "JPEG")
                counts_after[source_id] += 1
        except Exception as e:
            logger.warning(f"Failed to copy/convert file {src_path}: {e}")

    # 1. Merging Garbage Classification V2
    gcv2_dir = raw_dir / "gcv2/original"
    if not gcv2_dir.exists():
        gcv2_dir = raw_dir / "gcv2"
        
    if gcv2_dir.exists():
        for class_dir in gcv2_dir.iterdir():
            if class_dir.is_dir() and class_dir.name not in ["original", "standardized_256", "standardized_384"]:
                orig_name = class_dir.name
                target_mapped = GCV2_MAP.get(orig_name.lower())
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".ppm"]:
                        copy_file(img_path, target_mapped, "gcv2", orig_name)

    # 2. Merging Garbage Classification (12 classes)
    gc12_dir = raw_dir / "gc12"
    if gc12_dir.exists():
        # Might contain nested directories depending on Kaggle extract structure
        # Walk directories to find class folders containing image files
        for root, dirs, files in os.walk(gc12_dir):
            root_path = Path(root)
            if root_path.name.lower() in GC12_MAP:
                orig_name = root_path.name
                target_mapped = GC12_MAP.get(orig_name.lower())
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".ppm")):
                        copy_file(root_path / file, target_mapped, "gc12", orig_name)

    # 3. Merging TACO Cropped Objects
    taco_cropped_dir = raw_dir / "taco/cropped"
    if taco_cropped_dir.exists():
        for class_dir in taco_cropped_dir.iterdir():
            if class_dir.is_dir():
                orig_name = class_dir.name
                target_mapped = TACO_MAP.get(orig_name)
                for img_path in class_dir.glob("*.jpg"):
                    copy_file(img_path, target_mapped, "taco", orig_name)

    logger.info(f"Remap & merge complete.")
    logger.info(f"Source counts (raw) before mapping: {dict(counts_before)}")
    logger.info(f"Source counts after mapping and validation: {dict(counts_after)}")
    
    return {
        "gcv2_before": counts_before["gcv2"],
        "gcv2_after": counts_after["gcv2"],
        "gc12_before": counts_before["gc12"],
        "gc12_after": counts_after["gc12"],
        "taco_before": counts_before["taco"],
        "taco_after": counts_after["taco"],
    }


def compute_phash_worker(file_path: Path) -> tuple[Path, str, int] | None:
    """Worker function to compute perceptual hash and resolution of an image."""
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            phash = imagehash.phash(img)
            return file_path, str(phash), w * h
    except Exception:
        return None


def run_deduplication(merged_dir: Path, logs_dir: Path) -> int:
    """Computes pHash for all images in parallel, groups duplicates

    (Hamming Distance <= 8), and keeps the one with highest resolution.
    """
    logger.info("Starting Perceptual Hashing and Deduplication...")
    all_files = []
    for target in TARGET_CLASSES:
        all_files.extend(list((merged_dir / target).glob("*.jpg")))

    logger.info(f"Computing perceptual hashes for {len(all_files)} images...")
    
    # Run hashing in parallel
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as pool:
        results = list(tqdm(pool.imap(compute_phash_worker, all_files), total=len(all_files), desc="Hashing Images"))

    # Parse results
    valid_hashes = []
    for res in results:
        if res:
            file_path, hash_str, resolution = res
            # Convert hex hash_str to 64-bit int
            hash_int = int(hash_str, 16)
            valid_hashes.append({
                "filepath": file_path,
                "hash_int": hash_int,
                "resolution": resolution,
                "class": file_path.parent.name
            })

    logger.info("Identifying near-duplicates using Pigeonhole LSH clustering...")
    
    # LSH indexing: Partition 64-bit hash into 4 chunks of 16-bits
    # If Hamming distance <= 8, at least one chunk matches exactly
    buckets = [defaultdict(list) for _ in range(4)]
    for idx, item in enumerate(valid_hashes):
        val = item["hash_int"]
        item_id = idx
        # Extracted 16-bit keys
        k0 = val & 0xFFFF
        k1 = (val >> 16) & 0xFFFF
        k2 = (val >> 32) & 0xFFFF
        k3 = (val >> 48) & 0xFFFF
        
        buckets[0][k0].append(item_id)
        buckets[1][k1].append(item_id)
        buckets[2][k2].append(item_id)
        buckets[3][k3].append(item_id)

    # Find candidates
    def get_hamming_distance(v1: int, v2: int) -> int:
        return bin(v1 ^ v2).count("1")

    duplicate_pairs = set()
    for idx, item in enumerate(valid_hashes):
        val = item["hash_int"]
        candidates = set()
        
        # Pull candidate indices from buckets
        k0 = val & 0xFFFF
        k1 = (val >> 16) & 0xFFFF
        k2 = (val >> 32) & 0xFFFF
        k3 = (val >> 48) & 0xFFFF
        
        candidates.update(buckets[0][k0])
        candidates.update(buckets[1][k1])
        candidates.update(buckets[2][k2])
        candidates.update(buckets[3][k3])
        
        for cand_id in candidates:
            if cand_id > idx:
                cand_item = valid_hashes[cand_id]
                # Optional: duplicates across classes or same class?
                # The requirements state zero duplicates between all pairs
                dist = get_hamming_distance(val, cand_item["hash_int"])
                if dist <= 8:
                    duplicate_pairs.add((idx, cand_id, dist))

    logger.info(f"Found {len(duplicate_pairs)} duplicate relationships.")

    # Find connected components of duplicate graph
    parent = list(range(len(valid_hashes)))

    def find_parent(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find_parent(parent[i])
        return parent[i]

    def union_nodes(i: int, j: int) -> None:
        root_i = find_parent(i)
        root_j = find_parent(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for idx1, idx2, _ in duplicate_pairs:
        union_nodes(idx1, idx2)

    # Group items by component
    components = defaultdict(list)
    for idx in range(len(valid_hashes)):
        root = find_parent(idx)
        components[root].append(idx)

    # Handle duplicates and log
    removed_count = 0
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(logs_dir / "duplicates_removed.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["removed_filepath", "retained_filepath", "distance"])

        for root, indices in components.items():
            if len(indices) <= 1:
                continue
            
            # Sort items by resolution descending, then filepath
            # We want to retain indices[0]
            sorted_indices = sorted(indices, key=lambda x: (-valid_hashes[x]["resolution"], valid_hashes[x]["filepath"].name))
            retained_idx = sorted_indices[0]
            retained_item = valid_hashes[retained_idx]
            
            # Discard the rest
            for discard_idx in sorted_indices[1:]:
                discard_item = valid_hashes[discard_idx]
                discard_path = discard_item["filepath"]
                
                if discard_path.exists():
                    # Calculate actual distance to log
                    dist = get_hamming_distance(retained_item["hash_int"], discard_item["hash_int"])
                    writer.writerow([
                        discard_path.relative_to(merged_dir.parent),
                        retained_item["filepath"].relative_to(merged_dir.parent),
                        dist
                    ])
                    discard_path.unlink()
                    removed_count += 1

    logger.info(f"Deduplication completed. Removed {removed_count} near-duplicate images.")
    return removed_count


def validate_and_filter(merged_dir: Path, logs_dir: Path) -> int:
    """Filters out corrupt, small (<64x64), or completely grayscale (std < 5) images."""
    logger.info("Starting image validation and quality filtering...")
    
    all_files = []
    for target in TARGET_CLASSES:
        all_files.extend(list((merged_dir / target).glob("*.jpg")))

    removed_count = 0
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(logs_dir / "quality_removed.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "reason"])

        for filepath in tqdm(all_files, desc="Filtering Images"):
            reason = None
            try:
                with Image.open(filepath) as img:
                    img.verify()
                
                # Reopen to check dimensions and color channels since verify closes the handle
                with Image.open(filepath) as img:
                    w, h = img.size
                    if w < 64 or h < 64:
                        reason = f"undersized ({w}x{h})"
                    else:
                        # Color check
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        arr = np.array(img, dtype=np.float32)
                        
                        # Compute standard deviation of color channel differences
                        diff_rg = arr[:, :, 0] - arr[:, :, 1]
                        diff_gb = arr[:, :, 1] - arr[:, :, 2]
                        diff_br = arr[:, :, 2] - arr[:, :, 0]
                        color_std = np.sqrt(np.mean(diff_rg**2 + diff_gb**2 + diff_br**2))
                        
                        if color_std < 5:
                            reason = f"grayscale (color_std={color_std:.2f})"
            except Exception as e:
                reason = f"corrupt ({e})"

            if reason:
                writer.writerow([filepath.relative_to(merged_dir.parent), reason])
                filepath.unlink()
                removed_count += 1

    logger.info(f"Quality filter completed. Removed {removed_count} low-quality/corrupt images.")
    return removed_count


def balance_classes(merged_dir: Path) -> int:
    """Balances class distributions.

    Augments underrepresented classes (< 500 images) up to a minimum of 1000
    images.
    """
    logger.info("Starting class balancing and data augmentation...")
    augmented_count = 0

    for target in TARGET_CLASSES:
        class_dir = merged_dir / target
        images = list(class_dir.glob("*.jpg"))
        initial_count = len(images)
        
        logger.info(f"Class '{target}': {initial_count} images.")
        
        if initial_count == 0:
            logger.warning(f"Class '{target}' is empty. Cannot augment.")
            continue
            
        if initial_count < 500:
            logger.info(f"Class '{target}' is underrepresented ({initial_count} < 500). Augmenting to 1000...")
            needed = 1000 - initial_count
            
            # Seed random for reproducible augmentations
            rng = random.Random(42)
            
            for _ in range(needed):
                # Pick random image to augment
                src_path = rng.choice(images)
                try:
                    with Image.open(src_path) as img:
                        # Random augmentations:
                        # 1. Horizontal flip (50% chance)
                        if rng.random() > 0.5:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            
                        # 2. Rotation ±15 degrees
                        angle = rng.uniform(-15, 15)
                        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
                        
                        # 3. Brightness Jitter ±20%
                        brightness_factor = rng.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Brightness(img)
                        img = enhancer.enhance(brightness_factor)
                        
                        # Save augmented file
                        unique_id = uuid.uuid4().hex[:6]
                        dest_name = f"aug_{src_path.stem}_{unique_id}.jpg"
                        img.convert("RGB").save(class_dir / dest_name, "JPEG")
                        augmented_count += 1
                except Exception as e:
                    logger.warning(f"Failed to augment image {src_path}: {e}")

    logger.info(f"Class balancing complete. Added {augmented_count} augmented images.")
    return augmented_count


def train_val_test_split(merged_dir: Path, final_dir: Path, logs_dir: Path) -> dict[str, dict[str, int]]:
    """Splits images into train/val/test splits (70/15/15) and generates split manifest."""
    logger.info("Splitting dataset into train, val, and test splits (70/15/15)...")
    
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        for target in TARGET_CLASSES:
            (final_dir / split / target).mkdir(parents=True, exist_ok=True)

    manifest_data = []
    distribution = {target: {"train": 0, "val": 0, "test": 0} for target in TARGET_CLASSES}

    # For each class, split independently
    for target in TARGET_CLASSES:
        class_dir = merged_dir / target
        images = sorted(list(class_dir.glob("*.jpg")))
        
        # Shuffle dataset with stable seed
        rng = random.Random(42)
        rng.shuffle(images)
        
        total = len(images)
        train_end = int(total * 0.70)
        val_end = train_end + int(total * 0.15)
        
        for idx, img_path in enumerate(images):
            if idx < train_end:
                split_name = "train"
            elif idx < val_end:
                split_name = "val"
            else:
                split_name = "test"
                
            dest_path = final_dir / split_name / target / img_path.name
            
            try:
                shutil.copy2(img_path, dest_path)
                distribution[target][split_name] += 1
                manifest_data.append({
                    "filepath": str(dest_path.relative_to(final_dir.parent)),
                    "class": target,
                    "split": split_name
                })
            except Exception as e:
                logger.error(f"Error copying {img_path} to split {split_name}: {e}")

    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(logs_dir / "split_manifest.csv", index=False)
    
    logger.info("Stratified dataset splitting complete. Saved split manifest.")
    return distribution


def generate_summary_report(
    logs_dir: Path,
    raw_stats: dict[str, int],
    removed_duplicates: int,
    removed_quality: int,
    augmented_count: int,
    distribution: dict[str, dict[str, int]]
) -> None:
    """Generates the text summary report and prints class distribution table."""
    logger.info("Generating summary report...")
    
    report_path = logs_dir / "dataset_summary.txt"
    
    # Calculate totals
    train_total = sum(d["train"] for d in distribution.values())
    val_total = sum(d["val"] for d in distribution.values())
    test_total = sum(d["test"] for d in distribution.values())
    overall_total = train_total + val_total + test_total

    report_lines = [
        "============================================================",
        "          AI WASTE CLASSIFICATION PIPELINE SUMMARY           ",
        "============================================================",
        f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- SOURCE DATASETS STATS ---",
        f"Garbage Classification V2:          {raw_stats.get('gcv2_before', 0)} loaded -> {raw_stats.get('gcv2_after', 0)} mapped",
        f"Garbage Classification (12 classes): {raw_stats.get('gc12_before', 0)} loaded -> {raw_stats.get('gc12_after', 0)} mapped",
        f"TACO Bounding Box Crops:            {raw_stats.get('taco_before', 0)} loaded -> {raw_stats.get('taco_after', 0)} mapped",
        "",
        "--- CLEANING & PREPROCESSING STATS ---",
        f"Perceptual Hashing (pHash) near-duplicates removed: {removed_duplicates}",
        f"Quality/dimension/grayscale filters removed:        {removed_quality}",
        f"Balancing data augmentations added:                 {augmented_count}",
        "",
        "--- FINAL SPLIT TOTALS ---",
        f"Training set:   {train_total} images",
        f"Validation set: {val_total} images",
        f"Testing set:    {test_total} images",
        f"Overall Total:  {overall_total} images",
        "",
        "--- CLASS DISTRIBUTION TABLE ---",
        f"{'Class Name':<16} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8} | Status",
        "-" * 65
    ]

    print("\n" + "\n".join(report_lines[:23]))
    print(report_lines[21])
    print(report_lines[22])

    for target in TARGET_CLASSES:
        dist = distribution[target]
        t = dist["train"]
        v = dist["val"]
        te = dist["test"]
        tot = t + v + te
        status = "OK" if tot >= 1000 else "UNDERREPRESENTED"
        
        class_line = f"{target:<16} | {t:<8} | {v:<8} | {te:<8} | {tot:<8} | {status}"
        report_lines.append(class_line)
        print(class_line)

    report_lines.append("============================================================")
    print("============================================================\n")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Pipeline complete. Summary report saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquires and preprocesses waste datasets.")
    parser.add_argument("--mock-size", type=int, default=0, help="Generate mock datasets of mock-size for testing pipeline logic.")
    args = parser.parse_args()

    # Configure paths
    base_dir = Path("data")
    raw_dir = base_dir / "raw"
    merged_dir = base_dir / "merged"
    final_dir = base_dir / "final"
    logs_dir = base_dir / "logs"

    setup_logger(logs_dir)
    logger.info("Initializing Data Processing Pipeline...")

    # Step 1: Download & Extract (or Generate Mock Data)
    if args.mock_size > 0:
        generate_mock_datasets(raw_dir, args.mock_size)
    else:
        try:
            download_datasets(raw_dir)
        except Exception:
            logger.error("Download failed. To run in local mock mode for verification, run this script with --mock-size <N>.")
            return

    # Step 2: TACO Cropping
    crop_taco_objects(raw_dir / "taco")

    # Step 3: Remap & Copy
    raw_stats = remap_and_copy(raw_dir, merged_dir)

    # Step 4: Perceptual Hashing & Deduplication
    removed_duplicates = run_deduplication(merged_dir, logs_dir)

    # Step 5: Quality Check
    removed_quality = validate_and_filter(merged_dir, logs_dir)

    # Step 6: Balancing & Augmentations
    augmented_count = balance_classes(merged_dir)

    # Step 7: Train/Val/Test Split
    distribution = train_val_test_split(merged_dir, final_dir, logs_dir)

    # Step 8: Summary Report
    generate_summary_report(logs_dir, raw_stats, removed_duplicates, removed_quality, augmented_count, distribution)


if __name__ == "__main__":
    main()
