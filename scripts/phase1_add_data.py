"""
Phase 1: Download additional datasets and merge them into data/final/.

Datasets:
  1. feyzazkefe/trashnet   — cardboard, glass, metal, paper, plastic, trash (~2500 imgs)
  2. techsash/waste-classification-data — organic, recyclable (~25K imgs, but only mapped classes used)
  3. sapal6/waste-classification-data-v2 — additional organic/recyclable

Only images that map cleanly to our 8 taxonomy classes are kept.
No synthetic or hallucinated data is added.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase1_data")

# ── Our 8 target classes ────────────────────────────────────────────────────
CLASSES = {"battery", "cardboard", "glass", "metal", "organic", "paper", "plastic", "textile"}

# ── Class mappings: source_label (lowercase) → our class ───────────────────
LABEL_MAP = {
    # TrashNet labels
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": None,          # "trash" is an undefined catch-all — skip

    # techsash / sapal6 — folder names
    "o": "organic",         # techsash uses O/R binary
    "r": None,              # "recyclable" is too broad — skip (can't tell which class)
    "organic": "organic",
    "recyclable": None,

    # Common alternatives across merged Kaggle datasets
    "cardboards": "cardboard",
    "glasses": "glass",
    "metals": "metal",
    "papers": "paper",
    "plastics": "plastic",
    "clothes": "textile",
    "textiles": "textile",
    "biological": "organic",
    "bio": "organic",
    "shoes": None,          # no matching class — skip
    "batteries": "battery",
    "battery": "battery",
    "trash bag": None,
    "cup": None,
}

RAW_DIR  = Path("data/raw_extra")
WORK_DIR = Path("data/work_extra")   # staged before dedup
FINAL    = Path("data/final")

# ── Load existing pHashes to avoid adding duplicates of what we already have ─
def load_existing_hashes(final_dir: Path) -> set:
    log.info("Loading existing image hashes from data/final/ ...")
    hashes = set()
    for img_path in final_dir.rglob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                h = str(imagehash.phash(Image.open(img_path).convert("RGB")))
                hashes.add(h)
            except Exception:
                pass
    log.info("Loaded %d existing hashes.", len(hashes))
    return hashes


def download_dataset(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        log.info("Dataset %s already exists in %s, skipping download.", slug, dest)
        return True
    log.info("Downloading %s ...", slug)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", slug, "-p", str(dest), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error("Download failed for %s:\n%s", slug, result.stderr)
        return False
    log.info("Downloaded %s OK.", slug)
    return True


def map_and_stage(src_root: Path, work_dir: Path, existing_hashes: set) -> dict[str, int]:
    """Walk src_root, apply LABEL_MAP, quality-filter, dedup against existing, copy to work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    skipped = 0

    for img_path in src_root.rglob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        # Determine label from immediate parent folder name
        folder = img_path.parent.name.lower().strip()
        our_class = LABEL_MAP.get(folder)
        if our_class is None:
            skipped += 1
            continue

        # Quality check
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if w < 64 or h < 64:
                skipped += 1
                continue
        except (UnidentifiedImageError, Exception):
            skipped += 1
            continue

        # Dedup against existing dataset
        try:
            ph = str(imagehash.phash(img))
        except Exception:
            skipped += 1
            continue

        if ph in existing_hashes:
            skipped += 1
            continue

        # Stage file
        dest_class_dir = work_dir / our_class
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_class_dir / img_path.name
        # Avoid name collisions
        if dest_file.exists():
            dest_file = dest_class_dir / f"{img_path.stem}_{counts[our_class]}{img_path.suffix}"

        shutil.copy2(img_path, dest_file)
        existing_hashes.add(ph)  # mark as seen
        counts[our_class] += 1

    log.info("Staged: %s | Skipped: %d", dict(counts), skipped)
    return counts


def merge_into_final(work_dir: Path, final_dir: Path) -> dict[str, int]:
    """Copy staged new images into data/final/train/ with a safe rename scheme."""
    merged = defaultdict(int)
    # work_dir contains subdirectories for each dataset
    for dataset_dir in work_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for cls_dir in dataset_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            if cls not in CLASSES:
                continue
            dest = final_dir / "train" / cls
            dest.mkdir(parents=True, exist_ok=True)
            # Find a safe start index
            existing = list(dest.glob("*"))
            idx = len(existing)
            for f in cls_dir.iterdir():
                if f.is_file():
                    new_name = dest / f"extra_{idx:05d}{f.suffix}"
                    shutil.copy2(f, new_name)
                    idx += 1
                    merged[cls] += 1
    return merged


def main():
    datasets = [
        ("feyzazkefe/trashnet",               RAW_DIR / "trashnet"),
        ("techsash/waste-classification-data", RAW_DIR / "techsash"),
        ("sapal6/waste-classification-data-v2",RAW_DIR / "sapal6"),
    ]

    log.info("=== PHASE 1: Additional Data Download & Merge ===")

    # Load existing hashes
    existing_hashes = load_existing_hashes(FINAL)

    total_new = defaultdict(int)

    for slug, dest in datasets:
        if not download_dataset(slug, dest):
            log.warning("Skipping %s due to download failure.", slug)
            continue

        work = WORK_DIR / dest.name
        counts = map_and_stage(dest, work, existing_hashes)
        for cls, n in counts.items():
            total_new[cls] += n

    log.info("=== Staging complete. New images per class: %s", dict(total_new))

    # Merge into final/train/
    merged = merge_into_final(WORK_DIR, FINAL)
    log.info("=== Merged into data/final/train/: %s", dict(merged))

    # Print final class counts
    log.info("=== Final train class counts after merge:")
    for cls in sorted(CLASSES):
        cls_dir = FINAL / "train" / cls
        n = sum(1 for f in cls_dir.iterdir() if f.is_file()) if cls_dir.exists() else 0
        log.info("  %-12s : %d images", cls, n)


if __name__ == "__main__":
    main()
