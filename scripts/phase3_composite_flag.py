import os
import csv
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

from waste_classifier.hierarchical.hierarchy import get_stage1_label, STAGE1_CLASSES

TARGET_CLASSES = {"paper", "cardboard", "textile"}

TACO_KNOWN_COMPOSITE_PREFIXES = {
    "taco_Paper_cup",
    "taco_Meal_carton",
    "taco_Pizza_box",
    "taco_Wrapping_paper",
    "taco_Magazine_paper",
    "taco_Aluminium_blister_pack",
}

CONF_THRESHOLD = 0.85

SPLITS = [
    ("train", "data/final/train"),
    ("val",   "data/final/val"),
    ("test",  "data/final/test"),
]

THUMB_W    = 200
THUMB_H    = 200
LABEL_H    = 38
PAD        = 6
GRID_COLS  = 6
BATCH_SIZE = 30


class Stage1FilteredDataset(Dataset):
    def __init__(self, base_dataset, classes, transform):
        self.transform = transform
        self.samples   = []
        for path, t3 in base_dataset.samples:
            cls_name = classes[t3]
            if cls_name in TARGET_CLASSES:
                self.samples.append((path, t3, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t3, cls_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, t3, cls_name, path


def get_source_tag(filepath):
    name = os.path.basename(filepath)
    if name.startswith("gc12_"):
        return "gc12_"
    if name.startswith("gcv2_"):
        return "gcv2_"
    if name.startswith("taco_"):
        return "taco_"
    return "unknown"


def is_known_composite(filepath):
    name = os.path.basename(filepath)
    stem = name[: name.rfind("_")]
    return stem in TACO_KNOWN_COMPOSITE_PREFIXES


def make_contact_sheet(entries, out_path, font):
    n     = len(entries)
    rows  = math.ceil(n / GRID_COLS)
    sheet_w = GRID_COLS * (THUMB_W + PAD) + PAD
    sheet_h = rows * (THUMB_H + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw  = ImageDraw.Draw(sheet)

    for i, e in enumerate(entries):
        col = i % GRID_COLS
        row = i // GRID_COLS
        x   = PAD + col * (THUMB_W + PAD)
        y   = PAD + row * (THUMB_H + LABEL_H + PAD)

        try:
            img = Image.open(e["path"]).convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (THUMB_W, THUMB_H), (60, 60, 60))
        sheet.paste(img, (x, y))

        draw.rectangle([x, y + THUMB_H, x + THUMB_W, y + THUMB_H + LABEL_H], fill=(40, 40, 40))
        fname   = os.path.basename(e["path"])
        short   = (fname[:21] + "..") if len(fname) > 23 else fname
        caption = f"{short}\n{e['cls']} | {e['conf']*100:.1f}% | {e['src']}"
        draw.text((x + 2, y + THUMB_H + 2), caption, font=font, fill=(210, 210, 210))

    sheet.save(out_path, dpi=(150, 150))


def collate_fn(batch):
    imgs      = torch.stack([b[0] for b in batch])
    t3s       = [b[1] for b in batch]
    cls_names = [b[2] for b in batch]
    paths     = [b[3] for b in batch]
    return imgs, t3s, cls_names, paths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("artifacts/resnet18_stage1_baseline.pt", map_location=device))
    model.to(device)
    model.eval()

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 9)
    except Exception:
        font = ImageFont.load_default()

    out_dir = "results/phase3_review_batches"
    os.makedirs(out_dir, exist_ok=True)

    flagged_model    = []
    flagged_known    = []

    class_stats  = defaultdict(lambda: {"total": 0, "model_flagged": 0, "known_composite": 0})
    source_stats = defaultdict(lambda: {"total": 0, "model_flagged": 0, "known_composite": 0})

    for split_name, split_path in SPLITS:
        raw_ds = ImageFolder(root=split_path, transform=None)
        classes = raw_ds.classes

        ds     = Stage1FilteredDataset(raw_ds, classes, tfm)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

        with torch.no_grad():
            for imgs, t3s, cls_names, paths in loader:
                imgs    = imgs.to(device)
                outputs = model(imgs)
                probs   = torch.softmax(outputs, dim=-1)
                confs, preds = torch.max(probs, dim=-1)

                for j in range(len(paths)):
                    cls_name = cls_names[j]
                    t3       = t3s[j]
                    path     = paths[j]
                    pred_s1  = int(preds[j].item())
                    conf     = float(confs[j].item())
                    true_s1  = get_stage1_label(cls_name)
                    src      = get_source_tag(path)

                    class_stats[cls_name]["total"]  += 1
                    source_stats[(cls_name, src)]["total"] += 1

                    known = is_known_composite(path)
                    if known:
                        flagged_known.append({
                            "path": path, "cls": cls_name, "split": split_name,
                            "true_s1": true_s1, "pred_s1": pred_s1,
                            "conf": conf, "src": src,
                        })
                        class_stats[cls_name]["known_composite"]  += 1
                        source_stats[(cls_name, src)]["known_composite"] += 1

                    elif pred_s1 != true_s1 and conf >= CONF_THRESHOLD:
                        flagged_model.append({
                            "path": path, "cls": cls_name, "split": split_name,
                            "true_s1": true_s1, "pred_s1": pred_s1,
                            "conf": conf, "src": src,
                        })
                        class_stats[cls_name]["model_flagged"]  += 1
                        source_stats[(cls_name, src)]["model_flagged"] += 1

    flagged_model.sort(key=lambda x: x["conf"], reverse=True)

    for group_name, entries in [("model_flagged", flagged_model), ("known_composite", flagged_known)]:
        n_batches = math.ceil(len(entries) / BATCH_SIZE)
        for bi in range(n_batches):
            batch = entries[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
            fname = f"{group_name}_batch{bi+1:02d}_of{n_batches:02d}.png"
            make_contact_sheet(batch, os.path.join(out_dir, fname), font)

    csv_path = "results/phase3_flagged_images.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flag_type", "file_path", "split", "fine_class", "source_tag",
                         "true_label", "predicted_label", "confidence"])
        for e in flagged_model:
            writer.writerow(["model_flagged", e["path"], e["split"], e["cls"], e["src"],
                             STAGE1_CLASSES[e["true_s1"]], STAGE1_CLASSES[e["pred_s1"]],
                             f"{e['conf']:.6f}"])
        for e in flagged_known:
            writer.writerow(["known_composite", e["path"], e["split"], e["cls"], e["src"],
                             STAGE1_CLASSES[e["true_s1"]], STAGE1_CLASSES[e["pred_s1"]],
                             f"{e['conf']:.6f}"])

    print("=== Phase 3 Flagging Summary ===")
    print(f"Model-flagged (conf > {CONF_THRESHOLD}, disagrees with label): {len(flagged_model)}")
    print(f"Known-composite TACO (by filename rule):                       {len(flagged_known)}")
    print()
    print("--- Per class ---")
    for cls in sorted(TARGET_CLASSES):
        s = class_stats[cls]
        mf_pct = s["model_flagged"] / s["total"] * 100 if s["total"] else 0
        print(f"  {cls:12s} total={s['total']:4d}  model_flagged={s['model_flagged']:3d} ({mf_pct:5.1f}%)  "
              f"known_composite={s['known_composite']}")
    print()
    print("--- Per (class, source) ---")
    for (cls, src) in sorted(source_stats.keys()):
        s = source_stats[(cls, src)]
        mf_pct = s["model_flagged"] / s["total"] * 100 if s["total"] else 0
        print(f"  {cls:12s} {src:8s}  total={s['total']:4d}  model_flagged={s['model_flagged']:3d} ({mf_pct:5.1f}%)  "
              f"known_composite={s['known_composite']}")


if __name__ == "__main__":
    main()
