import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

from waste_classifier.hierarchical.hierarchy import get_stage1_label, STAGE1_CLASSES

class Stage1DatasetWrapper(Dataset):
    def __init__(self, base_dataset, classes):
        self.base_dataset = base_dataset
        self.classes = classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target3 = self.base_dataset[idx]
        class_name = self.classes[target3]
        target1 = get_stage1_label(class_name)
        return img, target1

def get_source_tag(filepath):
    filename = os.path.basename(filepath)
    if filename.startswith("gc12_"):
        return "gc12_"
    elif filename.startswith("gcv2_"):
        return "gcv2_"
    elif filename.startswith("taco_"):
        return "taco_"
    return "unknown"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    raw_test = ImageFolder(root="data/final/test", transform=val_test_transform)
    classes  = raw_test.classes

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("artifacts/resnet18_stage1_baseline.pt", map_location=device))
    model.to(device)
    model.eval()

    all_paths    = [path for path, _ in raw_test.samples]
    all_t3_class = [classes[t3] for _, t3 in raw_test.samples]
    all_targets  = []
    all_preds    = []
    all_confs    = []

    loader = DataLoader(Stage1DatasetWrapper(raw_test, classes), batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        for images, targets1 in loader:
            images   = images.to(device)
            outputs  = model(images)
            probs    = torch.softmax(outputs, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            all_targets.extend(targets1.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)
    all_confs   = np.array(all_confs)

    pc_errors = []
    for idx in range(len(raw_test)):
        t3_cls  = all_t3_class[idx]
        true_s1 = all_targets[idx]
        pred_s1 = all_preds[idx]
        if t3_cls not in ("paper", "cardboard"):
            continue
        if true_s1 == pred_s1:
            continue
        pc_errors.append({
            "idx":       idx,
            "file_path": all_paths[idx],
            "t3_class":  t3_cls,
            "true_s1":   true_s1,
            "pred_s1":   pred_s1,
            "conf":      float(all_confs[idx]),
            "src_tag":   get_source_tag(all_paths[idx]),
        })

    crosstab = defaultdict(lambda: {"total": 0, "errors": 0})
    for t3_cls in ("paper", "cardboard"):
        for src in ("gc12_", "gcv2_", "taco_"):
            crosstab[(t3_cls, src)]["total"] = 0

    for idx in range(len(raw_test)):
        t3_cls = all_t3_class[idx]
        if t3_cls not in ("paper", "cardboard"):
            continue
        src    = get_source_tag(all_paths[idx])
        true_s1 = all_targets[idx]
        pred_s1 = all_preds[idx]
        crosstab[(t3_cls, src)]["total"] += 1
        if true_s1 != pred_s1:
            crosstab[(t3_cls, src)]["errors"] += 1

    os.makedirs("results", exist_ok=True)

    with open("results/stage2b_paper_cardboard_by_source.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fine_class", "source_tag", "total_images", "error_count", "error_rate"])
        for t3_cls in ("paper", "cardboard"):
            for src in ("gc12_", "gcv2_", "taco_"):
                stats = crosstab[(t3_cls, src)]
                total = stats["total"]
                errs  = stats["errors"]
                rate  = errs / total if total > 0 else 0.0
                writer.writerow([t3_cls, src, total, errs, f"{rate:.6f}"])

    pc_errors.sort(key=lambda x: x["conf"], reverse=True)
    top20 = pc_errors[:20]

    with open("results/stage2b_paper_cardboard_hard_errors.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "true_label", "predicted_label", "confidence", "source_tag"])
        for e in top20:
            true_lbl = STAGE1_CLASSES[e["true_s1"]]
            pred_lbl = STAGE1_CLASSES[e["pred_s1"]]
            writer.writerow([e["file_path"], true_lbl, pred_lbl, f"{e['conf']:.6f}", e["src_tag"]])

    cols, rows = 4, 5
    thumb_w, thumb_h = 200, 200
    label_h = 36
    pad = 6
    sheet_w = cols * (thumb_w + pad) + pad
    sheet_h = rows * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    for i, e in enumerate(top20):
        col = i % cols
        row = i // cols
        x   = pad + col * (thumb_w + pad)
        y   = pad + row * (thumb_h + label_h + pad)

        try:
            img = Image.open(e["file_path"]).convert("RGB")
            img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (thumb_w, thumb_h), (80, 80, 80))

        sheet.paste(img, (x, y))

        draw      = ImageDraw.Draw(sheet)
        label_bg  = (50, 50, 50)
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=label_bg)

        fname   = os.path.basename(e["file_path"])
        short   = fname[:22] + ".." if len(fname) > 24 else fname
        caption = f"{short}\n{e['t3_class']} | {e['conf']*100:.1f}%"
        draw.text((x + 3, y + thumb_h + 2), caption, font=font, fill=(220, 220, 220))

    sheet.save("results/stage2b_paper_cardboard_contact_sheet.png", dpi=(150, 150))
    print(f"Contact sheet saved. {len(top20)} images.")
    print("Cross-tab and hard-error CSVs saved.")

if __name__ == "__main__":
    main()
