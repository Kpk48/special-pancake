import os
from torchvision.datasets import ImageFolder
from waste_classifier.hierarchical.hierarchy import get_stage1_label

def main():
    splits = ["train", "val", "test"]
    for split in splits:
        root_dir = f"data/final/{split}"
        dataset = ImageFolder(root=root_dir)
        classes = dataset.classes
        
        counts = {0: 0, 1: 0}
        for path, t3 in dataset.samples:
            cls_name = classes[t3]
            target1 = get_stage1_label(cls_name, path)
            counts[target1] += 1
            
        print(f"Split: {split:5s} | Biodegradable: {counts[0]:4d} | Non-Biodegradable: {counts[1]:4d} | Total: {len(dataset):5d}")

if __name__ == "__main__":
    main()
