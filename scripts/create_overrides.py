import os
import csv

def main():
    src_csv = "results/phase3_flagged_images.csv"
    dest_csv = "data/final/stage1_label_overrides.csv"
    
    overrides = []
    
    with open(src_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["file_path"]
            predicted = row["predicted_label"]
            
            label_idx = 0 if predicted == "biodegradable" else 1
            overrides.append((filepath, label_idx))
            
    os.makedirs(os.path.dirname(dest_csv), exist_ok=True)
    with open(dest_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "label_idx"])
        for filepath, idx in overrides:
            writer.writerow([filepath, idx])
            
    print(f"Generated overrides for {len(overrides)} files.")

if __name__ == "__main__":
    main()
