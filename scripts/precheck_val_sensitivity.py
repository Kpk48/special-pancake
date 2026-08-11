import os
import shutil
import csv
import json
import subprocess
import sys

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    overrides_file = os.path.join(root_dir, "data", "final", "stage1_label_overrides.csv")
    backup_file = os.path.join(root_dir, "data", "final", "stage1_label_overrides.csv.bak")

    # Step 0: Read existing overrides to report count
    with open(overrides_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    
    header = rows[0]
    data_rows = rows[1:]
    train_rows = [r for r in data_rows if "train" in r[0]]
    val_rows = [r for r in data_rows if "val" in r[0]]
    test_rows = [r for r in data_rows if "test" in r[0]]
    
    print(f"Original overrides count: {len(data_rows)} (Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)})")

    # Step 1: Backup original overrides
    shutil.copyfile(overrides_file, backup_file)
    print(f"Backed up original file to {backup_file}")

    try:
        # Step 2: Write temporary overrides file excluding val rows
        with open(overrides_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(train_rows)
        print(f"Wrote temporary overrides file with {len(train_rows)} train-only rows.")

        # Step 3: Run calibrate_stage1.py
        calib_script = os.path.join(root_dir, "scripts", "calibrate_stage1.py")
        res = subprocess.run([sys.executable, calib_script], capture_output=True, text=True, check=True)
        print("Calibrate script output (temp val-excluded version):")
        
        # Load output artifact
        calib_json = os.path.join(root_dir, "artifacts", "hierarchical", "stage1_calibration.json")
        with open(calib_json, "r", encoding="utf-8") as f:
            temp_result = json.load(f)
            
        print("\n=== TEMP (Val Overrides Excluded) Calibration Results ===")
        print(f"Optimal Acc Threshold: {temp_result['optimal_accuracy_threshold']}")
        print(f"Optimal F1 Threshold: {temp_result['optimal_f1_threshold']}")
        print(f"Val Acc at 0.55: {next(r['val_acc'] for r in temp_result['sweep_results'] if abs(r['threshold']-0.55)<1e-5):.4f}")
        print(f"Val Acc at 0.50: {next(r['val_acc'] for r in temp_result['sweep_results'] if abs(r['threshold']-0.50)<1e-5):.4f}")

    finally:
        # Step 4: Restore original file
        shutil.copyfile(backup_file, overrides_file)
        os.remove(backup_file)
        print("\nRestored original stage1_label_overrides.csv successfully.")

    # Step 5: Re-run calibrate_stage1.py with original file to restore baseline calibration JSON artifact
    res_orig = subprocess.run([sys.executable, calib_script], capture_output=True, text=True, check=True)
    with open(calib_json, "r", encoding="utf-8") as f:
        orig_result = json.load(f)

    print("\n=== RESTORED ORIGINAL (49 overrides) Calibration Results ===")
    print(f"Optimal Acc Threshold: {orig_result['optimal_accuracy_threshold']}")
    print(f"Optimal F1 Threshold: {orig_result['optimal_f1_threshold']}")

if __name__ == "__main__":
    main()
