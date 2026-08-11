import sys
import os
import json
import numpy as np

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def compute_mean_std(values):
    return float(np.mean(values)), float(np.std(values))

def main():
    abl_data = load_json("results/ablation_5seed_results.json")
    summary_data = load_json("scratch/ieee_data_summary.json")
    
    if abl_data is None:
        print("Ablation data not found yet.")
        return

    print("================================================================================")
    print("                      IEEE PAPER FINAL AUDIT & DATA SUMMARY                     ")
    print("================================================================================")

    # 1. LINEAGE RECONCILIATION
    print("\n1. LINEAGE RECONCILIATION")
    print("   - Checkpoint Lineage Confirmation:")
    print("     * Table I (Stage 1 Metrics): Post-stage1_v2_relabeled correction lineage (Phase 4 relabeled dataset).")
    print("     * Table II (Pipeline Metrics): Pre-relabeled canonical lineage (stage1.pt, stage2.pt, stage3.pt).")
    print("     * Table III (Multi-Scale Ablation): Post-relabeled Stage 1 (88.20%), mixed with pre-relabeled Stage 2/3.")
    print("     * Table IV (Conditioning Ablation): Pre-relabeled Stage 1 input to Stage 2/3.")
    print("   - Exact Discrepancy Explanation (87.97% vs 88.20%):")
    print("     * 88.20% represents Stage 1 test accuracy evaluated on the Phase 4 relabeled test set WITH 6 test-set label overrides applied (test target contamination).")
    print("     * 87.97% / 88.01% represents Stage 1 test accuracy evaluated on the UNCONTAMINATED test set (where the 6 test-set overrides were removed, restoring original ground truth).")
    
    if abl_data:
        ds_full_s1 = [x["dsconv"]["s1_acc"] * 100 for x in abl_data["dsconv"]]
        ds_full_s3 = [x["dsconv"]["s3_acc"] * 100 for x in abl_data["dsconv"]]
        ds_full_jt = [x["dsconv"]["joint_acc"] * 100 for x in abl_data["dsconv"]]
        
        ds_plain_s1 = [x["plain"]["s1_acc"] * 100 for x in abl_data["dsconv"]]
        ds_plain_s3 = [x["plain"]["s3_acc"] * 100 for x in abl_data["dsconv"]]
        ds_plain_jt = [x["plain"]["joint_acc"] * 100 for x in abl_data["dsconv"]]
        
        cond_s1 = [x["conditioned"]["s1_acc"] * 100 for x in abl_data["conditioning"]]
        cond_s3 = [x["conditioned"]["s3_acc"] * 100 for x in abl_data["conditioning"]]
        cond_jt = [x["conditioned"]["joint_acc"] * 100 for x in abl_data["conditioning"]]
        
        flat_s1 = [x["flat"]["s1_acc"] * 100 for x in abl_data["conditioning"]]
        flat_s3 = [x["flat"]["s3_acc"] * 100 for x in abl_data["conditioning"]]
        flat_jt = [x["flat"]["joint_acc"] * 100 for x in abl_data["conditioning"]]

        m1, s1 = compute_mean_std(ds_full_s1)
        m3, s3 = compute_mean_std(ds_full_s3)
        mj, sj = compute_mean_std(ds_full_jt)
        
        pm1, ps1 = compute_mean_std(ds_plain_s1)
        pm3, ps3 = compute_mean_std(ds_plain_s3)
        pmj, psj = compute_mean_std(ds_plain_jt)
        
        cm1, cs1 = compute_mean_std(cond_s1)
        cm3, cs3 = compute_mean_std(cond_s3)
        cmj, csj = compute_mean_std(cond_jt)

        fm1, fs1 = compute_mean_std(flat_s1)
        fm3, fs3 = compute_mean_std(flat_s3)
        fmj, fsj = compute_mean_std(flat_jt)

        print("\n   - N=5 Multi-Seed Ablation Results (Mean ± Std %):")
        print("     A. Multi-Scale Ablation (DSConv vs. Plain Conv):")
        print(f"        * Full DSConv:   Stage 1 Acc: {m1:.2f} ± {s1:.2f}%, Stage 3 Acc: {m3:.2f} ± {s3:.2f}%, Joint Acc: {mj:.2f} ± {sj:.2f}%")
        print(f"        * Plain Conv:    Stage 1 Acc: {pm1:.2f} ± {ps1:.2f}%, Stage 3 Acc: {pm3:.2f} ± {ps3:.2f}%, Joint Acc: {pmj:.2f} ± {psj:.2f}%")
        print(f"        * Delta:         Stage 1 Delta: +{m1-pm1:.2f} pp, Stage 3 Delta: +{m3-pm3:.2f} pp, Joint Delta: +{mj-pmj:.2f} pp")
        print("     B. Conditioning Ablation (Conditioned vs. Flat Heads):")
        print(f"        * Conditioned:   Stage 1 Acc: {cm1:.2f} ± {cs1:.2f}%, Stage 3 Acc: {cm3:.2f} ± {cs3:.2f}%, Joint Acc: {cmj:.2f} ± {csj:.2f}%")
        print(f"        * Flat Heads:    Stage 1 Acc: {fm1:.2f} ± {fs1:.2f}%, Stage 3 Acc: {fm3:.2f} ± {fs3:.2f}%, Joint Acc: {fmj:.2f} ± {fsj:.2f}%")
        print(f"        * Delta:         Stage 1 Delta: +{cm1-fm1:.2f} pp, Stage 3 Delta: +{cm3-fm3:.2f} pp, Joint Delta: +{cmj-fmj:.2f} pp")

if __name__ == "__main__":
    main()
