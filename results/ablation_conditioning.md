# Ablation Study — Conditioned Classification Heads

This ablation study evaluates the performance impact of conditioning downstream classifier heads (Stage 2 and Stage 3) on previous stage predicted class embeddings versus training flat independent heads without hierarchy conditioning.

## 1. Parameter Matching Verification

* **Stage 2 Model Parameters**: Conditioned `105,446` vs Flat `104,390` (Difference: `1.00%`, verified matched within 5%)
* **Stage 3 Model Parameters**: Conditioned `105,640` vs Flat `104,520` (Difference: `1.06%`, verified matched within 5%)

---

## 2. Results Summary (Stage 1 Threshold = 0.55)

Evaluated on the held-out test split (2,568 images) at calibrated Stage 1 decision threshold $t = 0.55$:

| Model Variant | Stage 2 Acc | Stage 2 F1 | Stage 3 Acc | Stage 3 F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Conditioned Heads (Hierarchical Embedding Baseline)** | **66.98%** | **0.5882** | **63.40%** | **0.5551** |
| **Flat Independent Heads (No Conditioning)** | 62.85% | 0.5442 | 58.71% | 0.5083 |
| **Conditioning Advantage (Delta)** | **+4.13 pp** | **+0.0440** | **+4.69 pp** | **+0.0468** |

---

## 3. Methodological & Analytical Notes
1. **Hierarchical Context Advantage**: Conditioning downstream heads on upstream stage predictions provides a **+4.13 to +4.69 percentage point** accuracy improvement in Stage 2 and Stage 3.
2. **Parameter Matching**: Parameter differences between conditioned and flat heads are $\le 1.06\%$, confirming that performance gains stem from hierarchical conditioning logic rather than increased network capacity.
3. **Single-Run Reporting Caveat**: Baseline figures reflect the canonical Phase 4 retrained checkpoint benchmarks. Ablation variant results are evaluated from single-seed training runs under identical hyperparameters and should be interpreted as indicative structural trends.
