# Model Evaluation and Paper Comparison Report

This report evaluates the performance of the **Hierarchical CNN** versus the **KNN Baseline** model. Results are structured by classification stages to mirror the metrics in Table 9 and Table 10 of Nahiduzzaman et al. (2025).

| Model | Stage | Classes | Precision (macro) | Recall (macro) | F1-Score | Accuracy | AUC | Params | Size (MB) | Inference Time (s/img) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KNN Baseline | Stage 1 | 2 | 0.3720 | 0.5000 | 0.4266 | 0.7440 | 0.5000 | 5016 | 0.1348 | 0.00010 |
| Hierarchical CNN | Stage 1 | 2 | 0.3720 | 0.5000 | 0.4266 | 0.7440 | 0.5778 | 96002 | 0.3864 | 0.01398 |
| KNN Baseline | Stage 2 | 6 | 0.0752 | 0.1667 | 0.1036 | 0.4512 | 0.5000 | 5016 | 0.1348 | 0.00010 |
| Hierarchical CNN | Stage 2 | 6 | 0.0174 | 0.1667 | 0.0315 | 0.1043 | 0.6014 | 105446 | 0.4232 | 0.01398 |
| KNN Baseline | Stage 3 | 11 | 0.0040 | 0.1250 | 0.0078 | 0.0320 | 0.5000 | 5016 | 0.1348 | 0.00010 |
| Hierarchical CNN | Stage 3 | 11 | 0.0130 | 0.1250 | 0.0236 | 0.1043 | 0.5560 | 105835 | 0.4246 | 0.01398 |
