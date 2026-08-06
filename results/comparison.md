# Model Evaluation and Paper Comparison Report

This report evaluates the performance of the **Hierarchical CNN** versus the **KNN Baseline** model. Results are structured by classification stages to mirror the metrics in Table 9 and Table 10 of Nahiduzzaman et al. (2025).

| Model | Stage | Classes | Precision (macro) | Recall (macro) | F1-Score | Accuracy | AUC | Params | Size (MB) | Inference Time (s/img) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KNN Baseline | Stage 1 | 2 | 0.3748 | 0.5000 | 0.4284 | 0.7496 | 0.5000 | 5016 | 0.1408 | 0.00034 |
| Hierarchical CNN | Stage 1 | 2 | 0.7511 | 0.8168 | 0.7638 | 0.7952 | 0.9033 | 96002 | 0.3872 | 0.00956 |
| KNN Baseline | Stage 2 | 6 | 0.0737 | 0.1667 | 0.1022 | 0.4420 | 0.5000 | 5016 | 0.1408 | 0.00034 |
| Hierarchical CNN | Stage 2 | 6 | 0.5882 | 0.6211 | 0.5882 | 0.6698 | 0.8344 | 105446 | 0.4240 | 0.00956 |
| KNN Baseline | Stage 3 | 8 | 0.0038 | 0.1250 | 0.0074 | 0.0304 | 0.5000 | 5016 | 0.1408 | 0.00034 |
| Hierarchical CNN | Stage 3 | 8 | 0.5492 | 0.5937 | 0.5551 | 0.6340 | 0.8251 | 105640 | 0.4248 | 0.00956 |
