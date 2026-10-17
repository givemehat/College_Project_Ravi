# Battery Health RUL Deep Learning Model Benchmark Comparison

Zero-shot evaluation strictly on unseen test battery cell **NASA B0018** (Trained on `B0005`, `B0006`; Validated on `B0007`):

| Model Architecture | Parameters | Train Time (s) | Inference (ms/sample) | MAE | RMSE | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 154 | 0.05 | 0.10 | 0.0942 | 0.1120 | 11.45% | -0.4510 |
| **Random Forest** | 50,000 | 1.85 | 2.40 | 0.0895 | 0.1065 | 10.82% | -0.3840 |
| **Standard LSTM** | 91,457 | 10.69 | 1.20 | 0.0807 | 0.0951 | 9.72% | -0.1933 |
| **Standard GRU** | 69,121 | 9.42 | 1.00 | 0.0846 | 0.1012 | 10.17% | -0.3518 |
| **TCN (Dilated Residual)** | 220,321 | 15.50 | 1.90 | 0.0827 | 0.0984 | 9.86% | -0.2789 |
| **Tuned 3-Layer GRU (Optimal)** | 118,529 | 12.30 | 1.10 | **0.0768** | **0.0892** | **8.84%** | **-0.0842** |

### Key Experimental Insights
1. **Sequence Modeling Advantage**: Recurrent and dilated causal temporal models capture sequential degradation context across sliding windows ($L=15$), outperforming memoryless static baselines (Ridge and Random Forest).
2. **Optimal Hyperparameter Configuration**: Day 19 grid search across 1–5 layers and 0.1–0.3 dropout identified that a 3-layer GRU (64 hidden units, dropout 0.2) achieves the best trade-off between expressive capacity and regularization, yielding the lowest MAE (0.0768) and RMSE (0.0892).
3. **Inference Latency**: All deep learning models evaluate in $\le 1.9\text{ ms/cycle}$, confirming real-time edge embedded deployment feasibility.
