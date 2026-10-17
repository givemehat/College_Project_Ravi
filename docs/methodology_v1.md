# Methodology Document: Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction

**Programme**: TIH, IIT Guwahati — 4-Week Online Internship Research Project  
**Track**: Group O4 — Deep Learning / Battery Prognostics / Sequential Modeling  
**Intern / Researcher**: Rajnish Singh  
**Mentor**: Preetodeep Dev  

---

## 1. Executive Summary & Problem Formulation

Predictive health monitoring and Remaining Useful Life (RUL) estimation are vital for the reliable operation of Lithium-Ion battery energy storage systems. Because battery aging is governed by non-linear electrochemical processes (including active material dissolution, lithium plating, and SEI layer growth), standard static regression models struggle to capture degradation dynamics over long operational cycles.

This research project delivers a reproducible temporal deep learning framework for:
1. **State of Health (SOH) Estimation**: Quantifying instantaneous discharge capacity relative to nominal capacity:
   $$\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$$
2. **Remaining Useful Life (RUL) Prediction**: Forecasting the number of operational discharge cycles remaining before capacity drops permanently below the End-of-Life (EOL) failure threshold ($1.40\text{ Ah}$ for B0005, B0006, and B0018; $1.50\text{ Ah}$ for B0007):
   $$\text{RUL}_k = \max(0, k_{\text{EOL}} - k)$$
3. **Addressing Physical Non-Linearities**:
   - **Capacity Regeneration**: Non-linear capacity rebound spikes caused by electrochemical relaxation during rest periods between cycles.
   - **Accelerated EOL "Knee" Transition**: Sharp transition from gradual linear aging to accelerated exponential degradation in late life.

---

## 2. Dataset & Zero-Leakage Experimental Setup

### 2.1 NASA PCoE Battery Aging Dataset
Experiments are conducted on the benchmark NASA Ames Prognostics Center of Excellence (PCoE) battery aging dataset:
- Cells: `B0005`, `B0006`, `B0007`, `B0018`.
- Chemistry: 18650 Cylindrical LiCoO2 cells cycled under constant-current / constant-voltage (CC-CV) charge (1.5A to 4.2V) and constant-current discharge (2.0A to cut-off voltages).

### 2.2 Strict Zero-Leakage Split
To ensure unbiased generalization without data leakage:
- **Training Cells**: `B0005` (168 cycles) & `B0006` (168 cycles).
- **Validation Cell**: `B0007` (168 cycles) — used for hyperparameter tuning and model checkpointing.
- **Unseen Test Cell**: `B0018` (132 cycles) — held out completely for zero-shot evaluation.
- **Normalization Safeguard**: Normalization parameters ($\min, \max$) are fitted **strictly** on training cells and applied out-of-sample to validation and test cells.

### 2.3 Feature Engineering & Temporal Windowing
- **Extracted Features**: Cycle mean/min voltage, discharge duration, temperature elevation ($\Delta T = T_{\text{max}} - T_{\text{start}}$), charge integration, and cycle capacity difference.
- **Sliding Sequence Window**: Sliding window of length $L = 15$ cycles to provide sequential temporal context for recurrent and convolutional models.

---

## 3. Evaluated Model Zoo

### 3.1 Baseline Models
- **Ridge Regression**: Linear baseline with $L_2$ regularization penalty (154 parameters). Provides a computational efficiency lower bound.
- **Random Forest Regressor**: 100-tree ensemble (50,000 parameters) capturing non-linear feature interactions without sequential recurrent memory.

### 3.2 Deep Sequence Models
- **Standard LSTM**: 2-layer Long Short-Term Memory network (91,457 parameters) with input, forget, and output gating mechanisms to model long-term temporal dependencies.
- **Standard GRU**: 2-layer Gated Recurrent Unit network (69,121 parameters) utilizing reset and update gates, achieving faster training and lower inference latency.
- **Temporal Convolutional Network (TCN)**: 3-layer residual causal 1D dilated convolutions with dilation factors $d \in \{1, 2, 4\}$ (220,321 parameters), enabling receptive field coverage without recurrent bottlenecking.
- **Tuned 3-Layer GRU (Optimal Candidate)**: 3 recurrent layers, 64 hidden units, dropout 0.2, and learning rate 0.001 (118,529 parameters). Identified via Day 19 grid search across 1–5 layers and 0.1–0.3 dropout ranges.

---

## 4. Quantitative Benchmark Results

Evaluated on unseen test battery cell **NASA B0018**:

| Model Architecture | Parameters | Train Time (s) | Inference (ms/sample) | MAE | RMSE | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 154 | 0.05 | 0.10 | 0.0942 | 0.1120 | 11.45% | -0.4510 |
| **Random Forest** | 50,000 | 1.85 | 2.40 | 0.0895 | 0.1065 | 10.82% | -0.3840 |
| **Standard LSTM** | 91,457 | 10.69 | 1.20 | 0.0807 | 0.0951 | 9.72% | -0.1933 |
| **Standard GRU** | 69,121 | 9.42 | 1.00 | 0.0846 | 0.1012 | 10.17% | -0.3518 |
| **TCN (Dilated Residual)** | 220,321 | 15.50 | 1.90 | 0.0827 | 0.0984 | 9.86% | -0.2789 |
| **Tuned 3-Layer GRU (Optimal)** | 118,529 | 12.30 | 1.10 | **0.0768** | **0.0892** | **8.84%** | **-0.0842** |

---

## 5. Summary & Deliverables

1. **Robust Zero-Leakage Pipeline**: Clean, reproducible preprocessing and sequence generation across 26 structured daily tasks.
2. **Model Selection Rationale**: The Tuned 3-Layer GRU demonstrated the strongest generalization performance (lowest MAE and RMSE) while maintaining edge-deployable inference latency ($1.10\text{ ms/cycle}$).
3. **Artifacts & Checkpoints**: Packaged deployment checkpoint (`best_gru_tuned.pth`) with serialized preprocessing scalers and comprehensive day-wise evaluation notebooks.
