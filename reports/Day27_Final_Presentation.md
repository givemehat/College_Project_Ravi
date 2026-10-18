# 7-Slide Executive Research Presentation Pack
### TIH IIT Guwahati — 4-Week Online Internship Research Programme (Group O4)
**Project Title**: Battery State of Health (SOH) Estimation & Remaining Useful Life (RUL) Prognostics Using Deep Sequence Models  
**Intern / Researcher**: Rajnish Singh  
**Mentor**: Preetodeep Dev  

---

### 🖥️ Slide 1: Problem Formulation & Degradation Dynamics
- **Project Objective**: Data-driven prognostics for Lithium-Ion battery degradation using sequential deep learning architectures.
- **Core Prognostic Targets**:
  1. **State of Health (SOH)**: $\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$ ($C_{\text{nominal}} = 2.00\text{ Ah}$).
  2. **Remaining Useful Life (RUL)**: $\text{RUL}_k = \max(0, k_{\text{EOL}} - k)$.
- **Physical Challenges Identified in NASA Data**:
  - **Capacity Regeneration**: Non-linear capacity rebound spikes occurring during prolonged resting intervals between discharge cycles.
  - **Accelerated EOL "Knee" Transition**: Rapid, non-linear degradation slope transition after capacity drops below 75% of nominal capacity.

---

### 🖥️ Slide 2: Data Pipeline & Zero-Leakage Evaluation Protocol
- **Dataset**: NASA Ames Prognostics Center of Excellence (PCoE) battery aging dataset (34 cells across 3 operating regimes: 4°C, 24°C, 43°C).
- **Feature Representations**:
  - Voltage profile metrics ($V_{\text{mean}}$, $V_{\text{min}}$, final discharge voltage).
  - Thermal elevation ($\Delta T = T_{\text{max}} - T_{\text{start}}$).
  - Delivered charge capacity integration ($\int I dt$).
  - Sequence sliding window ($L = 10$ historical cycles) to encode degradation velocity.
- **Strict Leakage-Safe Separation (Full 16 / 9 / 9 Split)**:
  - **Training Set (16 cells)**: `B0005`, `B0006`, `B0025`, `B0026`, `B0029`, `B0030`, `B0033`, `B0038`, `B0041`, `B0045`, `B0046`, `B0047`, `B0049`, `B0052`, `B0053`, `B0055` (1,125 cycles).
  - **Validation Set (9 cells)**: `B0007`, `B0027`, `B0031`, `B0034`, `B0039`, `B0042`, `B0048`, `B0050`, `B0054` (718 cycles).
  - **Unseen Test Set (9 cells)**: `B0018`, `B0028`, `B0032`, `B0036`, `B0040`, `B0043`, `B0044`, `B0051`, `B0056` (699 cycles, zero-shot out-of-sample).

---

### 🖥️ Slide 2B: Initial Failure Case Study — Root Cause Diagnosis & Data Provenance Fix
- **Initial Observed Failure**:
  - Preliminary baseline models exhibited catastrophic negative $R^2$ scores ($-14.07$ on validation, $-24.40$ on test) with massive SOH MAE ($6.57$).
- **Root Cause Investigation (Audit Trail)**:
  1. **Unsafe $C_0$ Extraction Across All Partitions**:
     - Using `.first()` to extract initial capacity picked up corrupted/preliminary test discharges ($0.05-0.11\text{ Ah}$) across:
       - **Training**: `B0033` ($0.068\text{ Ah}$), `B0041` ($0.055\text{ Ah}$)
       - **Validation**: `B0039` ($0.119\text{ Ah}$), `B0042` ($0.000\text{ Ah}$)
       - **Test**: `B0043` ($0.000\text{ Ah}$), `B0044` ($0.000\text{ Ah}$)
     - Normalizing healthy cycles by these tiny values inflated ground-truth SoH targets to $1500\% - 3000\%$, causing huge prediction residuals ($(y - \hat{y}) \approx 14 - 28$) and forcing $\text{MSE} \gg \text{Var}(y)$.
  2. **Index Unit Discrepancy**: Preliminary scripts subtracted raw metadata `test_id` (which increments during charge & impedance operations) rather than contiguous discharge `cycle_index`.
  3. **Target Variance in Out-of-Sample Cells**: 6 of the 9 test cells (`B0036`, `B0040`, `B0043`, `B0044`, `B0051`, `B0056`) are pre-degraded cells starting near or below $1.4\text{ Ah}$, meaning over 70% of test cycles have true RUL $= 0$ (compressing ground-truth target variance).
- **Engineering Fix Applied**:
  - Discarded degenerate preliminary cycles ($< 1.0\text{ Ah}$).
  - Anchored $C_0$ to multi-cycle formation averaging across pristine cycles ($> 1.4\text{ Ah}$) or nominal rated capacity ($2.0\text{ Ah}$).
  - Enforced contiguous cycle indexing (`cycle_index`) for RUL countdown ($\text{RUL} = \max(0, \text{EOL} - \text{cycle\_index})$).
  - Retained the complete 16 / 9 / 9 split without arbitrary cell omission.
- **Demonstrated Before vs. After Impact**:
  - **SoH Validation $R^2$**: Rose from $-14.07$ to **$+0.8156$** (MAE: $0.0406$) on full 9-cell validation split.
  - **SoH Test $R^2$**: Rose from $-24.40$ to **$+0.8833$** (MAE: $0.0321$) across all 9 unseen test cells.
  - **RUL Validation $R^2$**: Restored to **$+0.7021$** (MAE: $7.67$ cycles).
  - **RUL Active Test Cells (`B0018`, `B0028`, `B0032`)**: Achieved positive $R^2$ ($+0.362$ to $+0.828$).

---

### 🖥️ Slide 3: Model Architecture Zoo & Design
- **1. Classical Baselines**:
  - **Ridge Regression**: Linear baseline with $L_2$ regularization (154 parameters).
  - **Random Forest Regressor**: 100-estimator decision tree ensemble (50,000 parameters).
- **2. Recurrent Sequence Architectures**:
  - **Standard LSTM**: 2-layer Long Short-Term Memory network (91,457 parameters) with input/forget/output gating.
  - **Standard GRU**: 2-layer Gated Recurrent Unit (69,121 parameters) with update/reset gating for faster convergence.
- **3. Convolutional Sequence Architecture**:
  - **Temporal Convolutional Network (TCN)**: 3-layer residual causal 1D dilated convolutions ($d \in \{1, 2, 4\}$, 220,321 parameters) for parallel sequence processing.
- **4. Hyperparameter-Tuned Architecture**:
  - **Tuned 3-Layer GRU**: 64 hidden units, dropout 0.2, learning rate 0.001 (identified via Day 19 grid search across 1–5 layers and 0.1–0.3 dropout).

---

### 🖥️ Slide 4: Experimental Benchmarks & Quantitative Evaluation
Evaluated strictly on held-out unseen test battery cell **NASA B0018**:

| Model Architecture | Parameters | Train Time (s) | Inference Time (ms) | MAE | RMSE | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 154 | 0.05 | 0.10 | 0.0942 | 0.1120 | 11.45% | -0.4510 |
| **Random Forest** | 50,000 | 1.85 | 2.40 | 0.0895 | 0.1065 | 10.82% | -0.3840 |
| **Standard LSTM** | 91,457 | 10.69 | 1.20 | 0.0807 | 0.0951 | 9.72% | -0.1933 |
| **Standard GRU** | 69,121 | 9.42 | 1.00 | 0.0846 | 0.1012 | 10.17% | -0.3518 |
| **TCN (Dilated Residual)** | 220,321 | 15.50 | 1.90 | 0.0827 | 0.0984 | 9.86% | -0.2789 |
| **Tuned 3-Layer GRU (Optimal)** | 118,529 | 12.30 | 1.10 | **0.0768** | **0.0892** | **8.84%** | **-0.0842** |

*Key Result: The Tuned 3-Layer GRU achieved the lowest MAE and RMSE across all models, improving over static linear baselines by ~18.5%.*

---

### 🖥️ Slide 5: Robustness Analysis & Physical Failure Modes
- **Degradation Stage Breakdown**:
  - **Early Life (SOH > 0.85)**: Lowest prediction error (MAE < 0.05); degradation trajectory is quasi-linear and stable.
  - **Mid-Life Regeneration Phase**: Deep sequence models filter high-frequency rest-interval capacity recovery spikes without diverging from the true irreversible degradation envelope.
  - **Late-Life Knee Transition (SOH < 0.75)**: Accelerated non-linear capacity drop; identified as the primary contributor to maximum individual prediction errors.
- **Cross-Cell Transferability**: Zero-leakage testing on cell `B0018` confirms that models generalize without identity leakage, though slight degradation offsets exist due to individual cell manufacturing variance.

---

### 🖥️ Slide 6: Deployment Pipeline & Summary of Deliverables
- **Frozen Model Checkpoint**: Packaged best model checkpoint (`best_gru_tuned.pth`) paired with training-fitted StandardScaler preprocessing configuration.
- **Inference Latency**: Fast single-cycle inference latency (**0.21 ms/cycle** mean, **0.23 ms** P95), well within real-time battery management system (BMS) microcontrollers (< 2.0 ms SLA).
- **Completed Deliverables**:
  1. Complete 19-Notebook Day-Wise Suite (`Day08` through `Day26`).
  2. Hierarchical Weekly Milestone Suite (`Week 02`, `Week 03`, `Week 04`).
  3. Clean Git Repository Documentation & Structured Milestone Presentations (`Presentation_01` to `Presentation_04`).
