# Battery Health Monitoring (SOH) & Remaining Useful Life (RUL) Prediction

[![TIH IIT Guwahati](https://img.shields.io/badge/TIH%20IIT%20Guwahati-Internship%20Phase-00529B.svg)](https://tih.iitg.ac.in)
[![Track](https://img.shields.io/badge/Track-Group%20O4-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20PCoE-orange.svg)](https://data.nasa.gov/)

An end-to-end, reproducible deep learning framework for **Battery State of Health (SOH)** estimation and **Remaining Useful Life (RUL)** prognostics on lithium-ion battery degradation cycles.

Developed as part of the **TIH IIT Guwahati 4-Week Online Internship Research Project** (Group O4).

---

## 📌 Project Overview & Objectives

Accurate estimation of battery health and remaining lifetime is critical for ensuring operational safety and avoiding sudden system failure in energy storage applications. This project implements a comprehensive 26-day experimental pipeline using the benchmark NASA Ames Prognostics Center of Excellence (PCoE) battery aging dataset.

### Core Objectives:
1. **State of Health (SOH) Estimation**: Quantify instantaneous usable discharge capacity relative to nominal capacity:
   $$\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$$
2. **Remaining Useful Life (RUL) Forecasting**: Predict the number of operational discharge cycles remaining before capacity drops permanently below the End-of-Life (EOL) failure threshold ($1.40\text{ Ah}$ for B0005/B0006/B0018, $1.50\text{ Ah}$ for B0007):
   $$\text{RUL}_k = \max(0, k_{\text{EOL}} - k)$$
3. **Capacity Regeneration & Knee-Transition Modeling**: Address physical non-linearities including electrochemical capacity rebound spikes during rest intervals and accelerated degradation during the late-life "knee" transition.

---

## 🔬 Experimental Methodology & Pipeline

- **Dataset**: NASA PCoE Lithium-Ion Battery Aging Dataset (`B0005`, `B0006`, `B0007`, `B0018`).
- **Strict Zero-Leakage Evaluation**:
  - **Training Set**: Cells `B0005` & `B0006` (scalers fitted strictly on training data).
  - **Validation Set**: Cell `B0007` (used for hyperparameter selection and checkpoint tuning).
  - **Unseen Test Set**: Cell `B0018` (evaluated zero-shot out-of-sample).
- **Feature Engineering**: Extracted cycle-by-cycle summary features including terminal voltage statistics, thermal elevation ($\Delta T = T_{\text{max}} - T_{\text{start}}$), charge duration, and rolling statistics using a sequence sliding window ($L = 15$).
- **Model Zoo**:
  - **Baselines**: Ridge Regression and Random Forest Regressor.
  - **Sequence Models**: Standard LSTM and Standard GRU networks.
  - **Convolutional Network**: Temporal Convolutional Network (TCN) with dilated causal 1D convolutions (dilation factors $d \in \{1, 2, 4\}$).
  - **Tuned Model**: 3-Layer GRU (64 hidden units, dropout 0.2, learning rate 0.001) selected via comprehensive grid search across 1–5 layers and 0.1–0.3 dropout ranges.

---

## 📊 Quantitative Benchmark Results

Evaluated on unseen test battery cell **NASA B0018**:

| Model Architecture | Parameters | Train Time (s) | Inference Time (ms) | MAE | RMSE | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 154 | 0.05 | 0.10 | 0.0942 | 0.1120 | 11.45% | -0.4510 |
| **Random Forest** | 50,000 | 1.85 | 2.40 | 0.0895 | 0.1065 | 10.82% | -0.3840 |
| **Standard LSTM** | 91,457 | 10.69 | 1.20 | 0.0807 | 0.0951 | 9.72% | -0.1933 |
| **Standard GRU** | 69,121 | 9.42 | 1.00 | 0.0846 | 0.1012 | 10.17% | -0.3518 |
| **TCN (Dilated Residual)** | 220,321 | 15.50 | 1.90 | 0.0827 | 0.0984 | 9.86% | -0.2789 |
| **Tuned 3-Layer GRU (Optimal)** | 118,529 | 12.30 | 1.10 | **0.0768** | **0.0892** | **8.84%** | **-0.0842** |

---

## 📂 Repository Structure

```tree
├── notebooks/                       # Standardized day-by-day deliverables (Day 08 to Day 26)
│   ├── day_wise_notebooks/
│   │   ├── Day08_Data_Quality_Assessment/
│   │   ├── Day09_Temporal_Split_Review/
│   │   ├── Day10_Temporal_EDA/
│   │   ├── Day11_Sequence_Processing/
│   │   ├── Day12_Sequence_Pipeline/
│   │   ├── Day13_Baseline_Prognostics/
│   │   ├── Day14_Feature_Representation_Freeze/
│   │   ├── Day15_LSTM_SOH_Estimation/
│   │   ├── Day16_LSTM_vs_GRU_Comparison/
│   │   ├── Day17_TCN_Dilated_Causal_Convolution/
│   │   ├── Day18_Model_Benchmark_Comparison/
│   │   ├── Day19_Hyperparameter_Optimization/
│   │   ├── Day20_Cross_Cell_and_Stage_Robustness/
│   │   ├── Day21_Candidate_Model_Review/
│   │   ├── Day22_Frozen_Inference_Pipeline/
│   │   ├── Day23_Model_Packaging_and_API/
│   │   ├── Day24_Final_Presentation_Pack/
│   │   ├── Day25_Failure_Conditions_Diagnostics/
│   │   └── Day26_Comprehensive_Failure_Report/
│   └── hierarchical_notebooks/
├── ppt/                             # Milestone presentation slide decks
├── reports/                         # Summary presentations & milestone reports
│   └── Day27_Final_Presentation.md
├── results/                         # Evaluation metrics, figures, and plots
└── src/                             # Core Python data loaders and modeling utilities
```

---

## 👥 Authors & Acknowledgments

- **Intern / Researcher**: Rajnish Singh
- **Mentorship & Review**: Preetodeep Dev, TIH IIT Guwahati
- **Programme**: TIH IIT Guwahati 4-Week Online Internship Programme
