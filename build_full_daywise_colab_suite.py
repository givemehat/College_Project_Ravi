import os
import json
import shutil

colab_base = '/Users/rajnishsingh/Downloads/Google_collab'
battery_rul_nb = '/Users/rajnishsingh/Downloads/BATTERY RUL/notebooks'
iit_g_nb = '/Users/rajnishsingh/Downloads/IIT-G-Project'

# Mapping of days to source file, target subfolder and standardized notebook name
day_configs = [
    {
        "day": "Day08",
        "title": "Battery Data Quality Assessment & Sequence Alignment",
        "subfolder": "Day08_Data_Quality_Assessment",
        "nb_name": "Day08_Rajnish_Data_Quality_Assessment.ipynb",
        "src": "Day8_Rajnish_Cleaning_Alignment.ipynb"
    },
    {
        "day": "Day09",
        "title": "Temporal Split Review & Zero-Leakage Data Strategy",
        "subfolder": "Day09_Temporal_Split_Review",
        "nb_name": "Day09_Rajnish_Temporal_Split_Review.ipynb",
        "src": "day9_Rajnish_Temposhift.ipynb"
    },
    {
        "day": "Day10",
        "title": "Temporal EDA & Degradation Trajectory Visualizations",
        "subfolder": "Day10_Temporal_EDA",
        "nb_name": "Day10_Rajnish_Temporal_EDA.ipynb",
        "src": "day10_Rajnish_TempEDA.ipynb"
    },
    {
        "day": "Day11",
        "title": "Sequence Processing & Health Indicator Representations",
        "subfolder": "Day11_Sequence_Processing",
        "nb_name": "Day11_Rajnish_Sequence_Processing.ipynb",
        "src": "day11_Rajnish_Sequence_Processing.ipynb"
    },
    {
        "day": "Day12",
        "title": "Temporal Sliding Window Pipeline & PyTorch DataLoaders",
        "subfolder": "Day12_Sequence_Pipeline",
        "nb_name": "Day12_Rajnish_Sequence_Pipeline.ipynb",
        "src": "Day12_Rajnish_SequencePipeline.ipynb"
    },
    {
        "day": "Day13",
        "title": "Baseline Empirical Double-Exponential & Random Forest Prognostics",
        "subfolder": "Day13_Baseline_Prognostics",
        "nb_name": "Day13_Rajnish_Baseline_Prognostics.ipynb",
        "src": "Day13_Rajnish_BaselineProganstic.ipynb"
    },
    {
        "day": "Day14",
        "title": "Feature Representation Freeze & Multi-Feature Correlation",
        "subfolder": "Day14_Feature_Representation_Freeze",
        "nb_name": "Day14_Rajnish_Feature_Representation_Freeze.ipynb",
        "src": "Day14_Rajnish_KeyChange_New.ipynb"
    },
    {
        "day": "Day15",
        "title": "Multi-Layer LSTM Architecture for SOH & RUL Forecasting",
        "subfolder": "Day15_LSTM_SOH_Estimation",
        "nb_name": "Day15_Rajnish_LSTM_SOH_Estimation.ipynb",
        "src": "Day15_Rajnish_LSTM_SOH.ipynb"
    },
    {
        "day": "Day16",
        "title": "LSTM vs GRU Comparative Recurrent Modeling",
        "subfolder": "Day16_LSTM_vs_GRU_Comparison",
        "nb_name": "Day16_Rajnish_LSTM_vs_GRU_Comparison.ipynb",
        "src": "Day16_RajnishLSTM_GRU.ipynb"
    },
    {
        "day": "Day17",
        "title": "Temporal Convolutional Network (TCN with Dilated Causal 1D Convolutions)",
        "subfolder": "Day17_TCN_Dilated_Causal_Convolution",
        "nb_name": "Day17_Rajnish_TCN_SOH_RUL.ipynb",
        "src": "Day17_Rajnish_TCN_SOH_nw.ipynb"
    },
    {
        "day": "Day18",
        "title": "Multi-Model Comparative Benchmarking on Unseen Test Battery B0018",
        "subfolder": "Day18_Model_Benchmark_Comparison",
        "nb_name": "Day18_Rajnish_Model_Benchmark_Comparison.ipynb",
        "src": "Day18_Rajnish_ModelComparison_updated.ipynb"
    },
    {
        "day": "Day19",
        "title": "Hyperparameter Optimization, AdamW & Multi-Task Loss Tuning",
        "subfolder": "Day19_Hyperparameter_Optimization",
        "nb_name": "Day19_Rajnish_Hyperparameter_Optimization.ipynb",
        "src": "Day19_Rajnish_FineTuning_pynb.ipynb"
    },
    {
        "day": "Day20",
        "title": "Cross-Cell & Degradation-Stage Robustness Analysis",
        "subfolder": "Day20_Cross_Cell_and_Stage_Robustness",
        "nb_name": "Day20_Rajnish_Cross_Cell_and_Stage_Robustness.ipynb",
        "src": "Day20_Rajnish_Robustness.ipynb"
    },
    {
        "day": "Day21",
        "title": "Candidate Model Selection & Mentor Validation Review",
        "subfolder": "Day21_Candidate_Model_Review",
        "nb_name": "Day21_Rajnish_Candidate_Model_Review.ipynb",
        "src": "Day21_Candidate_Review.ipynb"
    },
    {
        "day": "Day22",
        "title": "Frozen Inference Pipeline & Edge Deployment Latency Benchmark",
        "subfolder": "Day22_Frozen_Inference_Pipeline",
        "nb_name": "Day22_Rajnish_Frozen_Inference_Pipeline.ipynb",
        "src": "Day22_Frozen_Pipeline.ipynb"
    },
    {
        "day": "Day24",
        "title": "Final Presentation Pack & Visual Synthesis",
        "subfolder": "Day24_Final_Presentation_Pack",
        "nb_name": "Day24_Rajnish_Final_Presentation_Pack.ipynb",
        "src": "Day24_Final_Presentation_Pack.ipynb"
    },
    {
        "day": "Day25",
        "title": "Failure Case Diagnostics & Worst-Case Error Analysis",
        "subfolder": "Day25_Failure_Conditions_Diagnostics",
        "nb_name": "Day25_Rajnish_Failure_Conditions_Diagnostics.ipynb",
        "src": "Day25_Rajnish_Failure_Conditions_Report.ipynb"
    },
    {
        "day": "Day26",
        "title": "Comprehensive Research Summary & Final AUV Operational Report",
        "subfolder": "Day26_Comprehensive_Failure_Report",
        "nb_name": "Day26_Rajnish_Comprehensive_Failure_Report.ipynb",
        "src": "Day26_Rajnish_Failure_Conditions_Report.ipynb"
    }
]

# 1. Create individual day subfolders and place clean day notebooks
print("--- 1. Creating Day-wise Subfolders and Notebooks ---")
for cfg in day_configs:
    src_path = os.path.join(colab_base, cfg['src'])
    if not os.path.exists(src_path):
        # check if in subfolders
        for root, dirs, files in os.walk(colab_base):
            if cfg['src'] in files:
                src_path = os.path.join(root, cfg['src'])
                break
                
    if os.path.exists(src_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        # Target directories
        targets = [
            os.path.join(colab_base, 'day_wise_notebooks', cfg['subfolder']),
            os.path.join(battery_rul_nb, 'day_wise_notebooks', cfg['subfolder']),
            os.path.join(iit_g_nb, 'day_wise_notebooks', cfg['subfolder'])
        ]
        
        for t_dir in targets:
            os.makedirs(t_dir, exist_ok=True)
            t_file = os.path.join(t_dir, cfg['nb_name'])
            with open(t_file, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
                
        print(f"Created {cfg['day']} in {cfg['subfolder']}/{cfg['nb_name']}")

# 2. Build the Comprehensive Master Notebook containing all combined day subsections
print("\n--- 2. Building Comprehensive Master Notebook (Day 08 to Day 26) ---")

master_cells = []

# Master Header
master_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Master Deep Learning Battery Health Monitoring & RUL Prediction Pipeline\n",
        "### TIH, IIT Guwahati — 4-Week Online Internship Research Project (Group O4)\n",
        "**Project Title**: Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles (AUVs)\n",
        "**Author / Intern**: Rajnish Singh\n",
        "**Architecture**: Complete End-to-End Execution Flow from **Day 08** to **Day 26**\n",
        "**Dataset**: NASA Ames Prognostics Center of Excellence (PCoE) Battery Aging Dataset (`cleaned_dataset`)\n",
        "\n",
        "---\n",
        "## 📑 Table of Contents\n",
        "- [⚙️ Global Imports & Hardware Detection](#setup)\n",
        "- [📅 Day 08: Battery Data Quality Assessment & Sequence Alignment](#day-08)\n",
        "- [📅 Day 09: Temporal Split Review & Zero-Leakage Data Strategy](#day-09)\n",
        "- [📅 Day 10: Temporal EDA & Degradation Trajectory Visualizations](#day-10)\n",
        "- [📅 Day 11: Sequence Processing & Health Indicator Representations](#day-11)\n",
        "- [📅 Day 12: Temporal Sliding Window Pipeline & PyTorch DataLoaders](#day-12)\n",
        "- [📅 Day 13: Baseline Empirical Double-Exponential & Random Forest Prognostics](#day-13)\n",
        "- [📅 Day 14: Feature Representation Freeze & Multi-Feature Correlation](#day-14)\n",
        "- [📅 Day 15: Multi-Layer LSTM Architecture for SOH & RUL Forecasting](#day-15)\n",
        "- [📅 Day 16: LSTM vs GRU Comparative Recurrent Modeling](#day-16)\n",
        "- [📅 Day 17: Temporal Convolutional Network (TCN with Dilated Causal 1D Convolutions)](#day-17)\n",
        "- [📅 Day 18: Multi-Model Comparative Benchmarking on Unseen Test Battery B0018](#day-18)\n",
        "- [📅 Day 19: Hyperparameter Optimization, AdamW & Multi-Task Loss Tuning](#day-19)\n",
        "- [📅 Day 20: Cross-Cell & Degradation-Stage Robustness Analysis](#day-20)\n",
        "- [📅 Day 21: Candidate Model Selection & Mentor Validation Review](#day-21)\n",
        "- [📅 Day 22: Frozen Inference Pipeline & Edge Deployment Latency Benchmark](#day-22)\n",
        "- [📅 Day 24: Final Presentation Pack & Visual Synthesis](#day-24)\n",
        "- [📅 Day 25: Failure Case Diagnostics & Worst-Case Error Analysis](#day-25)\n",
        "- [📅 Day 26: Comprehensive Research Summary & Final AUV Operational Report](#day-26)\n"
    ]
})

# Setup Cell
master_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "<a id=\"setup\"></a>\n",
        "## ⚙️ Global Environment Setup, Imports & Hardware Detection\n",
        "Configure Python environment, random seeds, PyTorch compute device (GPU / Apple Silicon MPS / CPU), and locate `cleaned_dataset`."
    ]
})

master_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import sys\n",
        "import time\n",
        "import random\n",
        "import copy\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from tabulate import tabulate\n",
        "\n",
        "from scipy.optimize import curve_fit\n",
        "from scipy.integrate import trapezoid as trapz_fn\n",
        "from scipy.signal import savgol_filter\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "\n",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
        "plt.rcParams['font.size'] = 11\n",
        "plt.rcParams['figure.dpi'] = 120\n",
        "\n",
        "def set_seed(seed=42):\n",
        "    random.seed(seed)\n",
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
        "    np.random.seed(seed)\n",
        "    torch.manual_seed(seed)\n",
        "    if torch.cuda.is_available():\n",
        "        torch.cuda.manual_seed_all(seed)\n",
        "    elif torch.backends.mps.is_available():\n",
        "        torch.mps.manual_seed(seed)\n",
        "\n",
        "set_seed(42)\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))\n",
        "print(f'Using Compute Device: {device}')\n",
        "\n",
        "def locate_dataset_dir():\n",
        "    candidates = [\n",
        "        'cleaned_dataset',\n",
        "        '../cleaned_dataset',\n",
        "        'data/cleaned_dataset',\n",
        "        '/Users/rajnishsingh/Downloads/IIT-G-Project/cleaned_dataset',\n",
        "        '/Users/rajnishsingh/Downloads/BATTERY RUL/data/cleaned_dataset'\n",
        "    ]\n",
        "    for c in candidates:\n",
        "        if os.path.exists(os.path.join(c, 'metadata.csv')):\n",
        "            return c\n",
        "    raise FileNotFoundError('cleaned_dataset directory with metadata.csv not found.')\n",
        "\n",
        "dataset_dir = locate_dataset_dir()\n",
        "print(f'Located cleaned_dataset directory at: {dataset_dir}')\n"
    ]
})

# Add individual days sequentially
for cfg in day_configs:
    src_path = os.path.join(colab_base, cfg['src'])
    if not os.path.exists(src_path):
        for root, dirs, files in os.walk(colab_base):
            if cfg['src'] in files:
                src_path = os.path.join(root, cfg['src'])
                break
                
    if os.path.exists(src_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            nb_day = json.load(f)
            
        day_anchor = cfg['day'].lower().replace('day', 'day-')
        master_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"---\n",
                f"<a id=\"{day_anchor}\"></a>\n",
                f"# 📅 {cfg['day']}: {cfg['title']}\n",
                f"**Milestone**: {cfg['day']} Deliverable (TIH IIT Guwahati 4-Week Workflow)\n"
            ]
        })
        
        for cell in nb_day.get('cells', []):
            master_cells.append(cell)

print(f"Total cells compiled in Master Notebook: {len(master_cells)}")

master_nb = {
    "cells": master_cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.10.0"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Save in all key paths
master_destinations = [
    os.path.join(colab_base, 'Master_Battery_Health_RUL_Day08_to_Day26.ipynb'),
    os.path.join(battery_rul_nb, 'Master_Battery_Health_RUL_Day08_to_Day26.ipynb'),
    os.path.join(iit_g_nb, 'Master_Battery_Health_RUL_Day08_to_Day26.ipynb')
]

for dest in master_destinations:
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(master_nb, f, indent=2)
    print(f"Saved compiled master notebook to: {dest}")

print("\n--- Day-wise Notebook Suite & Master Pipeline Built Successfully! ---")
