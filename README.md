Fairness-Aware Explainable AI for Responsible Algorithmic Trading

Repository accompanying the Springer Digital Finance submission:
“Fairness-Aware Explainable AI for Responsible Algorithmic Trading”

Project Overview

This repository provides an end-to-end experimental pipeline for fairness-aware binary classification of next-day S&P 500 direction, combining:

Predictive modeling

Fairness mitigation

Explainability

Calibration assessment

Strategy-level backtesting

The implementation includes:

LightGBM baseline classifier (XGBoost optional)

Fairness mitigation using Exponentiated Gradient with Demographic Parity (Fairlearn)

Group-level fairness analysis across volatility regimes

SHAP-based feature attribution (baseline vs. fairness-mitigated model)

Calibration diagnostics (reliability curve, Brier score)

Trading simulation and cumulative return comparison

MLflow experiment tracking

Research Contribution

This work operationalizes responsible AI for financial prediction by jointly evaluating:

Predictive utility

Group fairness under volatility-regime stratification

Model interpretability (SHAP)

Probability calibration (reliability + scoring)

Economic relevance via simple backtesting

The contribution is methodological and empirical: fairness constraints are tested in a time-ordered market setting; outcomes are interpreted statistically and economically.

Methodology Overview

Acquire S&P 500 historical data (^GSPC) via yfinance.

Clean and validate data (missingness, duplicates, chronological order).

Construct target variable next_day_up and feature set.

Define sensitive attribute group_volatility (low/med/high).

Apply time-ordered train/test split (no random shuffling).

Train baseline LightGBM model.

Train fairness-mitigated model using Exponentiated Gradient with Demographic Parity.

Evaluate overall and group-level metrics.

Compute SHAP explanations (baseline vs. fair).

Assess calibration (reliability curve + Brier score).

Backtest signals and compare cumulative performance.

Log artifacts and metrics (MLflow).

Repository Structure
.
├─ data/                               # Raw downloaded data (created by datafile.py)
├─ artifacts/                          # Figures, fairness tables, SHAP outputs
├─ mlruns/                             # MLflow experiment tracking
│
├─ datafile.py                         # Download ^GSPC via yfinance → data/sp500_data.csv
├─ clean_sp500.py                      # Clean and export to sp500_clean.csv
├─ fairness_check.ipynb                # End-to-end analysis notebook
├─ fairness_check.html                 # HTML export of the notebook
├─ sp500_clean.csv                     # Cleaned modeling table
│
├─ calibration_baseline.png
├─ figure8_calibration.png
├─ cm_baseline.png
├─ cm_fair.png
├─ pr_baseline.png
├─ fig3_next_day_up_diagram.png
├─ selection_rate_stability_over_time.png
│
├─ requirements.txt                    # Core, pinned dependencies
├─ requirements-notebooks.txt          # Optional: Jupyter extras (if running notebooks)
├─ LICENSE                             # Apache-2.0
├─ NOTICE                              # Apache-2.0 NOTICE
└─ README.md
Installation Instructions

Tested on Python 3.13.3 (Python 3.10+ recommended)

1) Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
. .venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
2) Install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
3) (Optional) Install notebook dependencies
pip install -r requirements-notebooks.txt
Data Source (S&P 500 via yfinance)

Instrument: S&P 500 index (^GSPC)

Interface: yfinance

Typical fields: Open, High, Low, Close, Volume, timestamp

Download script: datafile.py

Cleaned table used in experiments: sp500_clean.csv

Note: Market data is subject to provider terms; raw data is not redistributed here.
Re-download using the provided script for full reproducibility.

Reproducibility Instructions
Step-by-step pipeline
# Step 1: Download raw data → data/sp500_data.csv
python datafile.py

# Step 2: Clean and validate → sp500_clean.csv
python clean_sp500.py

# Step 3: Run the notebook
jupyter notebook fairness_check.ipynb
Optional: non-interactive execution
jupyter nbconvert --to notebook --execute fairness_check.ipynb \
  --output fairness_check.executed.ipynb \
  --ExecutePreprocessor.timeout=0
Workflow diagram (text)
yfinance (^GSPC)
  → data/sp500_data.csv
  → clean_sp500.py
  → sp500_clean.csv
  → feature + target construction (next_day_up)
  → volatility grouping (group_volatility)
  → time-ordered train/test split
  → baseline LightGBM + fairness-constrained model
  → evaluation (utility, DP gap, SHAP, calibration)
  → trading signals + backtest
  → artifacts/mlruns + saved figures
Fairness Framework

Fairness mitigation is implemented via:

fairlearn.reductions.ExponentiatedGradient

Constraint: DemographicParity

Sensitive groups are based on volatility regimes (group_volatility).

We report:

Overall utility (accuracy, balanced accuracy, PR/ROC where appropriate)

Group-disaggregated metrics (e.g., selection rate, accuracy)

Demographic parity gap before/after mitigation

This setup supports transparent inspection of utility–fairness trade-offs under market heterogeneity.

Explainability & Evaluation
Explainability

SHAP (TreeExplainer) for baseline and fair models

Global importance and dependence plots

Compare attribution shifts after fairness constraints

Evaluation

Utility: accuracy, balanced accuracy, confusion matrices

Fairness: selection rates + DP gap

Calibration: reliability curve, Brier score (ECE-style optional)

Economic: strategy backtest equity curves and stability

Results Summary

Fairness-constrained learning reduces group-level selection-rate disparity relative to the unconstrained baseline, with expected trade-offs in utility.
We document effects across predictive, calibration, and backtesting perspectives.
Full quantitative results appear in the manuscript tables and figures.

Code Availability

A versioned, archived snapshot of this repository will be available on Zenodo after publishing the GitHub release.

DOI:

https://doi.org/YOUR-DOI-HERE

(Replace after you archive Release v1.0.0 to Zenodo and obtain the DOI.)

Optional badge (after DOI):

[![DOI](https://zenodo.org/badge/DOI/YOUR-DOI-HERE.svg)](https://doi.org/YOUR-DOI-HERE)
License

This project is released under the Apache License 2.0.
See the included LICENSE and NOTICE files.
Downstream redistributions should preserve the NOTICE file as required by Apache-2.0.

Citation
@article{2026_fair_xai_trading,
  title   = {Fairness-Aware Explainable AI for Responsible Algorithmic Trading},
  author  = {Loveday Okwudiri Okoro and Anchal Garg and Evans Onwe},
  journal = {Digital Finance},
  year    = {2026},
  note    = {Under review / accepted version pending},
  doi     = {YOUR-DOI-HERE}
}
Contact

For correspondence, reproducibility queries, or reporting issues:

Corresponding author: Loveday Okwudiri Okoro

Email: lovedayo@acm.org

Use GitHub Issues for technical discussion and traceability