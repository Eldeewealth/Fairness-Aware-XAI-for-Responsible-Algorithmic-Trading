# Fairness-Aware Explainable AI for Responsible Algorithmic Trading

Repository accompanying the Springer *Digital Finance* submission:

**“Fairness-Aware Explainable AI for Responsible Algorithmic Trading”**

---

## Project Overview

This repository provides an end-to-end experimental pipeline for **fairness-aware binary classification of next-day S&P 500 direction**, combining:

- Predictive modeling  
- Fairness mitigation  
- Explainability  
- Calibration assessment  
- Strategy-level backtesting  

The implementation includes:

- **LightGBM** baseline classifier  
- **Fairlearn Exponentiated Gradient** with **Demographic Parity**  
- Group-level fairness analysis across volatility regimes  
- **SHAP**-based feature attribution (baseline vs. fairness-mitigated)  
- **Calibration** diagnostics (reliability curve, Brier score)  
- Trading simulation & cumulative return comparison  
- **MLflow** experiment tracking  

---

## Research Contribution

This work operationalizes **responsible AI** for financial prediction by jointly evaluating:

- Predictive utility  
- Group fairness under volatility-regime stratification  
- Model interpretability (SHAP)  
- Probability calibration  
- Economic relevance via backtesting  

The contribution is **methodological and empirical**: fairness constraints are applied in a **time-ordered financial context**, and results are interpreted statistically and economically.

---

## Methodology Overview

1. Acquire S&P 500 historical data (`^GSPC`) via `yfinance`.  
2. Clean & validate data (missingness, duplicates, ordering).  
3. Construct target variable **`next_day_up`** and feature set.  
4. Define sensitive attribute **`group_volatility`** (low/med/high).  
5. Apply **time-ordered** train/test split (no shuffling).  
6. Train baseline **LightGBM** model.  
7. Train fairness‑mitigated model using **Exponentiated Gradient + Demographic Parity**.  
8. Evaluate overall & group-level metrics.  
9. Compute **SHAP** explanations for both models.  
10. Assess calibration (reliability curve + scoring rules).  
11. Run trading simulation & compare returns.  
12. Log artifacts & metrics using **MLflow**.  

---

## Repository Structure
.
├─ data/                               # Raw downloaded data (created by datafile.py)
├─ artifacts/                          # Figures, fairness tables, SHAP outputs
├─ mlruns/                             # MLflow tracking outputs
│
├─ datafile.py                         # Download S&P 500 (^GSPC) via yfinance → data/sp500_data.csv
├─ clean_sp500.py                      # Clean & export to sp500_clean.csv
│
├─ fairness_check.ipynb                # Main end-to-end notebook
├─ fairness_check.html                 # HTML export of notebook
│
├─ sp500_clean.csv                     # Cleaned dataset for modeling
│
├─ calibration_baseline.png
├─ figure8_calibration.png
├─ cm_baseline.png
├─ cm_fair.png
├─ pr_baseline.png
├─ fig3_next_day_up_diagram.png
├─ selection_rate_stability_over_time.png
│
├─ requirements.txt                    # Core pinned dependencies
├─ requirements-notebooks.txt          # Optional: Jupyter dependencies
│
├─ LICENSE                             # Apache-2.0 license
├─ NOTICE                              # Apache-2.0 NOTICE
└─ README.md

---

## Installation Instructions

**Tested on Python 3.13.3 (Python 3.10+ recommended)**

```bash
# 1) Create and activate a virtual environment
python -m venv .venv

# Windows
. .venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 2) Install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Optional: install notebook dependencies
pip install -r requirements-notebooks.txt


Data Source (S&P 500 via yfinance)

Instrument: S&P 500 index (^GSPC)
Provider: yfinance
Typical fields: OHLCV + timestamp
Download script: datafile.py
Cleaned table: sp500_clean.csv

Note: Market data is not redistributed. Re-download using the provided script for full reproducibility.

Reproducibility Instructions
Step-by-step pipeline
Shell# Step 1: Download raw datapython datafile.py# Step 2: Clean and validate datapython clean_sp500.py# Step 3: Run the notebookjupyter notebook fairness_check.ipynbShow more lines
Optional: non-interactive execution
Shelljupyter nbconvert --to notebook --execute fairness_check.ipynb \  --output fairness_check.executed.ipynb --ExecutePreprocessor.timeout=0Show more lines

Reproducibility Workflow Diagram
yfinance (^GSPC)
   → raw CSV (data/sp500_data.csv)
   → cleaning/validation (clean_sp500.py)
   → modeling table (sp500_clean.csv)
   → feature + target construction (next_day_up)
   → volatility group assignment (group_volatility)
   → time-ordered train/test split
   → baseline LightGBM + fairness-constrained model
   → evaluation (utility, DP gap, fairness metrics, SHAP, calibration)
   → trading signal generation + backtest
   → MLflow artifacts + saved figures/tables


Fairness Framework
Fairness mitigation uses:

fairlearn.reductions.ExponentiatedGradient
Constraint: DemographicParity

Sensitive groups: volatility regimes (group_volatility)
We report:

Selection rate by group
Accuracy by group
Demographic Parity gap
Utility–fairness trade-offs

This supports transparent analysis under heterogeneous market conditions.

Explainability & Evaluation
Explainability

SHAP (TreeExplainer)
Global feature importance
Dependence plots
Comparison of baseline vs. fairness-mitigated explanations

Evaluation

Utility: accuracy, balanced accuracy, confusion matrices
Fairness: selection rates, DP gap
Calibration: reliability curve, Brier score
Economic: growth curves (backtesting)


Results Summary
Fairness‑constrained learning reduces group-level selection disparity versus the baseline, with expected utility trade-offs. Effects are examined across predictive accuracy, calibration behavior, and economic backtesting impacts.

Code Availability
A versioned, archived snapshot of this repository will be available on Zenodo after publishing the GitHub release.
DOI: https://doi.org/YOUR-DOI-HERE
(Replace once Zenodo generates your DOI.)

License
This project is licensed under the Apache License 2.0.
See LICENSE and NOTICE (required for downstream redistribution).

Citation
BibTeX@article{2026_fair_xai_trading,  title   = {Fairness-Aware Explainable AI for Responsible Algorithmic Trading},  author  = {Loveday Okwudiri Okoro and Anchal Garg and Evans Onwe},  journal = {Digital Finance},  year    = {2026},  note    = {Under review / accepted version pending},  doi     = {YOUR-DOI-HERE}}Show more lines

Contact
For correspondence or reproducibility issues:

Loveday Okwudiri Okoro
Email: lovedayo@acm.org
Use GitHub Issues for technical discussion