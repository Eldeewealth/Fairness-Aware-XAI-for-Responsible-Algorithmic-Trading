# Fairness-Aware Explainable AI for Responsible Algorithmic Trading

Repository accompanying the Springer *Digital Finance* submission:

**“Fairness-Aware Explainable AI for Responsible Algorithmic Trading”**

## Project Overview
This repository provides an end-to-end experimental pipeline for fairness-aware binary classification of next-day S&P 500 direction, combining predictive modeling, fairness mitigation, explainability, calibration assessment, and strategy-level backtesting.

The implementation includes:
- LightGBM baseline classifier
- Fairness mitigation using Exponentiated Gradient with Demographic Parity
- Group-level fairness analysis across volatility regimes
- SHAP-based feature attribution analysis
- Calibration diagnostics
- Trading simulation and cumulative return comparison

## Research Contribution
This work operationalizes responsible AI for financial prediction by jointly evaluating:
- Predictive utility
- Group fairness under market regime stratification
- Model interpretability
- Probability calibration
- Economic relevance via trading simulation

The contribution is methodological and empirical: fairness constraints are tested within a realistic time-ordered market setting, and outcomes are interpreted both statistically and economically.

## Methodology Overview
1. Acquire S&P 500 historical data (`^GSPC`) via `yfinance`.
2. Clean and validate data (missingness, duplicates, chronological ordering).
3. Construct target variable `next_day_up` and feature set.
4. Define sensitive attribute `group_volatility` (low/med/high regime groups).
5. Apply time-ordered train/test split (no random shuffling).
6. Train baseline LightGBM model.
7. Train fairness-mitigated model using Exponentiated Gradient with Demographic Parity constraint.
8. Evaluate overall and group-level metrics.
9. Compute SHAP explanations for baseline and fair models.
10. Assess calibration (curve and proper scoring rules).
11. Run trading simulation and compare cumulative performance.
12. Log artifacts and metrics (MLflow).

## Repository Structure
```text
.
├─ datafile.py                     # Download S&P 500 data (yfinance) to data/sp500_data.csv
├─ clean_sp500.py                  # Data cleaning and export to sp500_clean.csv
├─ fairness_check.ipynb            # Main end-to-end analysis notebook
├─ data/
│  └─ sp500_data.csv               # Raw downloaded data
├─ sp500_clean.csv                 # Cleaned dataset used in modeling
├─ artifacts/                      # Saved figures and fairness metric tables
├─ mlruns/                         # MLflow tracking outputs
└─ *.png / *.html                  # Exported plots and report artifacts
```

## Installation Instructions
Python version: **3.10+**

```bash
# 1) Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm fairlearn shap yfinance mlflow jupyter
```

## Dependencies (Python Libraries)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `fairlearn`
- `shap`
- `yfinance`
- `mlflow`
- `jupyter`

## Data Source Description (S&P 500 via yfinance)
- Instrument: S&P 500 index (`^GSPC`)
- Provider interface: `yfinance`
- Typical fields: Open, High, Low, Close, Volume, Date index
- Download script: `datafile.py`
- Cleaned output used in experiments: `sp500_clean.csv`

Data are publicly accessible market data and should be redownloaded to ensure consistency with current provider formatting and availability.

## Reproducibility Instructions
### Step-by-step pipeline
```bash
# Step 1: Download raw data
python datafile.py

# Step 2: Clean and validate data
python clean_sp500.py

# Step 3: Run the full experimental notebook
jupyter notebook fairness_check.ipynb
```

Optional non-interactive execution:
```bash
jupyter nbconvert --to notebook --execute fairness_check.ipynb --output fairness_check.executed.ipynb
```

### Reproducibility workflow diagram (textual)
```text
yfinance (^GSPC)
   -> raw CSV (data/sp500_data.csv)
   -> cleaning/validation (clean_sp500.py)
   -> modeling table (sp500_clean.csv)
   -> feature + target construction (next_day_up)
   -> volatility group assignment (sensitive attribute)
   -> time-ordered train/test split
   -> baseline LightGBM + fairness-constrained model
   -> evaluation:
      (accuracy/balanced accuracy, DP gap, group metrics, SHAP, calibration)
   -> trading signal generation + backtest
   -> figures/tables + MLflow artifacts
```

## Fairness Framework Explanation
Fairness mitigation is implemented via:
- `fairlearn.reductions.ExponentiatedGradient`
- Fairness constraint: `DemographicParity`

Sensitive groups are based on volatility regimes (`group_volatility`). The analysis reports both:
- Overall predictive performance
- Group-disaggregated metrics (e.g., selection rate and accuracy by group)
- Demographic parity gap before/after mitigation

This setup allows transparent inspection of utility-fairness trade-offs under market heterogeneity.

## Explainability and Evaluation
Explainability:
- SHAP (`TreeExplainer`) is used for feature attribution in both baseline and fairness-mitigated models.
- Comparative attribution profiles are used to assess whether fairness mitigation materially alters decision logic.

Evaluation:
- Classification metrics: accuracy, balanced accuracy
- Fairness metrics: group selection rates, demographic parity gap
- Calibration diagnostics: calibration curve and proper scoring measures (e.g., Brier score, ECE-style analysis)
- Confusion matrices and supporting visual diagnostics

## Trading Simulation Description
Predicted class probabilities/signals are converted into trading actions in a simplified backtesting framework. The notebook compares strategy growth trajectories between baseline and fairness-mitigated models over the test period. This links statistical model behavior to downstream economic outcomes.

## Results Summary
Consistent with the paper, fairness-constrained learning reduces disparity in group-level selection behavior relative to the unconstrained baseline. This improvement is accompanied by expected utility trade-offs, which are documented through predictive, calibration, and backtesting perspectives. Full quantitative details are reported in the manuscript tables and figures.

## Citation
```bibtex
@article{PLACEHOLDER2026_fair_xai_trading,
  title   = {Fairness-Aware Explainable AI for Responsible Algorithmic Trading},
  author  = {Author, First and Author, Second and Author, Third},
  journal = {Digital Finance},
  year    = {2026},
  note    = {Under review / accepted version pending},
  doi     = {TBD}
}
```

## License
Add the project license file (`LICENSE`) and specify the applicable terms here (e.g., MIT, Apache-2.0, or CC BY-NC for code/data documentation separation).

## Contact Information
For correspondence, reproducibility queries, or reporting issues:
- Corresponding author: **[Name, Affiliation]**
- Email: **[email@institution.edu]**
- Repository issues: use the GitHub Issues tab for traceable technical discussion.
