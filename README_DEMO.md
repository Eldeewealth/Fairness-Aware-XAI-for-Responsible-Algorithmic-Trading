Demo: Fairness-Aware XAI — Quick Run

This repository contains a lightweight demo script `demo.py` that runs a short end-to-end pipeline:
- Loads `sp500_clean.csv` (must be present in the repository root)
- Creates the `next_day_up` target and a volatility-based sensitive group
- Trains a small LightGBM baseline model
- Applies ExponentiatedGradient (Demographic Parity) mitigation
- Prints and saves a short CSV summary (`demo_results.csv`)

How to run (Windows PowerShell):

```powershell
# activate your virtualenv if needed
& ".\.venv\Scripts\Activate.ps1"

# install requirements if not already installed
python -m pip install -r requirements.txt

# run the demo (uses most recent 1000 rows by default)
python demo.py --data sp500_clean.csv --limit 1000 --output demo_results.csv
```

Notes:
- The demo uses reduced model size (`n_estimators=60`) and a sample limit to keep runtime short.
- If you want a full run, remove `--limit` or increase it; expect longer training time.
- If imports fail, install the packages in `requirements.txt` or the subset: `lightgbm fairlearn scikit-learn pandas numpy`.

Want me to run the demo here and show the output? Reply and I can execute it and report results (requires the Python environment to have the needed packages).