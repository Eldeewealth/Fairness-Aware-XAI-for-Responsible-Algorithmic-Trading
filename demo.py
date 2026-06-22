# Lightweight demo for the Fairness-Aware XAI project
# Quick runnable script that executes the main pipeline steps

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
except Exception as e:
    print("Missing dependencies or import error:\n", e)
    print("Install required packages from requirements.txt (or see README_DEMO.md)")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "sp500_clean.csv"

FEATURES = ["open", "high", "low", "volume"]


def load_and_prepare(data_path, sample_limit=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # basic feature/target creation (next-day direction)
    df["close_next"] = df["close"].shift(-1)
    df["next_day_up"] = (df["close_next"] > df["close"]).astype(int)
    df = df.dropna(subset=["close_next"]).reset_index(drop=True)

    # compute rolling volatility and coarse groups (fast)
    df["log_return"] = np.log(df["close"]).diff()
    df["volatility"] = df["log_return"].rolling(window=20, min_periods=5).std()
    df = df.dropna().reset_index(drop=True)
    df["group_volatility"] = pd.qcut(df["volatility"], q=3, labels=["low", "med", "high"])
    df["group_volatility"] = pd.Categorical(df["group_volatility"], categories=["low", "med", "high"], ordered=True)

    if sample_limit is not None and len(df) > sample_limit:
        # keep most recent rows (time-ordered demo)
        df = df.iloc[-sample_limit:].reset_index(drop=True)

    return df


def time_ordered_split(df, frac=0.8):
    N = len(df)
    split_idx = int(frac * N)

    X = df[FEATURES].copy()
    y = df["next_day_up"].astype(int).copy()

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    sens_train = df.iloc[:split_idx]["group_volatility"].astype(str)
    sens_test = df.iloc[split_idx:]["group_volatility"].astype(str)

    return X_train, X_test, y_train, y_test, sens_train, sens_test


def train_and_evaluate(X_train, X_test, y_train, y_test, sens_train, sens_test):
    # small, fast model for demo
    base = LGBMClassifier(n_estimators=60, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
    base.fit(X_train, y_train)

    y_prob_base = base.predict_proba(X_test)[:, 1]
    y_pred_base = (y_prob_base >= 0.5).astype(int)

    baseline_accuracy = accuracy_score(y_test, y_pred_base)
    baseline_bal = balanced_accuracy_score(y_test, y_pred_base)

    # fairness mitigation (demographic parity)
    dp = DemographicParity()
    fair = ExponentiatedGradient(estimator=base, constraints=dp, eps=0.05)
    fair.fit(X_train, y_train, sensitive_features=sens_train)
    y_pred_fair = fair.predict(X_test).astype(int)

    fair_accuracy = accuracy_score(y_test, y_pred_fair)
    fair_bal = balanced_accuracy_score(y_test, y_pred_fair)

    results = {
        "baseline_accuracy": float(baseline_accuracy),
        "baseline_balanced_accuracy": float(baseline_bal),
        "fair_accuracy": float(fair_accuracy),
        "fair_balanced_accuracy": float(fair_bal)
    }

    return results, base, fair, y_prob_base, y_pred_base, y_pred_fair


def save_results(results, out_csv="demo_results.csv"):
    df = pd.DataFrame([results])
    df.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")


def main(args):
    print("Demo: loading and preparing data (this should be quick)")
    df = load_and_prepare(args.data, sample_limit=args.limit)

    # quick checks
    missing = [c for c in FEATURES + ["next_day_up", "group_volatility"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns after preparation: {missing}")

    X_train, X_test, y_train, y_test, sens_train, sens_test = time_ordered_split(df, frac=0.8)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    results, base, fair, y_prob_base, y_pred_base, y_pred_fair = train_and_evaluate(X_train, X_test, y_train, y_test, sens_train, sens_test)

    print("\n=== Demo results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    save_results(results, out_csv=args.output)
    print("Demo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quick demo for fairness-aware trading pipeline")
    parser.add_argument("--data", default=DATA_PATH, help="Path to cleaned CSV (default: sp500_clean.csv)")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to keep for fast demo (most recent)")
    parser.add_argument("--output", default="demo_results.csv", help="CSV file to write demo summary")
    args = parser.parse_args()
    main(args)
