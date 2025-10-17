import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate

# Load dataset
df = pd.read_csv("sp500_clean.csv", parse_dates=["date"])

# Feature Engineering
df["volatility"] = df["high"] - df["low"]
df["price_change"] = df["close"] - df["open"]
df["group_volume"] = pd.qcut(df["volume"], q=3, labels=["low", "medium", "high"])
df["group_volatility"] = pd.qcut(df["volatility"], q=3, labels=["low", "medium", "high"])
df["group_price_change"] = pd.qcut(df["price_change"], q=3, labels=["low", "medium", "high"])

# Binary Target
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df.dropna(inplace=True)

# Feature Matrix and Target
features = ["open", "high", "low", "volume"]
X = df[features]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Base Model
model = LGBMClassifier(max_depth=5, num_leaves=31, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# SHAP Explainability
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)[1] if isinstance(explainer.shap_values(X_test), list) else explainer.shap_values(X_test)
shap.summary_plot(shap_vals, X_test, feature_names=features, plot_type="bar")

# Histogram of Volume
plt.figure(figsize=(8, 4))
sns.histplot(df["volume"], bins=30, kde=True)
plt.title("Histogram of Trading Volume")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Time Trends for Volume, Volatility, and Price Change
for col, group in zip(["volume", "volatility", "price_change"],
                      ["group_volume", "group_volatility", "group_price_change"]):
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="date", y=col, hue=group)
    plt.title(f"{col.title()} Trends by Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Group-wise Performance Function
def group_metrics(group_col):
    idx_df = df.iloc[X_test.index]
    for label in idx_df[group_col].dropna().unique():
        group_idx = idx_df[idx_df[group_col] == label].index
        if not group_idx.empty:
            y_true_group = y_test.loc[group_idx]
            y_pred_group = pd.Series(y_pred, index=y_test.index).loc[group_idx]
            acc = accuracy_score(y_true_group, y_pred_group)
            prec = precision_score(y_true_group, y_pred_group, zero_division=0)
            rec = recall_score(y_true_group, y_pred_group, zero_division=0)
            print(f"\n📊 {group_col} = {label}")
            print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

print("\n==== Group Performance by Volume ====")
group_metrics("group_volume")

# Fairness Mitigation with Fairlearn
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    estimator=LGBMClassifier(max_depth=5, num_leaves=31),
    constraints=constraint
)
mitigator.fit(X_train, y_train, sensitive_features=df.loc[X_train.index, "group_volume"].astype(str))
y_pred_fair = mitigator.predict(X_test)

# Fairness MetricFrame Report
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=df.loc[X_test.index, "group_volume"].astype(str)
)

print("\n🎯 Fairness Metrics (Post-Mitigation):")
print(metric_frame.by_group)

# === FIGURE 4: Fairness Metrics Plot ===
# -# === Figure 4: Accuracy (bars) + Selection Rate (lines), Before vs After ===

# 1) Compute MetricFrames
mf_before = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df.loc[X_test.index, "group_volume"].astype(str)
).by_group.copy()

mf_after = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=df.loc[X_test.index, "group_volume"].astype(str)
).by_group.copy()

# 2) Order groups consistently
order = ["low", "medium", "high"]
mf_before = mf_before.reindex([g for g in order if g in mf_before.index])
mf_after  = mf_after.reindex([g for g in order if g in mf_after.index])
groups = [g.capitalize() for g in mf_before.index]
x = np.arange(len(groups))
w = 0.35

# 3) Build the plot
fig, ax1 = plt.subplots(figsize=(9, 5))

# Bars: Accuracy (left=before, right=after)
b1 = ax1.bar(x - w/2, mf_before["accuracy"].values, width=w, label="Accuracy (Before)")
b2 = ax1.bar(x + w/2, mf_after["accuracy"].values,  width=w, label="Accuracy (After)")
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Volume Group")
ax1.set_xticks(x)
ax1.set_xticklabels(groups)
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", linestyle=":", alpha=0.35)

# 4) Twin axis for Selection Rate (lines)
ax2 = ax1.twinx()
ax2.plot(x, mf_before["selection_rate"].values, "o--", label="Selection Rate (Before)")
ax2.plot(x, mf_after["selection_rate"].values,  "o--", label="Selection Rate (After)", color="red")
ax2.set_ylabel("Selection Rate")
ax2.set_ylim(0, 1.0)

# 5) Legends (merge both axes)
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=2, frameon=False)

plt.title("Figure 4 – Fairness Metrics Comparison by Volume Group (Before vs After Mitigation)")
plt.tight_layout()
# plt.savefig("figure4_fairness_before_after.png", dpi=300)
# plt.show()

# 6) (Optional) value labels on bars and points
def add_bar_labels(bars, ax):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

def add_point_labels(ax, xs, ys):
    for xi, yi in zip(xs, ys):
        ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

add_bar_labels(b1, ax1)
add_bar_labels(b2, ax1)
add_point_labels(ax2, x, mf_before["selection_rate"].values)
add_point_labels(ax2, x, mf_after["selection_rate"].values)
