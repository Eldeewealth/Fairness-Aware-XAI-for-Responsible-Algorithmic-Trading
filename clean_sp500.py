import pandas as pd

# Load CSV (skipping first two rows) and assign correct column names manually
column_names = ["Date", "Close", "High", "Low", "Open", "Volume"]
df = pd.read_csv("data/sp500_data.csv", skiprows=2, names=column_names, parse_dates=["Date"])

# 1. Preview the dataset
print("📌 Initial rows:")
print(df.head())

# 2. Check column names and types
print("\n🔍 Data types:")
print(df.dtypes)

# 3. Check for missing values before cleaning
print("\n🧹 Missing values BEFORE cleaning:")
print(df.isnull().sum())

# 4. Drop rows with missing values (if any)
df_clean = df.dropna()

# 5. Ensure chronological order
df_clean = df_clean.sort_values(by="Date").reset_index(drop=True)

# 6. Check for duplicate dates
duplicates = df_clean[df_clean.duplicated(subset=["Date"])]
if not duplicates.empty:
    print("\n⚠️ Duplicates found. Dropping them.")
    df_clean = df_clean.drop_duplicates(subset=["Date"])
else:
    print("\n✅ No duplicate dates found.")

# 7. Rename columns (optional formatting)
df_clean.columns = [col.strip().lower() for col in df_clean.columns]

# 8. Save cleaned file
df_clean.to_csv("sp500_clean.csv", index=False)
print("\n✅ Cleaned data saved as 'sp500_clean.csv'")

# 9. Print shape and sample
print("\n🧾 Cleaned data shape:", df_clean.shape)
print("\n📌 Sample data:")
print(df_clean.tail())

# 🔍 10. Check for missing values AFTER cleaning + show row indices if any
print("\n🧪 Missing values AFTER cleaning:")
missing_summary = df_clean.isnull().sum()
print(missing_summary)

print("\n📍 Indices of missing values per column (if any):")
for col in df_clean.columns:
    missing_indices = df_clean[df_clean[col].isnull()].index.tolist()
    if missing_indices:
        print(f"{col}: {missing_indices}")
    else:
        print(f"{col}: No missing values")
