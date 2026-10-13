
from pathlib import Path

import pandas as pd

# ============================================================
# CUSTOMERS LIKE YOU - PHASE 1 DATA CLEANING
# Assignment tasks: DE2 and DE3
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Explicit non-product codes identified in the assignment brief.
NON_PRODUCT_CODES = {
    "POST",
    "M",
    "D",
    "BANK CHARGES",
}

# ------------------------------------------------------------
# 1. Load and normalize the raw dataset
# ------------------------------------------------------------

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

sheets = pd.read_excel(RAW_FILE, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

# Standardize column names.
df.columns = [str(c).strip() for c in df.columns]

# Map common column-name variations to standard names.
column_map = {
    "InvoiceNo": "Invoice",
    "Invoice No": "Invoice",
    "UnitPrice": "Price",
    "CustomerID": "Customer ID",
    "CustomerID ": "Customer ID",
}

df = df.rename(columns=column_map)

required_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns missing: {sorted(missing_columns)}"
    )

df["Invoice"] = df["Invoice"].astype("string").str.strip()
df["StockCode"] = df["StockCode"].astype("string").str.strip()
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Customer ID"] = pd.to_numeric(
    df["Customer ID"], errors="coerce"
)

# ------------------------------------------------------------
# 2. Cleaning log
# ------------------------------------------------------------

cleaning_log = []


def record_step(rule, before, after, reason):
    removed = before - after

    cleaning_log.append({
        "rule": rule,
        "rows_before": before,
        "rows_after": after,
        "rows_removed": removed,
        "reason": reason,
    })

    print(
        f"{rule}: removed {removed:,} rows "
        f"({before:,} -> {after:,})"
    )


print("=" * 65)
print("CUSTOMERS LIKE YOU - CLEANING AND TIME SPLIT")
print("=" * 65)

print(f"Raw rows: {len(df):,}")

# ------------------------------------------------------------
# 3. Remove cancellations
# ------------------------------------------------------------

before = len(df)

is_cancellation = df["Invoice"].str.upper().str.startswith(
    "C", na=False
)

df = df.loc[~is_cancellation].copy()

record_step(
    "Remove cancellation invoices",
    before,
    len(df),
    "Invoice numbers starting with C",
)

# ------------------------------------------------------------
# 4. Remove non-positive or invalid quantities
# ------------------------------------------------------------

before = len(df)

df = df.loc[
    df["Quantity"].notna() & (df["Quantity"] > 0)
].copy()

record_step(
    "Remove invalid quantities",
    before,
    len(df),
    "Quantity must be greater than zero",
)

# ------------------------------------------------------------
# 5. Remove non-positive or invalid prices
# ------------------------------------------------------------
# Decision: Exclude non-positive quantities, including return lines.
# Limitation: Original purchases associated with later returns
# may remain in the dataset. This baseline does not reverse them.
# This decision is documented for future sensitivity analysis.

before = len(df)

df = df.loc[
    df["Price"].notna() & (df["Price"] > 0)
].copy()

record_step(
    "Remove invalid prices",
    before,
    len(df),
    "Unit price must be greater than zero",
)

# ------------------------------------------------------------
# 6. Remove known non-product codes
# ------------------------------------------------------------

before = len(df)

normalized_codes = (
    df["StockCode"].str.upper().str.strip()
)

df = df.loc[
    ~normalized_codes.isin(NON_PRODUCT_CODES)
].copy()

record_step(
    "Remove known non-product codes",
    before,
    len(df),
    "Exclude POST, M, D and BANK CHARGES",
)

# ------------------------------------------------------------
# 7. Remove missing customer IDs
# ------------------------------------------------------------

before = len(df)

df = df.loc[df["Customer ID"].notna()].copy()

record_step(
    "Remove missing customer IDs",
    before,
    len(df),
    "Customer-level recommendations require a customer ID",
)

# ------------------------------------------------------------
# 8. Validate remaining essential fields
# ------------------------------------------------------------

before = len(df)

df = df.loc[
    df["Invoice"].notna()
    & df["StockCode"].notna()
    & df["StockCode"].ne("")
    & df["InvoiceDate"].notna()
].copy()

record_step(
    "Remove invalid invoice, product code or date",
    before,
    len(df),
    "Essential transaction fields must be present",
)

# Description is deliberately not used as a mandatory filter:
# a valid product code may have a missing description.

df["Customer ID"] = df["Customer ID"].astype("int64")

# ------------------------------------------------------------
# 9. Create the chronological train/test split
# ------------------------------------------------------------

latest_date = df["InvoiceDate"].max()

# Test period begins two calendar months before the latest
# transaction timestamp in the cleaned data.
cutoff_date = latest_date - pd.DateOffset(months=2)

train = df.loc[df["InvoiceDate"] < cutoff_date].copy()
test = df.loc[df["InvoiceDate"] >= cutoff_date].copy()

print("\nTime-based split")
print("-" * 65)
print("Earliest cleaned transaction:", df["InvoiceDate"].min())
print("Latest cleaned transaction:", latest_date)
print("Test cutoff:", cutoff_date)
print("Training rows:", f"{len(train):,}")
print("Test rows:", f"{len(test):,}")

if train.empty or test.empty:
    raise ValueError("Training or test data is empty. Check dates.")

# ------------------------------------------------------------
# 10. Identify genuinely unseen customer-product purchases
# ------------------------------------------------------------

# Only training customer-product pairs are used to determine
# whether a test purchase is new to that customer.

train_pairs = train[
    ["Customer ID", "StockCode"]
].drop_duplicates()

train_pairs = train_pairs.assign(seen_in_train=True)

test_with_history = test.merge(
    train_pairs,
    on=["Customer ID", "StockCode"],
    how="left",
    validate="many_to_one",
)

unseen_test = test_with_history.loc[
    test_with_history["seen_in_train"].isna()
].drop(columns=["seen_in_train"])

print("\nUnseen-product evaluation candidates")
print("-" * 65)
print("Test transaction rows:", f"{len(test):,}")
print("Unseen customer-product rows:", f"{len(unseen_test):,}")
print(
    "Customers in training:",
    f"{train['Customer ID'].nunique():,}",
)
print(
    "Customers in test:",
    f"{test['Customer ID'].nunique():,}",
)

# ------------------------------------------------------------
# 11. Save processed data and audit reports
# ------------------------------------------------------------

# CSV is used here for straightforward inspection.
# These files may contain customer IDs. Keep them local.
df.to_csv(
    OUTPUT_DIR / "clean_transactions.csv",
    index=False,
)
train.to_csv(
    OUTPUT_DIR / "train_transactions.csv",
    index=False,
)
test.to_csv(
    OUTPUT_DIR / "test_transactions.csv",
    index=False,
)
unseen_test.to_csv(
    OUTPUT_DIR / "test_unseen_products.csv",
    index=False,
)

pd.DataFrame(cleaning_log).to_csv(
    REPORT_DIR / "cleaning_log.csv",
    index=False,
)

# Record the split details for reproducibility.
split_summary = pd.DataFrame([{
    "raw_rows": sum(len(sheet) for sheet in sheets.values()),
    "clean_rows": len(df),
    "training_rows": len(train),
    "test_rows": len(test),
    "unseen_test_rows": len(unseen_test),
    "latest_cleaned_date": str(latest_date),
    "test_cutoff": str(cutoff_date),
    "training_customers": train["Customer ID"].nunique(),
    "test_customers": test["Customer ID"].nunique(),
}])

split_summary.to_csv(
    REPORT_DIR / "split_summary.csv",
    index=False,
)

print("\nSaved files")
print("-" * 65)
print("Clean data:       data/processed/clean_transactions.csv")
print("Training data:    data/processed/train_transactions.csv")
print("Test data:        data/processed/test_transactions.csv")
print("Unseen purchases: data/processed/test_unseen_products.csv")
print("Cleaning log:     reports/cleaning_log.csv")
print("Split summary:    reports/split_summary.csv")

print("\nPhase 1 cleaning and split completed.")