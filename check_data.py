
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
FILE_PATH = Path("data/raw/online_retail_II.xlsx")

# --------------------------------------------------
# 1. Check that the raw file exists
# --------------------------------------------------
if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {FILE_PATH.resolve()}\n"
        "Download and extract Online Retail II first."
    )

print("=" * 60)
print("CUSTOMERS LIKE YOU - DATA INSPECTION")
print("=" * 60)
print("File:", FILE_PATH.resolve())
print(f"File size: {FILE_PATH.stat().st_size / (1024**2):.2f} MB")

# --------------------------------------------------
# 2. Load both Excel sheets
# --------------------------------------------------
sheets = pd.read_excel(FILE_PATH, sheet_name=None)

for sheet_name, sheet_df in sheets.items():
    print(f"\nSheet: {sheet_name}")
    print("Rows:", len(sheet_df))
    print("Columns:", sheet_df.columns.tolist())

# Combine sheets
df = pd.concat(sheets.values(), ignore_index=True)

# --------------------------------------------------
# 3. Inspect schema and sample data
# --------------------------------------------------
print("\nCombined dataset")
print("-" * 60)
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.to_string())

print("\nFirst five rows:")
print(df.head().to_string())

# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------
print("\nMissing values by column:")
print(df.isna().sum().to_string())

print("\nMissing value percentages:")
print(
    (df.isna().mean() * 100)
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 5. Date range
# --------------------------------------------------
date_column = next(
    (
        c for c in df.columns
        if c.strip().lower() in {"invoicedate", "invoice date"}
    ),
    None,
)

if date_column:
    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce"
    )
    print("\nTransaction date range:")
    print("Earliest:", df[date_column].min())
    print("Latest:", df[date_column].max())

# --------------------------------------------------
# 6. Save a concise profile report
# --------------------------------------------------
report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)

profile = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_count": [int(df[c].isna().sum()) for c in df.columns],
    "missing_percent": [
        round(float(df[c].isna().mean() * 100), 2)
        for c in df.columns
    ],
})

profile.to_csv(report_dir / "data_profile.csv", index=False)

print("\nProfile saved to reports/data_profile.csv")
print("\nInspection completed successfully.")