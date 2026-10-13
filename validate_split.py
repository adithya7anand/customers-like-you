
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

train = pd.read_csv(
    DATA / "train_transactions.csv",
    parse_dates=["InvoiceDate"],
)
test = pd.read_csv(
    DATA / "test_transactions.csv",
    parse_dates=["InvoiceDate"],
)
unseen = pd.read_csv(
    DATA / "test_unseen_products.csv",
    parse_dates=["InvoiceDate"],
)

errors = []

# Check files and non-empty datasets
for name, frame in [
    ("training", train),
    ("test", test),
    ("unseen test", unseen),
]:
    print(f"{name}: {len(frame):,} rows")

if train.empty:
    errors.append("Training dataset is empty.")

if test.empty:
    errors.append("Test dataset is empty.")

if unseen.empty:
    print(
        "WARNING: No unseen test purchases. "
        "The recommendation evaluation may not be possible."
    )

# Check chronological split
if not train.empty and not test.empty:
    train_max = train["InvoiceDate"].max()
    test_min = test["InvoiceDate"].min()

    print("\nLatest training transaction:", train_max)
    print("Earliest test transaction:", test_min)

    if train_max >= test_min:
        errors.append(
            "Training and test dates overlap or are out of order."
        )

# Check for customer-product leakage in the unseen file
if not train.empty and not unseen.empty:
    train_pairs = pd.MultiIndex.from_frame(
        train[["Customer ID", "StockCode"]].drop_duplicates()
    )

    unseen_pairs = pd.MultiIndex.from_frame(
        unseen[["Customer ID", "StockCode"]]
    )

    overlap_count = unseen_pairs.isin(train_pairs).sum()

    print(
        "\nUnseen test rows whose customer-product pair "
        "appears in training:",
        int(overlap_count),
    )

    if overlap_count:
        errors.append(
            "The unseen test file contains previously seen "
            "customer-product pairs."
        )

# Check quantities, prices and customer IDs
for name, frame in [("training", train), ("test", test)]:
    if frame["Customer ID"].isna().any():
        errors.append(f"{name} has missing customer IDs.")

    if (frame["Quantity"] <= 0).any():
        errors.append(f"{name} has non-positive quantities.")

    if (frame["Price"] <= 0).any():
        errors.append(f"{name} has non-positive prices.")

print("\n" + "=" * 55)

if errors:
    print("VALIDATION FAILED")
    for error in errors:
        print("-", error)
    raise SystemExit(1)

print("VALIDATION PASSED")
print("Basic cleaning and time-split checks are successful.")