
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

# ============================================================
# CUSTOMERS LIKE YOU - SPARSE MATRIX BUILDER
# Assignment tasks: DE4 and DE5
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_transactions.csv"

MATRIX_FILE = DATA_DIR / "customer_product_matrix.npz"
CUSTOMERS_FILE = DATA_DIR / "customer_index.csv"
PRODUCTS_FILE = DATA_DIR / "product_index.csv"
REPORT_FILE = REPORT_DIR / "matrix_report.csv"
CONFIG_FILE = REPORT_DIR / "matrix_config.json"

MIN_CUSTOMERS_PER_PRODUCT = 10
RANDOM_SEED = 42

REPORT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()

# ------------------------------------------------------------
# 1. Load training data only
# ------------------------------------------------------------

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_FILE}"
    )

train = pd.read_csv(
    TRAIN_FILE,
    dtype={"StockCode": "string"},
)

required = {"Customer ID", "StockCode", "Quantity"}
if not required.issubset(train.columns):
    raise ValueError(
        f"Missing columns: {required - set(train.columns)}"
    )

train = train.dropna(
    subset=["Customer ID", "StockCode", "Quantity"]
).copy()

train["Customer ID"] = train["Customer ID"].astype("int64")
train["StockCode"] = train["StockCode"].str.strip()
train["Quantity"] = pd.to_numeric(
    train["Quantity"], errors="raise"
)

if (train["Quantity"] <= 0).any():
    raise ValueError(
        "Training contains non-positive quantities."
    )

print("Training transaction rows:", f"{len(train):,}")

# ------------------------------------------------------------
# 2. Aggregate purchases before applying log1p
# ------------------------------------------------------------

customer_product = (
    train.groupby(
        ["Customer ID", "StockCode"],
        as_index=False,
        sort=True,
    )
    .agg(total_quantity=("Quantity", "sum"))
)

# Retain products bought by at least 10 distinct customers.
product_customer_counts = (
    customer_product.groupby("StockCode")["Customer ID"]
    .nunique()
)

eligible_products = product_customer_counts[
    product_customer_counts >= MIN_CUSTOMERS_PER_PRODUCT
].index

customer_product = customer_product[
    customer_product["StockCode"].isin(eligible_products)
].copy()

if customer_product.empty:
    raise ValueError("No customer-product pairs remain.")

# Purchase value = log(1 + total quantity).
customer_product["value"] = np.log1p(
    customer_product["total_quantity"].astype(float)
)

# ------------------------------------------------------------
# 3. Build stable index mappings
# ------------------------------------------------------------

customers = np.sort(
    customer_product["Customer ID"].unique()
)

products = np.sort(
    customer_product["StockCode"].unique()
)

customer_to_row = {
    customer: i for i, customer in enumerate(customers)
}

product_to_col = {
    product: j for j, product in enumerate(products)
}

rows = customer_product["Customer ID"].map(
    customer_to_row
).to_numpy()

cols = customer_product["StockCode"].map(
    product_to_col
).to_numpy()

values = customer_product["value"].to_numpy(
    dtype=np.float64
)

# ------------------------------------------------------------
# 4. Construct CSR sparse matrix
# ------------------------------------------------------------

A = sparse.csr_matrix(
    (values, (rows, cols)),
    shape=(len(customers), len(products)),
    dtype=np.float64,
)

A.sum_duplicates()
A.eliminate_zeros()
A.sort_indices()

# ------------------------------------------------------------
# 5. Calculate matrix statistics
# ------------------------------------------------------------

n_customers, n_products = A.shape
total_cells = n_customers * n_products
nnz = A.nnz

sparsity = (
    1 - nnz / total_cells
    if total_cells
    else 0
)

dense_bytes = total_cells * A.dtype.itemsize
sparse_bytes = (
    A.data.nbytes
    + A.indices.nbytes
    + A.indptr.nbytes
)

elapsed_seconds = time.perf_counter() - start_time

# ------------------------------------------------------------
# 6. Save matrix and index mappings
# ------------------------------------------------------------

sparse.save_npz(MATRIX_FILE, A)

pd.DataFrame({
    "row_index": np.arange(len(customers)),
    "Customer ID": customers,
}).to_csv(CUSTOMERS_FILE, index=False)

pd.DataFrame({
    "column_index": np.arange(len(products)),
    "StockCode": products,
}).to_csv(PRODUCTS_FILE, index=False)

report = {
    "customers": int(n_customers),
    "products": int(n_products),
    "matrix_rows": int(n_customers),
    "matrix_columns": int(n_products),
    "nonzero_entries": int(nnz),
    "total_cells": int(total_cells),
    "sparsity_percent": round(sparsity * 100, 4),
    "dense_memory_mb": round(dense_bytes / (1024**2), 3),
    "sparse_memory_mb": round(sparse_bytes / (1024**2), 3),
    "memory_reduction_percent": round(
        (1 - sparse_bytes / dense_bytes) * 100, 3
    ) if dense_bytes else 0,
    "purchase_value": "log1p of aggregated quantity",
    "minimum_customers_per_product": MIN_CUSTOMERS_PER_PRODUCT,
    "matrix_format": "CSR",
    "dtype": str(A.dtype),
    "build_time_seconds": round(elapsed_seconds, 3),
    "random_seed": RANDOM_SEED,
}

pd.DataFrame([report]).to_csv(
    REPORT_FILE,
    index=False,
)

with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nMATRIX BUILD REPORT")
print("=" * 55)

for key, value in report.items():
    print(f"{key}: {value}")

print("\nFiles created:")
print(MATRIX_FILE)
print(CUSTOMERS_FILE)
print(PRODUCTS_FILE)
print(REPORT_FILE)
print(CONFIG_FILE)

print("\nSparse matrix construction completed.")