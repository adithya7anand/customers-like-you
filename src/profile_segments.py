
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CUSTOMER SEGMENT PROFILING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

TOP_PRODUCTS = 10

# ------------------------------------------------------------
# 1. Load internal cluster assignments and training data
# ------------------------------------------------------------

assignments = pd.read_csv(
    DATA / "customer_cluster_assignments_internal.csv"
)

train = pd.read_csv(
    DATA / "train_transactions.csv",
    dtype={"StockCode": "string"},
)

products = pd.read_csv(
    DATA / "product_index.csv",
    dtype={"StockCode": "string"},
)

print("=" * 65)
print("CUSTOMER SEGMENT PROFILING")
print("=" * 65)

print("Customers assigned to segments:", len(assignments))
print("Training transactions:", len(train))

# ------------------------------------------------------------
# 2. Calculate customer purchasing behaviour
# ------------------------------------------------------------

train["StockCode"] = train["StockCode"].astype(str)
train["line_value"] = train["Quantity"] * train["Price"]

customer_metrics = (
    train.groupby("Customer ID")
    .agg(
        orders=("Invoice", "nunique"),
        units_purchased=("Quantity", "sum"),
        spend=("line_value", "sum"),
        distinct_products=("StockCode", "nunique"),
    )
    .reset_index()
)

profiled = assignments.merge(
    customer_metrics,
    on="Customer ID",
    how="left",
)

# ------------------------------------------------------------
# 3. Create anonymous segment summary
# ------------------------------------------------------------

segment_summary = (
    profiled.groupby("cluster")
    .agg(
        customer_count=("Customer ID", "size"),
        avg_orders=("orders", "mean"),
        median_orders=("orders", "median"),
        avg_units=("units_purchased", "mean"),
        median_units=("units_purchased", "median"),
        avg_spend=("spend", "mean"),
        median_spend=("spend", "median"),
        avg_distinct_products=("distinct_products", "mean"),
    )
    .reset_index()
)

segment_summary["customer_percentage"] = (
    segment_summary["customer_count"]
    / segment_summary["customer_count"].sum()
    * 100
)

segment_summary.to_csv(
    REPORTS / "segment_profile_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# 4. Identify top products by segment
# ------------------------------------------------------------

# Count distinct customers buying each product within each segment.
# This identifies customer reach, not just units sold.

segment_transactions = train.merge(
    assignments,
    on="Customer ID",
    how="inner",
)

product_reach = (
    segment_transactions.groupby(
        ["cluster", "StockCode"]
    )["Customer ID"]
    .nunique()
    .reset_index(name="customers_buying_product")
)

segment_sizes = assignments.groupby("cluster").size()

product_reach["segment_customers"] = (
    product_reach["cluster"].map(segment_sizes)
)

product_reach["customer_penetration_pct"] = (
    product_reach["customers_buying_product"]
    / product_reach["segment_customers"]
    * 100
)

# Add product descriptions where available.
description_map = (
    train.dropna(subset=["Description"])
    .assign(StockCode=lambda df: df["StockCode"].astype(str))
    .groupby("StockCode")["Description"]
    .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "")
)

product_reach["description"] = (
    product_reach["StockCode"].map(description_map).fillna("")
)

product_reach = product_reach.sort_values(
    ["cluster", "customers_buying_product"],
    ascending=[True, False],
)

product_reach["rank_within_segment"] = (
    product_reach.groupby("cluster")["customers_buying_product"]
    .rank(method="first", ascending=False)
    .astype(int)
)

top_products = product_reach[
    product_reach["rank_within_segment"] <= TOP_PRODUCTS
].copy()

# Product codes and descriptions are retained for internal analysis.
# Review and anonymize any customer-level data before publication.
top_products.to_csv(
    REPORTS / "segment_top_products_internal.csv",
    index=False,
)

# ------------------------------------------------------------
# 5. Plot customer count by segment
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    segment_summary["cluster"].astype(str),
    segment_summary["customer_count"],
)

plt.title("Number of Customers by Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    REPORTS / "segment_profile_customer_counts.png",
    dpi=150,
)

plt.close()

# ------------------------------------------------------------
# 6. Plot median spend by segment
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    segment_summary["cluster"].astype(str),
    segment_summary["median_spend"],
)

plt.title("Median Spend by Segment")
plt.xlabel("Segment")
plt.ylabel("Median Spend")
plt.tight_layout()

plt.savefig(
    REPORTS / "segment_profile_median_spend.png",
    dpi=150,
)

plt.close()

# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("\nSEGMENT SUMMARY")
print("=" * 65)
print(segment_summary.round(2).to_string(index=False))

for cluster_id in sorted(top_products["cluster"].unique()):
    print(f"\nSEGMENT {cluster_id}: TOP PRODUCTS")
    print("-" * 65)

    view = top_products[
        top_products["cluster"] == cluster_id
    ][
        [
            "rank_within_segment",
            "StockCode",
            "description",
            "customers_buying_product",
            "customer_penetration_pct",
        ]
    ]

    print(view.to_string(index=False))

print("\nSaved reports:")
print(REPORTS / "segment_profile_summary.csv")
print(REPORTS / "segment_top_products_internal.csv")
print(REPORTS / "segment_profile_customer_counts.png")
print(REPORTS / "segment_profile_median_spend.png")
