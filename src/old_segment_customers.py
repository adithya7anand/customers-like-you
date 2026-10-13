
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ============================================================
# CUSTOMER SEGMENTATION USING SVD + K-MEANS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 200
RANDOM_SEED = 42
CLUSTERS_TO_TEST = [3, 4, 5, 6, 7, 8]

# ------------------------------------------------------------
# 1. Load the training matrix
# ------------------------------------------------------------

A = sparse.load_npz(
    DATA / "customer_product_matrix.npz"
).tocsr()

customers = pd.read_csv(DATA / "customer_index.csv")
products = pd.read_csv(DATA / "product_index.csv")

assert A.shape == (len(customers), len(products))

print("=" * 65)
print("CUSTOMER SEGMENTATION")
print("=" * 65)
print("Customer-product matrix:", A.shape)

# ------------------------------------------------------------
# 2. Fit SVD using training data only
# ------------------------------------------------------------

print(f"\nFitting SVD with {N_COMPONENTS} components...")

svd = TruncatedSVD(
    n_components=N_COMPONENTS,
    algorithm="randomized",
    n_iter=7,
    random_state=RANDOM_SEED,
)

customer_factors = svd.fit_transform(A)

print("Customer factor shape:", customer_factors.shape)
print(
    "Explained variance ratio:",
    round(float(svd.explained_variance_ratio_.sum()), 4),
)

np.save(
    MODELS / f"segmentation_customer_factors_k{N_COMPONENTS}.npy",
    customer_factors,
)

# ------------------------------------------------------------
# 3. Normalize customer factors
# ------------------------------------------------------------

# Normalization makes customer factor vectors comparable
# by direction rather than absolute magnitude.
customer_vectors = normalize(customer_factors)

# ------------------------------------------------------------
# 4. Compare possible numbers of clusters
# ------------------------------------------------------------

cluster_results = []
cluster_models = {}

print("\nTesting cluster counts...")

for k in CLUSTERS_TO_TEST:
    print(f"Testing {k} clusters...")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=10,
    )

    labels = model.fit_predict(customer_vectors)
    cluster_models[k] = (model, labels)

    # Sample customers for faster silhouette calculation.
    sample_size = min(1500, len(customer_vectors))

    score = silhouette_score(
        customer_vectors,
        labels,
        sample_size=sample_size,
        random_state=RANDOM_SEED,
    )

    cluster_sizes = pd.Series(labels).value_counts()

    cluster_results.append({
        "n_clusters": k,
        "silhouette_score": score,
        "smallest_cluster_size": int(cluster_sizes.min()),
        "largest_cluster_size": int(cluster_sizes.max()),
        "inertia": float(model.inertia_),
    })

    print(f"Silhouette score: {score:.4f}")

cluster_results_df = pd.DataFrame(cluster_results)

cluster_results_df.to_csv(
    REPORTS / "segmentation_cluster_comparison.csv",
    index=False,
)

# Select the cluster count with the highest silhouette score.
best_k = int(
    cluster_results_df.loc[
        cluster_results_df["silhouette_score"].idxmax(),
        "n_clusters",
    ]
)

best_model, best_labels = cluster_models[best_k]

print(f"\nSelected cluster count: {best_k}")

# ------------------------------------------------------------
# 5. Save customer cluster assignments
# ------------------------------------------------------------

# Customer IDs are kept only in this internal working file.
# Do not publish this file or use IDs in report figures.
assignments = pd.DataFrame({
    "Customer ID": customers["Customer ID"],
    "cluster": best_labels,
})

assignments.to_csv(
    DATA / "customer_cluster_assignments_internal.csv",
    index=False,
)

# ------------------------------------------------------------
# 6. Build customer-level purchase data for profiling
# ------------------------------------------------------------

train = pd.read_csv(
    DATA / "train_transactions.csv",
    dtype={"StockCode": "string"},
)

train["StockCode"] = train["StockCode"].astype(str)

# Aggregate purchase behaviour using training data only.
customer_profile = (
    train.groupby("Customer ID")
    .agg(
        orders=("Invoice", "nunique"),
        units_purchased=("Quantity", "sum"),
        spend=("Price", lambda x: 0.0),
    )
    .drop(columns=["spend"])
)

# Calculate spend separately using quantity multiplied by price.
train["line_value"] = train["Quantity"] * train["Price"]

spend = train.groupby("Customer ID")["line_value"].sum()

customer_profile["spend"] = spend

customer_profile = customer_profile.reset_index()

# Merge assignments with profile information.
profiled = assignments.merge(
    customer_profile,
    on="Customer ID",
    how="left",
)

# ------------------------------------------------------------
# 7. Create an anonymous cluster summary
# ------------------------------------------------------------

cluster_summary = (
    profiled.groupby("cluster")
    .agg(
        customer_count=("Customer ID", "size"),
        avg_orders=("orders", "mean"),
        avg_units=("units_purchased", "mean"),
        avg_spend=("spend", "mean"),
        median_spend=("spend", "median"),
    )
    .reset_index()
)

cluster_summary["customer_percentage"] = (
    cluster_summary["customer_count"]
    / cluster_summary["customer_count"].sum()
    * 100
)

cluster_summary.to_csv(
    REPORTS / "customer_segment_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# 8. Plot cluster sizes
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_summary["cluster"].astype(str),
    cluster_summary["customer_count"],
)

plt.title(f"Customer Segment Sizes (K={best_k})")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    REPORTS / "customer_segment_sizes.png",
    dpi=150,
)

plt.close()

# ------------------------------------------------------------
# 9. Plot average spend by segment
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_summary["cluster"].astype(str),
    cluster_summary["avg_spend"],
)

plt.title(f"Average Customer Spend by Segment (K={best_k})")
plt.xlabel("Segment")
plt.ylabel("Average Spend")
plt.tight_layout()

plt.savefig(
    REPORTS / "customer_segment_average_spend.png",
    dpi=150,
)

plt.close()

# ------------------------------------------------------------
# 10. Display results
# ------------------------------------------------------------

print("\nCLUSTER COMPARISON")
print("=" * 65)
print(cluster_results_df.to_string(index=False))

print("\nCUSTOMER SEGMENT SUMMARY")
print("=" * 65)
print(cluster_summary.round(2).to_string(index=False))

print("\nSaved:")
print(REPORTS / "segmentation_cluster_comparison.csv")
print(REPORTS / "customer_segment_summary.csv")
print(REPORTS / "customer_segment_sizes.png")
print(REPORTS / "customer_segment_average_spend.png")
print(DATA / "customer_cluster_assignments_internal.csv")
print(
    MODELS
    / f"segmentation_customer_factors_k{N_COMPONENTS}.npy"
)
