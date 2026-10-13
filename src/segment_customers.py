
from pathlib import Path
import time

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
# Final exploratory solution: 5 customer segments
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 200
RANDOM_SEED = 42

# Compare only cluster counts permitted by the assignment.
CLUSTERS_TO_TEST = [3, 4, 5, 6]

# Deliberately choose five groups for the final exploratory solution.
# The silhouette scores from earlier experiments were very low.
FINAL_N_CLUSTERS = 5

SILHOUETTE_SAMPLE_SIZE = 1500


def main():
    # --------------------------------------------------------
    # 1. Load the training matrix and mappings
    # --------------------------------------------------------

    print("=" * 68)
    print("CUSTOMER SEGMENTATION")
    print("=" * 68)

    matrix_path = DATA / "customer_product_matrix.npz"
    customer_index_path = DATA / "customer_index.csv"
    product_index_path = DATA / "product_index.csv"

    A = sparse.load_npz(matrix_path).tocsr()
    customers = pd.read_csv(customer_index_path)
    products = pd.read_csv(product_index_path)

    if A.shape != (len(customers), len(products)):
        raise ValueError(
            "Matrix dimensions do not match the customer and product mappings."
        )

    print("Customer-product matrix:", A.shape)
    print("Customers in matrix:", len(customers))
    print("Products in matrix:", len(products))
    print("Cluster counts to compare:", CLUSTERS_TO_TEST)
    print("Final cluster count:", FINAL_N_CLUSTERS)

    # --------------------------------------------------------
    # 2. Fit SVD using training data only
    # --------------------------------------------------------

    print(f"\nFitting Truncated SVD with {N_COMPONENTS} components...")

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_SEED,
    )

    svd_start = time.perf_counter()
    customer_factors = svd.fit_transform(A)
    svd_seconds = time.perf_counter() - svd_start

    explained_variance = float(
        svd.explained_variance_ratio_.sum()
    )

    print("Customer factor shape:", customer_factors.shape)
    print(f"SVD fit time: {svd_seconds:.3f} seconds")
    print(f"Explained variance ratio: {explained_variance:.4f}")

    factors_path = (
        MODELS / f"segmentation_customer_factors_k{N_COMPONENTS}.npy"
    )

    np.save(factors_path, customer_factors)

    # --------------------------------------------------------
    # 3. Normalize customer factors
    # --------------------------------------------------------

    # Normalization compares the direction of customer vectors
    # rather than allowing large vector magnitudes to dominate.
    customer_vectors = normalize(customer_factors)

    zero_vector_count = int(
        np.sum(np.linalg.norm(customer_vectors, axis=1) == 0)
    )

    print("Zero-vector customers:", zero_vector_count)

    # --------------------------------------------------------
    # 4. Compare cluster counts from 3 through 6
    # --------------------------------------------------------

    cluster_results = []
    cluster_models = {}

    sample_size = min(
        SILHOUETTE_SAMPLE_SIZE,
        len(customer_vectors),
    )

    print("\nComparing cluster counts...")

    for k in CLUSTERS_TO_TEST:
        print(f"\nTesting {k} clusters...")

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
        )

        kmeans_start = time.perf_counter()
        labels = model.fit_predict(customer_vectors)
        kmeans_seconds = time.perf_counter() - kmeans_start

        # Use a fixed random seed for a reproducible silhouette sample.
        score = silhouette_score(
            customer_vectors,
            labels,
            sample_size=sample_size,
            random_state=RANDOM_SEED,
        )

        cluster_sizes = pd.Series(labels).value_counts()

        cluster_results.append({
            "n_clusters": k,
            "silhouette_score": float(score),
            "smallest_cluster_size": int(cluster_sizes.min()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "cluster_size_std": float(cluster_sizes.std()),
            "zero_vector_customers": zero_vector_count,
            "svd_seconds": svd_seconds,
            "kmeans_fit_seconds": kmeans_seconds,
            "svd_components": N_COMPONENTS,
            "random_state": RANDOM_SEED,
            "silhouette_sample_size": sample_size,
        })

        cluster_models[k] = {
            "model": model,
            "labels": labels,
        }

        print(f"Silhouette score: {score:.6f}")
        print(f"Smallest cluster: {cluster_sizes.min()} customers")
        print(f"Largest cluster: {cluster_sizes.max()} customers")
        print(f"K-Means fit time: {kmeans_seconds:.3f} seconds")

    cluster_results_df = pd.DataFrame(cluster_results)

    comparison_path = REPORTS / "segmentation_cluster_comparison.csv"
    cluster_results_df.to_csv(comparison_path, index=False)

    print("\nCLUSTER COUNT COMPARISON")
    print("-" * 68)
    print(
        cluster_results_df[
            [
                "n_clusters",
                "silhouette_score",
                "smallest_cluster_size",
                "largest_cluster_size",
                "kmeans_fit_seconds",
            ]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. Select the final five-cluster solution
    # --------------------------------------------------------

    # This is a deliberate choice for a simpler assignment deliverable,
    # not a claim that five clusters are objectively optimal.
    selected = cluster_models[FINAL_N_CLUSTERS]
    best_labels = selected["labels"]

    selected_silhouette = cluster_results_df.loc[
        cluster_results_df["n_clusters"] == FINAL_N_CLUSTERS,
        "silhouette_score",
    ].iloc[0]

    print(f"\nSelected cluster count: {FINAL_N_CLUSTERS}")
    print(f"Selected solution silhouette: {selected_silhouette:.6f}")
    print(
        "Interpretation: exploratory segmentation; "
        "low silhouette scores indicate weak cluster separation."
    )

    # --------------------------------------------------------
    # 6. Save internal customer assignments
    # --------------------------------------------------------

    # This file contains customer IDs.
    # Keep it internal and do not submit or publish it.
    assignments = pd.DataFrame({
        "Customer ID": customers["Customer ID"],
        "cluster": best_labels,
    })

    assignments_path = (
        DATA / "customer_cluster_assignments_internal.csv"
    )

    assignments.to_csv(assignments_path, index=False)

    # --------------------------------------------------------
    # 7. Load training transactions and calculate profiles
    # --------------------------------------------------------

    print("\nCalculating customer purchase profiles...")

    train_path = DATA / "train_transactions.csv"

    train = pd.read_csv(
        train_path,
        dtype={"StockCode": "string"},
    )

    train["StockCode"] = train["StockCode"].astype(str)
    train["line_value"] = train["Quantity"] * train["Price"]

    customer_profile = (
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
        customer_profile,
        on="Customer ID",
        how="left",
    )

    # --------------------------------------------------------
    # 8. Create an anonymous five-segment summary
    # --------------------------------------------------------

    cluster_summary = (
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

    total_customers = cluster_summary["customer_count"].sum()

    cluster_summary["customer_percentage"] = (
        cluster_summary["customer_count"]
        / total_customers
        * 100
    )

    summary_path = REPORTS / "customer_segment_summary.csv"
    cluster_summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # 9. Plot customer count by segment
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        cluster_summary["cluster"].astype(str),
        cluster_summary["customer_count"],
    )

    plt.title("Customer Segment Sizes (K=5)")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    sizes_plot_path = REPORTS / "customer_segment_sizes.png"

    plt.savefig(sizes_plot_path, dpi=150)
    plt.close()

    # --------------------------------------------------------
    # 10. Plot median spend by segment
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        cluster_summary["cluster"].astype(str),
        cluster_summary["median_spend"],
    )

    plt.title("Median Customer Spend by Segment (K=5)")
    plt.xlabel("Segment")
    plt.ylabel("Median Spend")
    plt.tight_layout()

    spend_plot_path = REPORTS / "customer_segment_average_spend.png"

    plt.savefig(spend_plot_path, dpi=150)
    plt.close()

    # --------------------------------------------------------
    # 11. Print results
    # --------------------------------------------------------

    print("\nFIVE-CLUSTER CUSTOMER SUMMARY")
    print("=" * 68)
    print(cluster_summary.round(2).to_string(index=False))

    print("\nSaved outputs:")
    print(comparison_path)
    print(summary_path)
    print(sizes_plot_path)
    print(spend_plot_path)
    print(factors_path)
    print(assignments_path)

    print("\nReminder: customer_cluster_assignments_internal.csv")
    print("contains customer IDs and must remain internal.")


if __name__ == "__main__":
    main()
