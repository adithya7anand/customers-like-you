from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

MATRIX_PATH = ROOT / "data" / "processed" / "customer_product_matrix.npz"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
SVD_COMPONENTS = 200
CLUSTER_COUNTS = [5, 8]
SILHOUETTE_SAMPLE_SIZE = 1500


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Matrix not found: {MATRIX_PATH}")

    print("Loading customer-product matrix...")
    matrix = load_npz(MATRIX_PATH).tocsr()
    n_customers, n_products = matrix.shape
    print(f"Matrix shape: {matrix.shape}")

    # TruncatedSVD requires n_components < n_features
    n_components = min(SVD_COMPONENTS, n_products - 1, n_customers - 1)
    if n_components < SVD_COMPONENTS:
        print(f"Reducing SVD components from {SVD_COMPONENTS} to {n_components}.")

    print(f"Fitting TruncatedSVD with {n_components} components...")
    start = time.perf_counter()

    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=7,
        random_state=RANDOM_STATE,
    )
    customer_factors = svd.fit_transform(matrix)

    # Customers with no purchases produce all-zero vectors; normalize()
    # leaves them as zeros, so they would cluster together artificially.
    zero_rows = int((np.linalg.norm(customer_factors, axis=1) == 0).sum())
    if zero_rows:
        print(f"Warning: {zero_rows} customers have zero vectors (no purchases).")

    customer_factors = normalize(customer_factors)
    svd_seconds = time.perf_counter() - start
    print(f"SVD completed in {svd_seconds:.2f} seconds.")

    # Use the same sample for every silhouette calculation.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, n_customers)
    sample_indices = np.sort(
        rng.choice(n_customers, size=sample_size, replace=False)
    )
    sample_factors = customer_factors[sample_indices]

    results = []

    for k in CLUSTER_COUNTS:
        if k >= n_customers:
            print(f"Skipping k={k}: not enough customers.")
            continue

        print(f"\nEvaluating {k} clusters...")
        start = time.perf_counter()

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(customer_factors)
        fit_seconds = time.perf_counter() - start

        sample_labels = labels[sample_indices]
        n_sample_clusters = len(np.unique(sample_labels))

        # silhouette_score needs 2 <= n_labels <= n_samples - 1
        if 2 <= n_sample_clusters <= sample_size - 1:
            score = float(
                silhouette_score(sample_factors, sample_labels, metric="euclidean")
            )
        else:
            score = float("nan")
            print(
                f"Warning: sample contains {n_sample_clusters} cluster(s); "
                "silhouette score not computed."
            )

        cluster_sizes = np.bincount(labels, minlength=k)

        results.append({
            "n_clusters": k,
            "silhouette_score": score,
            "smallest_cluster_size": int(cluster_sizes.min()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "cluster_size_std": float(cluster_sizes.std()),
            "zero_vector_customers": zero_rows,
            "svd_seconds": svd_seconds,
            "kmeans_fit_seconds": fit_seconds,
            "svd_components": n_components,
            "random_state": RANDOM_STATE,
            "silhouette_sample_size": sample_size,
        })

        print(f"Silhouette score: {score:.6f}")
        print(f"Smallest cluster: {cluster_sizes.min()} customers")
        print(f"Largest cluster: {cluster_sizes.max()} customers")
        print(f"KMeans time: {fit_seconds:.2f} seconds")

    results_df = pd.DataFrame(results)
    output_path = REPORTS_DIR / "segmentation_5_vs_8.csv"
    results_df.to_csv(output_path, index=False)

    print("\nComparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved report to: {output_path}")

    print(
        "\nInterpretation reminder: silhouette scores close to zero "
        "indicate weak separation. Choose a solution using both "
        "cluster quality and business interpretability."
    )


if __name__ == "__main__":
    main()