from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "data" / "processed" / "customer_product_matrix.npz"
REPORTS_DIR = ROOT / "reports"

COMPONENTS = [10, 20, 50, 100, 200]
RANDOM_STATE = 42


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training matrix...")
    sparse_matrix = load_npz(MATRIX_PATH).tocsr()
    n_customers, n_products = sparse_matrix.shape

    print(f"Matrix shape: {sparse_matrix.shape}")

    # PCA centres the data. Use a dense matrix only if memory permits.
    dense_bytes = n_customers * n_products * np.dtype(np.float64).itemsize
    dense_mb = dense_bytes / (1024 ** 2)

    print(f"Approximate dense matrix size: {dense_mb:.2f} MB")

    # Guard against accidentally allocating a very large dense matrix.
    if dense_mb > 1000:
        raise MemoryError(
            "Dense PCA input would exceed 1,000 MB. "
            "Do not proceed without a memory-safe PCA approach."
        )

    print("Converting training matrix to dense format for PCA...")
    matrix = sparse_matrix.toarray().astype(np.float64, copy=False)

    results = []

    for k in COMPONENTS:
        if k >= min(n_customers, n_products):
            print(f"Skipping k={k}: invalid for this matrix.")
            continue

        print(f"\nFitting PCA with k={k}...")
        start = time.perf_counter()

        pca = PCA(
            n_components=k,
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )
        customer_factors = pca.fit_transform(matrix)

        fit_seconds = time.perf_counter() - start

        # Reconstruct the training matrix to estimate relative error.
        reconstructed = pca.inverse_transform(customer_factors)

        numerator = np.linalg.norm(matrix - reconstructed, ord="fro")
        denominator = np.linalg.norm(matrix, ord="fro")
        relative_error = numerator / denominator if denominator else np.nan

        # PCA stores components and mean; factors here include scores
        # and components. Report both arrays for a fair storage estimate.
        factor_bytes = (
            customer_factors.nbytes
            + pca.components_.nbytes
            + pca.mean_.nbytes
        )

        results.append({
            "method": "PCA",
            "n_components": k,
            "fit_seconds": fit_seconds,
            "explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
            "relative_reconstruction_error": float(relative_error),
            "factor_storage_mb": factor_bytes / (1024 ** 2),
            "dense_input_mb": dense_mb,
            "random_state": RANDOM_STATE,
        })

        print(f"Fit time: {fit_seconds:.2f} seconds")
        print(
            "Explained variance ratio: "
            f"{pca.explained_variance_ratio_.sum():.6f}"
        )
        print(f"Relative reconstruction error: {relative_error:.6f}")
        print(f"Factor storage: {factor_bytes / (1024 ** 2):.2f} MB")

        del pca, customer_factors, reconstructed

    results_df = pd.DataFrame(results)
    output_path = REPORTS_DIR / "pca_experiments.csv"
    results_df.to_csv(output_path, index=False)

    print("\nPCA experiment results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
