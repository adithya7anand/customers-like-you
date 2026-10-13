from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]

MATRIX_PATH = ROOT / "data" / "processed" / "customer_product_matrix.npz"
REPORTS_DIR = ROOT / "reports"

SVD_COMPONENTS = 200
N_CLUSTERS = 8
SEEDS = [42, 21, 7, 100, 123]
BASE_SEED = 42


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading customer-product matrix...")
    matrix = load_npz(MATRIX_PATH).tocsr()
    print(f"Matrix shape: {matrix.shape}")

    n_components = min(
        SVD_COMPONENTS,
        matrix.shape[0] - 1,
        matrix.shape[1] - 1,
    )

    print(f"Fitting SVD with {n_components} components...")
    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=7,
        random_state=BASE_SEED,
    )

    factors = normalize(svd.fit_transform(matrix))

    labels_by_seed = {}
    results = []

    for seed in SEEDS:
        print(f"\nFitting KMeans with seed {seed}...")
        start = time.perf_counter()

        model = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=seed,
            n_init=10,
        )
        labels = model.fit_predict(factors)

        elapsed = time.perf_counter() - start
        labels_by_seed[seed] = labels

        sizes = np.bincount(labels, minlength=N_CLUSTERS)

        results.append({
            "seed": seed,
            "n_clusters": N_CLUSTERS,
            "smallest_cluster_size": int(sizes.min()),
            "largest_cluster_size": int(sizes.max()),
            "cluster_size_std": float(sizes.std()),
            "kmeans_fit_seconds": elapsed,
        })

        print(f"Smallest cluster: {sizes.min()}")
        print(f"Largest cluster: {sizes.max()}")
        print(f"Runtime: {elapsed:.2f} seconds")

    # Compare every seed against the baseline seed 42.
    reference_labels = labels_by_seed[BASE_SEED]

    for row in results:
        seed = row["seed"]
        row["adjusted_rand_index_vs_seed_42"] = adjusted_rand_score(
            reference_labels,
            labels_by_seed[seed],
        )

    results_df = pd.DataFrame(results)
    output_path = REPORTS_DIR / "cluster_stability_k8.csv"
    results_df.to_csv(output_path, index=False)

    print("\nStability results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved report to: {output_path}")

    print(
        "\nInterpret ARI alongside the segment profiles. "
        "Higher agreement means more stable assignments; "
        "there is no single universal ARI cutoff for a good segmentation."
    )


if __name__ == "__main__":
    main()