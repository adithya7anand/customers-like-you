
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# ============================================================
# PHASE 2 - TRUNCATED SVD EXPERIMENTS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

MATRIX_FILE = DATA_DIR / "customer_product_matrix.npz"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_COUNTS = [10, 20, 50, 100, 200]
RANDOM_SEED = 42

# ------------------------------------------------------------
# 1. Load the validated sparse training matrix
# ------------------------------------------------------------

if not MATRIX_FILE.exists():
    raise FileNotFoundError(
        f"Matrix not found: {MATRIX_FILE}"
    )

A = sparse.load_npz(MATRIX_FILE).tocsr()

n_customers, n_products = A.shape

print("=" * 65)
print("TRUNCATED SVD EXPERIMENTS")
print("=" * 65)
print("Matrix shape:", A.shape)
print("Non-zero entries:", A.nnz)
print("Components to test:", COMPONENT_COUNTS)

if max(COMPONENT_COUNTS) >= min(A.shape):
    raise ValueError(
        "Component count must be smaller than both matrix dimensions."
    )

# Frobenius norm squared, without densifying the matrix.
matrix_norm_squared = float(A.multiply(A).sum())
matrix_norm = np.sqrt(matrix_norm_squared)

results = []
models = {}

# ------------------------------------------------------------
# 2. Fit each component count
# ------------------------------------------------------------

for k in COMPONENT_COUNTS:
    print(f"\nFitting TruncatedSVD with k={k}...")

    model = TruncatedSVD(
        n_components=k,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_SEED,
    )

    start = time.perf_counter()

    # Customer factors: one row per customer, k dimensions.
    customer_factors = model.fit_transform(A)

    fit_seconds = time.perf_counter() - start

    # Projected energy. TruncatedSVD components are approximately
    # orthonormal, so this gives an efficient residual estimate.
    projected_energy = float(
        np.square(customer_factors).sum()
    )

    residual_energy = max(
        matrix_norm_squared - projected_energy,
        0.0,
    )

    relative_reconstruction_error = (
        np.sqrt(residual_energy) / matrix_norm
        if matrix_norm > 0
        else 0.0
    )

    # Store both customer and product factors.
    product_factors = model.components_.T

    customer_factor_bytes = customer_factors.nbytes
    product_factor_bytes = product_factors.nbytes
    factor_bytes = customer_factor_bytes + product_factor_bytes

    result = {
        "components": k,
        "fit_time_seconds": round(fit_seconds, 3),
        "explained_variance_ratio": round(
            float(model.explained_variance_ratio_.sum()), 6
        ),
        "relative_reconstruction_error_estimate": round(
            float(relative_reconstruction_error), 6
        ),
        "customer_factor_mb": round(
            customer_factor_bytes / (1024**2), 3
        ),
        "product_factor_mb": round(
            product_factor_bytes / (1024**2), 3
        ),
        "combined_factor_mb": round(
            factor_bytes / (1024**2), 3
        ),
    }

    results.append(result)
    models[k] = model

    print("Fit time:", result["fit_time_seconds"], "seconds")
    print(
        "Explained variance ratio:",
        result["explained_variance_ratio"],
    )
    print(
        "Relative reconstruction error estimate:",
        result["relative_reconstruction_error_estimate"],
    )
    print("Combined factor size:", result["combined_factor_mb"], "MB")

# ------------------------------------------------------------
# 3. Save experiment results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_file = REPORT_DIR / "svd_experiments.csv"
results_df.to_csv(results_file, index=False)

print("\nExperiment summary:")
print(results_df.to_string(index=False))

# ------------------------------------------------------------
# 4. Plot reconstruction error versus components
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(
    results_df["components"],
    results_df["relative_reconstruction_error_estimate"],
    marker="o",
)
plt.xlabel("Number of components (k)")
plt.ylabel("Estimated relative reconstruction error")
plt.title("Truncated SVD: Reconstruction Error vs Components")
plt.grid(True)
plt.tight_layout()

plot_file = REPORT_DIR / "svd_reconstruction_error.png"
plt.savefig(plot_file, dpi=150)
plt.close()

# ------------------------------------------------------------
# 5. Plot fit time versus components
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(
    results_df["components"],
    results_df["fit_time_seconds"],
    marker="o",
)
plt.xlabel("Number of components (k)")
plt.ylabel("Fit time (seconds)")
plt.title("Truncated SVD: Fit Time vs Components")
plt.grid(True)
plt.tight_layout()

time_plot_file = REPORT_DIR / "svd_fit_time.png"
plt.savefig(time_plot_file, dpi=150)
plt.close()

# ------------------------------------------------------------
# 6. Save the k=50 model for the next stage
# ------------------------------------------------------------

# Save factors rather than using pickle. The next stage can load
# these arrays and the matching index files.
selected_k = 50
selected_model = models[selected_k]

np.save(
    MODEL_DIR / "svd_customer_factors_k50.npy",
    selected_model.transform(A),
)
np.save(
    MODEL_DIR / "svd_product_factors_k50.npy",
    selected_model.components_.T,
)

print("\nSaved:")
print(results_file)
print(plot_file)
print(time_plot_file)
print(MODEL_DIR / "svd_customer_factors_k50.npy")
print(MODEL_DIR / "svd_product_factors_k50.npy")

print("\nSVD experiments completed.")