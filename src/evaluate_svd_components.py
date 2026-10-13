
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# ============================================================
# EVALUATE SVD COMPONENT COUNTS AGAINST BEST-SELLER BASELINE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

COMPONENTS_TO_TEST = [10, 20, 50, 100, 200]
TOP_N = 10
RANDOM_SEED = 42
N_ITER = 7

# ------------------------------------------------------------
# 1. Load training matrix and mappings
# ------------------------------------------------------------

A = sparse.load_npz(
    DATA / "customer_product_matrix.npz"
).tocsr()

customers = pd.read_csv(DATA / "customer_index.csv")
products = pd.read_csv(DATA / "product_index.csv")

train = pd.read_csv(
    DATA / "train_transactions.csv",
    dtype={"StockCode": "string"},
)

unseen_test = pd.read_csv(
    DATA / "test_unseen_products.csv",
    dtype={"StockCode": "string"},
)

customer_ids = customers["Customer ID"].to_numpy()
product_codes = products["StockCode"].astype(str).to_numpy()

customer_to_row = {
    int(cid): i for i, cid in enumerate(customer_ids)
}

product_to_col = {
    str(code): j for j, code in enumerate(product_codes)
}

assert A.shape == (len(customers), len(products))

print("=" * 68)
print("SVD COMPONENT COMPARISON")
print("=" * 68)
print("Training matrix:", A.shape)
print("Components:", COMPONENTS_TO_TEST)

# ------------------------------------------------------------
# 2. Build held-out targets
# ------------------------------------------------------------

test_targets = {}

for item in unseen_test[
    ["Customer ID", "StockCode"]
].drop_duplicates().itertuples(index=False):

    cid = int(item[0])
    code = str(item[1])

    if cid not in customer_to_row:
        continue

    if code not in product_to_col:
        continue

    test_targets.setdefault(cid, set()).add(code)

eligible_customers = sorted(test_targets)

if not eligible_customers:
    raise ValueError("No eligible held-out test targets found.")

print("Customers evaluated:", len(eligible_customers))
print(
    "Eligible held-out pairs:",
    sum(len(targets) for targets in test_targets.values()),
)

# ------------------------------------------------------------
# 3. Corrected best-seller baseline
# ------------------------------------------------------------

sales = train.groupby("StockCode")["Quantity"].sum()
sales.index = sales.index.astype(str)

eligible_sales = sales.reindex(product_codes, fill_value=0)

# Rank every eligible product, not just the global top 10.
best_seller_order = np.argsort(
    -eligible_sales.to_numpy(),
    kind="stable",
)

best_seller_ranked_codes = product_codes[best_seller_order]

baseline_hits = 0
baseline_recommendation_count = 0

for cid in eligible_customers:
    row = customer_to_row[cid]

    seen_codes = {
        product_codes[j] for j in A[row].indices
    }

    baseline_recs = [
        code
        for code in best_seller_ranked_codes
        if code not in seen_codes
    ][:TOP_N]

    baseline_recommendation_count += len(baseline_recs)

    if test_targets[cid].intersection(baseline_recs):
        baseline_hits += 1

n_eval = len(eligible_customers)
baseline_hit_rate = baseline_hits / n_eval

print("\nCorrected baseline")
print("Customers with a hit:", baseline_hits)
print(f"Hit Rate@10: {baseline_hit_rate:.4f}")
print(
    "Average recommendations per customer:",
    f"{baseline_recommendation_count / n_eval:.2f}",
)

# ------------------------------------------------------------
# 4. Fit and evaluate each SVD model
# ------------------------------------------------------------

results = []

for k in COMPONENTS_TO_TEST:
    print("\n" + "-" * 60)
    print(f"Evaluating SVD with {k} components")

    model = TruncatedSVD(
        n_components=k,
        algorithm="randomized",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
    )

    fit_start = time.perf_counter()
    customer_factors = model.fit_transform(A)
    fit_seconds = time.perf_counter() - fit_start

    # Product factors are the transposed SVD components.
    product_factors = model.components_.T

    hits = 0
    recommendation_count = 0

    scoring_start = time.perf_counter()

    for cid in eligible_customers:
        row = customer_to_row[cid]

        # Approximate reconstructed purchase scores.
        scores = customer_factors[row] @ product_factors.T

        # Exclude products already purchased in training.
        seen = A[row].indices
        scores[seen] = -np.inf

        top_indices = np.argsort(-scores)[:TOP_N]
        recommended_codes = product_codes[top_indices]

        recommendation_count += len(recommended_codes)

        if test_targets[cid].intersection(recommended_codes):
            hits += 1

    scoring_seconds = time.perf_counter() - scoring_start
    hit_rate = hits / n_eval

    results.append({
        "method": "Truncated SVD",
        "components": k,
        "customers_evaluated": n_eval,
        "svd_hits_at_10": hits,
        "svd_hit_rate_at_10": hit_rate,
        "baseline_hits_at_10": baseline_hits,
        "baseline_hit_rate_at_10": baseline_hit_rate,
        "hit_rate_difference": hit_rate - baseline_hit_rate,
        "fit_time_seconds": fit_seconds,
        "recommendation_scoring_seconds": scoring_seconds,
        "average_recommendations_per_customer":
            recommendation_count / n_eval,
        "explained_variance_ratio":
            float(model.explained_variance_ratio_.sum()),
    })

    print(f"Fit time: {fit_seconds:.3f} seconds")
    print(f"Explained variance: {model.explained_variance_ratio_.sum():.4f}")
    print(f"Hits@10: {hits} / {n_eval}")
    print(f"Hit Rate@10: {hit_rate:.4f}")
    print(f"Baseline Hit Rate@10: {baseline_hit_rate:.4f}")

# ------------------------------------------------------------
# 5. Save and display comparison
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

output_path = REPORTS / "svd_recommendation_comparison.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 68)
print("FINAL COMPARISON")
print("=" * 68)

print(
    results_df[
        [
            "components",
            "svd_hits_at_10",
            "svd_hit_rate_at_10",
            "baseline_hit_rate_at_10",
            "hit_rate_difference",
            "fit_time_seconds",
            "explained_variance_ratio",
        ]
    ].to_string(index=False)
)

print("\nSaved:", output_path)
