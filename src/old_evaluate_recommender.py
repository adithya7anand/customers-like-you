
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PHASE 2 - RECOMMENDATIONS AND HELD-OUT EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(parents=True, exist_ok=True)

K = 50
TOP_N = 10

# ------------------------------------------------------------
# 1. Load matrix, mappings and factor matrices
# ------------------------------------------------------------

A = sparse.load_npz(
    DATA / "customer_product_matrix.npz"
).tocsr()

customers = pd.read_csv(DATA / "customer_index.csv")
products = pd.read_csv(DATA / "product_index.csv")

customer_factors = np.load(
    MODELS / f"svd_customer_factors_k{K}.npy"
)

product_factors = np.load(
    MODELS / f"svd_product_factors_k{K}.npy"
)

train = pd.read_csv(
    DATA / "train_transactions.csv",
    dtype={"StockCode": "string"},
)

unseen_test = pd.read_csv(
    DATA / "test_unseen_products.csv",
    dtype={"StockCode": "string"},
)

assert A.shape == (len(customers), len(products))
assert customer_factors.shape == (len(customers), K)
assert product_factors.shape == (len(products), K)

customer_ids = customers["Customer ID"].to_numpy()
product_codes = products["StockCode"].astype(str).to_numpy()

customer_to_row = {
    int(cid): i for i, cid in enumerate(customer_ids)
}

product_to_col = {
    str(code): j for j, code in enumerate(product_codes)
}

print("Loaded matrix:", A.shape)
print("Customer factors:", customer_factors.shape)
print("Product factors:", product_factors.shape)

# ------------------------------------------------------------
# 2. Find the 10 most similar products
# ------------------------------------------------------------

P = normalize(product_factors)

similarity = cosine_similarity(P)

np.fill_diagonal(similarity, -np.inf)

similar_rows = []

for j, code in enumerate(product_codes):
    top_indices = np.argsort(
        -similarity[j]
    )[:TOP_N]

    for rank, idx in enumerate(top_indices, start=1):
        similar_rows.append({
            "product_code": code,
            "rank": rank,
            "similar_product_code": product_codes[idx],
            "cosine_similarity": float(similarity[j, idx]),
        })

similar_df = pd.DataFrame(similar_rows)

similar_df.to_csv(
    REPORTS / "similar_products_top10.csv",
    index=False,
)

print("\nSimilar-product recommendations saved.")

# ------------------------------------------------------------
# 3. Create the top-10 best-seller baseline
# ------------------------------------------------------------

# Aggregate the original training quantities by product.
# Only products in the fitted matrix are eligible.
sales = (
    train.groupby("StockCode")["Quantity"]
    .sum()
)

sales.index = sales.index.astype(str)

eligible_sales = sales.reindex(
    product_codes, fill_value=0
)

# OLD LOGIC - COMMENTED OUT, RETAINED FOR REFERENCE
# This selected only the global top 10 products.
# best_seller_indices = np.argsort(
#     -eligible_sales.to_numpy()
# )[:TOP_N]
#
# best_sellers = set(
#     product_codes[best_seller_indices]
# )

# NEW LOGIC
# Rank ALL eligible products by training-set popularity.
# Each customer will skip products already purchased and
# receive the first 10 remaining products from this ranking.
best_seller_order = np.argsort(
    -eligible_sales.to_numpy(),
    kind="stable"
)

best_seller_ranked_codes = product_codes[best_seller_order]

# ------------------------------------------------------------
# 4. Create held-out targets for eligible customers/products
# ------------------------------------------------------------

test_targets = {}

for row in unseen_test[
    ["Customer ID", "StockCode"]
].drop_duplicates().itertuples(index=False):

    cid = int(row[0])
    code = str(row[1])

    # Evaluate only customers and products represented
    # in the training matrix.
    if cid not in customer_to_row:
        continue

    if code not in product_to_col:
        continue

    test_targets.setdefault(cid, set()).add(code)

# Use only customers with at least one eligible held-out target.
eligible_customers = sorted(test_targets)

if not eligible_customers:
    raise ValueError(
        "No eligible held-out customer-product targets. "
        "Check the index mappings and test split."
    )

print("Customers evaluated:", len(eligible_customers))
print(
    "Eligible held-out customer-product pairs:",
    sum(len(v) for v in test_targets.values()),
)

# ------------------------------------------------------------
# 5. Recommend products from SVD scores
# ------------------------------------------------------------

def recommend_for_customer(customer_id, n=TOP_N):
    row = customer_to_row[customer_id]

    # Approximate reconstructed purchase scores.
    scores = customer_factors[row] @ product_factors.T

    # Do not recommend products already purchased in training.
    seen = A[row].indices
    scores[seen] = -np.inf

    top = np.argsort(-scores)[:n]
    return product_codes[top]


# ------------------------------------------------------------
# 6. Evaluate hit rate at 10 against best sellers
# ------------------------------------------------------------

svd_hits = 0
baseline_hits = 0
recommendation_rows = []

start = time.perf_counter()

for cid in eligible_customers:
    targets = test_targets[cid]

    predicted = recommend_for_customer(cid, TOP_N)

    if targets.intersection(predicted):
        svd_hits += 1

    # Best-seller baseline, excluding training purchases.
    row = customer_to_row[cid]
    seen_codes = {
        product_codes[j] for j in A[row].indices
    }

    # OLD LOGIC - COMMENTED OUT, RETAINED FOR REFERENCE
    # This considered only the global top 10 before filtering
    # products the customer had already purchased.
    #
    # baseline_recs = [
    #     code for code in product_codes[best_seller_indices]
    #     if code not in seen_codes
    # ][:TOP_N]
    #
    # if targets.intersection(baseline_recs):
    #     baseline_hits += 1

    # NEW LOGIC
    # First exclude purchased products, then take the top 10.
    baseline_recs = [
        code
        for code in best_seller_ranked_codes
        if code not in seen_codes
    ][:TOP_N]
    if len(baseline_recs) < TOP_N:
        print(
            f"Warning: Customer row {row} received "
            f"only {len(baseline_recs)} baseline recommendations"
        )

    if targets.intersection(baseline_recs):
        baseline_hits += 1

    # Preserve the existing customer recommendation output.
    for rank, code in enumerate(predicted, start=1):
        recommendation_rows.append({
            "customer_index": row,
            "rank": rank,
            "recommended_product_code": code,
        })

scoring_seconds = time.perf_counter() - start

n_eval = len(eligible_customers)

svd_hit_rate = svd_hits / n_eval
baseline_hit_rate = baseline_hits / n_eval

results = pd.DataFrame([{
    "method": "Truncated SVD",
    "components": K,
    "customers_evaluated": n_eval,
    "svd_hits_at_10": svd_hits,
    "svd_hit_rate_at_10": svd_hit_rate,
    "baseline_hits_at_10": baseline_hits,
    "baseline_hit_rate_at_10": baseline_hit_rate,
    "hit_rate_difference": svd_hit_rate - baseline_hit_rate,
    "recommendation_scoring_seconds": scoring_seconds,
}])

results.to_csv(
    REPORTS / "recommender_evaluation.csv",
    index=False,
)

pd.DataFrame(recommendation_rows).to_csv(
    REPORTS / "customer_recommendations_top10.csv",
    index=False,
)

print("\nEVALUATION RESULTS")
print("=" * 60)
print(results.to_string(index=False))

print("\nSaved:")
print(REPORTS / "recommender_evaluation.csv")
print(REPORTS / "customer_recommendations_top10.csv")
print(REPORTS / "similar_products_top10.csv")