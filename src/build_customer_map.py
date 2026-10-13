
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.decomposition import TruncatedSVD


# ============================================================
# FINAL CUSTOMER MAP AND COUNTRY PROFILES
# Uses five-cluster assignments from the training data.
# No customer IDs are displayed in the map.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

SEGMENT_NAMES = {
    0: "Games, Toys & Gifts",
    1: "Bags & Storage",
    2: "Decorative Homeware",
    3: "Baking & Cake Decoration",
    4: "Broad-Mix, Lower-Frequency",
}


def main():
    print("=" * 68)
    print("CUSTOMER MAP AND COUNTRY PROFILES")
    print("=" * 68)

    # --------------------------------------------------------
    # 1. Load the training matrix and internal assignments
    # --------------------------------------------------------

    matrix_path = DATA / "customer_product_matrix.npz"
    customer_path = DATA / "customer_index.csv"
    assignments_path = DATA / "customer_cluster_assignments_internal.csv"
    train_path = DATA / "train_transactions.csv"

    A = sparse.load_npz(matrix_path).tocsr()
    customers = pd.read_csv(customer_path)
    assignments = pd.read_csv(assignments_path)

    if len(assignments) != len(customers):
        raise ValueError(
            "Assignment count does not match the customer index."
        )

    if assignments["cluster"].nunique() != 5:
        raise ValueError(
            "Expected five clusters. Run src/segment_customers.py first."
        )

    # --------------------------------------------------------
    # 2. Project customers into two dimensions
    # --------------------------------------------------------

    print("\nProjecting customers onto two SVD components...")

    map_svd = TruncatedSVD(
        n_components=2,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_SEED,
    )

    coordinates = map_svd.fit_transform(A)

    map_data = pd.DataFrame({
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "cluster": assignments["cluster"].to_numpy(),
    })

    map_data["segment_name"] = map_data["cluster"].map(SEGMENT_NAMES)

    print("Customers plotted:", len(map_data))
    print(
        "Variance captured by two map components:",
        f"{map_svd.explained_variance_ratio_.sum():.4f}",
    )

    # --------------------------------------------------------
    # 3. Create anonymous customer map
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in sorted(SEGMENT_NAMES):
        subset = map_data[map_data["cluster"] == cluster_id]

        ax.scatter(
            subset["x"],
            subset["y"],
            s=12,
            alpha=0.55,
            label=f'{cluster_id}: {SEGMENT_NAMES[cluster_id]}',
        )

    ax.set_title(
        "Customer Map: Five Exploratory Segments",
        fontsize=15,
    )
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    ax.legend(
        title="Customer segments",
        loc="best",
        fontsize=8,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    map_path = REPORTS / "final_customer_map_2d.png"
    fig.savefig(map_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 4. Summarize countries by segment
    # --------------------------------------------------------

    print("\nCalculating country profiles from training transactions...")

    train = pd.read_csv(
        train_path,
        dtype={"StockCode": "string"},
    )

    required_columns = {"Customer ID", "Country"}
    if not required_columns.issubset(train.columns):
        raise ValueError(
            "Training data must contain Customer ID and Country columns."
        )

    # Determine each customer's most frequently recorded country.
    customer_country = (
        train.dropna(subset=["Customer ID", "Country"])
        .groupby(["Customer ID", "Country"])
        .size()
        .reset_index(name="transaction_rows")
        .sort_values(
            ["Customer ID", "transaction_rows", "Country"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["Customer ID"])
    )

    country_assignments = assignments.merge(
        customer_country[["Customer ID", "Country"]],
        on="Customer ID",
        how="left",
    )

    country_counts = (
        country_assignments.dropna(subset=["Country"])
        .groupby(["cluster", "Country"])["Customer ID"]
        .nunique()
        .reset_index(name="customers")
    )

    segment_sizes = assignments.groupby("cluster").size()

    country_counts["segment_customer_count"] = (
        country_counts["cluster"].map(segment_sizes)
    )

    country_counts["segment_percentage"] = (
        country_counts["customers"]
        / country_counts["segment_customer_count"]
        * 100
    )

    country_counts["segment_name"] = (
        country_counts["cluster"].map(SEGMENT_NAMES)
    )

    country_counts = country_counts.sort_values(
        ["cluster", "customers"],
        ascending=[True, False],
    )

    # Keep the top five countries per segment.
    country_counts["country_rank"] = (
        country_counts.groupby("cluster")["customers"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top_countries = country_counts[
        country_counts["country_rank"] <= 5
    ].copy()

    # This report contains aggregate country counts, not customer IDs.
    country_path = REPORTS / "segment_top_countries.csv"
    top_countries.to_csv(country_path, index=False)

    # --------------------------------------------------------
    # 5. Display results
    # --------------------------------------------------------

    print("\nTOP COUNTRIES BY SEGMENT")
    print("=" * 68)

    for cluster_id in sorted(SEGMENT_NAMES):
        print(
            f"\nSegment {cluster_id}: "
            f"{SEGMENT_NAMES[cluster_id]}"
        )

        view = top_countries[
            top_countries["cluster"] == cluster_id
        ][
            ["Country", "customers", "segment_percentage"]
        ]

        print(view.round(2).to_string(index=False))

    print("\nSaved outputs:")
    print(map_path)
    print(country_path)
    print("\nThe customer map contains no customer IDs.")


if __name__ == "__main__":
    main()
