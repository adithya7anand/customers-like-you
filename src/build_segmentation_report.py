from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

SEGMENT_SUMMARY = REPORTS_DIR / "segment_profile_summary.csv"
CLUSTER_COMPARISON = REPORTS_DIR / "segmentation_5_vs_8.csv"
STABILITY_REPORT = REPORTS_DIR / "cluster_stability_k8.csv"

OUTPUT_CSV = REPORTS_DIR / "final_segmentation_summary.csv"
OUTPUT_SIZES_PNG = REPORTS_DIR / "final_segment_sizes.png"
OUTPUT_SPEND_PNG = REPORTS_DIR / "final_segment_median_spend.png"


def main():
    required_files = [
        SEGMENT_SUMMARY,
        CLUSTER_COMPARISON,
        STABILITY_REPORT,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required report not found: {path}\n"
                "Run the relevant segmentation scripts first."
            )

    # --------------------------------------------------
    # 2. Load existing reports
    # --------------------------------------------------
    segments = pd.read_csv(SEGMENT_SUMMARY)
    comparison = pd.read_csv(CLUSTER_COMPARISON)
    stability = pd.read_csv(STABILITY_REPORT)

    print("Loaded segmentation reports.")
    print(f"Segment profile rows: {len(segments)}")

    # --------------------------------------------------
    # 3. Identify key comparison results
    # --------------------------------------------------
    comparison_5 = comparison.loc[
        comparison["n_clusters"] == 5
    ]
    comparison_8 = comparison.loc[
        comparison["n_clusters"] == 8
    ]

    if comparison_5.empty or comparison_8.empty:
        raise ValueError(
            "The five-versus-eight-cluster comparison is incomplete."
        )

    score_5 = float(comparison_5.iloc[0]["silhouette_score"])
    score_8 = float(comparison_8.iloc[0]["silhouette_score"])

    non_reference_stability = stability.loc[
        stability["seed"] != 42,
        "adjusted_rand_index_vs_seed_42",
    ].dropna()

    if non_reference_stability.empty:
        raise ValueError(
            "No alternative-seed stability results were found."
        )

    # --------------------------------------------------
    # 4. Add presentation-friendly metrics
    # --------------------------------------------------
    # Use existing column names where possible.
    # Do not include customer IDs in this report.

    segments = segments.copy()

    print("\nAvailable segment profile columns:")
    print(segments.columns.tolist())

    # --------------------------------------------------
    # 5. Save a compact validation summary
    # --------------------------------------------------
    validation_summary = pd.DataFrame([
        {
            "analysis": "Five-cluster solution",
            "metric": "Silhouette score",
            "value": score_5,
            "interpretation": (
                "Low separation; exploratory use only"
            ),
        },
        {
            "analysis": "Eight-cluster solution",
            "metric": "Silhouette score",
            "value": score_8,
            "interpretation": (
                "Low separation; exploratory use only"
            ),
        },
        {
            "analysis": "Eight-cluster stability",
            "metric": "Minimum alternative-seed ARI",
            "value": float(non_reference_stability.min()),
            "interpretation": (
                "Shows agreement with seed 42; some assignments vary"
            ),
        },
        {
            "analysis": "Eight-cluster stability",
            "metric": "Maximum alternative-seed ARI",
            "value": float(non_reference_stability.max()),
            "interpretation": (
                "Shows agreement with seed 42; not proof of business validity"
            ),
        },
    ])

    validation_path = REPORTS_DIR / "segmentation_validation_summary.csv"
    validation_summary.to_csv(validation_path, index=False)

    # --------------------------------------------------
    # 6. Create a presentation-ready segment profile CSV
    # --------------------------------------------------
    # Exclude any identifier columns as a precaution.
    private_columns = [
        column for column in segments.columns
        if "customer_id" in column.lower()
        or column.lower() in {"customer id", "customerid"}
    ]

    public_segments = segments.drop(
        columns=private_columns,
        errors="ignore",
    )

    public_segments.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------
    # 7. Discover available metric columns
    # --------------------------------------------------
    def find_column(candidates):
        for candidate in candidates:
            if candidate in public_segments.columns:
                return candidate
        return None

    count_col = find_column([
        "customer_count",
        "customers",
        "n_customers",
        "size",
    ])

    spend_col = find_column([
        "median_spend",
        "median_total_spend",
        "median_spend_per_customer",
    ])

    label_col = find_column([
        "cluster",
        "cluster_id",
        "segment",
        "segment_id",
    ])

    # --------------------------------------------------
    # 8. Plot customer counts, if available
    # --------------------------------------------------
    if count_col and label_col:
        plot_data = public_segments.sort_values(label_col)

        plt.figure(figsize=(9, 5))
        plt.bar(
            plot_data[label_col].astype(str),
            plot_data[count_col],
        )
        plt.title("Customer Count by Exploratory Segment")
        plt.xlabel("Segment")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.savefig(OUTPUT_SIZES_PNG, dpi=200)
        plt.close()

        print(f"Saved: {OUTPUT_SIZES_PNG}")
    else:
        print(
            "\nCustomer-count chart skipped because expected "
            "column names were not found."
        )

    # --------------------------------------------------
    # 9. Plot median spend, if available
    # --------------------------------------------------
    if spend_col and label_col:
        plot_data = public_segments.sort_values(label_col)

        plt.figure(figsize=(9, 5))
        plt.bar(
            plot_data[label_col].astype(str),
            plot_data[spend_col],
        )
        plt.title("Median Customer Spend by Exploratory Segment")
        plt.xlabel("Segment")
        plt.ylabel("Median Spend")
        plt.tight_layout()
        plt.savefig(OUTPUT_SPEND_PNG, dpi=200)
        plt.close()

        print(f"Saved: {OUTPUT_SPEND_PNG}")
    else:
        print(
            "\nMedian-spend chart skipped because expected "
            "column names were not found."
        )

    # --------------------------------------------------
    # 10. Print findings
    # --------------------------------------------------
    print("\nSegmentation validation summary")
    print(validation_summary.to_string(index=False))

    print("\nFinal segment summary")
    print(public_segments.to_string(index=False))

    print(f"\nSaved segment summary: {OUTPUT_CSV}")
    print(f"Saved validation summary: {validation_path}")

    print(
        "\nNote: low silhouette scores mean that the segments "
        "should be described as exploratory purchasing profiles."
    )


if __name__ == "__main__":
    main()