
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def main():
    print("Loading PCA and SVD experiment results...")

    pca_exp = pd.read_csv(REPORTS / "pca_experiments.csv")
    svd_exp = pd.read_csv(REPORTS / "svd_experiments.csv")

    pca_rec = pd.read_csv(REPORTS / "pca_recommendation_comparison.csv")
    svd_rec = pd.read_csv(REPORTS / "svd_recommendation_comparison.csv")

    # Standardize the component-count column.
    for df in [pca_exp, svd_exp]:
        if "components" not in df.columns:
            if "n_components" in df.columns:
                df.rename(columns={"n_components": "components"}, inplace=True)
            else:
                raise ValueError("Could not find component count in experiment results.")

    # Select and rename PCA experiment metrics.
    pca_metrics = pca_exp[
        [
            "components",
            "fit_seconds",
            "explained_variance_ratio",
            "relative_reconstruction_error",
            "factor_storage_mb",
            "dense_input_mb",
        ]
    ].rename(columns={
        "fit_seconds": "pca_fit_seconds",
        "explained_variance_ratio": "pca_explained_variance",
        "relative_reconstruction_error": "pca_reconstruction_error",
        "factor_storage_mb": "pca_factor_storage_mb",
    })

    # SVD experiment files may use a different reconstruction-error
    # column name, so detect the available column.
    svd_error_candidates = [
        "relative_reconstruction_error",
        "relative_reconstruction_error_estimate",
        "reconstruction_error_estimate",
        "reconstruction_error",
    ]
    svd_error_column = next(
        (col for col in svd_error_candidates if col in svd_exp.columns),
        None,
    )

    if svd_error_column is None:
        raise ValueError(
            "Could not find the SVD reconstruction-error column. "
            f"Available columns: {list(svd_exp.columns)}"
        )

    svd_metrics = svd_exp[
        [
            "components",
            "fit_time_seconds",
            "explained_variance_ratio",
            svd_error_column,
            "combined_factor_mb",
        ]
    ].rename(columns={
        "fit_time_seconds": "svd_fit_seconds",
        "explained_variance_ratio": "svd_explained_variance",
        svd_error_column: "svd_reconstruction_error",
        "combined_factor_mb": "svd_factor_storage_mb",
    })

    # Recommendation results.
    pca_recommendations = pca_rec[
        [
            "components",
            "pca_hit_rate_at_10",
            "baseline_hit_rate_at_10",
        ]
    ].rename(columns={
        "pca_hit_rate_at_10": "pca_hit_rate_at_10",
    })

    svd_recommendations = svd_rec[
        [
            "components",
            "svd_hit_rate_at_10",
        ]
    ]

    # Merge all metrics by component count.
    comparison = pca_metrics.merge(
        svd_metrics, on="components", how="inner"
    )

    comparison = comparison.merge(
        pca_recommendations, on="components", how="inner"
    )

    comparison = comparison.merge(
        svd_recommendations, on="components", how="inner"
    )

    comparison["pca_minus_svd_hit_rate"] = (
        comparison["pca_hit_rate_at_10"]
        - comparison["svd_hit_rate_at_10"]
    )

    comparison["baseline_hit_rate_at_10"] = (
        comparison["baseline_hit_rate_at_10"]
    )

    comparison["pca_improvement_over_baseline"] = (
        comparison["pca_hit_rate_at_10"]
        - comparison["baseline_hit_rate_at_10"]
    )

    comparison["svd_improvement_over_baseline"] = (
        comparison["svd_hit_rate_at_10"]
        - comparison["baseline_hit_rate_at_10"]
    )

    comparison = comparison.sort_values("components")

    output_path = REPORTS / "pca_vs_svd_comparison.csv"
    comparison.to_csv(output_path, index=False)

    display_columns = [
        "components",
        "pca_hit_rate_at_10",
        "svd_hit_rate_at_10",
        "baseline_hit_rate_at_10",
        "pca_fit_seconds",
        "svd_fit_seconds",
        "pca_explained_variance",
        "svd_explained_variance",
        "pca_reconstruction_error",
        "svd_reconstruction_error",
        "pca_factor_storage_mb",
        "svd_factor_storage_mb",
    ]

    print("\nPCA VS SVD COMPARISON")
    print("=" * 100)
    print(
        comparison[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
