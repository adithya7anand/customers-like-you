
# Customers Like You: PCA or SVD

## 1. Project overview

This project uses the UCI Online Retail II transaction dataset to explore customer-product patterns, compare dimensionality-reduction methods, evaluate product recommendations, and identify exploratory customer segments.

The project compares Principal Component Analysis (PCA) with Truncated Singular Value Decomposition (SVD) using component counts of 10, 20, 50, 100, and 200.

## 2. Objectives

- Profile and clean the transaction data.
- Create a chronological training/test split.
- Build a sparse customer-product matrix using training data only.
- Compare PCA and Truncated SVD reconstruction and computational performance.
- Evaluate Hit Rate@10 against a best-seller baseline.
- Create an exploratory customer segmentation with 3 to 6 groups.
- Produce an anonymous two-dimensional customer map and segment profiles.

## 3. Environment and setup

Recommended environment: Python 3.11 or another version supported by the installed dependencies.

Activate the project virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the documented dependencies:

```powershell
python -m pip install -r requirements.txt
```

The raw Online Retail II workbook is expected at:

`data/raw/online_retail_II.xlsx`

Do not commit private customer-level data, customer identifiers, credentials, or environment-specific files to a public repository.

## 4. Reproducing the pipeline

Run commands from the project root.

### Step 1: Clean data and create the chronological split

```powershell
python src/clean_data.py
```

This produces cleaned transactions, training and test transactions, held-out unseen-product targets, and cleaning/split reports.

The split holds out the latest two-month period relative to the latest cleaned transaction timestamp. Recommendation evaluation uses products that are present in the eligible product catalogue but unseen for the relevant customer in training.

### Step 2: Build the training customer-product matrix

```powershell
python build_matrix.py
```

The matrix is built from training transactions only. The pipeline aggregates customer-product quantities, applies a log transformation, and filters products with insufficient training-customer support.

### Step 3: Run dimensionality-reduction experiments

```powershell
python src/run_svd_experiments.py
python src/run_pca_experiments.py
```

Results are saved under `reports/`.

### Step 4: Evaluate recommendations

```powershell
python src/evaluate_svd_components.py
python src/evaluate_pca_components.py
python src/build_method_comparison.py
```

These scripts compare recommendations against a best-seller baseline using the same eligible test customers and held-out customer-product pairs.

### Step 5: Run customer segmentation and profiling

```powershell
python src/segment_customers.py
python src/profile_segments.py
python src/build_customer_map.py
```

The segmentation uses training-matrix SVD factors followed by K-Means. Five clusters are selected deliberately for a simpler exploratory solution. The silhouette score is low, so the clusters should not be interpreted as definitive natural customer types.

### Step 6: Run the analysis notebook

Open the project in Jupyter Notebook and run:

`notebooks/customers_like_you_analysis.ipynb`

Execute all cells from top to bottom. The notebook loads the generated reports and figures, displays the main PCA/SVD comparisons, recommendation results, and segmentation findings, and documents the project's conclusions and limitations.

Run the data preparation and experiment scripts first if the reports have not yet been generated.


## 5. Data and matrix summary

The recorded run produced the following results:
- Raw dataset: 1,067,371 rows across the two Online Retail II workbook sheets, as recorded during the initial data profile.
- Clean transactions: 802,965 rows.
- Training transactions: 684,357 rows.
- Test transactions: 118,608 rows.
- Held-out unseen-product rows: 72,924.
- Customer-product matrix: 5,483 customers by 3,690 products.
- Non-zero matrix entries: 414,008.
- Matrix sparsity: approximately 97.95%.
- Dense matrix size: approximately 154.36 MB.
- Sparse matrix storage: approximately 4.76 MB.

Counts and sizes refer to the recorded run and may vary if the source data or pipeline changes.

## 6. PCA versus Truncated SVD

At 200 components, the recorded results were:

| Metric | PCA | Truncated SVD |
|---|---:|---:|
| Explained variance ratio | 57.35% | 57.36% |
| Reconstruction error | 0.6388 | 0.6388 |
| Fit time | 1.97 s | 1.30 s |
| Factor storage | 14.03 MB | 14.00 MB |
| Hit Rate@10 | 40.04% | 39.94% |

Fit times are measurements from one local run, not benchmark guarantees. PCA uses a dense input matrix, while Truncated SVD works directly with the sparse matrix.

### Recommendation evaluation

The best-seller baseline achieved a Hit Rate@10 of 17.41%. At 200 components, PCA achieved 40.04% and Truncated SVD achieved 39.94% on the recorded evaluation.

The recommendation results show a small measured PCA advantage, while SVD offers sparse-input handling and faster measured fitting. The difference in Hit Rate@10 is small and should not be treated as conclusive.

The component-count comparison uses the same held-out evaluation data. For an unbiased final performance estimate, select the model using a separate validation set and reserve an untouched test set for final reporting.

## 7. Customer segmentation

The final exploratory solution contains five clusters:

| Cluster | Customers | Share |
|---|---:|---:|
| 0 | 997 | 18.18% |
| 1 | 675 | 12.31% |
| 2 | 1,204 | 21.96% |
| 3 | 700 | 12.77% |
| 4 | 1,907 | 34.78% |

Provisional product-based descriptions:

- Cluster 0: Games, Toys & Gifts.
- Cluster 1: Bags & Storage.
- Cluster 2: Decorative Homeware.
- Cluster 3: Baking & Cake Decoration.
- Cluster 4: Broad-Mix, Lower-Frequency Buyers.

The five-cluster silhouette score is approximately 0.0173, indicating weak separation. These labels are descriptive interpretations of product patterns, not validated customer personas.

The two-dimensional customer map captures approximately 11.47% of the variance. It is intended for exploration and does not preserve all customer similarities.

## 8. Main output files

- `reports/cleaning_log.csv`
- `reports/split_summary.csv`
- `reports/matrix_report.csv`
- `reports/pca_experiments.csv`
- `reports/svd_experiments.csv`
- `reports/pca_recommendation_comparison.csv`
- `reports/svd_recommendation_comparison.csv`
- `reports/pca_vs_svd_comparison.csv`
- `reports/segmentation_cluster_comparison.csv`
- `reports/customer_segment_summary.csv`
- `reports/segment_top_products_internal.csv`
- `reports/segment_top_countries.csv`
- `reports/final_customer_map_2d.png`

## 9. Privacy and limitations

Customer IDs are needed for internal joins and evaluation but must not appear in submitted figures, slides, or the memo. Internal files containing customer IDs should not be shared publicly.

Other limitations include weak cluster separation, information loss in the two-dimensional map, local-run timing variability, and the need for an untouched test set to establish unbiased recommendation performance.

## 10. Contributions

**Contributor: Adi Anand**

- Data preparation, cleaning, chronological train/test split, and customer-product matrix construction.
- PCA and Truncated SVD experiments, reconstruction analysis, and computational performance comparison.
- Recommendation evaluation using Hit Rate@10 against the best-seller baseline.
- Customer segmentation, segment profiling, and customer-map generation.
- Analysis notebook, project documentation, interpretation of findings, and preparation of the memo and presentation.

## 11. Reproducibility

The main experiments use fixed random seeds where configured. Run the commands from the project root and retain the generated CSV reports and figures.

The recorded output files are results from the current local run. Re-running the pipeline may change timings and, depending on the environment, numerical results.
