# Credit Demand in Afghan Firms: An OOP Data Analysis Pipeline

## Dataset Description

This project analyzes the **World Bank Enterprise Survey for Afghanistan** (2025),
covering 426 firms across multiple sectors and regions. The dataset captures firm-level
information on credit access, financial status, ownership structure, innovation,
workforce composition, and performance metrics.

The analysis investigates **what drives credit demand** among Afghan firms; specifically,
what distinguishes firms that express a need for credit from those that do not.

The target variable is constructed from the survey's finance questions:
- **No need (0):** Firms that do not need credit (219 firms)
- **Demand (1):** Firms that sought credit — whether discouraged, approved, rejected,
  or still in process (179 firms)
- **Excluded:** 28 firms with ambiguous status (withdrawn, unknown outcome)

## OOP Concepts Used
- **Encapsulation:** Functionality organized into DataLoader, DataPreprocessor,
  Analyzer, ModelTrainer, and Visualizer classes
- **Abstraction:** main.py executes the pipeline through high-level method calls
  without exposing implementation details
- **Static Methods:** Used where behavior belongs to the class rather than an instance
  (e.g., path validation and sector collapsing in DataPreprocessor)
- **Modular Design:** Each component has a single responsibility and can be tested
  independently

## Pipeline Explanation

The project follows a modular OOP architecture with six Python modules:

### `src/data_loader.py` — DataLoader
Dynamically discovers and loads CSV and XLSX files from the `data/` directory
using `os` and `glob`. Handles missing files gracefully with error logging. Merges
additional variables from the full survey dataset (`output.csv`) and loads variable/value
label mappings from `var_labels.csv` and `value_labels_by_variable.csv` to dynamically
decode coded survey variables.

### `src/utils.py` — Constants & Helpers
Stores sentinel value definitions (-9, -7 for missing; -8 for refusal in specific
columns), column groupings (finance, firm, performance, innovation, workforce), and
helper functions for label mapping. Acts as the shared configuration for the pipeline.

### `src/preprocessing.py` — DataPreprocessor
Cleans the raw data in four steps using pandas and numpy:
1. **Sentinel recoding:** Replaces -9 and -7 with NaN across all numeric columns;
   -8 only in column k11
2. **Type conversion:** 30 categorical columns to `category` dtype, 29 numeric validated
3. **Feature engineering:** Creates `firm_age`, log-transforms for skewed variables
   (d2, n3, a6c, l4a1, l4a2, l4b), `credit_demand` target from `finance_status`,
   and `sector_broad_3` collapsed from ISIC codes
4. **Missing value treatment:** Columns with >20% missingness are dropped (14 columns),
   then median imputation for remaining numeric and explicit "Missing" category for
   categorical variables

Also includes a diagnostic `detect_outliers_iqr()` method that flags outliers using the
1.5 × IQR rule. The pipeline detected 573 outlier flags across 221 unique rows, with
raw sales (d2, n3) and workforce variables (l4a1, l4a2, l4b) showing the most outliers.

### `src/analysis.py` — Analyzer & ModelTrainer
**Analyzer** performs EDA: summary statistics (excluding raw pre-log variables, sampling
weights, and year variables), distribution/skewness analysis, correlation matrix, and
group comparisons (by region, firm size, sector, legal status). Includes cross-tabulations
of financing sources and sales by firm size and region. Automatically collects key insights.

**ModelTrainer** implements a binary classification (credit demand vs. no need) using
XGBoost with a 64-combination grid search over 6 hyperparameters (eta, max_depth,
min_child_weight, subsample, colsample_bytree, gamma) with early stopping
(early_stopping_rounds=30, n_estimators=500). Evaluated with 5-fold stratified
cross-validation and AUC-ROC scoring. Uses `drop_first=False` for one-hot encoding
to keep all dummy categories visible in feature importance. Feature labels are loaded
dynamically from the label files rather than hardcoded.

### `src/visualization.py` — Visualizer
Generates nine publication-quality plots using matplotlib and seaborn, saved to `outputs/`:
1. Distribution histograms with KDE for key numeric variables
2. Correlation heatmap of log-transformed variables
3. Credit demand rate by region
4. Credit demand rate by firm size
5. Finance status category breakdown
6. Sales distribution comparison (demand vs. no need)
7. Credit demand rate by sector
8. Working capital financing sources by demand status (grouped bar)
9. XGBoost feature importance (top 10 predictors with labels)

### `src/main.py` — Pipeline Orchestrator
Ties all modules together: load → preprocess → outlier detection → EDA → model → visualize.
Includes an **argparse CLI interface** for controlling pipeline execution:

```bash
python3 src/main.py                          # default paths
python3 src/main.py --data data --output outputs  # custom paths
```

## How to Run

```bash
# 1. Clone or download the project
cd opp_project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python3 src/main.py

# With custom paths:
python3 src/main.py --data path/to/data --output path/to/outputs

# Or run individual modules:
python3 src/data_loader.py
python3 src/preprocessing.py
python3 src/analysis.py
python3 src/visualization.py
```

## Key Results

### EDA Insights
- **Skewness:** 12 variables are highly skewed (|skew| > 2), justifying
  log-transformation (e.g. raw sales skewness 17.6 → log sales 0.35)
- **Outliers:** IQR detection flagged 573 outlier values across 221 rows,
  concentrated in raw sales and workforce variables
- **Correlation:** Notable correlations include k3a ~ k3i (r = -0.81),
  indicating firms substitute between internal funds and owners' equity.
  XGBoost handles correlated features well since it selects the best split
  independently at each node, so both variables are retained
- **Regional disparity:** East & Southeast has the highest credit demand
  rate (58.5%), North the lowest (40.7%)
- **Sector:** Industry firms demand credit at 50%, nearly double the rate
  of Services (30.2%) and Trade (32.8%)
- **Firm size:** Small firms have the highest demand rate (46.2%),
  large firms the lowest (39.1%)
- **Financing structure:** Demand firms rely less on internal funds
  (76.6% vs 87.9%) and more on external sources like supplier credit
  and non-bank finance.

### Why XGBoost?

XGBoost was chosen because it captures nonlinear interactions between features
that linear models miss, includes built-in regularization to prevent overfitting
on small samples (n=398), and supports grid search with early stopping for
systematic hyperparameter optimization. It is also the same model used in the
author's thesis, allowing direct comparison of results between R and Python
implementations.

### ML Results

| Metric | Value |
|--------|-------|
| Model | XGBoost with 64-combination grid search + early stopping |
| Task | Credit Demand vs. No Need |
| Sample size | 398 firms (219 no need, 179 demand) |
| Features | 55 (after one-hot encoding with drop_first=False) |
| Baseline rate | 0.550 |
| **XGBoost CV AUC** | **0.724 (+/- 0.016)** |
| **Beats baseline?** | **YES** |
| Best parameters | eta=0.01, max_depth=3, min_child_weight=1, subsample=0.7, colsample_bytree=0.7, gamma=0.1 |
| Best trees | Selected via early stopping (mean across folds) |

The top predictors of credit demand are working capital financing variables:
internal funds (k3a), supplier credit (k3f), owners' equity (k3i), and
non-bank finance (k3e) — four of the top six features. R&D spending (h8),
Industry sector, East & Southeast region, and log sales also appear in the
top 10. These findings are consistent with the author's thesis, which achieved
CV AUC 0.710 on the same robustness model using R and XGBoost.

## Project Structure

```
credit-demand-pipeline/
├── data/                     # place downloaded dataset files here
├── outputs/                  # generated visualizations
│   ├── 01_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_demand_by_region.png
│   ├── 04_demand_by_size.png
│   ├── 05_finance_status.png
│   ├── 06_sales_by_demand.png
│   ├── 07_demand_by_sector.png
│   ├── 08_financing_sources.png
│   └── 09_feature_importance.png
├── src/
│   ├── data_loader.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── visualization.py
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Data

The original Enterprise Survey data files are **not included** in this repository.

This project is based on the **World Bank Enterprise Surveys**. The dataset has been excluded from this repository to comply with the dataset's terms of use and redistribution policy.

To run the project, download the required dataset from the World Bank Enterprise Surveys website and place the following files in the `data/` directory:

- `output.csv`
- `survey_reduced_df.csv`
- `var_labels.csv`
- `value_labels_by_variable.csv`

Dataset:
https://www.enterprisesurveys.org