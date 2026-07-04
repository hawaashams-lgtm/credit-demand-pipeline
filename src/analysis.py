"""
analysis.py
-----------
Exploratory Data Analysis and Machine Learning for the
Credit Demand pipeline. Contains:
  - Analyzer:     distribution, correlation, group comparisons
  - ModelTrainer: Stage 1 classification with XGBoost
"""

import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils import (
    FINANCE_COLS, FIRM_COLS, PERFORMANCE_COLS,
    INNOVATION_COLS, WORKFORCE_COLS, CATEGORICAL_COLS, ID_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# EDA
# ======================================================================
class Analyzer:
    """
    Performs Exploratory Data Analysis on the cleaned dataset.
    Produces summary statistics, distribution analysis,
    correlation matrix, and group-based comparisons.
    """

    def __init__(self):
        self.insights = []

    def run(self, df: pd.DataFrame) -> None:
        """Execute the full EDA pipeline."""
        logger.info("=" * 60)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 60)

        self.summary_statistics(df)
        self.distribution_analysis(df)
        self.correlation_analysis(df)
        self.group_comparisons(df)
        self.print_insights()

    def summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Print and return descriptive statistics."""
        logger.info("\n--- Summary Statistics ---")
        print(f"\nDataset shape: {df.shape}")

        print("\n[Target] finance_status distribution:")
        print(df["finance_status"].value_counts())


        print("\n[Target] Credit Demand (Stage 1):")
        class_0 = ["no_need"]
        class_1 = ["discouraged", "approved_full", "approved_partial",
                   "rejected", "in_process"]
        demand_counts = df["finance_status"].isin(class_1).sum()
        no_need_counts = df["finance_status"].isin(class_0).sum()
        other_counts = len(df) - demand_counts - no_need_counts
        print(f"  no_need:  {no_need_counts}")
        print(f"  demand:   {demand_counts}")
        print(f"  excluded: {other_counts}")

        exclude = set(ID_COLS) | {"wmedian", "wweak", "b5", "b6b", "d8",
                                  "d2", "n3", "a6c", "l4a1", "l4a2", "l4b"}
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df[
            [c for c in numeric_df.columns
             if c not in exclude and not c.startswith("credit_")]
        ]
        desc = numeric_df.describe().T
        desc["missing"] = df.shape[0] - numeric_df.count()
        print(f"\nNumeric variables: {len(numeric_df.columns)}")
        print(desc[["count", "mean", "std", "min", "50%", "max", "missing"]])

        return desc

    def distribution_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze distributions of key variables."""
        logger.info("\n--- Distribution Analysis ---")

        exclude = set(ID_COLS) | {"wmedian", "wweak", "b5", "b6b", "d8",
                                  "d2", "n3", "a6c", "l4a1", "l4a2", "l4b"}
        skew_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in exclude and not c.startswith("credit_")]

        skewness = df[skew_cols].skew().sort_values(ascending=False)
        print("\nSkewness of key numeric variables:")
        print(skewness.to_string())

        highly_skewed = skewness[skewness.abs() > 2]
        if len(highly_skewed) > 0:
            self.insights.append(
                f"INSIGHT 1 — Skewness: {len(highly_skewed)} variables are "
                f"highly skewed (|skew| > 2): {list(highly_skewed.index)}. "
                f"Log-transformation is applied to reduce this."
            )

        print("\nFirm size distribution (a6a):")
        print(df["a6a"].value_counts().sort_index())

        print("\nRegion distribution (a3a):")
        print(df["a3a"].value_counts().sort_index())

        return skewness.to_dict()

    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for key numeric variables."""
        logger.info("\n--- Correlation Analysis ---")

        exclude = set(ID_COLS) | {"wmedian", "wweak", "b5", "b6b", "d8",
                                  "d2", "n3", "a6c", "l4a1", "l4a2", "l4b"}
        corr_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in exclude and not c.startswith("credit_")]

        corr_matrix = df[corr_cols].corr()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        strong = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "var1", "level_1": "var2", 0: "corr"})
        )
        strong["abs_corr"] = strong["corr"].abs()
        strong = strong.sort_values("abs_corr", ascending=False)

        print("\nTop 10 pairwise correlations:")
        print(strong.head(10).to_string(index=False))

        top_pair = strong.iloc[0]
        self.insights.append(
            f"INSIGHT 2 — Correlation: The strongest numeric correlation is "
            f"between {top_pair['var1']} and {top_pair['var2']} "
            f"(r = {top_pair['corr']:.3f}). XGBoost handles correlated "
            f"features well, so both are retained."
        )

        return corr_matrix

    def group_comparisons(self, df: pd.DataFrame) -> None:
        """Compare credit demand across groups."""
        logger.info("\n--- Group Comparisons ---")
        df_demand = df[df["credit_demand"].notna()].copy()

        print("\n[1] Credit demand vs. no need - median sales:")
        group_sales = (
            df_demand.groupby("credit_demand")["log_d2"]
            .agg(["median", "mean", "count"])
        )
        print(group_sales)

        print("\n[2] Credit demand rate by firm size (a6a):")
        size_rates = (
            df_demand.groupby("a6a")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "demand_rate"})
        )
        print(size_rates)

        print("\n[3] Credit demand rate by region (a3a):")
        region_rates = (
            df_demand.groupby("a3a")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "demand_rate"})
        )
        print(region_rates)


        print("\n[4] Credit demand rate by sector (sector_broad_3):")
        sector_rates = (
            df_demand.groupby("sector_broad_3")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "demand_rate"})
        )
        print(sector_rates)

        print("\n[5] Financing sources by credit demand and firm size:")
        finance_by_size = (
            df_demand
            .groupby(["a6a", "credit_demand"])[["k3a", "k3i", "k3f", "k3e"]]
            .mean()
            .round(1)
        )

        print(finance_by_size)

        print("\n[6] Sales by firm size and credit demand:")
        sales_by_size = (
            df_demand
            .groupby(["a6a", "credit_demand"])["log_d2"]
            .agg(["median", "mean", "count"])
            .round(2)
        )

        print(sales_by_size)

        print("\n[7] Credit demand rate by region and firm size:")
        region_size_rates = (
            df_demand
            .groupby(["a3a", "a6a"])["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "demand_rate"})
        )

        print("\n[8] Financing sources by region and credit demand:")
        finance_by_region = (
            df_demand
            .groupby(["a3a", "credit_demand"])[["k3a", "k3i", "k3f", "k3e"]]
            .mean()
            .round(1)
        )
        print(finance_by_region)

        print("\n[9] Sales by region and credit demand:")
        sales_by_region = (
            df_demand
            .groupby(["a3a", "credit_demand"])["log_d2"]
            .agg(["median", "mean", "count"])
            .round(2)
        )
        print(sales_by_region)

        region_size_rates["demand_rate"] = (
                region_size_rates["demand_rate"] * 100
        ).round(1)

        print(region_size_rates)


        print("\n[10] Credit demand rate by legal status (b1):")
        legal_labels = {1: "Shareholding with shares traded",
                        2: "Shareholding with non-traded shares",
                        3: "Sole proprietorship",
                        4: "Partnership",
                        5: "Limited partnership"}
        legal_rates = (
            df_demand.groupby("b1")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "demand_rate"})
        )
        legal_rates.index = legal_rates.index.map(
            lambda x: legal_labels.get(x, x)
        )
        print(legal_rates)


        if len(size_rates) > 0:
            highest = size_rates["demand_rate"].idxmax()
            lowest = size_rates["demand_rate"].idxmin()
            self.insights.append(
                f"INSIGHT 3 — Group comparison: Firm size category '{highest}' "
                f"has the highest credit demand rate "
                f"({size_rates.loc[highest, 'demand_rate']:.1%}), "
                f"while '{lowest}' has the lowest "
                f"({size_rates.loc[lowest, 'demand_rate']:.1%}). "
                f"Firm size codes: 1=Small, 2=Medium, 3=Large."
            )


    def print_insights(self) -> None:
        """Print all collected insights from the EDA."""
        logger.info("\n" + "=" * 60)
        logger.info("KEY INSIGHTS")
        logger.info("=" * 60)
        for insight in self.insights:
            print(f"\n  {insight}")
        print()


# ======================================================================
# MACHINE LEARNING — Classification
# ======================================================================
class ModelTrainer:
    """
    Stage 1 classification: predict credit demand vs no need
    using XGBoost with grid search tuning.
    The model excludes target-close finance variables
    (k82a, k82b, k30, m1a, k15d).
    """

    # Feature set for the credit demand classification model
    NUMERIC_FEATURES = [
        "d3b", "d3c", "b2a", "b2b", "b2c", "firm_age",
        "log_a6c", "log_d2", "log_n3", "b7",
        "k3a", "k3bc", "k3e", "k3f", "k3dgh", "k3i", "k3j", "k33", "k38",
    ]

    CATEGORICAL_FEATURES = [
        "k6", "a3a", "b4", "b7a", "b8",
        "h1", "h5", "h8", "b1", "e1", "e312",
        "sector_broad_3", "k21",
    ]

    def __init__(self, var_labels=None, val_labels=None):
        self.result = None
        self.var_labels = var_labels or {}
        self.val_labels = val_labels or {}

    def build_data(self, df: pd.DataFrame):
        """
        Build Stage 1 dataset: no_need (0) vs demand (1).
        Demand includes discouraged, approved, rejected, in_process.
        """
        class_0 = ["no_need"]
        class_1 = ["discouraged", "approved_full", "approved_partial",
                    "rejected", "in_process"]

        df_model = df[df["finance_status"].isin(class_0 + class_1)].copy()
        df_model["target"] = df_model["finance_status"].apply(
            lambda x: 0 if x in class_0 else 1
        )
        y = df_model["target"].values

        all_features = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        available = [f for f in all_features if f in df_model.columns]

        X = df_model[available].copy()

        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
        X = X.fillna(0)

        logger.info(
            f"\nStage 1 — Credit Demand vs. No Need"
            f"\n  n={len(y)} (0=no_need: {sum(y==0)}, 1=demand: {sum(y==1)})"
            f"\n  Features: {X.shape[1]}"
        )
        return X, y

    def train(self, X, y) -> dict:
        """
        Train XGBoost with grid search and 5-fold stratified
        cross-validation using a fixed hyperparameter grid.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Full grid search with 64 hyperparameter combinations
        from itertools import product

        grid_values = {
            "eta": [0.01, 0.05],
            "max_depth": [2, 3],
            "min_child_weight": [1, 3],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "gamma": [0, 0.1],
        }

        keys = list(grid_values.keys())
        param_grid = [dict(zip(keys, v)) for v in product(*grid_values.values())]

        logger.info(f"\nGrid search: testing {len(param_grid)} combinations...")
        best_auc = -1
        best_params = None
        best_scores = None
        best_iteration = None

        for i, params in enumerate(param_grid):
            model = XGBClassifier(
                n_estimators=500,
                eval_metric="auc",
                early_stopping_rounds=30,
                max_depth=params["max_depth"],
                learning_rate=params["eta"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                gamma=params["gamma"],

                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
            scores = []
            best_iterations = []

            for train_idx, valid_idx in cv.split(X, y):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False,
                )

                preds = model.predict_proba(X_valid)[:, 1]
                scores.append(roc_auc_score(y_valid, preds))
                best_iterations.append(model.best_iteration + 1)

            scores = np.array(scores)
            mean_best_iteration = int(np.mean(best_iterations))

            mean_auc = scores.mean()
            if (i + 1) % 10 == 0 or mean_auc > best_auc:
                logger.info(
                    f"  [{i + 1}/{len(param_grid)}] "
                    f"eta={params['eta']}, depth={params['max_depth']}, "
                    f"mcw={params['min_child_weight']}, "
                    f"sub={params['subsample']}, col={params['colsample_bytree']}, "
                    f"gamma={params['gamma']} "
                    f"-> AUC={mean_auc:.4f}"
                )
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params
                best_scores = scores
                best_iteration = mean_best_iteration

        logger.info(f"\n  Best AUC: {best_auc:.4f}")
        logger.info(f"  Best params: {best_params}")

        # Retrain best model on full data
        best_model = XGBClassifier(
            n_estimators=best_iteration,
            max_depth=best_params["max_depth"],
            learning_rate=best_params["eta"],
            min_child_weight=best_params["min_child_weight"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            gamma=best_params["gamma"],
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        best_model.fit(X, y)

        baseline = max(y.mean(), 1 - y.mean())

        self.result = {
            "n": len(y),
            "baseline": baseline,
            "best_params": best_params,
            "best_iteration": best_iteration,
            "auc_mean": best_scores.mean(),
            "auc_std": best_scores.std(),
            "model": best_model,
            "feature_names": list(X.columns),
        }

        self._print_results()
        self._print_feature_importance()
        return self.result

    def _print_results(self) -> None:
        """Pretty-print classification results."""
        r = self.result
        p = r["best_params"]
        print(f"\n{'='*55}")
        print(f"  STAGE 1 — Credit Demand vs. No Need (XGBoost)")
        print(f"{'='*55}")
        print(f"  Sample size:      {r['n']}")
        print(f"  Baseline rate:    {r['baseline']:.3f}")
        print(f"  Best params:      eta={p['eta']}, depth={p['max_depth']}, "
              f"mcw={p['min_child_weight']}")
        print(f"  Best trees:       {r['best_iteration']}")
        print(f"  XGBoost AUC:      {r['auc_mean']:.4f} "
              f"(+/- {r['auc_std']:.4f})")
        beats = "YES" if r["auc_mean"] > r["baseline"] else "NO"
        print(f"  Beats baseline?   {beats}")

    def _print_feature_importance(self, top_n: int = 10) -> None:
        """Print top features from the model with dynamic labels."""
        model = self.result["model"]
        features = self.result["feature_names"]

        # Labels for derived features not in var_labels file
        derived_labels = {
            "log_d2": "Log total annual sales",
            "log_n3": "Log sales 3 years ago",
            "log_a6c": "Log firm size",
            "log_l4a1": "Log skilled workers",
            "log_l4a2": "Log semi-skilled workers",
            "log_l4b": "Log low-skilled workers",
            "firm_age": "Firm age (years)",
        }

        def get_label(feat_name):
            # Check derived labels first
            if feat_name in derived_labels:
                return derived_labels[feat_name]

            # Check raw variable labels
            if feat_name in self.var_labels:
                return self.var_labels[feat_name]

            # Try to split one-hot encoded name (e.g. k6_1, a3a_4)
            for cat_var in self.CATEGORICAL_FEATURES:
                prefix = cat_var + "_"
                if feat_name.startswith(prefix):
                    value = feat_name[len(prefix):]
                    var_label = self.var_labels.get(cat_var, cat_var)
                    # Look up the value label
                    if cat_var in self.val_labels:
                        try:
                            val_label = self.val_labels[cat_var].get(
                                int(float(value)), value
                            )
                        except (ValueError, TypeError):
                            val_label = value
                    else:
                        val_label = value
                    return f"{var_label}: {val_label}"

            return feat_name

        imp_df = pd.DataFrame({
            "feature": features,
            "label": [get_label(f) for f in features],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print(f"\nTop {top_n} features:")
        print(imp_df.head(top_n).to_string(index=False))


# ======================================================================
# Quick test
# ======================================================================
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load — merge extra variables from full dataset
    loader = DataLoader(DATA_DIR)
    df = loader.load_csv(os.path.join(DATA_DIR, "survey_reduced_df.csv"))
    extra_vars = ["b7", "k21", "k3a", "k3bc", "k3e", "k3f",
                   "k3dgh", "k3i", "k3j", "k33", "k38"]
    df = loader.merge_extra_variables(df, extra_cols=extra_vars)

    # Preprocess
    pre = DataPreprocessor()
    df_clean = pre.process(df)

    # EDA
    analyzer = Analyzer()
    analyzer.run(df_clean)

    # ML
    var_labels = loader.load_variable_labels()
    val_labels = loader.load_value_labels()
    trainer = ModelTrainer(var_labels=var_labels, val_labels=val_labels)
    X, y = trainer.build_data(df_clean)
    trainer.train(X, y)
