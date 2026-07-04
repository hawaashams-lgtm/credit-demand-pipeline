"""
visualization.py
----------------
Visualization module for the Credit Demand pipeline.
Produces distribution plots, correlation heatmap, and
custom analytical visualizations. All plots are saved to
an 'outputs' directory.
"""

import os
import logging
from textwrap import fill

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates and saves publication-quality visualizations
    for the Enterprise Survey credit demand analysis.

    Parameters
    ----------
    output_dir : str
        Directory where plots will be saved.
    var_labels : dict
        Dictionary of variable labels loaded from var_labels.csv.
    val_labels : dict
        Dictionary of value labels loaded from value_labels_by_variable.csv.
    """

    def __init__(self, output_dir: str = "outputs", var_labels=None, val_labels=None):
        self.output_dir = output_dir
        self.var_labels = var_labels or {}
        self.val_labels = val_labels or {}

        os.makedirs(output_dir, exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams["figure.dpi"] = 150

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, model_result: dict = None) -> None:
        """Generate all required visualizations."""
        logger.info("=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)

        self.plot_distributions(df)
        self.plot_correlation_heatmap(df)
        self.plot_demand_by_region(df)
        self.plot_demand_by_size(df)
        self.plot_finance_status(df)
        self.plot_sales_by_demand(df)
        self.plot_demand_by_sector(df)
        self.plot_financing_sources(df)

        if model_result:
            self.plot_feature_importance(model_result)

        logger.info(f"\nAll plots saved to '{self.output_dir}/'")

    # ------------------------------------------------------------------
    # Dynamic label helpers
    # ------------------------------------------------------------------
    def _variable_label(self, variable: str, fallback: str = None) -> str:
        """Return variable label from var_labels if available."""
        return self.var_labels.get(variable, fallback or variable)

    def _value_label(self, variable: str, value, fallback=None) -> str:
        """Return value label from val_labels if available."""
        mapping = self.val_labels.get(variable, {})

        possible_keys = [value]

        try:
            possible_keys.append(int(float(value)))
            possible_keys.append(float(value))
            possible_keys.append(str(int(float(value))))
        except (ValueError, TypeError):
            possible_keys.append(str(value))

        for key in possible_keys:
            if key in mapping:
                return mapping[key]

        return fallback if fallback is not None else str(value)

    def _feature_label(self, feature: str) -> str:
        """Return readable label for raw, derived, or one-hot encoded features."""

        # Derived features do not exist in the original label files,
        # so they still need short manual labels.
        derived_labels = {
            "log_d2": "Log total annual sales",
            "log_n3": "Log sales 3 years ago",
            "log_a6c": "Log firm size",
            "log_l4a1": "Log skilled workers",
            "log_l4a2": "Log semi-skilled workers",
            "log_l4b": "Log low-skilled workers",
            "firm_age": "Firm age (years)",
            "sector_broad_3": "Sector",
        }

        if feature in derived_labels:
            return derived_labels[feature]

        # Raw variable labels from var_labels.csv
        if feature in self.var_labels:
            return self.var_labels[feature]

        # One-hot encoded categorical features, e.g. a3a_4, h8_1, b1_5
        categorical_vars = [
            "k6", "a3a", "b4", "b7a", "b8",
            "h1", "h5", "h8", "b1", "e1", "e312",
            "sector_broad_3", "k21",
        ]

        for var in categorical_vars:
            prefix = var + "_"
            if feature.startswith(prefix):
                value = feature[len(prefix):]
                var_label = self._variable_label(var, var)
                value_label = self._value_label(var, value, value)
                return f"{var_label}: {value_label}"

        return feature

    # ------------------------------------------------------------------
    # 1. Distribution plots
    # ------------------------------------------------------------------
    def plot_distributions(self, df: pd.DataFrame) -> None:
        """Histogram + KDE for key numeric variables."""
        logger.info("Plotting distributions...")

        vars_to_plot = [
            "log_d2",
            "log_n3",
            "firm_age",
            "log_l4a1",
        ]
        vars_to_plot = [v for v in vars_to_plot if v in df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for i, var in enumerate(vars_to_plot):
            ax = axes[i]
            label = self._feature_label(var)

            sns.histplot(df[var], kde=True, bins=30, ax=ax, color="steelblue")
            ax.set_title(label)
            ax.set_xlabel("")
            ax.set_ylabel("Count")

        plt.suptitle(
            "Distribution of Key Numeric Variables",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        self._save("01_distributions.png")

    # ------------------------------------------------------------------
    # 2. Correlation heatmap
    # ------------------------------------------------------------------
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Heatmap of correlations among log-transformed variables."""
        logger.info("Plotting correlation heatmap...")

        corr_cols = [
            "log_d2", "log_n3", "log_a6c",
            "log_l4a1", "log_l4a2", "log_l4b", "firm_age",
        ]
        corr_cols = [c for c in corr_cols if c in df.columns]

        corr = df[corr_cols].corr()

        # Dynamic readable labels for heatmap axes
        readable_cols = [self._feature_label(c) for c in corr.columns]
        corr.index = readable_cols
        corr.columns = readable_cols

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            linewidths=0.5,
        )

        ax.set_title(
            "Correlation Matrix — Log-Transformed Variables",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save("02_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # 3. Credit demand rate by region
    # ------------------------------------------------------------------
    def plot_demand_by_region(self, df: pd.DataFrame) -> None:
        """Bar chart of credit demand rate by region."""
        logger.info("Plotting demand rate by region...")
        df_demand = df[df["credit_demand"].notna()].copy()

        region_data = (
            df_demand.groupby("a3a")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "rate"})
            .reset_index()
        )

        region_data["label"] = region_data["a3a"].apply(
            lambda x: fill(str(self._value_label("a3a", x, x)), 12)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            region_data["label"],
            region_data["rate"],
            color=["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"],
            edgecolor="white",
            linewidth=1.5,
        )

        for bar, count in zip(bars, region_data["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={int(count)}",
                ha="center",
                fontsize=10,
            )

        ax.set_ylabel("Credit Demand Rate")
        ax.set_xlabel(self._variable_label("a3a", "Region"))
        ax.set_title("Credit Demand Rate by Region", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.70)

        ax.axhline(
            y=df_demand["credit_demand"].mean(),
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Overall mean",
        )

        ax.legend()
        plt.tight_layout()
        self._save("03_demand_by_region.png")

    # ------------------------------------------------------------------
    # 4. Credit demand rate by firm size
    # ------------------------------------------------------------------
    def plot_demand_by_size(self, df: pd.DataFrame) -> None:
        """Bar chart of credit demand rate by firm size."""
        logger.info("Plotting demand rate by firm size...")
        df_demand = df[df["credit_demand"].notna()].copy()

        size_data = (
            df_demand.groupby("a6a")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "rate"})
            .reset_index()
        )

        size_data["label"] = size_data["a6a"].apply(
            lambda x: self._value_label("a6a", x, x)
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(
            size_data["label"],
            size_data["rate"],
            color=["#4878cf", "#d65f5f", "#6acc65"],
            edgecolor="white",
            linewidth=1.5,
            width=0.5,
        )

        for bar, count in zip(bars, size_data["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={int(count)}",
                ha="center",
                fontsize=10,
            )

        ax.set_ylabel("Credit Demand Rate")
        ax.set_xlabel(self._variable_label("a6a", "Firm Size"))
        ax.set_title("Credit Demand Rate by Firm Size", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.60)

        ax.axhline(
            y=df_demand["credit_demand"].mean(),
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Overall mean",
        )

        ax.legend()
        plt.tight_layout()
        self._save("04_demand_by_size.png")

    # ------------------------------------------------------------------
    # 5. Finance status breakdown
    # ------------------------------------------------------------------
    def plot_finance_status(self, df: pd.DataFrame) -> None:
        """Horizontal bar chart of finance_status categories."""
        logger.info("Plotting finance status distribution...")

        status_counts = df["finance_status"].value_counts()

        colors = []
        for status in status_counts.index:
            if status == "no_need":
                colors.append("#6acc65")
            elif status in ["approved_full", "approved_partial"]:
                colors.append("#4878cf")
            elif status in ["discouraged", "rejected"]:
                colors.append("#d65f5f")
            else:
                colors.append("#cccccc")

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            status_counts.index,
            status_counts.values,
            color=colors,
            edgecolor="white",
            linewidth=1,
        )

        for bar, val in zip(bars, status_counts.values):
            ax.text(
                bar.get_width() + 2,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("Number of Firms")
        ax.set_title(
            "Distribution of Finance Status Categories",
            fontsize=13,
            fontweight="bold",
        )
        ax.invert_yaxis()
        plt.tight_layout()
        self._save("05_finance_status.png")

    # ------------------------------------------------------------------
    # 6. Sales distribution by credit demand status
    # ------------------------------------------------------------------
    def plot_sales_by_demand(self, df: pd.DataFrame) -> None:
        """KDE plot comparing log sales for demand vs no need."""
        logger.info("Plotting sales by demand status...")

        fig, ax = plt.subplots(figsize=(9, 5))
        df_demand = df[df["credit_demand"].notna()].copy()

        no_need = df_demand[df_demand["credit_demand"] == 0]["log_d2"]
        demand = df_demand[df_demand["credit_demand"] == 1]["log_d2"]

        sns.kdeplot(
            no_need,
            ax=ax,
            label="No need (0)",
            color="#4878cf",
            fill=True,
            alpha=0.3,
        )
        sns.kdeplot(
            demand,
            ax=ax,
            label="Demand (1)",
            color="#d65f5f",
            fill=True,
            alpha=0.3,
        )

        ax.axvline(no_need.median(), color="#4878cf", linestyle="--", alpha=0.7)
        ax.axvline(demand.median(), color="#d65f5f", linestyle="--", alpha=0.7)

        ax.set_xlabel(self._feature_label("log_d2"))
        ax.set_ylabel("Density")
        ax.set_title(
            "Sales Distribution: Credit Demand vs. No Need",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()
        self._save("06_sales_by_demand.png")

    # ------------------------------------------------------------------
    # 7. Credit demand rate by sector
    # ------------------------------------------------------------------
    def plot_demand_by_sector(self, df: pd.DataFrame) -> None:
        """Bar chart of credit demand rate by sector."""
        logger.info("Plotting demand rate by sector...")
        df_demand = df[df["credit_demand"].notna()].copy()

        sector_data = (
            df_demand.groupby("sector_broad_3")["credit_demand"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "rate"})
            .reset_index()
        )

        sector_data = sector_data.sort_values("rate", ascending=False)

        sector_data["label"] = sector_data["sector_broad_3"].apply(
            lambda x: fill(str(x), 12)
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(
            sector_data["label"],
            sector_data["rate"],
            color=["#4878cf", "#d65f5f", "#6acc65"],
            edgecolor="white",
            linewidth=1.5,
            width=0.5,
        )

        for bar, count in zip(bars, sector_data["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={int(count)}",
                ha="center",
                fontsize=10,
            )

        ax.set_ylabel("Credit Demand Rate")
        ax.set_xlabel(self._feature_label("sector_broad_3"))
        ax.set_title("Credit Demand Rate by Sector", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.65)

        ax.axhline(
            y=df_demand["credit_demand"].mean(),
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Overall mean",
        )

        ax.legend()
        plt.tight_layout()
        self._save("07_demand_by_sector.png")

    # ------------------------------------------------------------------
    # 8. Financing sources — demand vs no need
    # ------------------------------------------------------------------
    def plot_financing_sources(self, df: pd.DataFrame) -> None:
        """Grouped bar chart comparing working capital financing sources by demand status."""
        logger.info("Plotting financing sources by demand status...")

        df_demand = df[df["credit_demand"].notna()].copy()

        sources = ["k3a", "k3i", "k3f", "k3e"]
        sources = [s for s in sources if s in df_demand.columns]

        short_labels = {
            "k3a": "Internal\nfunds",
            "k3i": "Owners'\nequity",
            "k3f": "Supplier\ncredit",
            "k3e": "Non-bank\nfinance",
        }

        means = df_demand.groupby("credit_demand")[sources].mean()

        x = np.arange(len(sources))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars_0 = ax.bar(
            x - width / 2,
            means.loc[0],
            width,
            label="No need (0)",
            color="#4878cf",
            edgecolor="white",
            linewidth=1.5,
        )

        bars_1 = ax.bar(
            x + width / 2,
            means.loc[1],
            width,
            label="Demand (1)",
            color="#d65f5f",
            edgecolor="white",
            linewidth=1.5,
        )

        for bars in [bars_0, bars_1]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([short_labels.get(s, s) for s in sources])

        ax.set_ylabel("Mean % of Working Capital")
        ax.set_xlabel("Financing Source")
        ax.set_title(
            "Working Capital Financing by Credit Demand Status",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")

        plt.tight_layout()
        self._save("08_financing_sources.png")


    # ------------------------------------------------------------------
    # 9. Feature importance from XGBoost model
    # ------------------------------------------------------------------
    def plot_feature_importance(self, result: dict) -> None:
        """Horizontal bar chart of top 10 XGBoost feature importances."""
        logger.info("Plotting feature importance...")

        model = result["model"]
        features = result["feature_names"]

        imp_df = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(10)

        short_labels = {
            "k3a": "k3a — Internal funds (%)",
            "k3f": "k3f — Supplier credit (%)",
            "k3i": "k3i — Owners/equity (%)",
            "k3e": "k3e — Non-bank finance (%)",
            "log_n3": "log_n3 — Sales 3 yrs ago",
            "log_d2": "log_d2 — Current sales",
            "sector_broad_3_Industry": "Sector — Industry",
            "sector_broad_3_Services": "Sector — Services",
            "a3a_4": "Region — East & Southeast",
            "h8_1": "R&D spending — Yes",
            "e1_1": "Main market — Local",
            "b1_3": "Legal status — Sole proprietorship",
            "b1_5": "Legal status — Limited partnership",
            "k6_2": "No bank account",
            "b7a_2": "Top manager not female",
        }

        imp_df["label"] = imp_df["feature"].map(short_labels).fillna(imp_df["feature"])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            imp_df["label"],
            imp_df["importance"],
            color="#4878cf",
            edgecolor="white",
            linewidth=1,
        )

        ax.set_xlabel("Feature Importance")
        ax.set_title(
            "Top 10 Predictors of Credit Demand (XGBoost)",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        self._save("09_feature_importance.png")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _save(self, filename: str) -> None:
        """Save the current figure and close it."""
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {path}")


# ======================================================================
# Quick test
# ======================================================================
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

    # Load
    loader = DataLoader(DATA_DIR)
    df = loader.load_csv(os.path.join(DATA_DIR, "survey_reduced_df.csv"))

    extra_vars = [
        "b7", "k21", "k3a", "k3bc", "k3e", "k3f",
        "k3dgh", "k3i", "k3j", "k33", "k38",
    ]
    df = loader.merge_extra_variables(df, extra_cols=extra_vars)

    # Load labels dynamically
    var_labels = loader.load_variable_labels()
    val_labels = loader.load_value_labels()

    # Preprocess
    pre = DataPreprocessor()
    df_clean = pre.process(df)

    # Visualize
    viz = Visualizer(
        output_dir=OUTPUT_DIR,
        var_labels=var_labels,
        val_labels=val_labels,
    )
    viz.run(df_clean)