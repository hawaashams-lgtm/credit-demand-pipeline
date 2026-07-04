"""
preprocessing.py
----------------
Data cleaning and transformation pipeline for the Enterprise Survey.
Handles sentinel value recoding, type conversion, feature engineering
(firm_age, log transforms, sector collapsing), outlier detection,
and missing value treatment.
"""

import logging
import numpy as np
import pandas as pd

from utils import (
    SENTINEL_CODES, REFUSAL_CODE, REFUSAL_COLUMNS,
    CATEGORICAL_COLS, CONTINUOUS_COLS, ID_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Cleans and transforms the raw Enterprise Survey DataFrame.

    Usage
    -----
    pre = DataPreprocessor()
    df_clean = pre.process(df)
    """

    def __init__(self):
        self.steps_applied = []

    # ==================================================================
    # MAIN PIPELINE — called from main.py
    # ==================================================================
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline and return cleaned DataFrame.
        """
        df = df.copy()
        logger.info(f"Starting preprocessing — shape: {df.shape}")

        df = self.recode_sentinels(df)
        df = self.convert_types(df)
        df = self.engineer_features(df)
        df = self.handle_missing(df)

        logger.info(f"Preprocessing complete — shape: {df.shape}")
        logger.info(f"Steps applied: {self.steps_applied}")
        return df

    # ==================================================================
    # STEP 1 — Recode sentinel values to NaN
    # ==================================================================
    def recode_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace Enterprise Survey sentinel codes with NaN.
        -9 and -7 are replaced across all numeric columns.
        -8 is replaced only in specific columns (k11).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        count_before = df[numeric_cols].isin(SENTINEL_CODES).sum().sum()

        # -9 and -7 across all numeric columns
        for code in SENTINEL_CODES:
            df[numeric_cols] = df[numeric_cols].replace(code, np.nan)

        # -8 only for specific columns
        refusal_count = 0
        for col in REFUSAL_COLUMNS:
            if col in df.columns:
                mask = df[col] == REFUSAL_CODE
                refusal_count += mask.sum()
                df.loc[mask, col] = np.nan

        logger.info(
            f"Recoded {count_before} sentinel values (-9, -7) to NaN"
        )
        logger.info(
            f"Recoded {refusal_count} refusal codes (-8) to NaN in {REFUSAL_COLUMNS}"
        )
        self.steps_applied.append("recode_sentinels")
        return df

    # ==================================================================
    # STEP 2 — Convert column types
    # ==================================================================
    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical columns to 'category' dtype and ensure
        continuous columns are numeric.
        """
        cat_converted = 0
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
                cat_converted += 1

        num_converted = 0
        for col in CONTINUOUS_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                num_converted += 1

        logger.info(
            f"Converted {cat_converted} categorical, {num_converted} numeric columns"
        )
        self.steps_applied.append("convert_types")
        return df

    # ==================================================================
    # STEP 3 — Feature engineering
    # ==================================================================
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived variables for the analysis:
        - firm_age from b5 (year began operations)
        - log transforms for skewed variables
        - sector_broad_3 from ISIC codes
        """
        # --- Firm age ---
        if "b5" in df.columns:
            df["firm_age"] = 2025 - df["b5"]
            logger.info("Created 'firm_age' = 2025 - b5")

        # --- Log transforms for skewed variables ---
        log_vars = {
            "d2": "log_d2",           # total annual sales
            "n3": "log_n3",           # sales 3 years ago
            "a6c": "log_a6c",         # screener size
            "l4a1": "log_l4a1",       # skilled workers
            "l4a2": "log_l4a2",       # semi-skilled workers
            "l4b": "log_l4b",         # low-skilled workers
        }
        for raw, logged in log_vars.items():
            if raw in df.columns:
                df[logged] = np.log1p(df[raw].clip(lower=0))

        logger.info(f"Created log-transformed columns: {list(log_vars.values())}")

        # --- Credit demand target ---
        demand_statuses = [
            "discouraged", "approved_full", "approved_partial",
            "rejected", "in_process",
        ]
        no_need_statuses = ["no_need"]

        if "finance_status" in df.columns:
            df["credit_demand"] = np.nan
            df.loc[df["finance_status"].isin(no_need_statuses), "credit_demand"] = 0
            df.loc[df["finance_status"].isin(demand_statuses), "credit_demand"] = 1
            logger.info("Created 'credit_demand' from finance_status")


        # --- Sector collapse (ISIC rev 4 codes) ---
        if "d1a2_v4" in df.columns:
            df["sector_broad_3"] = df["d1a2_v4"].apply(self._collapse_sector)
            df["sector_broad_3"] = df["sector_broad_3"].astype("category")
            logger.info("Created 'sector_broad_3' from ISIC codes")

        self.steps_applied.append("engineer_features")
        return df

    @staticmethod
    def _collapse_sector(isic_code) -> str:
        """
        Collapse ISIC Rev 4 codes into three broad sectors:
        - Industry: Manufacturing (10-33) + Construction (41-43)
        - Trade:    Wholesale/Retail (45-47)
        - Services: Everything else
        """
        if pd.isna(isic_code):
            return "Unknown"
        code = int(isic_code)
        two_digit = code // 100 if code >= 1000 else code // 10

        if 10 <= two_digit <= 33 or 41 <= two_digit <= 43:
            return "Industry"
        elif 45 <= two_digit <= 47:
            return "Trade"
        else:
            return "Services"

    # ==================================================================
    # STEP 4 — Handle missing values
    # ==================================================================
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Numeric columns: impute with median
        - Categorical columns: fill with 'Missing' category
        Report missingness before and after.
        """
        missing_before = df.isnull().sum().sum()

        # Drop columns with >20% missing (consistent with thesis)
        threshold = 0.20
        n_rows = len(df)
        cols_to_drop = []
        for col in df.columns:
            if col in ID_COLS or col == "credit_demand":
                continue
            miss_rate = df[col].isnull().sum() / n_rows
            if miss_rate > threshold:
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(
                f"Dropped {len(cols_to_drop)} columns with >{threshold:.0%} "
                f"missing: {cols_to_drop}"
            )

        # Numeric imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ID_COLS
                        and c != "credit_demand"]
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Categorical imputation
        cat_cols = df.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                if "Missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("Missing")
                df[col] = df[col].fillna("Missing")

        missing_after = df.isnull().sum().sum()
        logger.info(
            f"Missing values: {missing_before} before -> {missing_after} after"
        )
        self.steps_applied.append("handle_missing")
        return df

    # ==================================================================
    # STANDALONE UTILITIES
    # ==================================================================
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Flag outliers using the IQR method (1.5 * IQR rule).

        Returns a boolean DataFrame (same shape) where True = outlier.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c not in ID_COLS]

        outlier_flags = pd.DataFrame(False, index=df.index, columns=columns)

        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_flags[col] = (df[col] < lower) | (df[col] > upper)

        total = outlier_flags.sum().sum()
        n_rows = outlier_flags.any(axis=1).sum()
        logger.info(
            f"IQR outliers: {total} total flags across {n_rows} unique rows"
        )
        return outlier_flags

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return summary statistics for the cleaned dataset."""
        summary = pd.DataFrame({
            "dtype": df.dtypes,
            "non_null": df.notnull().sum(),
            "null": df.isnull().sum(),
            "null_pct": (df.isnull().sum() / len(df) * 100).round(1),
            "unique": df.nunique(),
        })
        return summary


# ======================================================================
# Quick test
# ======================================================================
if __name__ == "__main__":
    import os
    from data_loader import DataLoader

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load
    loader = DataLoader(DATA_DIR)
    df = loader.load_csv(os.path.join(DATA_DIR, "survey_reduced_df.csv"))

    # Preprocess
    pre = DataPreprocessor()
    df_clean = pre.process(df)

    # Show results
    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"\nNew columns added:")
    new_cols = [c for c in df_clean.columns if c not in df.columns]
    for c in new_cols:
        print(f"  {c}")

    print(f"\nMissing values remaining: {df_clean.isnull().sum().sum()}")

    # Quick outlier check
    print("\n--- Outlier Detection ---")
    outliers = pre.detect_outliers_iqr(df_clean)
    top_outlier_cols = outliers.sum().sort_values(ascending=False).head(5)
    print(f"Top 5 columns by outlier count:\n{top_outlier_cols}")












