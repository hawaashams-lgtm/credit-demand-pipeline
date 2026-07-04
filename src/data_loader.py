"""
data_loader.py
--------------
Reusable data ingestion system for the Enterprise Survey pipeline.
Dynamically discovers and loads files from a given directory,
supporting CSV, JSON, and XLSX formats. Also loads variable and
value label mappings to decode the coded survey variables.
"""

import os
import glob
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Discovers and loads datasets from a target directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing data files.
    """

    SUPPORTED_FORMATS = {".csv", ".json", ".xlsx"}

    def __init__(self, data_path: str):
        self.data_path = self._validate_path(data_path)
        self.files = self._discover_files()

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_path(path: str) -> str:
        """Check that the directory exists and return its absolute path."""
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            raise FileNotFoundError(f"Data directory not found: {abs_path}")
        return abs_path

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def _discover_files(self) -> dict:
        """Scan the data directory and group discovered files by extension."""
        found = {}
        for ext in self.SUPPORTED_FORMATS:
            pattern = os.path.join(self.data_path, f"*{ext}")
            matches = sorted(glob.glob(pattern))
            if matches:
                found[ext] = matches

        total = sum(len(v) for v in found.values())
        logger.info(f"Discovered {total} supported file(s) in '{self.data_path}'")
        for ext, paths in found.items():
            for p in paths:
                logger.info(f"  {ext}  ->  {os.path.basename(p)}")

        if total == 0:
            logger.warning("No supported files found. Check the data path.")

        return found

    # ------------------------------------------------------------------
    # Individual loaders
    # ------------------------------------------------------------------
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a single CSV file into a DataFrame."""
        self._check_file_exists(filepath)
        logger.info(f"Loading CSV: {os.path.basename(filepath)}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a single JSON file into a DataFrame."""
        self._check_file_exists(filepath)
        logger.info(f"Loading JSON: {os.path.basename(filepath)}")
        df = pd.read_json(filepath, **kwargs)
        logger.info(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def load_xlsx(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a single XLSX file into a DataFrame."""
        self._check_file_exists(filepath)
        logger.info(f"Loading XLSX: {os.path.basename(filepath)}")
        df = pd.read_excel(filepath, **kwargs)
        logger.info(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _load_single_file(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Route a file to the correct loader based on its extension."""
        ext = os.path.splitext(filepath)[1].lower()
        loaders = {
            ".csv": self.load_csv,
            ".json": self.load_json,
            ".xlsx": self.load_xlsx,
        }
        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file format: '{ext}'")
        return loader(filepath, **kwargs)

    # ------------------------------------------------------------------
    # Bulk loader
    # ------------------------------------------------------------------
    def load_all(self, ext_filter: str = None) -> dict:
        """
        Load every discovered file (or only those matching ext_filter)
        and return them in a dict keyed by filename (without extension).
        """
        datasets = {}
        target_files = {}

        if ext_filter:
            if ext_filter not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format filter: '{ext_filter}'")
            target_files = {ext_filter: self.files.get(ext_filter, [])}
        else:
            target_files = self.files

        for ext, paths in target_files.items():
            for filepath in paths:
                name = os.path.splitext(os.path.basename(filepath))[0]
                try:
                    datasets[name] = self._load_single_file(filepath)
                except Exception as e:
                    logger.error(f"Failed to load '{filepath}': {e}")

        logger.info(f"Loaded {len(datasets)} dataset(s) total.")
        return datasets

    # ------------------------------------------------------------------
    # Label loaders (specific to Enterprise Survey data)
    # ------------------------------------------------------------------
    def merge_extra_variables(
            self, df: pd.DataFrame, full_filename: str = "output.csv",
            extra_cols: list = None, merge_on: str = "idstd",
    ) -> pd.DataFrame:
        """
        Merge additional columns from the full dataset into
        the reduced dataframe.
        """
        if extra_cols is None:
            extra_cols = []

        full_path = os.path.join(self.data_path, full_filename)
        self._check_file_exists(full_path)

        cols_to_load = [merge_on] + extra_cols
        df_full = pd.read_csv(full_path, usecols=cols_to_load)
        logger.info(
            f"Merging {extra_cols} from '{full_filename}' on '{merge_on}'"
        )

        df = df.merge(df_full, on=merge_on, how="left")
        logger.info(f"  -> Shape after merge: {df.shape}")
        return df



    def load_variable_labels(self, filename: str = "var_labels.csv") -> dict:
        """
        Load variable labels as {variable_code: label} dict.
        Example: {'d2': 'Total Annual Sales', 'a6a': 'Sampling Size'}
        """
        filepath = os.path.join(self.data_path, filename)
        self._check_file_exists(filepath)
        df = pd.read_csv(filepath)
        mapping = dict(zip(df["variable"], df["label"]))
        logger.info(f"Loaded {len(mapping)} variable labels.")
        return mapping

    def load_value_labels(self, filename: str = "value_labels_by_variable.csv") -> dict:
        """
        Load value labels as nested dict:
        {variable_code: {numeric_code: text_label}}
        Example: {'a6a': {1: 'Small', 2: 'Medium', 3: 'Large'}}
        """
        filepath = os.path.join(self.data_path, filename)
        self._check_file_exists(filepath)
        df = pd.read_csv(filepath)
        mapping = {}
        for var, group in df.groupby("variable"):
            mapping[var] = dict(zip(group["code"], group["text"]))
        logger.info(f"Loaded value labels for {len(mapping)} variables.")
        return mapping

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_file_exists(filepath: str):
        """Raise FileNotFoundError if a specific file does not exist."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")


# ======================================================================
# Quick test — runs only when this file is executed directly
# ======================================================================
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    loader = DataLoader(DATA_DIR)

    # Load the reduced survey dataset
    df = loader.load_csv(os.path.join(DATA_DIR, "survey_reduced_df.csv"))
    print(f"\nReduced dataset shape: {df.shape}")
    print(df.head(3))

    # Load label mappings
    var_labels = loader.load_variable_labels()
    val_labels = loader.load_value_labels()
    print(f"\nSample variable label: d2 -> '{var_labels.get('d2')}'")
    print(f"Sample value labels:  a6a -> {val_labels.get('a6a')}")
