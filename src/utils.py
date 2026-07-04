"""
utils.py
--------
Shared constants and helper functions used across the pipeline.
Contains dataset-specific configurations like sentinel codes,
column groupings, and utility functions for label mapping.
"""

# ======================================================================
# SENTINEL VALUES
# ======================================================================
# The Enterprise Survey uses special codes instead of leaving blanks.
# These all need to be treated as missing data during preprocessing.

SENTINEL_VALUES = {
    -9: "Don't know (spontaneous)",
    -7: "Not applicable / Not in business",
}

SENTINEL_CODES = list(SENTINEL_VALUES.keys())  # [-9, -7]

# -8 ("Refusal") is only recoded for specific columns
REFUSAL_CODE = -8
REFUSAL_COLUMNS = ["k11"]


# ======================================================================
# COLUMN GROUPINGS
# ======================================================================
# These group the 53 columns in the reduced survey dataset by their role
# so other modules can grab the right columns without hardcoding
# variable names everywhere.

# Identifiers — not used in analysis, just for tracking rows
ID_COLS = ["idstd", "id"]

# Sampling weights — used for weighted statistics
WEIGHT_COLS = ["wmedian", "wweak"]

# Target variables — what we want to predict / analyze
TARGET_COLS = ["finance_status", "credit_demand"]

# Credit / finance variables used in the demand analysis
FINANCE_COLS = [
    "k162a",   # Applied for loan?
    "k162b",   # Applied for line of credit?
    "k17",     # Main reason for not applying
    "k20a1",   # Outcome of most recent application
    "k6",      # Has checking/saving account?
    "k82a",    # Has loan from financial institution?
    "k82b",    # Has line of credit?
    "k9",      # Type of financial institution
    "k11",     # Loan value at time of approval
    "k13",     # Collateral required?
    "k14a",    # Collateral type: land/buildings
    "k14b",    # Collateral type: equipment
    "k14c",    # Collateral type: accounts
    "k14d",    # Collateral type: personal assets
    "k14e",    # Collateral type: other
    "k15a",    # Value of collateral required
    "k32",     # Days to receive loan decision
]

# Firm characteristics
FIRM_COLS = [
    "a2", "a3a", "a3c", "a6a", "a6c",
    "b4", "b4a", "b5", "b6b", "b7a", "b8",
    "b1", "b2a", "b2b", "b2c",
    "b7",    # top manager years of experience
    "k21",   # audited financials
]

# Performance variables
PERFORMANCE_COLS = [
    "d1a2_v4", # ISIC industry code
    "d2",      # Total annual sales (last FY)
    "d3b",     # % sales: indirect exports
    "d3c",     # % sales: direct exports
    "d8",      # Year first exported
    "d1a3",    # Main product % of total sales
    "n3",      # Sales 3 years ago
    "e1",      # Main market scope
    "e312",    # Market competition level
]

# Innovation variables
INNOVATION_COLS = [
    "h1",      # New products introduced (last 3 yrs)?
    "h5",      # New process introduced?
    "h8",      # Spent on R&D?
    "h9",      # How much spent on R&D?
]

# Workforce variables
WORKFORCE_COLS = [
    "l4a1",    # Skilled production workers
    "l4a2",    # Semi-skilled production workers
    "l4b",     # Low-skilled production workers
]

# Categorical columns — these hold codes, not real numbers
CATEGORICAL_COLS = [
    "finance_status", "a2", "a3a", "a3c", "a6a", "b4", "b7a", "b8",
    "b1", "k162a", "k162b", "k17", "k20a1", "k6", "k82a", "k82b",
    "k9", "k13", "k14a", "k14b", "k14c", "k14d", "k14e",
    "h1", "h5", "h8", "e1", "e312", "d1a2_v4", "k21",
]

# Continuous / numeric columns — real numbers
CONTINUOUS_COLS = [
    "d2", "n3", "k11", "k15a", "k32", "b4a", "b5", "b6b",
    "a6c", "d3b", "d3c", "h9", "l4a1", "l4a2", "l4b",
    "b2a", "b2b", "b2c", "d1a3", "b7",
    "k3a", "k3bc", "k3e", "k3f", "k3dgh", "k3i", "k3j", "k33", "k38",
]


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def apply_variable_labels(df, var_labels: dict) -> None:
    """
    Print a readable summary of column names alongside their labels.

    Parameters
    ----------
    df : pd.DataFrame
    var_labels : dict from DataLoader.load_variable_labels()
    """
    print(f"\n{'Column':<15} {'Label'}")
    print("-" * 70)
    for col in df.columns:
        label = var_labels.get(col, "(no label)")
        print(f"{col:<15} {label}")


def apply_value_labels(series, val_labels: dict, variable: str):
    """
    Map coded values in a Series to their text labels.

    Parameters
    ----------
    series : pd.Series
        Column with numeric codes (e.g. 1, 2, 3).
    val_labels : dict
        Full value labels dict from DataLoader.load_value_labels().
    variable : str
        The variable name to look up (e.g. 'a6a').

    Returns
    -------
    pd.Series with text labels instead of codes.
    """
    mapping = val_labels.get(variable, {})
    if not mapping:
        return series
    return series.map(mapping)


def get_column_group(group_name: str) -> list:
    """
    Return a column group by name.
    Useful when other modules need columns without importing constants.
    """
    groups = {
        "id": ID_COLS,
        "weight": WEIGHT_COLS,
        "target": TARGET_COLS,
        "finance": FINANCE_COLS,
        "firm": FIRM_COLS,
        "performance": PERFORMANCE_COLS,
        "innovation": INNOVATION_COLS,
        "workforce": WORKFORCE_COLS,
        "categorical": CATEGORICAL_COLS,
        "continuous": CONTINUOUS_COLS,
    }
    if group_name not in groups:
        raise ValueError(
            f"Unknown group '{group_name}'. Choose from: {list(groups.keys())}"
        )
    return groups[group_name]


# ======================================================================
# Quick test
# ======================================================================
if __name__ == "__main__":
    print("Sentinel codes:", SENTINEL_CODES)
    print(f"\nFinance columns ({len(FINANCE_COLS)}):", FINANCE_COLS[:5], "...")
    print(f"Firm columns ({len(FIRM_COLS)}):", FIRM_COLS[:5], "...")
    print(f"Categorical columns ({len(CATEGORICAL_COLS)}):", CATEGORICAL_COLS[:5], "...")
    print(f"\nget_column_group('target') -> {get_column_group('target')}")
