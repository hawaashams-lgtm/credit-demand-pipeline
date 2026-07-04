"""
main.py
-------
Main pipeline script for the Credit Demand analysis.
Orchestrates: data loading → preprocessing → EDA → ML → visualization.

Usage:
    python src/main.py
    python src/main.py --data data --output outputs
"""

import os
import argparse
from data_loader import DataLoader
from preprocessing import DataPreprocessor
from analysis import Analyzer, ModelTrainer
from visualization import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Credit Demand Analysis Pipeline"
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Path to the data directory"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output directory"
    )

    return parser.parse_args()

def main():
    # Paths
    args = parse_args()

    BASE_DIR = os.path.dirname(__file__)

    DATA_DIR = args.data or os.path.join(BASE_DIR, "..", "data")
    OUTPUT_DIR = args.output or os.path.join(BASE_DIR, "..", "outputs")

    # 1. Load data
    loader = DataLoader(DATA_DIR)
    df = loader.load_csv(os.path.join(DATA_DIR, "survey_reduced_df.csv"))
    extra_vars = ["b7", "k21", "k3a", "k3bc", "k3e", "k3f",
                  "k3dgh", "k3i", "k3j", "k33", "k38"]
    df = loader.merge_extra_variables(df, extra_cols=extra_vars)
    var_labels = loader.load_variable_labels()
    val_labels = loader.load_value_labels()

    # 2. Preprocess
    pre = DataPreprocessor()
    df_clean = pre.process(df)

    # 3. EDA
    analyzer = Analyzer()
    analyzer.run(df_clean)

    # Diagnostic outlier detection (after EDA)
    outliers = pre.detect_outliers_iqr(df_clean)
    top_outlier_cols = outliers.sum().sort_values(ascending=False).head(10)
    print("\nTop columns by IQR outlier count:")
    print(top_outlier_cols)

    # 4. ML
    trainer = ModelTrainer(var_labels=var_labels, val_labels=val_labels)
    X, y = trainer.build_data(df_clean)
    result = trainer.train(X, y)

    # 5. Visualize
    viz = Visualizer(
        output_dir=OUTPUT_DIR,
        var_labels=var_labels,
        val_labels=val_labels,
    )
    viz.run(df_clean, model_result=result)

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()















