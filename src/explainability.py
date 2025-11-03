"""
explainability.py
-----------------
Generates SHAP-based explanations for the trained model.
Includes:
- Global feature importance
- Detailed summary plots
- Local explanations (force + waterfall)
- Dependence plots for top features
"""

# Imports
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load as joblib_load
from logger_config import setup_logger

# Configuration
MODEL_PATH = "./models/trained_model.joblib"
TEST_PATH = "./data/processed/test_preprocessed.csv"
TARGET_COL = "DEFAULT"
OUTPUT_DIR = "./reports/figures"
UNSCALED_TEST_PATH = "./data/processed/X_test.csv"

# Initialize Logger
logger = setup_logger("explainability_logger", "logs/explainability.log")


# Function: load_model_and_data

def load_model_and_data(model_path: str, test_path: str, target_col: str, unscaled_path: str):
    """
    Loads the trained model and test dataset.

    Args:
        model_path (str): Path to the saved model (.joblib).
        test_path (str): Path to the preprocessed test dataset.
        target_col (str): Name of the target column.
        unscaled_path (str): Path to the unscaled original test dataset.

    Returns:
        tuple: (model, X_test, y_test, X_test_original)
    """
    try:
        logger.info("Loading trained model and test data...")
        model = joblib_load(model_path)
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        X_test_original = pd.read_csv(unscaled_path)
        logger.info(f"Data loaded successfully: {X_test.shape[0]} samples, {X_test.shape[1]} features.")
        return model, X_test, y_test, X_test_original
    except Exception as e:
        logger.exception(f"Error loading model or data: {e}")
        raise


# Function: generate_shap_explanations

def generate_shap_explanations(model, X_test: pd.DataFrame):
    """
    Computes SHAP values using a model-agnostic explainer.

    Args:
        model: Trained model object.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: (explainer, shap_values)
    """
    try:
        logger.info("Initializing SHAP explainability...")
        shap.initjs()
        explainer = shap.Explainer(model.predict, X_test)
        shap_values = explainer(X_test)
        logger.info("SHAP values generated successfully.")
        return explainer, shap_values
    except Exception as e:
        logger.exception(f"Error generating SHAP values: {e}")
        raise


# Function: global_explainability

def global_explainability(shap_values, X_test: pd.DataFrame, output_dir: str):
    """
    Generates and saves global SHAP plots (bar + summary).

    Args:
        shap_values: Computed SHAP values.
        X_test (pd.DataFrame): Test features.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Generating global SHAP plots...")

        # Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP Summary - Bar Plot)")
        bar_path = os.path.join(output_dir, "shap_feature_importance_bar.png")
        plt.tight_layout()
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP bar plot: {bar_path}")

        # Detailed Summary Plot (Beeswarm)
        shap.summary_plot(shap_values, X_test, show=False)
        summary_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.tight_layout()
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot: {summary_path}")

    except Exception as e:
        logger.exception(f"Error generating global explainability plots: {e}")
        raise


# Function: local_explainability

def local_explainability(shap_values, X_test: pd.DataFrame, sample_idx: int = 10, output_dir: str = "./reports/figures"):
    """
    Generates local SHAP force and waterfall plots for a specific sample.

    Args:
        shap_values: Computed SHAP values.
        X_test (pd.DataFrame): Test features.
        sample_idx (int): Index of the sample to explain.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Generating local SHAP explanations for sample index {sample_idx}...")

        sample = X_test.iloc[[sample_idx]]
        logger.debug(f"Sample data for index {sample_idx}:\n{sample}")

        # Force Plot
        shap.force_plot(shap_values[sample_idx, :], sample, matplotlib=True)
        force_path = os.path.join(output_dir, f"shap_local_force_{sample_idx}.png")
        plt.savefig(force_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved local SHAP force plot: {force_path}")

        # Waterfall Plot
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        waterfall_path = os.path.join(output_dir, f"shap_local_waterfall_{sample_idx}.png")
        plt.savefig(waterfall_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved local SHAP waterfall plot: {waterfall_path}")

    except Exception as e:
        logger.exception(f"Error generating local SHAP plots: {e}")
        raise


# Function: dependence_plots

def dependence_plots(shap_values, X_test: pd.DataFrame, output_dir: str = "./reports/figures", top_n: int = 5):
    """
    Generates SHAP dependence plots for top N most important features.

    Args:
        shap_values: SHAP values for test samples.
        X_test (pd.DataFrame): Test features.
        output_dir (str): Directory to save plots.
        top_n (int): Number of top features to plot.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Generating SHAP dependence plots...")

        # Extract numpy array safely from SHAP Explanation object
        shap_values_array = shap_values.values if hasattr(shap_values, "values") else shap_values

        # Identify top features
        top_features = np.argsort(np.abs(shap_values_array).mean(0))[-top_n:]

        for feature_idx in top_features:
            feature_name = X_test.columns[feature_idx]
            logger.info(f"Creating dependence plot for feature: {feature_name}")

            shap.dependence_plot(feature_name, shap_values_array, X_test, show=False)
            dep_path = os.path.join(output_dir, f"shap_dependence_{feature_name}.png")
            plt.savefig(dep_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved dependence plot for feature {feature_name}: {dep_path}")

    except Exception as e:
        logger.exception(f"Error generating SHAP dependence plots: {e}")
        raise



if __name__ == "__main__":
    logger.info("Starting Explainability Pipeline")

    try:
        # Load model and data
        model, X_test, y_test, X_test_original = load_model_and_data(
            MODEL_PATH, TEST_PATH, TARGET_COL, UNSCALED_TEST_PATH
        )

        # Generate SHAP explanations
        explainer, shap_values = generate_shap_explanations(model, X_test)

        # Global explainability
        global_explainability(shap_values, X_test_original, OUTPUT_DIR)

        # Local explainability
        local_explainability(shap_values, X_test_original, sample_idx=10, output_dir=OUTPUT_DIR)

        # Dependence plots
        dependence_plots(shap_values, X_test_original, output_dir=OUTPUT_DIR, top_n=5)

        logger.info("Explainability analysis completed successfully!")

    except Exception as e:
        logger.exception(f"Error in Explainability Pipeline: {e}")

    logger.info("Explainability Pipeline Finished")
