# =============================================================================
# src/model_evaluator.py
# =============================================================================
# PURPOSE:
#   Deep evaluation of the trained model.
#   Goes beyond simple accuracy to understand model behavior.
#
# OUTPUTS:
#   1. Confusion Matrix Chart
#   2. ROC Curve Chart
#   3. Feature Importance Chart
#   4. Detailed Classification Report
#   5. Business Insights
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import logging
import joblib
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: load_saved_model()
# PURPOSE: Load the saved model and scaler from disk
# -----------------------------------------------------------------------------
def load_saved_model(config: dict) -> Tuple[Any, Any, list]:
    """
    Load saved model, scaler and feature names from disk.

    Args:
        config (dict): Configuration dictionary

    Returns:
        Tuple: (model, scaler, feature_names)
    """

    models_dir = config["paths"]["models_dir"]

    model_path    = os.path.join(models_dir, "best_model.pkl")
    scaler_path   = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.pkl")

    # Check all files exist
    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    # Load files using joblib
    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    metadata      = joblib.load(metadata_path)

    logger.info(f"Model loaded: {metadata['model_name']}")
    logger.info(f"Features: {len(feature_names)}")

    return model, scaler, feature_names


# -----------------------------------------------------------------------------
# FUNCTION 2: plot_confusion_matrix()
# PURPOSE: Visualize prediction results as a heatmap
#
# CONFUSION MATRIX TELLS US:
# How many customers were correctly/incorrectly classified
# in each category
# -----------------------------------------------------------------------------
def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    config: dict
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        config (dict): Configuration dictionary
    """

    logger.info("Generating Confusion Matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract values
    TN = cm[0][0]  # True Negatives  (Stayed → Predicted Stayed)
    FP = cm[0][1]  # False Positives (Stayed → Predicted Churned)
    FN = cm[1][0]  # False Negatives (Churned → Predicted Stayed)
    TP = cm[1][1]  # True Positives  (Churned → Predicted Churned)

    logger.info(f"  True Negatives  (Correct Stay):    {TN:,}")
    logger.info(f"  False Positives (Wrong Churn):     {FP:,}")
    logger.info(f"  False Negatives (Missed Churners): {FN:,}")
    logger.info(f"  True Positives  (Correct Churn):   {TP:,}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -----------------------------------------------------------------
    # LEFT: Raw counts confusion matrix
    # -----------------------------------------------------------------
    labels = [["TN\n(Correct Stay)", "FP\n(False Alarm)"],
              ["FN\n(Missed!)", "TP\n(Caught Churner)"]]

    sns.heatmap(
        cm,
        annot=np.array([[f"{TN}\n{labels[0][0]}", f"{FP}\n{labels[0][1]}"],
                        [f"{FN}\n{labels[1][0]}", f"{TP}\n{labels[1][1]}"]]),
        fmt="",
        cmap="Blues",
        ax=axes[0],
        linewidths=2,
        cbar=True,
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actual Stay", "Actual Churn"]
    )
    axes[0].set_title(
        f"Confusion Matrix\n{model_name}",
        fontweight="bold",
        pad=15
    )

    # -----------------------------------------------------------------
    # RIGHT: Percentage confusion matrix
    # Shows percentages instead of raw counts
    # -----------------------------------------------------------------
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        ax=axes[1],
        linewidths=2,
        cbar=True,
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actual Stay", "Actual Churn"]
    )
    axes[1].set_title(
        "Confusion Matrix (%)\nRow-wise percentages",
        fontweight="bold",
        pad=15
    )

    plt.suptitle("Model Prediction Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save chart
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, "08_confusion_matrix.png")
    fig.savefig(filepath, dpi=config["eda"]["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Confusion matrix saved: {filepath}")


# -----------------------------------------------------------------------------
# FUNCTION 3: plot_roc_curve()
# PURPOSE: Show how well model separates churners from non-churners
#
# ROC CURVE:
# X-axis: False Positive Rate (how often we wrongly predict churn)
# Y-axis: True Positive Rate (how often we correctly predict churn)
#
# Perfect model: curve goes to top-left corner
# Random model: diagonal line (AUC = 0.5)
# Our model: somewhere in between (higher = better!)
# AUC = Area Under Curve (0.5 = random, 1.0 = perfect)
# -----------------------------------------------------------------------------
def plot_roc_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    config: dict
) -> None:
    """
    Plot ROC curve for the model.

    Args:
        y_test: True labels
        y_prob: Predicted probabilities
        model_name (str): Name of the model
        config (dict): Configuration dictionary
    """

    logger.info("Generating ROC Curve...")

    # Calculate ROC curve points
    # fpr = False Positive Rates at different thresholds
    # tpr = True Positive Rates at different thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -----------------------------------------------------------------
    # LEFT: ROC Curve
    # -----------------------------------------------------------------
    axes[0].plot(
        fpr, tpr,
        color="#e74c3c",
        linewidth=2,
        label=f"{model_name} (AUC = {auc_score:.4f})"
    )

    # Random classifier line (baseline — no better than guessing)
    axes[0].plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random Classifier (AUC = 0.5)"
    )

    # Perfect classifier point
    axes[0].scatter([0], [1], color="green", s=100, zorder=5, label="Perfect Point")

    axes[0].set_xlabel("False Positive Rate\n(Wrongly predicted churn)")
    axes[0].set_ylabel("True Positive Rate\n(Correctly predicted churn)")
    axes[0].set_title(f"ROC Curve\nAUC = {auc_score:.4f}", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Shade area under curve
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")

    # -----------------------------------------------------------------
    # RIGHT: Precision-Recall Curve
    # Better for imbalanced datasets!
    # Shows tradeoff between precision and recall
    # -----------------------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    axes[1].plot(
        recall, precision,
        color="#3498db",
        linewidth=2,
        label=f"{model_name}"
    )

    # Baseline (random classifier on imbalanced data)
    baseline = y_test.mean()
    axes[1].axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        label=f"Baseline (Churn Rate = {baseline:.2f})"
    )

    axes[1].set_xlabel("Recall\n(Of all actual churners, how many caught?)")
    axes[1].set_ylabel("Precision\n(Of predicted churners, how many correct?)")
    axes[1].set_title("Precision-Recall Curve\n(Better for imbalanced data)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(recall, precision, alpha=0.1, color="#3498db")

    plt.suptitle("Model Performance Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(config["paths"]["reports_dir"], "09_roc_curve.png")
    fig.savefig(filepath, dpi=config["eda"]["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  ROC curve saved: {filepath}")


# -----------------------------------------------------------------------------
# FUNCTION 4: plot_feature_importance()
# PURPOSE: Show which features matter most for predictions
#
# WHY THIS MATTERS?
# Tells the BUSINESS which customer attributes drive churn.
# "Age and Balance matter most" → Business knows where to focus!
# -----------------------------------------------------------------------------
def plot_feature_importance(
    model: Any,
    feature_names: list,
    model_name: str,
    config: dict
) -> None:
    """
    Plot feature importance chart.

    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model
        config (dict): Configuration dictionary
    """

    logger.info("Generating Feature Importance chart...")

    # -----------------------------------------------------------------
    # Not all models have feature_importances_
    # Tree-based models (Random Forest, Gradient Boosting) do
    # Logistic Regression has coef_ instead
    # SVM and KNN don't have either
    # -----------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = model.feature_importances_
        importance_type = "Feature Importance"

    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression)
        importances = np.abs(model.coef_[0])
        importance_type = "Absolute Coefficient"

    else:
        logger.warning(f"  {model_name} doesn't support feature importance")
        return

    # Create DataFrame for easy sorting
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    # Take top 15 features
    importance_df = importance_df.tail(15)

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.8, len(importance_df))
    )

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Add value labels on bars
    for bar, value in zip(bars, importance_df["Importance"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9
        )

    ax.set_xlabel(importance_type)
    ax.set_title(
        f"Top 15 Most Important Features\n{model_name}",
        fontweight="bold",
        pad=15
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(config["paths"]["reports_dir"], "10_feature_importance.png")
    fig.savefig(filepath, dpi=config["eda"]["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Feature importance saved: {filepath}")


# -----------------------------------------------------------------------------
# FUNCTION 5: print_classification_report()
# PURPOSE: Print detailed per-class performance metrics
# -----------------------------------------------------------------------------
def print_classification_report(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str
) -> None:
    """
    Print detailed classification report.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
    """

    logger.info("\n" + "=" * 60)
    logger.info(f"CLASSIFICATION REPORT: {model_name}")
    logger.info("=" * 60)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Stayed (0)", "Churned (1)"]
    )

    # Print each line of report
    for line in report.split("\n"):
        if line.strip():
            logger.info(f"  {line}")

    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# FUNCTION 6: generate_business_insights()
# PURPOSE: Translate model results into business language
#
# WHY?
# Business people don't understand "F1 Score = 0.62"
# They understand "We correctly identified 62% of customers about to leave"
# This function bridges the gap between ML and business!
# -----------------------------------------------------------------------------
def generate_business_insights(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str
) -> None:
    """
    Generate business-friendly insights from model results.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model_name (str): Name of the model
    """

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    total_churners = TP + FN
    caught_churners = TP
    missed_churners = FN
    false_alarms = FP

    catch_rate = (caught_churners / total_churners * 100) if total_churners > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("BUSINESS INSIGHTS")
    logger.info("=" * 60)
    logger.info(f"\n  Model: {model_name}")
    logger.info(f"\n  Out of {total_churners} customers who actually churned:")
    logger.info(f"  ✅ Correctly identified: {caught_churners} ({catch_rate:.1f}%)")
    logger.info(f"  ❌ Missed (will churn undetected): {missed_churners}")
    logger.info(f"\n  False Alarms (predicted churn but stayed): {false_alarms}")
    logger.info(f"\n  💡 Business Value:")
    logger.info(f"     If average customer is worth $1,000/year:")
    logger.info(f"     Catching {caught_churners} churners = ${caught_churners*1000:,} retained!")
    logger.info(f"     Missing {missed_churners} churners = ${missed_churners*1000:,} lost")
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# FUNCTION 7: evaluate_pipeline()
# PURPOSE: Master function — runs complete evaluation
# -----------------------------------------------------------------------------
def evaluate_pipeline(
    df: pd.DataFrame,
    config: dict
) -> None:
    """
    Complete model evaluation pipeline.

    Args:
        df (pd.DataFrame): Engineered DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load saved model
    model, scaler, feature_names = load_saved_model(config)

    # Step 2: Prepare test data
    target_column = config["data"]["target_column"]

    # Get features and target
    X = df[feature_names]
    y = df[target_column]

    # Scale features using saved scaler
    X_scaled = scaler.transform(X)

    # Split same way as training (same random_state = same split!)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_scaled, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    # Step 3: Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Load model name from metadata
    import joblib
    metadata = joblib.load(os.path.join(config["paths"]["models_dir"], "model_metadata.pkl"))
    model_name = metadata["model_name"]

    # Step 4: Generate all evaluation outputs
    plot_confusion_matrix(y_test, y_pred, model_name, config)
    plot_roc_curve(y_test, y_prob, model_name, config)
    plot_feature_importance(model, feature_names, model_name, config)
    print_classification_report(y_test, y_pred, model_name)
    generate_business_insights(y_test, y_pred, y_prob, model_name)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION COMPLETE!")
    logger.info(f"Charts saved to: {config['paths']['reports_dir']}")
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import load_config, setup_logging, load_data
    from src.data_cleaner import clean_data
    from src.feature_engineer import engineer_features

    # Step 1: Load config
    config = load_config()

    # Step 2: Setup logging
    setup_logging(config)

    # Step 3: Load and prepare data
    df_raw = load_data(config)
    df_cleaned = clean_data(df_raw, config)
    df_engineered = engineer_features(df_cleaned, config)

    # Step 4: Run evaluation
    evaluate_pipeline(df_engineered, config)

    logger.info("\nmodel_evaluator.py completed successfully!")