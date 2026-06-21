# =============================================================================
# src/model_trainer.py
# =============================================================================
# PURPOSE:
#   Train multiple ML models and save the best one.
#
# MODELS TRAINED:
#   1. Logistic Regression
#   2. Random Forest
#   3. Gradient Boosting
#   4. Support Vector Machine
#   5. K-Nearest Neighbors
#
# STEPS:
#   1. Prepare features (X) and target (y)
#   2. Scale features using StandardScaler
#   3. Split into train/test sets
#   4. Train all models
#   5. Evaluate each model
#   6. Save best model to disk
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import logging
import joblib                          # For saving/loading models to disk
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

# Scikit-learn: The main ML library in Python
# Each import is a different tool we need
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: prepare_features()
# PURPOSE: Split DataFrame into Features (X) and Target (y)
#
# WHAT IS X AND y?
# X = Input features (everything the model uses to make predictions)
#     Example: Age, Balance, CreditScore, Geography...
#
# y = Target variable (what we want to predict)
#     Example: Exited (0 or 1)
#
# Think of it like:
# X = Question paper (all the information)
# y = Answer key (the correct answer)
# Model learns: given X, predict y
# -----------------------------------------------------------------------------
def prepare_features(
    df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target vector y.

    Args:
        df (pd.DataFrame): Engineered DataFrame
        config (dict): Configuration dictionary

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) — features and target
    """

    logger.info("Preparing features and target...")

    # Get target column name from config
    target_column: str = config["data"]["target_column"]

    # -----------------------------------------------------------------
    # Check target column exists
    # -----------------------------------------------------------------
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found! "
            f"Available: {df.columns.tolist()}"
        )

    # -----------------------------------------------------------------
    # X = ALL columns EXCEPT the target
    # drop() removes the target column, leaving only features
    # axis=1 means drop a column (not a row)
    # -----------------------------------------------------------------
    X: pd.DataFrame = df.drop(columns=[target_column])

    # -----------------------------------------------------------------
    # y = ONLY the target column
    # This is what we want to predict
    # -----------------------------------------------------------------
    y: pd.Series = df[target_column]

    # -----------------------------------------------------------------
    # Make sure all features are numeric
    # ML models cannot handle text or mixed types
    # -----------------------------------------------------------------
    X = X.select_dtypes(include=[np.number])

    logger.info(f"  Features (X) shape: {X.shape}")
    logger.info(f"  Target (y) shape: {y.shape}")
    logger.info(f"  Feature columns: {X.columns.tolist()}")
    logger.info(f"  Target distribution:")
    logger.info(f"    Class 0 (Stayed):  {(y==0).sum():,}")
    logger.info(f"    Class 1 (Churned): {(y==1).sum():,}")

    return X, y


# -----------------------------------------------------------------------------
# FUNCTION 2: split_data()
# PURPOSE: Split data into training and testing sets
#
# WHY SPLIT?
# We need data the model has NEVER seen to test it fairly.
# If we test on training data, the model just "memorizes" answers.
# That's like giving students the exam questions beforehand!
#
# ANALOGY:
# Training set = Textbook (model studies this)
# Test set = Exam (model is tested on this — never seen before!)
# -----------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        config (dict): Configuration dictionary

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """

    logger.info("Splitting data into train/test sets...")

    # Get split settings from config
    test_size: float = config["data"]["test_size"]       # 0.2 = 20%
    random_state: int = config["data"]["random_state"]   # 42

    # -----------------------------------------------------------------
    # train_test_split() randomly divides data
    #
    # test_size=0.2 means:
    #   80% → training set
    #   20% → test set
    #
    # random_state=42 means:
    #   Same random split every time (reproducible!)
    #
    # stratify=y means:
    #   Keep same churn ratio in both train and test sets
    #   Without this, test set might accidentally have all churners!
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y              # Maintain class distribution
    )

    logger.info(f"  Training set: {X_train.shape[0]:,} samples")
    logger.info(f"  Test set:     {X_test.shape[0]:,} samples")
    logger.info(f"  Train churn rate: {y_train.mean()*100:.1f}%")
    logger.info(f"  Test churn rate:  {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# FUNCTION 3: scale_features()
# PURPOSE: Normalize all features to same scale
#
# WHY SCALE?
# Problem:
#   CreditScore: 300-850   (range 550)
#   Balance:     0-250000  (range 250000)
#   Tenure:      0-10      (range 10)
#
# Without scaling, Balance dominates because its numbers are huge!
# Model thinks Balance is 25,000x more important than Tenure.
# That's mathematically WRONG.
#
# StandardScaler Solution:
# For each feature: (value - mean) / standard_deviation
# Result: Every feature has Mean=0 and Std=1
# Now all features are equally weighted!
#
# IMPORTANT RULE:
# Fit scaler on TRAINING data only!
# Then transform BOTH train and test.
# Why? If we fit on test data too, we're "leaking" test info to model.
# -----------------------------------------------------------------------------
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features

    Returns:
        Tuple: (X_train_scaled, X_test_scaled, scaler)
    """

    logger.info("Scaling features using StandardScaler...")

    # Create scaler object
    scaler = StandardScaler()

    # -----------------------------------------------------------------
    # fit_transform() on training data:
    # fit() = calculate mean and std FROM training data
    # transform() = apply scaling TO training data
    # -----------------------------------------------------------------
    X_train_scaled = scaler.fit_transform(X_train)

    # -----------------------------------------------------------------
    # transform() only on test data:
    # Uses mean and std calculated FROM training data
    # NOT recalculated from test data (that would be data leakage!)
    # -----------------------------------------------------------------
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame (scaler returns numpy array)
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    logger.info("  Features scaled successfully!")
    logger.info(f"  Sample mean after scaling: {X_train_scaled.mean().mean():.4f} (should be ~0)")
    logger.info(f"  Sample std after scaling:  {X_train_scaled.std().mean():.4f} (should be ~1)")

    return X_train_scaled, X_test_scaled, scaler


# -----------------------------------------------------------------------------
# FUNCTION 4: build_models()
# PURPOSE: Create all ML model objects with settings from config
#
# WHY BUILD SEPARATELY?
# Keeps model creation separate from training.
# Easy to add or remove models — just add/remove from this function!
# -----------------------------------------------------------------------------
def build_models(config: dict) -> Dict[str, Any]:
    """
    Create all ML model objects using config parameters.

    Args:
        config (dict): Configuration dictionary

    Returns:
        Dict[str, Any]: Dictionary of model name → model object
    """

    logger.info("Building ML models...")

    # Get model settings from config
    lr_config = config["models"]["logistic_regression"]
    rf_config = config["models"]["random_forest"]
    gb_config = config["models"]["gradient_boosting"]
    svm_config = config["models"]["svm"]
    knn_config = config["models"]["knn"]

    # -----------------------------------------------------------------
    # Create model dictionary
    # Each key = model name (for display)
    # Each value = model object (with hyperparameters from config)
    # -----------------------------------------------------------------
    models = {

        # LOGISTIC REGRESSION
        # C = regularization (prevents overfitting)
        # max_iter = max steps to find solution
        "Logistic Regression": LogisticRegression(
            C=lr_config["C"],
            max_iter=lr_config["max_iter"],
            random_state=lr_config["random_state"]
        ),

        # RANDOM FOREST
        # n_estimators = number of trees
        # max_depth = how deep each tree can grow
        # n_jobs=-1 = use all CPU cores
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_config["n_estimators"],
            max_depth=rf_config["max_depth"],
            min_samples_split=rf_config["min_samples_split"],
            random_state=rf_config["random_state"],
            n_jobs=rf_config["n_jobs"]
        ),

        # GRADIENT BOOSTING
        # learning_rate = how fast model learns
        # n_estimators = number of boosting stages
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb_config["n_estimators"],
            learning_rate=gb_config["learning_rate"],
            max_depth=gb_config["max_depth"],
            random_state=gb_config["random_state"]
        ),

        # SUPPORT VECTOR MACHINE
        # kernel="rbf" = handles non-linear data
        # probability=True needed to get confidence scores
        "SVM": SVC(
            C=svm_config["C"],
            kernel=svm_config["kernel"],
            random_state=svm_config["random_state"],
            probability=True       # Enables predict_proba() for confidence
        ),

        # K-NEAREST NEIGHBORS
        # n_neighbors = how many neighbors to consider
        "KNN": KNeighborsClassifier(
            n_neighbors=knn_config["n_neighbors"]
        )
    }

    logger.info(f"  Built {len(models)} models: {list(models.keys())}")
    return models


# -----------------------------------------------------------------------------
# FUNCTION 5: evaluate_model()
# PURPOSE: Calculate all performance metrics for one model
#
# METRICS EXPLAINED:
#
# ACCURACY: Out of all predictions, how many were correct?
#   = (Correct predictions) / (Total predictions)
#   Problem: Misleading for imbalanced data!
#
# PRECISION: Out of customers we PREDICTED would churn,
#            how many ACTUALLY churned?
#   = True Positives / (True Positives + False Positives)
#   High precision = fewer false alarms
#
# RECALL: Out of customers who ACTUALLY churned,
#         how many did we correctly PREDICT?
#   = True Positives / (True Positives + False Negatives)
#   High recall = fewer missed churners
#
# F1 SCORE: Balance between Precision and Recall
#   = 2 * (Precision * Recall) / (Precision + Recall)
#   Best metric for imbalanced datasets!
#
# ROC-AUC: How well model separates classes (0.5=random, 1.0=perfect)
# -----------------------------------------------------------------------------
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate a trained model and return all metrics.

    Args:
        model: Trained sklearn model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test labels
        model_name (str): Name for logging

    Returns:
        Dict[str, float]: Dictionary of metric name → score
    """

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Get probability scores (confidence)
    # predict_proba() returns [[prob_class0, prob_class1], ...]
    # [:, 1] gets probability of class 1 (churn) for each customer
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate all metrics
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob)
    }

    # Log results
    logger.info(f"\n  {model_name} Results:")
    logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")
    logger.info(f"    F1 Score:  {metrics['f1_score']:.4f}  ← PRIMARY METRIC")
    logger.info(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


# -----------------------------------------------------------------------------
# FUNCTION 6: train_all_models()
# PURPOSE: Train every model and collect their results
# -----------------------------------------------------------------------------
def train_all_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict
) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Train all models and evaluate them.

    Args:
        models: Dictionary of model objects
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
        config: Configuration dictionary

    Returns:
        Tuple: (trained_models, results)
    """

    logger.info("=" * 60)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 60)

    trained_models = {}
    results = {}
    cv_folds = config["evaluation"]["cv_folds"]

    for model_name, model in models.items():
        logger.info(f"\nTraining: {model_name}...")

        try:
            # ---------------------------------------------------------
            # STEP 1: Train the model
            # .fit() = the actual learning process
            # Model looks at X_train and y_train
            # Finds patterns that map X → y
            # ---------------------------------------------------------
            model.fit(X_train, y_train)
            logger.info(f"  {model_name} trained successfully!")

            # ---------------------------------------------------------
            # STEP 2: Cross Validation
            # Instead of one train/test split, we do K splits
            # Train on K-1 folds, test on remaining fold
            # Repeat K times, average the scores
            # This gives more reliable performance estimate!
            # ---------------------------------------------------------
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="f1"
            )

            logger.info(
                f"  Cross-validation F1 scores: "
                f"{cv_scores.round(4)}"
            )
            logger.info(
                f"  CV Mean: {cv_scores.mean():.4f} "
                f"(+/- {cv_scores.std():.4f})"
            )

            # ---------------------------------------------------------
            # STEP 3: Evaluate on test set
            # ---------------------------------------------------------
            metrics = evaluate_model(
                model, X_test, y_test, model_name
            )

            # Add CV score to metrics
            metrics["cv_f1_mean"] = cv_scores.mean()
            metrics["cv_f1_std"] = cv_scores.std()

            # Store results
            trained_models[model_name] = model
            results[model_name] = metrics

        except Exception as error:
            logger.error(f"  Error training {model_name}: {error}")
            continue

    return trained_models, results


# -----------------------------------------------------------------------------
# FUNCTION 7: select_best_model()
# PURPOSE: Compare all models and select the best one
# -----------------------------------------------------------------------------
def select_best_model(
    results: Dict[str, Dict],
    config: dict
) -> str:
    """
    Select the best model based on primary metric.

    Args:
        results: Dictionary of model results
        config: Configuration dictionary

    Returns:
        str: Name of the best model
    """

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    # Primary metric to compare (from config: "f1_score")
    primary_metric = config["evaluation"]["primary_metric"]

    # Print comparison table header
    logger.info(
        f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1 Score':>10} {'ROC-AUC':>10}"
    )
    logger.info("-" * 75)

    # Print each model's results
    for model_name, metrics in results.items():
        logger.info(
            f"{model_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f}"
        )

    logger.info("-" * 75)

    # Find best model by primary metric
    best_model_name = max(
        results,
        key=lambda name: results[name][primary_metric]
    )

    best_score = results[best_model_name][primary_metric]

    logger.info(f"\n🏆 BEST MODEL: {best_model_name}")
    logger.info(f"   {primary_metric}: {best_score:.4f}")

    return best_model_name


# -----------------------------------------------------------------------------
# FUNCTION 8: save_model()
# PURPOSE: Save trained model and scaler to disk using joblib
#
# WHY SAVE?
# Training takes time. We don't want to retrain every time.
# Save once → Load anytime → Make predictions instantly!
#
# WHY JOBLIB?
# joblib is optimized for saving large numpy arrays (which models contain)
# Faster than Python's built-in pickle for ML models
# -----------------------------------------------------------------------------
def save_model(
    model: Any,
    scaler: StandardScaler,
    model_name: str,
    feature_names: list,
    config: dict
) -> None:
    """
    Save the trained model and scaler to disk.

    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        model_name (str): Name of the model
        feature_names (list): List of feature column names
        config (dict): Configuration dictionary
    """

    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"  Model saved: {model_path}")

    # Save the scaler (we need this to scale new input data too!)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler saved: {scaler_path}")

    # Save feature names (we need to know which features the model expects)
    feature_path = os.path.join(models_dir, "feature_names.pkl")
    joblib.dump(feature_names, feature_path)
    logger.info(f"  Feature names saved: {feature_path}")

    # Save model metadata (name, scores, etc.)
    metadata = {
        "model_name": model_name,
        "feature_count": len(feature_names),
        "feature_names": feature_names
    }
    metadata_path = os.path.join(models_dir, "model_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    logger.info(f"  Metadata saved: {metadata_path}")

    logger.info(f"\nAll model files saved to: {models_dir}")


# -----------------------------------------------------------------------------
# FUNCTION 9: train_pipeline()
# PURPOSE: Master function — runs the complete training pipeline
# -----------------------------------------------------------------------------
def train_pipeline(
    df: pd.DataFrame,
    config: dict
) -> Tuple[Any, StandardScaler, Dict, str]:
    """
    Complete model training pipeline.

    Args:
        df (pd.DataFrame): Engineered feature DataFrame
        config (dict): Configuration dictionary

    Returns:
        Tuple: (best_model, scaler, all_results, best_model_name)
    """

    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Prepare features
    X, y = prepare_features(df, config)

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y, config)

    # Step 3: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test
    )

    # Step 4: Build models
    models = build_models(config)

    # Step 5: Train all models
    trained_models, results = train_all_models(
        models,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        config
    )

    # Step 6: Select best model
    best_model_name = select_best_model(results, config)
    best_model = trained_models[best_model_name]

    # Step 7: Save best model
    logger.info(f"\nSaving best model: {best_model_name}")
    save_model(
        best_model,
        scaler,
        best_model_name,
        X.columns.tolist(),
        config
    )

    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 60)

    return best_model, scaler, results, best_model_name


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

    # Step 3: Load raw data
    df_raw = load_data(config)

    # Step 4: Clean data
    df_cleaned = clean_data(df_raw, config)

    # Step 5: Engineer features
    df_engineered = engineer_features(df_cleaned, config)

    # Step 6: Train models
    best_model, scaler, results, best_model_name = train_pipeline(
        df_engineered, config
    )

    logger.info(f"\nBest Model: {best_model_name}")
    logger.info("\nmodel_trainer.py completed successfully!")