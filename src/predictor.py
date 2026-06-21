# =============================================================================
# src/predictor.py
# =============================================================================
# PURPOSE:
#   Make predictions for NEW customers using the saved model.
#   This is what the Streamlit app calls when a user clicks "Predict".
#
# FLOW:
#   New Customer Data (raw input)
#           ↓
#   Apply Feature Engineering
#           ↓
#   Scale Features
#           ↓
#   Load Saved Model
#           ↓
#   Predict (0=Stay, 1=Churn)
#           ↓
#   Return Prediction + Confidence Score
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


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: load_model_artifacts()
# PURPOSE: Load all saved model files from disk
#
# WHY "ARTIFACTS"?
# In ML, "artifacts" = all files produced during training:
#   - The trained model itself
#   - The scaler (to scale new input data)
#   - Feature names (to ensure correct column order)
#   - Metadata (model name, version etc.)
# -----------------------------------------------------------------------------
def load_model_artifacts(config: dict) -> Dict[str, Any]:
    """
    Load all saved model artifacts from disk.

    Args:
        config (dict): Configuration dictionary

    Returns:
        Dict containing model, scaler, feature_names, metadata
    """

    logger.info("Loading model artifacts...")

    models_dir = config["paths"]["models_dir"]

    # Define paths
    paths = {
        "model":    os.path.join(models_dir, "best_model.pkl"),
        "scaler":   os.path.join(models_dir, "scaler.pkl"),
        "features": os.path.join(models_dir, "feature_names.pkl"),
        "metadata": os.path.join(models_dir, "model_metadata.pkl")
    }

    # Check all files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model file missing: {path}\n"
                f"Please run model_trainer.py first!"
            )

    # Load all artifacts
    artifacts = {
        "model":         joblib.load(paths["model"]),
        "scaler":        joblib.load(paths["scaler"]),
        "feature_names": joblib.load(paths["features"]),
        "metadata":      joblib.load(paths["metadata"])
    }

    logger.info(f"  Model loaded: {artifacts['metadata']['model_name']}")
    logger.info(f"  Features expected: {len(artifacts['feature_names'])}")

    return artifacts


# -----------------------------------------------------------------------------
# FUNCTION 2: engineer_single_customer()
# PURPOSE: Apply feature engineering to ONE customer's raw data
#
# WHY THIS FUNCTION?
# During training, we engineered features on 10,000 rows.
# Now for prediction, we need to do the EXACT SAME transformations
# on just 1 customer's data.
# The features must match exactly what the model was trained on!
# -----------------------------------------------------------------------------
def engineer_single_customer(
    customer_data: Dict[str, Any],
    config: dict
) -> pd.DataFrame:
    """
    Apply feature engineering to a single customer's raw data.

    Args:
        customer_data (Dict): Raw customer data as dictionary
            Example: {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 80000.0
            }
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: Single-row DataFrame with all engineered features
    """

    logger.info("Engineering features for new customer...")

    # -----------------------------------------------------------------
    # Convert input dictionary to DataFrame
    # We wrap in [] to make it a list of one dictionary
    # This creates a DataFrame with 1 row
    # -----------------------------------------------------------------
    df = pd.DataFrame([customer_data])

    # -----------------------------------------------------------------
    # STEP 1: Create Age Group
    # Same logic as feature_engineer.py
    # -----------------------------------------------------------------
    age = float(customer_data["Age"])
    if age <= 30:
        age_group = 0      # Young
    elif age <= 45:
        age_group = 1      # Middle-aged
    elif age <= 60:
        age_group = 2      # Senior
    else:
        age_group = 3      # Elder

    df["Age_Group"] = age_group

    # -----------------------------------------------------------------
    # STEP 2: Credit Score Category
    # -----------------------------------------------------------------
    credit = float(customer_data["CreditScore"])
    if credit < 580:
        credit_cat = 0     # Poor
    elif credit < 670:
        credit_cat = 1     # Fair
    elif credit < 740:
        credit_cat = 2     # Good
    elif credit < 800:
        credit_cat = 3     # Very Good
    else:
        credit_cat = 4     # Excellent

    df["Credit_Category"] = credit_cat

    # -----------------------------------------------------------------
    # STEP 3: Ratio Features
    # -----------------------------------------------------------------
    balance = float(customer_data["Balance"])
    salary  = float(customer_data["EstimatedSalary"])
    products = float(customer_data["NumOfProducts"])

    df["Balance_Salary_Ratio"]  = balance / (salary + 1)
    df["Balance_Per_Product"]   = balance / (products + 1)
    df["CreditScore_Age_Ratio"] = credit / (age + 1)

    # -----------------------------------------------------------------
    # STEP 4: Interaction Features
    # -----------------------------------------------------------------
    active = float(customer_data["IsActiveMember"])
    tenure = float(customer_data["Tenure"])

    df["Age_Balance_Interaction"] = age * balance
    df["Products_Activity_Score"] = products * active
    df["Has_Zero_Balance"]        = 1 if balance == 0 else 0
    df["Tenure_Age_Ratio"]        = tenure / (age + 1)

    # -----------------------------------------------------------------
    # STEP 5: Encode Gender
    # Male → 0, Female → 1
    # -----------------------------------------------------------------
    df["Gender"] = 0 if customer_data["Gender"] == "Male" else 1

    # -----------------------------------------------------------------
    # STEP 6: Encode Geography (One-Hot)
    # Germany is reference (both = 0 means Germany)
    # Geo_France: 1 if France, else 0
    # Geo_Spain:  1 if Spain, else 0
    # -----------------------------------------------------------------
    geography = customer_data["Geography"]
    df["Geo_France"] = 1 if geography == "France" else 0
    df["Geo_Spain"]  = 1 if geography == "Spain"  else 0

    # -----------------------------------------------------------------
    # Drop original Geography column
    # (we replaced it with Geo_France and Geo_Spain)
    # -----------------------------------------------------------------
    if "Geography" in df.columns:
        df = df.drop(columns=["Geography"])

    logger.info(f"  Engineered features created: {df.shape[1]} columns")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 3: predict_single_customer()
# PURPOSE: Make churn prediction for one customer
#
# RETURNS:
#   prediction: 0 (Stay) or 1 (Churn)
#   confidence: Probability score 0.0 to 1.0
#   risk_level: "Low", "Medium", "High"
#   explanation: Human-readable explanation
# -----------------------------------------------------------------------------
def predict_single_customer(
    customer_data: Dict[str, Any],
    config: dict
) -> Dict[str, Any]:
    """
    Predict churn probability for a single customer.

    Args:
        customer_data (Dict): Raw customer features
        config (dict): Configuration dictionary

    Returns:
        Dict containing prediction results:
            - prediction: 0 or 1
            - churn_probability: float (0.0 to 1.0)
            - stay_probability: float (0.0 to 1.0)
            - risk_level: "Low", "Medium", "High"
            - recommendation: Business action to take
            - model_name: Which model made the prediction
    """

    logger.info("Making prediction for new customer...")

    # -----------------------------------------------------------------
    # STEP 1: Load model artifacts
    # -----------------------------------------------------------------
    artifacts    = load_model_artifacts(config)
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    model_name   = artifacts["metadata"]["model_name"]

    # -----------------------------------------------------------------
    # STEP 2: Engineer features
    # -----------------------------------------------------------------
    df_engineered = engineer_single_customer(customer_data, config)

    # -----------------------------------------------------------------
    # STEP 3: Align columns with training features
    #
    # CRITICAL STEP!
    # The model expects features in EXACT same order as training.
    # If training had 20 features, prediction must also have 20.
    # If a feature is missing, fill with 0.
    # If extra features exist, drop them.
    # -----------------------------------------------------------------
    # Add missing columns with value 0
    for feature in feature_names:
        if feature not in df_engineered.columns:
            df_engineered[feature] = 0
            logger.warning(f"  Missing feature added with 0: {feature}")

    # Select only the features the model knows about, in correct order
    df_aligned = df_engineered[feature_names]

    # Convert all to float
    df_aligned = df_aligned.astype(float)

    logger.info(f"  Features aligned: {df_aligned.shape}")

    # -----------------------------------------------------------------
    # STEP 4: Scale features using saved scaler
    # Use transform() NOT fit_transform()
    # (We use the scaler fitted during training)
    # -----------------------------------------------------------------
    X_scaled = scaler.transform(df_aligned)

    # -----------------------------------------------------------------
    # STEP 5: Make prediction
    # predict() returns 0 or 1
    # predict_proba() returns [prob_stay, prob_churn]
    # -----------------------------------------------------------------
    prediction  = model.predict(X_scaled)[0]        # 0 or 1
    probability = model.predict_proba(X_scaled)[0]  # [p_stay, p_churn]

    churn_probability = float(probability[1])   # Probability of churn
    stay_probability  = float(probability[0])   # Probability of staying

    # -----------------------------------------------------------------
    # STEP 6: Determine risk level
    # Based on churn probability thresholds
    # -----------------------------------------------------------------
    if churn_probability < 0.30:
        risk_level = "Low Risk"
        risk_emoji = "🟢"
    elif churn_probability < 0.60:
        risk_level = "Medium Risk"
        risk_emoji = "🟡"
    else:
        risk_level = "High Risk"
        risk_emoji = "🔴"

    # -----------------------------------------------------------------
    # STEP 7: Generate business recommendation
    # -----------------------------------------------------------------
    if churn_probability < 0.30:
        recommendation = (
            "Customer is likely to stay. "
            "Maintain current engagement with regular check-ins."
        )
    elif churn_probability < 0.60:
        recommendation = (
            "Customer shows some churn signals. "
            "Consider offering loyalty rewards or a product upgrade."
        )
    else:
        recommendation = (
            "HIGH CHURN RISK! Immediate action required. "
            "Contact customer with a personalized retention offer. "
            "Consider fee waivers or premium service upgrades."
        )

    # -----------------------------------------------------------------
    # STEP 8: Package all results
    # -----------------------------------------------------------------
    result = {
        "prediction":         int(prediction),
        "churn_probability":  round(churn_probability * 100, 2),
        "stay_probability":   round(stay_probability * 100, 2),
        "risk_level":         risk_level,
        "risk_emoji":         risk_emoji,
        "recommendation":     recommendation,
        "model_name":         model_name,
        "will_churn":         bool(prediction == 1)
    }

    # Log results
    logger.info(f"\n  {'='*40}")
    logger.info(f"  PREDICTION RESULT")
    logger.info(f"  {'='*40}")
    logger.info(f"  Model Used:        {model_name}")
    logger.info(f"  Prediction:        {'CHURN' if prediction == 1 else 'STAY'}")
    logger.info(f"  Churn Probability: {churn_probability*100:.2f}%")
    logger.info(f"  Stay Probability:  {stay_probability*100:.2f}%")
    logger.info(f"  Risk Level:        {risk_emoji} {risk_level}")
    logger.info(f"  Recommendation:    {recommendation}")
    logger.info(f"  {'='*40}")

    return result


# -----------------------------------------------------------------------------
# FUNCTION 4: predict_batch()
# PURPOSE: Make predictions for MULTIPLE customers at once
#
# WHEN USEFUL?
# When a bank wants to scan ALL customers every month
# and identify who needs retention attention.
# -----------------------------------------------------------------------------
def predict_batch(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Make predictions for multiple customers at once.

    Args:
        df (pd.DataFrame): DataFrame with multiple customers
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: Original DataFrame with prediction columns added
    """

    logger.info(f"Making batch predictions for {len(df):,} customers...")

    # Load artifacts once (not for every customer!)
    artifacts     = load_model_artifacts(config)
    model         = artifacts["model"]
    scaler        = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    # Align columns
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    X = df[feature_names].astype(float)
    X_scaled = scaler.transform(X)

    # Make predictions for all customers
    predictions   = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Add results to DataFrame
    df["Predicted_Churn"]       = predictions
    df["Churn_Probability"]     = (probabilities * 100).round(2)

    # Add risk level for each customer
    def get_risk(prob):
        if prob < 30:
            return "Low Risk"
        elif prob < 60:
            return "Medium Risk"
        else:
            return "High Risk"

    df["Risk_Level"] = df["Churn_Probability"].apply(get_risk)

    # Summary statistics
    total        = len(df)
    predicted_churn = predictions.sum()
    churn_rate   = predicted_churn / total * 100

    logger.info(f"  Total customers:     {total:,}")
    logger.info(f"  Predicted churners:  {predicted_churn:,} ({churn_rate:.1f}%)")
    logger.info(f"  High risk:    {(df['Risk_Level']=='High Risk').sum():,}")
    logger.info(f"  Medium risk:  {(df['Risk_Level']=='Medium Risk').sum():,}")
    logger.info(f"  Low risk:     {(df['Risk_Level']=='Low Risk').sum():,}")

    return df


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# Test the predictor with a sample customer
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import load_config, setup_logging

    # Load config and setup logging
    config = load_config()
    setup_logging(config)

    # -----------------------------------------------------------------
    # Test with a SAMPLE customer
    # This simulates what the Streamlit app will send
    # -----------------------------------------------------------------

    # Test Customer 1: High Risk (likely to churn)
    high_risk_customer = {
        "CreditScore":     400,
        "Geography":       "Germany",
        "Gender":          "Female",
        "Age":             55,
        "Tenure":          2,
        "Balance":         150000.0,
        "NumOfProducts":   4,
        "HasCrCard":       1,
        "IsActiveMember":  0,
        "EstimatedSalary": 60000.0
    }

    # Test Customer 2: Low Risk (likely to stay)
    low_risk_customer = {
        "CreditScore":     750,
        "Geography":       "France",
        "Gender":          "Male",
        "Age":             30,
        "Tenure":          8,
        "Balance":         80000.0,
        "NumOfProducts":   2,
        "HasCrCard":       1,
        "IsActiveMember":  1,
        "EstimatedSalary": 100000.0
    }

    logger.info("\nTesting High Risk Customer:")
    result1 = predict_single_customer(high_risk_customer, config)

    logger.info("\nTesting Low Risk Customer:")
    result2 = predict_single_customer(low_risk_customer, config)

    # Compare results
    logger.info("\n" + "="*50)
    logger.info("COMPARISON")
    logger.info("="*50)
    logger.info(
        f"High Risk Customer: "
        f"{result1['risk_emoji']} {result1['risk_level']} "
        f"({result1['churn_probability']}% churn chance)"
    )
    logger.info(
        f"Low Risk Customer:  "
        f"{result2['risk_emoji']} {result2['risk_level']} "
        f"({result2['churn_probability']}% churn chance)"
    )

    logger.info("\npredictor.py completed successfully!")