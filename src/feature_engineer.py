# =============================================================================
# src/feature_engineer.py
# =============================================================================
# PURPOSE:
#   Create new meaningful features from existing columns.
#   Better features = Better model performance.
#
# WHAT WE CREATE:
#   1. Age Groups (Young, Middle-aged, Senior, Elder)
#   2. Credit Score Categories (Poor, Fair, Good, Excellent)
#   3. Balance to Salary Ratio
#   4. Age Balance Interaction
#   5. Products Activity Score
#   6. Encode categorical columns (convert text to numbers)
#
# WHY ENCODING?
#   ML models only understand NUMBERS.
#   "Male"/"Female" must become 0/1
#   "France"/"Spain"/"Germany" must become numbers
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: create_age_groups()
# PURPOSE: Convert continuous Age into meaningful categories
# -----------------------------------------------------------------------------
def create_age_groups(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create age group categories from the Age column.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with new Age_Group column
    """

    logger.info("Creating age groups...")

    age_bins: list = config["feature_engineering"]["age_bins"]
    age_labels: list = config["feature_engineering"]["age_labels"]

    # -----------------------------------------------------------------
    # pd.cut() divides a continuous column into discrete bins
    # Example:
    # Age = 25 → falls in bin (0, 30] → label "Young"
    # Age = 42 → falls in bin (30, 45] → label "Middle-aged"
    # -----------------------------------------------------------------
    df["Age_Group"] = pd.cut(
        df["Age"].astype(float),
        bins=age_bins,
        labels=age_labels,
        include_lowest=True
    )

    # Convert to string type for consistency
    df["Age_Group"] = df["Age_Group"].astype(str)

    # Show distribution of age groups
    logger.info("  Age Group distribution:")
    age_dist = df["Age_Group"].value_counts()
    for group, count in age_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"    {group}: {count:,} ({percentage:.1f}%)")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 2: create_credit_score_categories()
# PURPOSE: Convert CreditScore number into meaningful categories
# -----------------------------------------------------------------------------
def create_credit_score_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create credit score categories from CreditScore column.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with new Credit_Category column
    """

    logger.info("Creating credit score categories...")

    def categorize_credit(score: float) -> str:
        """Convert a credit score number to a category string."""
        if score < 580:
            return "Poor"
        elif score < 670:
            return "Fair"
        elif score < 740:
            return "Good"
        elif score < 800:
            return "Very Good"
        else:
            return "Excellent"

    df["Credit_Category"] = df["CreditScore"].astype(float).apply(categorize_credit)

    logger.info("  Credit Category distribution:")
    credit_dist = df["Credit_Category"].value_counts()
    for category, count in credit_dist.items():
        percentage = (count / len(df)) * 100
        logger.info(f"    {category}: {count:,} ({percentage:.1f}%)")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 3: create_ratio_features()
# PURPOSE: Create ratio features that capture relationships between columns
# -----------------------------------------------------------------------------
def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features from existing numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with new ratio columns
    """

    logger.info("Creating ratio features...")

    # -----------------------------------------------------------------
    # Convert all columns to float first to avoid type errors
    # This is the KEY fix — ensures all arithmetic works correctly
    # -----------------------------------------------------------------
    balance = df["Balance"].astype(float)
    salary = df["EstimatedSalary"].astype(float)
    credit = df["CreditScore"].astype(float)
    age = df["Age"].astype(float)
    products = df["NumOfProducts"].astype(float)

    # Feature 1: Balance to Salary Ratio
    # How much of their salary do they keep in the bank?
    df["Balance_Salary_Ratio"] = balance / (salary + 1)
    logger.info("  Created: Balance_Salary_Ratio")

    # Feature 2: Balance per Product
    # How much balance does each product represent?
    df["Balance_Per_Product"] = balance / (products + 1)
    logger.info("  Created: Balance_Per_Product")

    # Feature 3: Credit Score to Age Ratio
    # How good is their credit score relative to their age?
    df["CreditScore_Age_Ratio"] = credit / (age + 1)
    logger.info("  Created: CreditScore_Age_Ratio")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 4: create_interaction_features()
# PURPOSE: Capture combined effects of two columns together
# -----------------------------------------------------------------------------
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between pairs of columns.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with new interaction columns
    """

    logger.info("Creating interaction features...")

    # -----------------------------------------------------------------
    # Convert ALL columns to float before any arithmetic
    # This prevents the TypeError we saw earlier
    # -----------------------------------------------------------------
    age = df["Age"].astype(float)
    balance = df["Balance"].astype(float)
    products = df["NumOfProducts"].astype(float)
    active = df["IsActiveMember"].astype(float)
    tenure = df["Tenure"].astype(float)

    # Feature 1: Age x Balance Interaction
    # Older customers with high balance = very valuable
    df["Age_Balance_Interaction"] = age * balance
    logger.info("  Created: Age_Balance_Interaction")

    # Feature 2: Products x Active Member Score
    # Active members with multiple products = very loyal
    df["Products_Activity_Score"] = products * active
    logger.info("  Created: Products_Activity_Score")

    # Feature 3: Has Zero Balance Flag
    # Customers with zero balance are often about to leave!
    df["Has_Zero_Balance"] = (balance == 0).astype(int)
    logger.info("  Created: Has_Zero_Balance")

    # Feature 4: Tenure to Age Ratio
    # How long have they been a customer relative to their age?
    df["Tenure_Age_Ratio"] = tenure / (age + 1)
    logger.info("  Created: Tenure_Age_Ratio")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 5: encode_categorical_columns()
# PURPOSE: Convert text columns to numbers
#
# TWO ENCODING METHODS:
# 1. LABEL ENCODING - for binary columns (2 unique values)
#    Male → 0, Female → 1
#
# 2. ONE-HOT ENCODING - for columns with 3+ unique values
#    Geography_France: 1 or 0
#    Geography_Spain:  1 or 0
#    (Germany is the reference — when both are 0, it means Germany)
# -----------------------------------------------------------------------------
def encode_categorical_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Encode categorical columns into numerical format.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns
    """

    logger.info("Encoding categorical columns...")

    # -----------------------------------------------------------------
    # LABEL ENCODING for Gender (binary column)
    # Male → 0, Female → 1
    # -----------------------------------------------------------------
    if "Gender" in df.columns:
        gender_mapping = {"Male": 0, "Female": 1}
        df["Gender"] = df["Gender"].map(gender_mapping)
        # Fill any unmapped values with 0
        df["Gender"] = df["Gender"].fillna(0).astype(int)
        logger.info("  Gender encoded: Male=0, Female=1")

    # -----------------------------------------------------------------
    # ONE-HOT ENCODING for Geography (3 categories)
    # pd.get_dummies() creates binary columns for each category
    # drop_first=True removes one column to avoid multicollinearity
    # -----------------------------------------------------------------
    if "Geography" in df.columns:
        geography_dummies = pd.get_dummies(
            df["Geography"].astype(str),
            drop_first=True,
            prefix="Geo"
        )

        # Convert boolean columns to integer (True→1, False→0)
        geography_dummies = geography_dummies.astype(int)

        # Add new columns to DataFrame
        df = pd.concat([df, geography_dummies], axis=1)

        # Remove original Geography column
        df = df.drop(columns=["Geography"])

        logger.info(
            f"  Geography one-hot encoded: "
            f"{geography_dummies.columns.tolist()}"
        )

    # -----------------------------------------------------------------
    # Encode Age_Group (our newly created column)
    # Young=0, Middle-aged=1, Senior=2, Elder=3
    # -----------------------------------------------------------------
    if "Age_Group" in df.columns:
        age_group_mapping = {
            "Young": 0,
            "Middle-aged": 1,
            "Senior": 2,
            "Elder": 3
        }
        df["Age_Group"] = df["Age_Group"].map(age_group_mapping)
        df["Age_Group"] = df["Age_Group"].fillna(0).astype(int)
        logger.info("  Age_Group encoded: Young=0, Middle-aged=1, Senior=2, Elder=3")

    # -----------------------------------------------------------------
    # Encode Credit_Category (our newly created column)
    # Poor=0, Fair=1, Good=2, Very Good=3, Excellent=4
    # -----------------------------------------------------------------
    if "Credit_Category" in df.columns:
        credit_mapping = {
            "Poor": 0,
            "Fair": 1,
            "Good": 2,
            "Very Good": 3,
            "Excellent": 4
        }
        df["Credit_Category"] = df["Credit_Category"].map(credit_mapping)
        df["Credit_Category"] = df["Credit_Category"].fillna(0).astype(int)
        logger.info("  Credit_Category encoded: Poor=0, Fair=1, Good=2, VeryGood=3, Excellent=4")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 6: engineer_features()
# PURPOSE: Master function that runs ALL feature engineering steps
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run all feature engineering steps in correct order.

    Args:
        df (pd.DataFrame): Cleaned DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """

    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    original_columns = df.shape[1]
    logger.info(f"Columns before engineering: {original_columns}")

    # Step 1: Create age groups
    df = create_age_groups(df, config)

    # Step 2: Create credit score categories
    df = create_credit_score_categories(df)

    # Step 3: Create ratio features
    df = create_ratio_features(df)

    # Step 4: Create interaction features
    df = create_interaction_features(df)

    # Step 5: Encode categorical columns (MUST be last!)
    # Why last? Because steps 1-4 create new categorical columns
    # that also need encoding
    df = encode_categorical_columns(df, config)

    # -----------------------------------------------------------------
    # Final check for any NaN values accidentally created
    # Some operations like division can create NaN
    # We fill all remaining NaN with 0 to be safe
    # -----------------------------------------------------------------
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"NaN values found after engineering: {nan_count}")
        logger.warning("Filling all NaN with 0...")
        df = df.fillna(0)
    else:
        logger.info("No NaN values created during engineering")

    # -----------------------------------------------------------------
    # Convert entire DataFrame to float for ML compatibility
    # ML models need all values to be numeric
    # -----------------------------------------------------------------
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            logger.warning(f"Could not convert column {col} to float")

    # Final summary
    new_columns = df.shape[1]
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE!")
    logger.info(f"Columns before: {original_columns}")
    logger.info(f"Columns after:  {new_columns}")
    logger.info(f"New features created: {new_columns - original_columns}")
    logger.info(f"All columns: {df.columns.tolist()}")
    logger.info("=" * 60)

    return df


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# Run this file directly to test:
#   python src/feature_engineer.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import load_config, setup_logging, load_data
    from src.data_cleaner import clean_data

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

    # Step 6: Show results
    logger.info("\nFirst 5 rows after feature engineering:")
    print(df_engineered.head())

    logger.info(f"\nFinal shape: {df_engineered.shape}")
    logger.info(f"All columns: {df_engineered.columns.tolist()}")
    logger.info("\nfeature_engineer.py completed successfully!")