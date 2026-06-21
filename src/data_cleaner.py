# =============================================================================
# src/data_cleaner.py
# =============================================================================
# PURPOSE:
#   Clean the raw dataset and prepare it for feature engineering.
#   This file handles:
#     1. Dropping useless columns
#     2. Fixing data types
#     3. Handling outliers
#     4. Removing duplicates
#     5. Saving cleaned data
#
# RULE: We NEVER modify the original raw data file.
#       We always save cleaned data to data/processed/
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os                          # For creating folders
import logging                     # For professional logging
from pathlib import Path           # For file path handling

import pandas as pd                # For DataFrame operations
import numpy as np                 # For numerical operations


# -----------------------------------------------------------------------------
# LOGGER
# Each file gets its own logger named after the file
# This makes it easy to trace which file wrote which log message
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: drop_useless_columns()
# PURPOSE: Remove columns that provide NO value to our ML model
#
# WHY?
# - RowNumber: Just 1,2,3,4... tells the model nothing
# - CustomerId: Random ID number, no pattern
# - Surname: Last name has no relation to churn behavior
# Keeping them would confuse the model and waste memory!
# -----------------------------------------------------------------------------
def drop_useless_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop columns that are not useful for machine learning.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with useless columns removed
    """

    logger.info("Step 1: Dropping useless columns...")

    # Get the list of ID columns to drop from config
    # This reads: ["RowNumber", "CustomerId", "Surname"]
    columns_to_drop: list = config["data"]["id_columns"]

    # ---------------------------------------------------------------------
    # Before dropping, check which columns actually EXIST in the DataFrame
    # Why? Because if a column doesn't exist and we try to drop it,
    # pandas will throw an error!
    # ---------------------------------------------------------------------
    existing_columns_to_drop = [
        col for col in columns_to_drop    # For each column in our drop list
        if col in df.columns              # Only keep it if it exists in DataFrame
    ]

    # This is called a "List Comprehension" — a short way to write a for loop
    # Long version:
    # existing_columns_to_drop = []
    # for col in columns_to_drop:
    #     if col in df.columns:
    #         existing_columns_to_drop.append(col)

    if existing_columns_to_drop:
        # drop() removes columns
        # axis=1 means "drop columns" (axis=0 would mean "drop rows")
        # inplace=False means "return a new DataFrame, don't modify original"
        df = df.drop(columns=existing_columns_to_drop, axis=1)
        logger.info(f"  Dropped columns: {existing_columns_to_drop}")
    else:
        logger.warning("  No useless columns found to drop!")

    logger.info(f"  Remaining columns: {df.columns.tolist()}")
    logger.info(f"  New shape: {df.shape}")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 2: fix_data_types()
# PURPOSE: Ensure each column has the correct data type
#
# WHY?
# Sometimes pandas reads numbers as text (object type).
# ML models can only work with numbers, not text.
# We need to make sure everything is the right type.
# -----------------------------------------------------------------------------
def fix_data_types(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Fix data types for all columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with corrected data types
    """

    logger.info("Step 2: Fixing data types...")

    # Get column lists from config
    numerical_columns: list = config["data"]["numerical_columns"]
    categorical_columns: list = config["data"]["categorical_columns"]

    # ---------------------------------------------------------------------
    # Fix numerical columns
    # pd.to_numeric() converts a column to numbers
    # errors='coerce' means: if a value can't be converted, make it NaN
    # (instead of crashing)
    # ---------------------------------------------------------------------
    for column in numerical_columns:
        if column in df.columns:
            # Store original dtype for logging
            original_dtype = df[column].dtype

            # Convert to numeric
            df[column] = pd.to_numeric(df[column], errors='coerce')

            # Log if dtype changed
            if df[column].dtype != original_dtype:
                logger.info(
                    f"  {column}: {original_dtype} → {df[column].dtype}"
                )

    # ---------------------------------------------------------------------
    # Fix categorical columns
    # astype(str) converts to string
    # .str.strip() removes extra spaces from beginning and end
    # Example: "  Male  " → "Male"
    # ---------------------------------------------------------------------
    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    logger.info("  Data types fixed successfully!")

    # Print a summary of all column types
    logger.info("\n  Column types after fixing:")
    for col, dtype in df.dtypes.items():
        logger.info(f"    {col:<25} {str(dtype)}")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 3: handle_missing_values()
# PURPOSE: Fill in or remove missing values (NaN values)
#
# WHY?
# ML models CANNOT handle NaN values — they will crash!
# We need a strategy for each type of column:
#   - Numerical columns: Fill with MEDIAN (middle value)
#     Why median and not mean? Because mean is affected by outliers!
#     Example: Salaries [30k, 35k, 40k, 1000k]
#     Mean = 276k (misleading!), Median = 37.5k (realistic!)
#   - Categorical columns: Fill with MODE (most common value)
# -----------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with no missing values
    """

    logger.info("Step 3: Handling missing values...")

    # Count missing values before handling
    total_missing_before = df.isnull().sum().sum()
    logger.info(f"  Missing values before: {total_missing_before}")

    # If no missing values, skip this step
    if total_missing_before == 0:
        logger.info("  No missing values found! Skipping this step.")
        return df

    # Get column lists from config
    numerical_columns: list = config["data"]["numerical_columns"]
    categorical_columns: list = config["data"]["categorical_columns"]

    # ---------------------------------------------------------------------
    # Handle numerical columns — fill with MEDIAN
    # ---------------------------------------------------------------------
    for column in numerical_columns:
        if column in df.columns:
            missing_count = df[column].isnull().sum()

            if missing_count > 0:
                # Calculate the median (middle value when sorted)
                median_value = df[column].median()

                # fillna() replaces NaN with the given value
                df[column] = df[column].fillna(median_value)

                logger.info(
                    f"  {column}: Filled {missing_count} "
                    f"missing values with median ({median_value:.2f})"
                )

    # ---------------------------------------------------------------------
    # Handle categorical columns — fill with MODE (most common value)
    # .mode() returns a Series, [0] gets the first (most common) value
    # ---------------------------------------------------------------------
    for column in categorical_columns:
        if column in df.columns:
            missing_count = df[column].isnull().sum()

            if missing_count > 0:
                # mode() returns most frequent value
                mode_value = df[column].mode()[0]

                df[column] = df[column].fillna(mode_value)

                logger.info(
                    f"  {column}: Filled {missing_count} "
                    f"missing values with mode ('{mode_value}')"
                )

    # Count missing values after handling
    total_missing_after = df.isnull().sum().sum()
    logger.info(f"  Missing values after: {total_missing_after}")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 4: remove_duplicates()
# PURPOSE: Remove exact duplicate rows
#
# WHY?
# Duplicate rows make the model "memorize" certain customers
# instead of learning general patterns. This is called OVERFITTING.
# -----------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """

    logger.info("Step 4: Removing duplicate rows...")

    # Count duplicates before removal
    duplicate_count = df.duplicated().sum()
    logger.info(f"  Duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:
        rows_before = len(df)

        # drop_duplicates() removes exact duplicate rows
        # keep='first' means keep the first occurrence, remove the rest
        df = df.drop_duplicates(keep='first')

        rows_after = len(df)
        logger.info(
            f"  Removed {rows_before - rows_after} duplicate rows"
        )
    else:
        logger.info("  No duplicates found!")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 5: handle_outliers()
# PURPOSE: Cap extreme values using the IQR method
#
# WHY CAP instead of REMOVE?
# If we remove outlier rows, we lose data.
# Instead we CAP them — bring them to a reasonable boundary.
#
# THE IQR METHOD:
# Q1 = 25th percentile (25% of values are below this)
# Q3 = 75th percentile (75% of values are below this)
# IQR = Q3 - Q1 (the middle 50% range)
#
# Lower Bound = Q1 - (1.5 × IQR)
# Upper Bound = Q3 + (1.5 × IQR)
#
# Values below Lower Bound → set to Lower Bound
# Values above Upper Bound → set to Upper Bound
# This is called "CAPPING" or "WINSORIZING"
# -----------------------------------------------------------------------------
def handle_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Handle outliers in numerical columns using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: DataFrame with outliers capped
    """

    logger.info("Step 5: Handling outliers using IQR method...")

    numerical_columns: list = config["data"]["numerical_columns"]

    for column in numerical_columns:
        if column in df.columns:

            # -------------------------------------------------------------
            # Calculate Q1, Q3, and IQR
            # .quantile(0.25) = 25th percentile
            # .quantile(0.75) = 75th percentile
            # -------------------------------------------------------------
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate bounds
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            # Count outliers before capping
            outliers_count = (
                (df[column] < lower_bound) |   # Below lower bound
                (df[column] > upper_bound)      # OR above upper bound
            ).sum()

            if outliers_count > 0:
                # np.clip() caps values between lower and upper bounds
                # Values below lower_bound → become lower_bound
                # Values above upper_bound → become upper_bound
                df[column] = np.clip(
                    df[column],
                    lower_bound,
                    upper_bound
                )

                logger.info(
                    f"  {column}: Capped {outliers_count} outliers "
                    f"[{lower_bound:.2f} - {upper_bound:.2f}]"
                )
            else:
                logger.info(f"  {column}: No outliers found")

    return df


# -----------------------------------------------------------------------------
# FUNCTION 6: save_cleaned_data()
# PURPOSE: Save the cleaned DataFrame to data/processed/ folder
#
# WHY SAVE?
# So we don't have to clean data every single time we run the project.
# Future files (feature_engineer.py, model_trainer.py) can directly
# load the clean version without re-cleaning.
# -----------------------------------------------------------------------------
def save_cleaned_data(df: pd.DataFrame, config: dict) -> None:
    """
    Save the cleaned DataFrame to the processed data folder.

    Args:
        df (pd.DataFrame): Cleaned DataFrame to save
        config (dict): Configuration dictionary

    Returns:
        None
    """

    # Get the save path from config
    save_path: str = config["paths"]["processed_data"]

    logger.info(f"Saving cleaned data to: {save_path}")

    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # to_csv() saves DataFrame as CSV file
    # index=False means don't save the row numbers (0,1,2,3...)
    df.to_csv(save_path, index=False, encoding='utf-8')

    # Verify the file was saved
    if Path(save_path).exists():
        file_size = Path(save_path).stat().st_size / 1024  # Size in KB
        logger.info(f"Cleaned data saved successfully!")
        logger.info(f"File size: {file_size:.2f} KB")
    else:
        raise Exception(f"Failed to save cleaned data to {save_path}")


# -----------------------------------------------------------------------------
# FUNCTION 7: clean_data()
# PURPOSE: Master function that runs ALL cleaning steps in order
#
# WHY A MASTER FUNCTION?
# Other files only need to call ONE function: clean_data()
# They don't need to know about all the individual steps.
# This is called "ABSTRACTION" — hiding complexity behind a simple interface.
# -----------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run all data cleaning steps in the correct order.

    This is the main function other files will call.
    It orchestrates all cleaning steps.

    Args:
        df (pd.DataFrame): Raw DataFrame
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: Fully cleaned DataFrame
    """

    logger.info("=" * 60)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    # Record original shape for comparison at the end
    original_shape = df.shape
    logger.info(f"Original shape: {original_shape}")

    # ---------------------------------------------------------------------
    # Run all cleaning steps in order
    # Each step takes a DataFrame, cleans it, returns cleaned DataFrame
    # The output of one step becomes the input of the next step
    # This is called a PIPELINE
    # ---------------------------------------------------------------------

    # Step 1: Drop useless columns
    df = drop_useless_columns(df, config)

    # Step 2: Fix data types
    df = fix_data_types(df, config)

    # Step 3: Handle missing values
    df = handle_missing_values(df, config)

    # Step 4: Remove duplicates
    df = remove_duplicates(df)

    # Step 5: Handle outliers
    df = handle_outliers(df, config)

    # Step 6: Save cleaned data
    save_cleaned_data(df, config)

    # Final summary
    logger.info("=" * 60)
    logger.info("DATA CLEANING COMPLETE!")
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Cleaned shape:  {df.shape}")
    logger.info(
        f"Columns removed: "
        f"{original_shape[1] - df.shape[1]}"
    )
    logger.info("=" * 60)

    return df


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# Run this file directly to test it:
#   python src/data_cleaner.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # We need to import data_loader to get the data
    # sys.path.insert() tells Python where to find our modules
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import load_config, setup_logging, load_data

    # Step 1: Load config
    config = load_config()

    # Step 2: Setup logging
    setup_logging(config)

    # Step 3: Load raw data
    df_raw = load_data(config)

    # Step 4: Clean the data
    df_cleaned = clean_data(df_raw, config)

    # Step 5: Show result
    logger.info("\nFirst 5 rows of cleaned data:")
    print(df_cleaned.head())

    logger.info("\ndata_cleaner.py completed successfully!")