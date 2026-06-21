# =============================================================================
# src/data_loader.py
# =============================================================================
# PURPOSE:
#   This file is responsible for ONE thing only — loading data.
#   It reads the CSV file, validates it, and returns a pandas DataFrame.
#   Every other file in this project will use this file to get data.
#
# CONCEPTS USED:
#   - Logging (professional print statements)
#   - Type hints (telling Python what data types to expect)
#   - Exception handling (handling errors gracefully)
#   - Docstrings (explaining what functions do)
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# Think of imports like "tools from a toolbox"
# Each import gives us access to extra functionality
# -----------------------------------------------------------------------------

import os                          # os = Operating System tools (file paths, folders)
import logging                     # logging = professional way to track what's happening
from pathlib import Path           # Path = modern way to handle file paths (works on ALL OS)
from typing import Optional, Tuple # typing = helps us write type hints

import pandas as pd                # pandas = most important data science library
import yaml                        # yaml = reads our config.yaml file


# -----------------------------------------------------------------------------
# LOGGER SETUP
# We create a logger specifically for this file.
# __name__ is a Python special variable — it gives this logger the name
# of this file (src.data_loader), making logs easy to trace.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: setup_logging()
# PURPOSE: Set up the logging system for the entire project.
# This function is called ONCE at the start of the project.
# After this, every logger.info(), logger.error() etc. works properly.
# -----------------------------------------------------------------------------
def setup_logging(config: dict) -> None:
    """
    Configure the logging system using settings from config.yaml.

    What is a docstring?
    This text between triple quotes is called a docstring.
    It explains what the function does, what it takes as input,
    and what it returns. Professional code ALWAYS has docstrings.

    Args:
        config (dict): The full configuration dictionary loaded from config.yaml

    Returns:
        None: This function sets up logging but doesn't return anything
    """

    # Extract logging settings from config
    # config["logging"]["level"] gives us "INFO" (a string)
    log_level_str: str = config["logging"]["level"]

    # config["logging"]["filename"] gives us "customer_analytics.log"
    log_filename: str = config["logging"]["filename"]

    # config["logging"]["format"] gives us the message format string
    log_format: str = config["logging"]["format"]

    # config["paths"]["logs_dir"] gives us "logs/"
    logs_dir: str = config["paths"]["logs_dir"]

    # ---------------------------------------------------------------------
    # Create the logs/ folder if it doesn't exist
    # exist_ok=True means: don't crash if folder already exists
    # ---------------------------------------------------------------------
    os.makedirs(logs_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Convert string "INFO" to actual logging level number
    # logging.getLevelName("INFO") returns the number 20
    # This is what Python's logging system understands internally
    # ---------------------------------------------------------------------
    log_level = logging.getLevelName(log_level_str)

    # ---------------------------------------------------------------------
    # Full path to log file: "logs/customer_analytics.log"
    # os.path.join() combines folder + filename correctly on ALL operating systems
    # On Windows: logs\customer_analytics.log
    # On Mac/Linux: logs/customer_analytics.log
    # ---------------------------------------------------------------------
    log_filepath: str = os.path.join(logs_dir, log_filename)

    # ---------------------------------------------------------------------
    # basicConfig() is the main setup function for logging
    # It configures:
    #   - level: minimum severity to record (INFO and above)
    #   - format: how each log message looks
    #   - handlers: WHERE to write logs (file AND console)
    # ---------------------------------------------------------------------
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            # Handler 1: Write logs to a FILE (permanent record)
            logging.FileHandler(log_filepath, encoding="utf-8"),

            # Handler 2: Also show logs in CONSOLE (terminal)
            # This way you see logs in real-time while running
            logging.StreamHandler()
        ]
    )

    # Now that logging is set up, we can use it!
    logger.info("=" * 60)
    logger.info("Logging system initialized successfully")
    logger.info(f"Log file location: {log_filepath}")
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# FUNCTION 2: load_config()
# PURPOSE: Read config.yaml and return it as a Python dictionary
#
# Why a separate function?
# Because MULTIPLE files need the config. Having one function that loads
# config means if the config structure changes, we fix it in ONE place.
# -----------------------------------------------------------------------------
def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load and return the project configuration from config.yaml.

    Args:
        config_path (str): Path to the config file.
                          Default value is "config/config.yaml"
                          The "=" means this argument is optional —
                          if you don't pass it, it uses the default.

    Returns:
        dict: Configuration as a Python dictionary

    Raises:
        FileNotFoundError: If config.yaml doesn't exist
        yaml.YAMLError: If config.yaml has syntax errors
    """

    # ---------------------------------------------------------------------
    # Path(config_path) creates a Path object
    # .exists() checks if the file actually exists on disk
    # This is BETTER than just trying to open it and catching errors
    # because we can give a more helpful error message
    # ---------------------------------------------------------------------
    if not Path(config_path).exists():
        # This creates a detailed error message
        error_msg = f"Configuration file not found at: {config_path}"

        # logger.error() records this as an ERROR level log
        logger.error(error_msg)

        # raise stops the program and shows this error
        # FileNotFoundError is a built-in Python exception type
        raise FileNotFoundError(error_msg)

    # ---------------------------------------------------------------------
    # Try to open and read the YAML file
    # "try/except" = attempt something, catch errors if they happen
    # ---------------------------------------------------------------------
    try:
        # open() opens the file
        # encoding="utf-8" handles special characters (emojis etc.)
        # "as file" gives the opened file a name we can use
        with open(config_path, encoding="utf-8") as file:

            # yaml.safe_load() converts YAML text → Python dictionary
            # safe_load is safer than load() — prevents code injection
            config = yaml.safe_load(file)

        logger.info(f"Configuration loaded successfully from: {config_path}")

        # Return the config dictionary to whoever called this function
        return config

    except yaml.YAMLError as error:
        # This catches YAML syntax errors (wrong indentation etc.)
        error_msg = f"Error parsing config.yaml: {error}"
        logger.error(error_msg)
        raise yaml.YAMLError(error_msg)


# -----------------------------------------------------------------------------
# FUNCTION 3: load_data()
# PURPOSE: Load the CSV dataset and return it as a pandas DataFrame
#
# What is a DataFrame?
# Think of it like an Excel spreadsheet in Python.
# It has rows and columns and we can do powerful operations on it.
# -----------------------------------------------------------------------------
def load_data(config: dict) -> pd.DataFrame:
    """
    Load the raw CSV dataset into a pandas DataFrame.

    Args:
        config (dict): Configuration dictionary (from load_config())

    Returns:
        pd.DataFrame: The loaded dataset as a DataFrame

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        Exception: For any other unexpected errors
    """

    # Extract the file path from config
    # This reads: config → paths → raw_data → "data/raw/customer_data.csv"
    raw_data_path: str = config["paths"]["raw_data"]

    logger.info(f"Attempting to load dataset from: {raw_data_path}")

    # ---------------------------------------------------------------------
    # Check if the file exists before trying to open it
    # This gives us a clear, helpful error message
    # ---------------------------------------------------------------------
    if not Path(raw_data_path).exists():

        # IMPORTANT: Our config says "customer_data.csv" but Kaggle
        # downloaded it as "Churn_Modelling.csv"
        # So we check for the actual filename too!
        actual_path = "data/raw/Churn_Modelling.csv"

        if Path(actual_path).exists():
            logger.warning(
                f"File not found at {raw_data_path}. "
                f"Using actual file at {actual_path}"
            )
            raw_data_path = actual_path
        else:
            error_msg = f"Dataset not found at: {raw_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # ---------------------------------------------------------------------
    # Try to load the CSV file
    # pd.read_csv() is the pandas function to read CSV files
    # It automatically detects column names from the first row
    # ---------------------------------------------------------------------
    try:
        logger.info("Reading CSV file...")

        # pd.read_csv() reads the file and creates a DataFrame
        df: pd.DataFrame = pd.read_csv(raw_data_path)

        logger.info("CSV file read successfully!")

        # Return the DataFrame
        return df

    except pd.errors.EmptyDataError:
        error_msg = "The CSV file is empty!"
        logger.error(error_msg)
        raise pd.errors.EmptyDataError(error_msg)

    except Exception as error:
        # This catches ANY other unexpected error
        error_msg = f"Unexpected error loading data: {error}"
        logger.error(error_msg)
        raise Exception(error_msg)


# -----------------------------------------------------------------------------
# FUNCTION 4: validate_data()
# PURPOSE: Check that the loaded data has what we expect
#
# Why validate?
# What if someone gives us a wrong CSV file?
# What if the target column is missing?
# Validation catches these problems EARLY with helpful messages.
# -----------------------------------------------------------------------------
def validate_data(df: pd.DataFrame, config: dict) -> bool:
    """
    Validate that the DataFrame has the expected structure.

    Args:
        df (pd.DataFrame): The loaded DataFrame to validate
        config (dict): Configuration dictionary

    Returns:
        bool: True if valid, raises Exception if invalid
    """

    logger.info("Validating dataset structure...")

    # ---------------------------------------------------------------------
    # Check 1: Is the DataFrame empty?
    # df.empty returns True if there are no rows
    # ---------------------------------------------------------------------
    if df.empty:
        raise ValueError("Dataset is empty — no rows found!")

    # ---------------------------------------------------------------------
    # Check 2: Does the target column exist?
    # The target column is what we're trying to predict (Exited)
    # If it's missing, we can't train any model!
    # ---------------------------------------------------------------------
    target_column: str = config["data"]["target_column"]

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset! "
            f"Available columns: {df.columns.tolist()}"
        )

    # ---------------------------------------------------------------------
    # Check 3: Do we have enough rows?
    # ML models need a minimum amount of data to learn patterns
    # Less than 100 rows is usually not enough
    # ---------------------------------------------------------------------
    min_rows = 100
    if len(df) < min_rows:
        raise ValueError(
            f"Dataset has only {len(df)} rows. "
            f"Minimum required: {min_rows}"
        )

    logger.info(f"✓ Dataset has {len(df):,} rows")          # :, adds comma formatting
    logger.info(f"✓ Dataset has {len(df.columns)} columns")
    logger.info(f"✓ Target column '{target_column}' found")
    logger.info("Dataset validation passed!")

    return True


# -----------------------------------------------------------------------------
# FUNCTION 5: get_data_summary()
# PURPOSE: Generate a summary report of the dataset
#
# This is the "first look" at your data — understanding what you have
# before doing anything else is a KEY data science practice.
# -----------------------------------------------------------------------------
def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to summarize

    Returns:
        None: Prints summary to console and logs
    """

    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)

    # Shape = (number of rows, number of columns)
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ---------------------------------------------------------------------
    # Memory usage
    # df.memory_usage(deep=True).sum() gives total memory in BYTES
    # We convert to MB (divide by 1024 twice)
    # This is important for large datasets
    # ---------------------------------------------------------------------
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"Memory Usage: {memory_mb:.2f} MB")

    # ---------------------------------------------------------------------
    # Column information
    # df.dtypes shows the data type of each column
    # int64 = integer numbers
    # float64 = decimal numbers
    # object = text/strings
    # ---------------------------------------------------------------------
    logger.info("\nColumn Data Types:")
    for column, dtype in df.dtypes.items():
        logger.info(f"  {column:<20} {str(dtype)}")

    # ---------------------------------------------------------------------
    # Missing values check
    # df.isnull() returns True/False for each cell
    # .sum() counts the Trues per column
    # ---------------------------------------------------------------------
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    logger.info(f"\nTotal Missing Values: {total_missing}")

    if total_missing > 0:
        logger.warning("Missing values found!")
        for column, count in missing_values[missing_values > 0].items():
            percentage = (count / len(df)) * 100
            logger.warning(f"  {column}: {count} missing ({percentage:.1f}%)")
    else:
        logger.info("No missing values found!")

    # ---------------------------------------------------------------------
    # Duplicate rows check
    # df.duplicated() returns True for rows that are exact duplicates
    # .sum() counts them
    # ---------------------------------------------------------------------
    duplicate_count = df.duplicated().sum()
    logger.info(f"\nDuplicate Rows: {duplicate_count}")

    # ---------------------------------------------------------------------
    # Target column distribution
    # Shows how many customers churned vs stayed
    # value_counts() counts occurrences of each unique value
    # ---------------------------------------------------------------------
    logger.info("\nTarget Column Distribution:")
    target_counts = df.iloc[:, -1].value_counts()
    for value, count in target_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  Class {value}: {count:,} ({percentage:.1f}%)")

    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# This block runs ONLY when you run this file directly:
#   python src/data_loader.py
#
# It does NOT run when another file imports this file.
# This is a Python best practice — every file can be both
# imported AND run directly.
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Step 1: Load configuration
    config = load_config()

    # Step 2: Set up logging using config settings
    setup_logging(config)

    # Step 3: Load the dataset
    df = load_data(config)

    # Step 4: Validate the dataset
    validate_data(df, config)

    # Step 5: Print summary
    get_data_summary(df)

    # Step 6: Show first 5 rows
    logger.info("\nFirst 5 rows of dataset:")
    print(df.head())

    logger.info("\ndata_loader.py completed successfully!")