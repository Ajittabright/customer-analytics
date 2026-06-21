import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_config, setup_logging, load_data, get_data_summary
from src.data_cleaner import clean_data
from src.feature_engineer import engineer_features
from src.eda import run_eda
from src.model_trainer import train_pipeline
from src.model_evaluator import evaluate_pipeline


def run_pipeline():

    pipeline_start = time.time()

    print("=" * 60)
    print("CUSTOMER ANALYTICS AND SALES PREDICTION SYSTEM")
    print("=" * 60)

    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("PIPELINE STARTED")

    logger.info("[STEP 2/7] Loading raw data...")
    step_start = time.time()
    df_raw = load_data(config)
    get_data_summary(df_raw)
    logger.info(f"Step 2 done in {time.time()-step_start:.2f}s")

    logger.info("[STEP 3/7] Cleaning data...")
    step_start = time.time()
    df_cleaned = clean_data(df_raw, config)
    logger.info(f"Step 3 done in {time.time()-step_start:.2f}s")

    logger.info("[STEP 4/7] Running EDA...")
    step_start = time.time()
    run_eda(df_cleaned, config)
    logger.info(f"Step 4 done in {time.time()-step_start:.2f}s")

    logger.info("[STEP 5/7] Engineering features...")
    step_start = time.time()
    df_engineered = engineer_features(df_cleaned, config)
    logger.info(f"Step 5 done in {time.time()-step_start:.2f}s")

    logger.info("[STEP 6/7] Training ML models...")
    step_start = time.time()
    best_model, scaler, results, best_model_name = train_pipeline(df_engineered, config)
    logger.info(f"Step 6 done in {time.time()-step_start:.2f}s")

    logger.info("[STEP 7/7] Evaluating best model...")
    step_start = time.time()
    evaluate_pipeline(df_engineered, config)
    logger.info(f"Step 7 done in {time.time()-step_start:.2f}s")

    total_time = time.time() - pipeline_start

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Best Model: {best_model_name}")
    logger.info("Run: streamlit run app/streamlit_app.py")
    logger.info("=" * 60)

    return best_model_name, results


if __name__ == "__main__":
    run_pipeline()