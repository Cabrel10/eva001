# -*- coding: utf-8 -*-
"""
End-to-end test for the data pipeline.

This test simulates the full process:
1. Download raw data using api_manager.py.
2. Process the raw data using data_pipeline.py.
3. Validate the output Parquet file.
"""

import subprocess
import os
import pandas as pd
import pytest
import logging
from pathlib import Path

# Configure logging for validation
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
VALIDATION_LOG_FILE = LOG_DIR / "validation_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(VALIDATION_LOG_FILE, mode='a'), # Append to the log file
        logging.StreamHandler() # Also print to console
    ]
)

# --- Test Configuration ---
TOKEN = "ETHUSDT"
EXCHANGE = "binance" # Use a reliable exchange like Binance
START_DATE = "2024-01-01" # Use a longer period to ensure enough data
END_DATE = "2024-03-01"   # 2 months period
TIMEFRAME = "1h" # Using 1h timeframe for potentially faster download/processing

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
RAW_DATA_FILE = RAW_DATA_DIR / f"test_e2e_{TOKEN.lower()}_raw.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / f"test_e2e_{TOKEN.lower()}_final.parquet"

API_MANAGER_SCRIPT = Path("utils/api_manager.py")
DATA_PIPELINE_SCRIPT = Path("data/pipelines/data_pipeline.py")

EXPECTED_COLUMNS = 38 # As specified in the requirements

# Clean up previous test files before running
@pytest.fixture(scope="module", autouse=True)
def cleanup_files():
    """Remove test files before and after the test module runs."""
    if RAW_DATA_FILE.exists():
        RAW_DATA_FILE.unlink()
        logging.info(f"Removed previous raw test file: {RAW_DATA_FILE}")
    if PROCESSED_DATA_FILE.exists():
        PROCESSED_DATA_FILE.unlink()
        logging.info(f"Removed previous processed test file: {PROCESSED_DATA_FILE}")
    yield # Test runs here
    # Cleanup after test
    if RAW_DATA_FILE.exists():
        RAW_DATA_FILE.unlink()
        logging.info(f"Cleaned up raw test file: {RAW_DATA_FILE}")
    if PROCESSED_DATA_FILE.exists():
        PROCESSED_DATA_FILE.unlink()
        logging.info(f"Cleaned up processed test file: {PROCESSED_DATA_FILE}")


def run_command(command: list, step_name: str) -> bool:
    """Helper function to run a subprocess command and log results."""
    logging.info(f"--- Running Step: {step_name} ---")
    logging.info(f"Command: {' '.join(command)}")
    try:
        # Using shell=False is generally safer, command should be a list
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"STDERR:\n{result.stderr}") # Log stderr even on success as warnings
        logging.info(f"--- Step '{step_name}' completed successfully ---")
        return True
    except subprocess.CalledProcessError as e:
        # Log stdout/stderr specifically for CalledProcessError for better diagnosis
        logging.error(f"!!! Step '{step_name}' failed with CalledProcessError !!!")
        logging.error(f"Return Code: {e.returncode}")
        # Ensure stdout and stderr are logged if they exist
        stdout_content = e.stdout.strip() if e.stdout else "N/A"
        stderr_content = e.stderr.strip() if e.stderr else "N/A"
        logging.error(f"STDOUT:\n{stdout_content}")
        logging.error(f"STDERR:\n{stderr_content}")
        return False
    except FileNotFoundError:
        logging.error(f"!!! Step '{step_name}' failed: Script not found at {command[1]} !!!")
        return False
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        logging.error(f"!!! An unexpected error occurred during step '{step_name}': {e} !!!")
        # Log traceback for unexpected errors if possible (might require importing traceback)
        # import traceback
        # logging.error(traceback.format_exc())
        return False


def test_end_to_end_pipeline():
    """
    Tests the full data pipeline from raw data download to processed parquet file.
    """
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download Raw Data ---
    api_command = [
        "python", str(API_MANAGER_SCRIPT),
        "--token", TOKEN,
        "--exchange", EXCHANGE,
        "--start", START_DATE,
        "--end", END_DATE,
        "--timeframe", TIMEFRAME,
        "--output", str(RAW_DATA_FILE)
    ]
    success = run_command(api_command, "Download Raw Data")
    assert success, "Failed to download raw data."

    # --- Step 2: Validate Raw Data File ---
    logging.info("--- Running Step: Validate Raw Data File ---")
    assert RAW_DATA_FILE.exists(), f"Raw data file not found: {RAW_DATA_FILE}"
    assert RAW_DATA_FILE.stat().st_size > 0, f"Raw data file is empty: {RAW_DATA_FILE}"
    logging.info(f"Raw data file '{RAW_DATA_FILE}' exists and is not empty.")
    logging.info("--- Step 'Validate Raw Data File' completed successfully ---")


    # --- Step 3: Run Data Processing Pipeline ---
    # Note: Corrected arguments based on previous error message (--input, --output)
    pipeline_command = [
        "python", str(DATA_PIPELINE_SCRIPT),
        "--input", str(RAW_DATA_FILE),
        "--output", str(PROCESSED_DATA_FILE)
    ]
    success = run_command(pipeline_command, "Run Data Processing Pipeline")
    assert success, "Failed to run the data processing pipeline."

    # --- Step 4: Validate Processed Data File ---
    logging.info("--- Running Step: Validate Processed Data File ---")
    assert PROCESSED_DATA_FILE.exists(), f"Processed data file not found: {PROCESSED_DATA_FILE}"
    assert PROCESSED_DATA_FILE.stat().st_size > 0, f"Processed data file is empty: {PROCESSED_DATA_FILE}"
    logging.info(f"Processed data file '{PROCESSED_DATA_FILE}' exists and is not empty.")

    # Load and validate shape
    try:
        df_processed = pd.read_parquet(PROCESSED_DATA_FILE)
        logging.info(f"Successfully loaded processed data. Shape: {df_processed.shape}")
        num_rows, num_cols = df_processed.shape
        assert num_rows > 0, "Processed DataFrame has no rows."
        assert num_cols == EXPECTED_COLUMNS, f"Expected {EXPECTED_COLUMNS} columns, but found {num_cols}."
        logging.info(f"Processed DataFrame has {num_rows} rows and the expected {num_cols} columns.")
        logging.info(f"Columns found: {df_processed.columns.tolist()}")

    except Exception as e:
        logging.error(f"!!! Failed to load or validate the processed Parquet file: {e} !!!")
        pytest.fail(f"Failed to load or validate the processed Parquet file: {e}")

    logging.info("--- Step 'Validate Processed Data File' completed successfully ---")
    logging.info("====== End-to-End Pipeline Test Passed ======")


# To run this test:
# pytest tests/validation/test_end_to_end_pipeline.py
