"""Independent CSV loaders for the IEEE-CIS Fraud Detection dataset."""

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_RAW_DATA_DIR = Path("data/raw")


def _load_csv(filename: str, data_dir: Path) -> pd.DataFrame:
    """Load one CSV file from the raw-data directory."""
    file_path = data_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(
            f"IEEE-CIS data file not found: {file_path}. Place {filename} in {data_dir}."
        )

    dataframe = pd.read_csv(file_path)
    LOGGER.info("Loaded %s with shape %s", file_path.name, dataframe.shape)
    return dataframe


def load_train_transaction(data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Load the training transaction data."""
    return _load_csv("train_transaction.csv", data_dir)


def load_train_identity(data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Load the training identity data."""
    return _load_csv("train_identity.csv", data_dir)


def load_test_transaction(data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Load the test transaction data."""
    return _load_csv("test_transaction.csv", data_dir)


def load_test_identity(data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Load the test identity data."""
    return _load_csv("test_identity.csv", data_dir)
