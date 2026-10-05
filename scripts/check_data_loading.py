"""Verify that the local IEEE-CIS training CSV files can be loaded."""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fraud_detection.data import load_train_identity, load_train_transaction

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _log_summary(name: str, dataframe: pd.DataFrame) -> None:
    """Log a compact summary without printing dataframe rows."""
    LOGGER.info(
        "%s: shape=%s, column_count=%d, first_columns=%s",
        name,
        dataframe.shape,
        len(dataframe.columns),
        dataframe.columns[:5].tolist(),
    )


def main() -> None:
    """Load and summarize the two IEEE-CIS training files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    datasets: tuple[tuple[str, Callable[[Path], pd.DataFrame]], ...] = (
        ("train_transaction", load_train_transaction),
        ("train_identity", load_train_identity),
    )
    for name, loader in datasets:
        dataframe = loader(RAW_DATA_DIR)
        _log_summary(name, dataframe)
        del dataframe


if __name__ == "__main__":
    main()
