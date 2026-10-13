"""Private development and synthetic sources; prediction rows never include labels."""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "models/frozen_v1/feature_contract.json"
FULL_TRAIN_ROWS = 590_540
TRAIN_ROWS = int(FULL_TRAIN_ROWS * 0.70)
VALIDATION_ROWS = int(FULL_TRAIN_ROWS * 0.15)
DEVELOPMENT_PREFIX_ROWS = TRAIN_ROWS + VALIDATION_ROWS


@dataclass(frozen=True, slots=True)
class SourceTransaction:
    source_transaction_id: int
    transaction_dt: int
    payload: dict
    fraud_label: bool | None


def contract_columns() -> tuple[str, ...]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return tuple(contract["raw_columns"])


def _python_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_validation_source(
    transaction_path: Path, identity_path: Path, limit: int
) -> list[SourceTransaction]:
    """Load the 70%-85% chronological development slice; never read HOLDOUT rows."""
    if not transaction_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("Local TRAIN transaction and identity CSV files are required")
    if limit < 1:
        raise ValueError("limit must be positive")
    frame = pd.read_csv(
        transaction_path,
        nrows=DEVELOPMENT_PREFIX_ROWS,
        low_memory=False,
    )
    if len(frame) < DEVELOPMENT_PREFIX_ROWS:
        raise ValueError(
            "TRAIN transaction CSV is shorter than expected; refusing an ambiguous split"
        )
    if "isFraud" not in frame or "TransactionDT" not in frame:
        raise ValueError("Development source is missing TransactionDT or isFraud")
    if (
        not frame["TransactionDT"].is_monotonic_increasing
        or not frame["TransactionID"].is_monotonic_increasing
    ):
        raise ValueError(
            "Development prefix is not chronological; refusing to inspect HOLDOUT rows"
        )
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
    train_ids = frame.iloc[:TRAIN_ROWS]["TransactionID"].astype(int)
    validation = frame.iloc[TRAIN_ROWS:DEVELOPMENT_PREFIX_ROWS].copy()
    validation_ids = set(validation["TransactionID"].astype(int))
    max_validation_id = max(validation_ids)
    if max(train_ids) >= min(validation_ids):
        raise ValueError("TRAIN and VALIDATION IDs do not provide a safe identity prefix")

    # Inspect only identity IDs to find the end of the allowed prefix. Then
    # read feature columns only through that row, never through the next split.
    id_chunks = pd.read_csv(
        identity_path,
        usecols=["TransactionID"],
        chunksize=50_000,
    )
    identity_prefix_rows = 0
    last_id = None
    for chunk in id_chunks:
        ids = pd.to_numeric(chunk.TransactionID, errors="raise")
        if last_id is not None and ids.iloc[0] < last_id:
            raise ValueError(
                "Identity rows are not ordered by TransactionID; refusing broad data access"
            )
        allowed = ids <= max_validation_id
        identity_prefix_rows += int(allowed.sum())
        last_id = ids.iloc[-1]
        if not allowed.all():
            break
    columns = {"TransactionID", "DeviceType", "DeviceInfo"}
    identity = pd.read_csv(
        identity_path,
        nrows=identity_prefix_rows,
        usecols=lambda name: name in columns or name.startswith("id_"),
        low_memory=False,
    )
    identity = identity.loc[identity.TransactionID.isin(validation_ids)]
    validation = validation.merge(
        identity, on="TransactionID", how="left", suffixes=("", "_identity")
    )
    for column in identity.columns:
        if column == "TransactionID":
            continue
        suffixed = f"{column}_identity"
        if suffixed in validation:
            if column in validation:
                validation[column] = validation[column].where(
                    validation[column].notna(), validation[suffixed]
                )
            else:
                validation[column] = validation[suffixed]
            validation.drop(columns=[suffixed], inplace=True)

    raw_columns = contract_columns()
    for column in raw_columns:
        if column not in validation:
            validation[column] = None
    validation = validation.head(limit)
    rows = []
    for row in validation.to_dict(orient="records"):
        source_id = int(row["TransactionID"])
        transaction_dt = int(row["TransactionDT"])
        label = bool(row["isFraud"])
        payload = {column: _python_value(row[column]) for column in raw_columns}
        rows.append(SourceTransaction(source_id, transaction_dt, payload, label))
    return rows


def make_synthetic_source(limit: int, seed: int = 42) -> list[SourceTransaction]:
    """Build artificial, schema-complete transactions without development data."""
    if limit < 1:
        raise ValueError("limit must be positive")
    rng = random.Random(seed)
    columns = contract_columns()
    # Synthetic-only relative clock: later independent live runs follow prior runs.
    base_dt = int(time.time())
    rows = []
    for position in range(limit):
        source_id = position + 1
        payload = dict.fromkeys(columns)
        payload.update(
            {
                "TransactionID": source_id,
                # Same-timestamp rows are ordered by source/simulation ID.
                "TransactionDT": base_dt,
                "TransactionAmt": round(rng.uniform(5, 600), 2),
                "ProductCD": rng.choice(("W", "C", "R", "H")),
                "card1": 10_000 + position % 17,
                "card2": 100 + position % 13,
                "card3": 150,
                "card4": rng.choice(("visa", "mastercard")),
                "card5": 226,
                "card6": rng.choice(("debit", "credit")),
                "addr1": 100 + position % 21,
                "addr2": 87,
                "P_emaildomain": rng.choice(("gmail.com", "yahoo.com", "hotmail.com")),
                "R_emaildomain": rng.choice(("gmail.com", "anonymous.com")),
                "DeviceType": rng.choice(("desktop", "mobile")),
                "DeviceInfo": rng.choice(("Windows", "iOS", "Android")),
            }
        )
        label = position % 9 == 0
        rows.append(SourceTransaction(source_id, payload["TransactionDT"], payload, label))
    return rows


def prepare_inference_payload(source: SourceTransaction, simulation_id: int, scenario: str) -> dict:
    """Return a sanitized copy with optional explicit drift perturbations."""
    payload = dict(source.payload)
    payload.pop("isFraud", None)
    payload["TransactionID"] = simulation_id
    if scenario == "drifted":
        amount = payload.get("TransactionAmt")
        if amount is not None:
            payload["TransactionAmt"] = round(float(amount) * 8.0 + 10_000, 2)
        ordinal = simulation_id % 4
        if ordinal == 0:
            payload["ProductCD"] = "SIMULATOR_UNSEEN"
        if ordinal in (0, 1):
            payload["card4"] = None
    if "isFraud" in payload:
        raise AssertionError("Fraud label leaked into inference payload")
    return payload
