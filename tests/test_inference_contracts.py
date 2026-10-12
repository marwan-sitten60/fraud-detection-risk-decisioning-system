import pandas as pd
import pytest

from fraud_detection.inference.contracts import validate_transactions

RAW = ("TransactionID", "TransactionDT", "TransactionAmt", "card1")


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [2, 3],
            "TransactionDT": [10, 11],
            "TransactionAmt": [20.0, 30.0],
            "card1": [100, None],
        }
    )


def test_input_contract_accepts_label_free_rows_without_mutation() -> None:
    source = transactions()
    original = source.copy(deep=True)
    checked = validate_transactions(source, RAW)
    pd.testing.assert_frame_equal(source, original)
    assert checked is not source
    assert tuple(checked.columns) == RAW


@pytest.mark.parametrize("target", ["isFraud", "y_true"])
def test_target_field_is_rejected(target: str) -> None:
    source = transactions()
    source[target] = 1
    with pytest.raises(ValueError, match="target fields"):
        validate_transactions(source, RAW)


def test_missing_and_duplicate_ids_fail_clearly() -> None:
    with pytest.raises(ValueError, match="missing required raw columns"):
        validate_transactions(transactions().drop(columns="card1"), RAW)
    duplicate = transactions()
    duplicate.loc[1, "TransactionID"] = duplicate.loc[0, "TransactionID"]
    with pytest.raises(ValueError, match="unique"):
        validate_transactions(duplicate, RAW)


def test_invalid_amount_and_order_fail_clearly() -> None:
    invalid = transactions()
    invalid.loc[0, "TransactionAmt"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_transactions(invalid, RAW)
    with pytest.raises(ValueError, match="ordered"):
        validate_transactions(transactions().iloc[::-1], RAW)


def test_invalid_numeric_field_fails_clearly() -> None:
    invalid = transactions()
    invalid["card1"] = invalid["card1"].astype(object)
    invalid.loc[0, "card1"] = "not-a-number"
    with pytest.raises(ValueError, match="card1 must contain numeric"):
        validate_transactions(invalid, RAW, numeric_columns=("card1",))
