"""Package-level smoke tests."""


def test_package_import() -> None:
    """The installed top-level package should import successfully."""
    import fraud_detection

    assert fraud_detection.__version__ == "0.1.0"
