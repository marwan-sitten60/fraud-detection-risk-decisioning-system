"""Configuration and validation for local traffic simulation runs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    scenario: str = "healthy"
    source: str = "validation"
    max_transactions: int = 32
    batch_size: int = 8
    target_rps: float | None = None
    time_scale: float = 86400.0
    seed: int = 42
    label_delay_min_days: float = 0.01
    label_delay_max_days: float = 0.01
    labels_enabled: bool | None = None
    base_url: str = "http://127.0.0.1:8000"
    run_id: str | None = None
    retry_limit: int = 2
    timeout_seconds: float = 180.0
    report_dir: str = "reports/simulation_runs"
    write_latest_report: bool = True
    train_transaction_path: str = "data/raw/train_transaction.csv"
    train_identity_path: str = "data/raw/train_identity.csv"

    def __post_init__(self) -> None:
        if self.scenario not in {"healthy", "drifted"}:
            raise ValueError("scenario must be healthy or drifted")
        if self.source not in {"validation", "synthetic"}:
            raise ValueError("source must be validation or synthetic")
        if self.max_transactions < 1 or self.batch_size < 1:
            raise ValueError("max_transactions and batch_size must be positive")
        if self.target_rps is not None and self.target_rps <= 0:
            raise ValueError("target_rps must be positive when configured")
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if self.label_delay_min_days < 0 or self.label_delay_max_days < 0:
            raise ValueError("label delays cannot be negative")
        if self.label_delay_min_days > self.label_delay_max_days:
            raise ValueError("minimum label delay cannot exceed maximum delay")
        if self.retry_limit < 0 or self.retry_limit > 5:
            raise ValueError("retry_limit must be between 0 and 5")

    @property
    def deliver_labels(self) -> bool:
        if self.scenario == "drifted":
            return self.labels_enabled is True
        return self.labels_enabled is not False
