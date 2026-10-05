"""Environment-backed application settings."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    environment: str
    database_url: str | None
    mlflow_tracking_uri: str
    model_path: Path

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        """Build settings from process variables, optionally loading a local env file first."""
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        environment = os.getenv("ENVIRONMENT", "local").strip().lower()
        if not environment:
            raise ValueError("ENVIRONMENT must not be empty")

        model_path = os.getenv("MODEL_PATH", "models/model.joblib").strip()
        if not model_path:
            raise ValueError("MODEL_PATH must not be empty")

        return cls(
            environment=environment,
            database_url=os.getenv("DATABASE_URL") or None,
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            model_path=Path(model_path),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process-wide settings."""
    return Settings.from_env()
