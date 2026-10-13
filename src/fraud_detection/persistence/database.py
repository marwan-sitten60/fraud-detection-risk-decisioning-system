"""Database connection, schema gate, and read-only replay queries."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from fraud_detection.persistence.models import Base, InferenceRequest, Prediction, Transaction

REVISION = "20261013_01"


class PersistenceStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, url: str) -> "PersistenceStore":
        return cls(create_engine(url, pool_pre_ping=True))

    def check_ready(self) -> None:
        """Require a live connection and exactly the current Alembic schema."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            inspector = inspect(connection)
            expected = set(Base.metadata.tables)
            if not expected.issubset(set(inspector.get_table_names())):
                raise RuntimeError("Persistence schema is incomplete")
            if "alembic_version" not in inspector.get_table_names():
                raise RuntimeError("Persistence migration is missing")
            versions = (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
            )
            if versions != [REVISION]:
                raise RuntimeError("Persistence schema revision does not match the application")

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.sessions() as session, session.begin():
            yield session

    def replay_stream(self, bundle_version: str) -> list[tuple[InferenceRequest, list[dict]]]:
        """Return successful operations with original boundaries and row order."""
        with self.sessions() as session:
            requests = session.scalars(
                select(InferenceRequest)
                .where(
                    InferenceRequest.bundle_version == bundle_version,
                    InferenceRequest.status == "success",
                )
                .order_by(InferenceRequest.sequence)
            ).all()
            stream = []
            for request in requests:
                rows = session.scalars(
                    select(Transaction)
                    .where(Transaction.first_seen_request_sequence == request.sequence)
                    .order_by(Transaction.row_position)
                ).all()
                if len(rows) != request.batch_size:
                    raise RuntimeError("Recovery request has an incomplete transaction batch")
                stream.append((request, [row.raw_payload for row in rows]))
            return stream

    def prediction_by_transaction(
        self, transaction_id: int
    ) -> tuple[Prediction, InferenceRequest] | None:
        with self.sessions() as session:
            row = session.execute(
                select(Prediction, InferenceRequest)
                .join(InferenceRequest, Prediction.request_sequence == InferenceRequest.sequence)
                .where(Prediction.transaction_id == transaction_id)
            ).first()
            return (row[0], row[1]) if row else None
