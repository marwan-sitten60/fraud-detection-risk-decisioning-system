"""Deterministic event construction for sequential chronological replay."""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from fraud_detection.simulation.config import SimulationConfig
from fraud_detection.simulation.source import SourceTransaction


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    event_type: str
    offset_seconds: float
    sequence: int
    transaction_indexes: tuple[int, ...] = ()
    label_index: int | None = None


def build_event_schedule(
    transactions: list[SourceTransaction],
    config: SimulationConfig,
    *,
    start_at: datetime | None = None,
) -> list[SimulationEvent]:
    """Build stable batch, label, and midpoint-monitoring events.

    TransactionDT and target RPS offer load on one accelerated simulator clock.
    Source TransactionDT is never rewritten in the request payload.
    """
    if not transactions:
        return []
    if start_at is None:
        start_at = datetime(2026, 1, 1)
    ordered = sorted(
        transactions, key=lambda item: (item.transaction_dt, item.source_transaction_id)
    )
    if ordered != transactions:
        raise ValueError("Source transactions must be prepared in chronological order")
    origin_dt = transactions[0].transaction_dt
    rng = random.Random(config.seed)
    events = []
    event_sequence = 0
    batch_due: dict[int, float] = {}
    batch_count = (len(transactions) + config.batch_size - 1) // config.batch_size
    for batch_number, start in enumerate(range(0, len(transactions), config.batch_size)):
        indices = tuple(range(start, min(start + config.batch_size, len(transactions))))
        last_idx = indices[-1]
        source_elapsed = (transactions[last_idx].transaction_dt - origin_dt) / config.time_scale
        rate_elapsed = (last_idx + 1) / config.target_rps if config.target_rps else 0.0
        due = max(source_elapsed, rate_elapsed)
        batch_due[batch_number] = due
        events.append(SimulationEvent("prediction", due, event_sequence, indices))
        event_sequence += 1

    if config.deliver_labels:
        for label_index, transaction in enumerate(transactions):
            if transaction.fraud_label is None:
                continue
            delay_days = rng.uniform(config.label_delay_min_days, config.label_delay_max_days)
            batch_number = label_index // config.batch_size
            prediction_due = batch_due[batch_number]
            due = prediction_due + delay_days * 86400.0 / config.time_scale
            events.append(SimulationEvent("label", due, event_sequence, label_index=label_index))
            event_sequence += 1

    midpoint = (batch_count + 1) // 2 - 1
    due = batch_due[midpoint]
    events.append(SimulationEvent("monitoring", due, event_sequence))
    events.sort(
        key=lambda event: (event.offset_seconds, _priority(event.event_type), event.sequence)
    )
    return events


def _priority(event_type: str) -> int:
    return {"prediction": 0, "label": 1, "monitoring": 2}[event_type]


def event_wall_time(start_at: datetime, offset_seconds: float) -> datetime:
    return start_at + timedelta(seconds=offset_seconds)
