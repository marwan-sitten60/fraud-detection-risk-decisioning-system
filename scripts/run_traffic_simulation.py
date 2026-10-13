"""Run a sequential HTTP traffic simulation against the local fraud API."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.simulation.config import SimulationConfig  # noqa: E402
from fraud_detection.simulation.runner import SimulationRunner  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", choices=("healthy", "drifted"), default="healthy")
    result.add_argument("--source", choices=("validation", "synthetic"), default="validation")
    result.add_argument("--max-transactions", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--target-rps", type=float)
    result.add_argument("--time-scale", type=float, default=86400.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--label-delay-min", type=float, default=0.01, help="minimum simulated days"
    )
    result.add_argument(
        "--label-delay-max", type=float, default=0.01, help="maximum simulated days"
    )
    result.add_argument("--labels-enabled", action="store_true", default=None)
    result.add_argument("--labels-disabled", action="store_false", dest="labels_enabled")
    result.set_defaults(labels_enabled=None)
    result.add_argument("--base-url", default="http://127.0.0.1:8000")
    result.add_argument("--retry-limit", type=int, default=2)
    result.add_argument("--timeout", type=float, default=180.0)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--report-dir", default="reports/simulation_runs")
    result.add_argument("--train-transaction-path", default="data/raw/train_transaction.csv")
    result.add_argument("--train-identity-path", default="data/raw/train_identity.csv")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        config = SimulationConfig(
            scenario=args.scenario,
            source=args.source,
            max_transactions=args.max_transactions,
            batch_size=args.batch_size,
            target_rps=args.target_rps,
            time_scale=args.time_scale,
            seed=args.seed,
            label_delay_min_days=args.label_delay_min,
            label_delay_max_days=args.label_delay_max,
            labels_enabled=args.labels_enabled,
            base_url=args.base_url,
            retry_limit=args.retry_limit,
            timeout_seconds=args.timeout,
            report_dir=args.report_dir,
            train_transaction_path=args.train_transaction_path,
            train_identity_path=args.train_identity_path,
        )
        summary = SimulationRunner(config).run(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("Simulation interrupted; a partial report was saved.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Simulation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"report=reports/simulation_runs/{summary.simulation_run_id}.json "
        f"scored={summary.successfully_scored_count}/{summary.configured_transaction_count} "
        f"labels={summary.labels_delivered}/{summary.labels_scheduled}"
    )
    return 0 if summary.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
