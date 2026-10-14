"""Guard the local container contract without requiring a Docker daemon."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_orders_services_and_keeps_stateful_api_single_worker() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"db", "migrate", "api", "dashboard"}
    assert services["migrate"]["depends_on"]["db"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert services["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["migrate"]["command"] == ["alembic", "upgrade", "head"]

    api_command = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert '"--workers", "1"' in api_command
    assert "/ready" in api_command
    assert "/ready" in services["api"]["healthcheck"]["test"][-1]
    assert services["dashboard"]["environment"]["FRAUD_API_BASE_URL"] == "http://api:8000"
    assert "_stcore/health" in services["dashboard"]["healthcheck"]["test"][-1]
    assert "FRAUD_DATABASE_URL" not in services["dashboard"].get("environment", {})


def test_compose_uses_private_named_database_volume_and_required_password() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["db"]["image"] == "postgres:17"
    assert "fraud_postgres_data:/var/lib/postgresql/data" in services["db"]["volumes"]
    assert "fraud_postgres_data" in compose["volumes"]
    assert "127.0.0.1:${POSTGRES_HOST_PORT:-5433}:5432" in services["db"]["ports"]
    assert "127.0.0.1:${API_PORT:-8000}:8000" in services["api"]["ports"]
    assert "127.0.0.1:${DASHBOARD_PORT:-8501}:8501" in services["dashboard"]["ports"]
    assert "${POSTGRES_PASSWORD:?" in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert services["dashboard"]["volumes"][0]["read_only"] is True


def test_docker_context_contains_runtime_bundle_but_excludes_local_data_and_secrets() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    api_dockerfile = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "**" in dockerignore
    assert "!models/frozen_v1/**" in dockerignore
    assert "!models/monitoring_v1/**" in dockerignore
    assert "data/" not in api_dockerfile
    assert " models/frozen_v1 ./models/frozen_v1" in api_dockerfile
    assert " models/monitoring_v1 ./models/monitoring_v1" in api_dockerfile
    assert "POSTGRES_PASSWORD=" in env_example
    assert "postgres_password=password" not in env_example.lower()

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "\n.env\n" in f"\n{ignored}\n"
