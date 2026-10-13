"""Load and verify the compact static TRAIN-derived monitoring reference."""

import json
from pathlib import Path

from fraud_detection.inference.bundle import sha256_file


class MonitoringReference:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        artifact = path / self.manifest["artifact"]
        if sha256_file(artifact) != self.manifest["sha256"]:
            raise ValueError("Monitoring reference checksum mismatch")
        self.data = json.loads(artifact.read_text(encoding="utf-8"))
        if self.data.get("reference_version") != "monitoring_v1":
            raise ValueError("Unsupported monitoring reference")

    @property
    def bundle_version(self) -> str:
        return str(self.data["bundle_version"])
