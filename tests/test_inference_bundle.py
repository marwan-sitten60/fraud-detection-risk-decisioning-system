import hashlib
import json

import pytest

from fraud_detection.inference.bundle import ordered_feature_hash, verify_bundle


def test_manifest_and_payload_checksums_detect_corruption(tmp_path) -> None:
    payload = tmp_path / "policy.json"
    payload.write_text('{"rules_enabled_by_default": false}\n')
    manifest = {
        "bundle_version": "frozen_v1",
        "artifact_files": {"policy": payload.name},
        "sha256": {"policy": hashlib.sha256(payload.read_bytes()).hexdigest()},
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "artifact_manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n"
    )
    assert verify_bundle(tmp_path) == manifest
    payload.write_text("corrupted")
    with pytest.raises(ValueError, match="policy.json"):
        verify_bundle(tmp_path)


def test_feature_hash_is_order_sensitive() -> None:
    assert ordered_feature_hash(("a", "b")) != ordered_feature_hash(("b", "a"))
