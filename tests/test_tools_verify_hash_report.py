import json
from pathlib import Path

import pytest

from tools.verify_hash_report import compute_sha256, validate_report


def _write_report(tmp_path: Path, artifact: Path, sha: str, python_version: str = "3.12.1") -> Path:
    report = {
        "build_info": {
            "python_version": python_version,
            "platform": "test-platform",
        },
        "extensions": {
            "example": {
                "path": str(artifact),
                "sha256": sha,
                "size_bytes": artifact.stat().st_size,
            }
        },
    }
    report_path = tmp_path / "build_hash_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def test_validate_report_round_trip(tmp_path: Path) -> None:
    artifact = tmp_path / "example.pyd"
    artifact.write_bytes(b"artifact-bytes")
    sha = compute_sha256(artifact)
    report_path = _write_report(tmp_path, artifact, sha)

    summary = validate_report(report_path, [tmp_path], require_all=True)

    assert summary["extensions"] == 1
    assert summary["python_version"].startswith("3.12")


def test_validate_report_detects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "example.pyd"
    artifact.write_bytes(b"artifact-bytes")
    sha = compute_sha256(artifact)
    report_path = _write_report(tmp_path, artifact, sha)

    artifact.write_bytes(b"modified")

    with pytest.raises(ValueError):
        validate_report(report_path, [tmp_path], require_all=True)


def test_validate_report_requires_python_312(tmp_path: Path) -> None:
    artifact = tmp_path / "example.pyd"
    artifact.write_bytes(b"artifact-bytes")
    sha = compute_sha256(artifact)
    report_path = _write_report(tmp_path, artifact, sha, python_version="3.11.9")

    with pytest.raises(ValueError):
        validate_report(report_path, [tmp_path], require_all=True)


def test_validate_report_flags_missing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "example.pyd"
    artifact.write_bytes(b"artifact-bytes")
    sha = compute_sha256(artifact)
    report_path = _write_report(tmp_path, artifact, sha)

    extra = tmp_path / "untracked.pyd"
    extra.write_bytes(b"extra-bytes")

    with pytest.raises(ValueError):
        validate_report(report_path, [tmp_path], require_all=True)
