import socket

import pytest

from services.runtime_security import (
    AdapterNetworkGuard,
    StrategyFSGuard,
)


def test_strategy_fs_guard_blocks_disallowed_writes(tmp_path):
    allowed = tmp_path / "logs"
    allowed.mkdir()

    guard = StrategyFSGuard(
        mode="read_only",
        allow_write_paths=[allowed],
        strategy_frame_predicate=lambda _: True,  # treat all frames as strategy frames
    )
    guard.install()
    try:
        allowed_file = allowed / "ok.txt"
        with open(allowed_file, "w", encoding="utf-8") as handle:
            handle.write("ok")

        blocked = tmp_path / "blocked" / "secret.txt"
        blocked.parent.mkdir(parents=True)
        with pytest.raises(PermissionError):
            with open(blocked, "w", encoding="utf-8") as handle:
                handle.write("nope")
    finally:
        guard.uninstall()


def test_adapter_network_guard_enforces_allowlist(monkeypatch):
    guard = AdapterNetworkGuard(
        allowed_hosts=["api.binance.com"],
        allowed_cidrs=["127.0.0.0/8"],
        allow_subdomains=True,
    )

    # Allowed host and CIDR
    guard.validate_host("api.binance.com")
    guard.validate_host("127.0.0.1")

    with pytest.raises(PermissionError):
        guard.validate_host("malicious.example.com")

    # Verify socket hook respects allowlist
    calls = []

    def _fake_connect(address, *args, **kwargs):
        calls.append(address)
        return "connected"

    monkeypatch.setattr(socket, "create_connection", _fake_connect)
    guard.install()
    try:
        assert socket.create_connection(("api.binance.com", 80)) == "connected"
        with pytest.raises(PermissionError):
            socket.create_connection(("bad.example.com", 80))
    finally:
        guard.uninstall()
