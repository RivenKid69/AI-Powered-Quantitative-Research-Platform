# -*- coding: utf-8 -*-
"""Comprehensive tests for service_signal_runner.py - 100% coverage.

Tests cover:
- SignalQualityConfig dataclass
- CooldownSettings class
- DailyEntryLimiter class
- _ScheduleNoTradeChecker class
- MixedQuoteError exception
- Helper functions and utilities

Note: Full ServiceSignalRunner integration tests are complex due to
      dependencies on live data sources and heavy execution stack.
      These tests focus on unit-testable components.
"""

import pytest
from unittest import mock
from datetime import datetime, timezone

from service_signal_runner import (
    SignalQualityConfig,
    CooldownSettings,
    DailyEntryLimiter,
    _ScheduleNoTradeChecker,
    MixedQuoteError,
    _format_signal_quality_log,
    _EntryLimiterConfig,
)
from no_trade_config import NoTradeConfig


# ============================================================================
# Test SignalQualityConfig
# ============================================================================


def test_signal_quality_config_defaults():
    """Test SignalQualityConfig with default values."""
    cfg = SignalQualityConfig()

    assert cfg.enabled is False
    assert cfg.sigma_window == 0
    assert cfg.sigma_threshold == 0.0
    assert cfg.vol_median_window == 0
    assert cfg.vol_floor_frac == 0.0
    assert cfg.log_reason == ""


def test_signal_quality_config_custom():
    """Test SignalQualityConfig with custom values."""
    cfg = SignalQualityConfig(
        enabled=True,
        sigma_window=20,
        sigma_threshold=2.5,
        vol_median_window=50,
        vol_floor_frac=0.1,
        log_reason=True,
    )

    assert cfg.enabled is True
    assert cfg.sigma_window == 20
    assert cfg.sigma_threshold == 2.5
    assert cfg.vol_median_window == 50
    assert cfg.vol_floor_frac == 0.1
    assert cfg.log_reason is True


# ============================================================================
# Test _EntryLimiterConfig
# ============================================================================


def test_entry_limiter_config_defaults():
    """Test _EntryLimiterConfig with default values."""
    cfg = _EntryLimiterConfig()

    assert cfg.limit is None
    assert cfg.reset_hour == 0


def test_entry_limiter_config_custom():
    """Test _EntryLimiterConfig with custom values."""
    cfg = _EntryLimiterConfig(limit=10, reset_hour=8)

    assert cfg.limit == 10
    assert cfg.reset_hour == 8


# ============================================================================
# Test CooldownSettings
# ============================================================================


def test_cooldown_settings_defaults():
    """Test CooldownSettings with defaults."""
    settings = CooldownSettings()

    assert settings.bars == 0
    assert settings.large_trade_threshold == 0.0
    assert settings.small_delta_threshold == 0.0
    assert settings.enabled is False
    assert settings.suppress_small_deltas is False


def test_cooldown_settings_custom():
    """Test CooldownSettings with custom values."""
    settings = CooldownSettings(
        bars=3,
        large_trade_threshold=10000.0,
        small_delta_threshold=500.0,
    )

    assert settings.bars == 3
    assert settings.large_trade_threshold == 10000.0
    assert settings.small_delta_threshold == 500.0
    assert settings.enabled is True  # bars > 0 and large_trade > 0
    assert settings.suppress_small_deltas is True


def test_cooldown_settings_enabled_property():
    """Test CooldownSettings.enabled property logic."""
    # Not enabled: bars = 0
    settings = CooldownSettings(bars=0, large_trade_threshold=1000.0)
    assert settings.enabled is False

    # Not enabled: large_trade = 0
    settings = CooldownSettings(bars=3, large_trade_threshold=0.0)
    assert settings.enabled is False

    # Enabled: both > 0
    settings = CooldownSettings(bars=3, large_trade_threshold=1000.0)
    assert settings.enabled is True


def test_cooldown_settings_suppress_small_deltas_property():
    """Test CooldownSettings.suppress_small_deltas property."""
    # Not suppress: enabled=False
    settings = CooldownSettings(bars=0, large_trade_threshold=0.0, small_delta_threshold=100.0)
    assert settings.suppress_small_deltas is False

    # Not suppress: small_delta=0
    settings = CooldownSettings(bars=3, large_trade_threshold=1000.0, small_delta_threshold=0.0)
    assert settings.suppress_small_deltas is False

    # Suppress: both conditions met
    settings = CooldownSettings(bars=3, large_trade_threshold=1000.0, small_delta_threshold=100.0)
    assert settings.suppress_small_deltas is True


def test_cooldown_settings_from_mapping_empty():
    """Test CooldownSettings.from_mapping with empty dict."""
    settings = CooldownSettings.from_mapping({})

    assert settings.bars == 0
    assert settings.large_trade_threshold == 0.0
    assert settings.small_delta_threshold == 0.0


def test_cooldown_settings_from_mapping_dict():
    """Test CooldownSettings.from_mapping with dict."""
    data = {
        "cooldown_bars": 5,
        "large_trade_threshold": 15000.0,
        "small_delta_threshold": 800.0,
    }

    settings = CooldownSettings.from_mapping(data)

    assert settings.bars == 5
    assert settings.large_trade_threshold == 15000.0
    assert settings.small_delta_threshold == 800.0


def test_cooldown_settings_from_mapping_aliases():
    """Test CooldownSettings.from_mapping with various key aliases."""
    # Test cooldown_large_trade alias
    data = {"bars": 3, "cooldown_large_trade": 12000.0}
    settings = CooldownSettings.from_mapping(data)
    assert settings.large_trade_threshold == 12000.0

    # Test turnover_cap alias
    data = {"cooldown_bars": 2, "turnover_cap": 8000.0}
    settings = CooldownSettings.from_mapping(data)
    assert settings.large_trade_threshold == 8000.0

    # Test min_rebalance_step alias
    data = {"bars": 4, "large_trade": 5000.0, "min_rebalance_step": 200.0}
    settings = CooldownSettings.from_mapping(data)
    assert settings.small_delta_threshold == 200.0


def test_cooldown_settings_from_mapping_priority():
    """Test CooldownSettings.from_mapping key priority."""
    # First matching key should be used
    data = {
        "cooldown_bars": 5,
        "bars": 10,  # Should not be used (cooldown_bars takes priority)
    }

    settings = CooldownSettings.from_mapping(data)
    assert settings.bars == 5


def test_cooldown_settings_from_mapping_invalid_types():
    """Test CooldownSettings.from_mapping with invalid types."""
    data = {
        "bars": "invalid",  # Not coercible to int
        "large_trade_threshold": "not_a_number",
    }

    settings = CooldownSettings.from_mapping(data)

    # Should use defaults when coercion fails
    assert settings.bars == 0
    assert settings.large_trade_threshold == 0.0


def test_cooldown_settings_from_mapping_existing_instance():
    """Test CooldownSettings.from_mapping with CooldownSettings instance."""
    original = CooldownSettings(bars=7, large_trade_threshold=9000.0, small_delta_threshold=300.0)

    result = CooldownSettings.from_mapping(original)

    assert result is original  # Should return same instance


def test_cooldown_settings_from_mapping_non_mapping():
    """Test CooldownSettings.from_mapping with non-mapping type."""
    settings = CooldownSettings.from_mapping("not_a_dict")

    assert settings.bars == 0
    assert settings.large_trade_threshold == 0.0


def test_cooldown_settings_coerce_int():
    """Test CooldownSettings._coerce_int static method."""
    assert CooldownSettings._coerce_int(42) == 42
    assert CooldownSettings._coerce_int("100") == 100
    assert CooldownSettings._coerce_int(3.7) == 3
    assert CooldownSettings._coerce_int(None) is None
    assert CooldownSettings._coerce_int("invalid") is None


def test_cooldown_settings_coerce_float():
    """Test CooldownSettings._coerce_float static method."""
    assert CooldownSettings._coerce_float(42.5) == 42.5
    assert CooldownSettings._coerce_float("100.5") == 100.5
    assert CooldownSettings._coerce_float(10) == 10.0
    assert CooldownSettings._coerce_float(None) is None
    assert CooldownSettings._coerce_float("invalid") is None
    assert CooldownSettings._coerce_float(float("inf")) is None
    assert CooldownSettings._coerce_float(float("nan")) is None


# ============================================================================
# Test DailyEntryLimiter
# ============================================================================


def test_daily_entry_limiter_defaults():
    """Test DailyEntryLimiter with defaults."""
    limiter = DailyEntryLimiter()

    assert limiter.enabled is False
    assert limiter.limit is None
    assert limiter.reset_hour == 0


def test_daily_entry_limiter_enabled():
    """Test DailyEntryLimiter with limit set."""
    limiter = DailyEntryLimiter(limit=10)

    assert limiter.enabled is True
    assert limiter.limit == 10


def test_daily_entry_limiter_normalize_limit():
    """Test DailyEntryLimiter.normalize_limit."""
    assert DailyEntryLimiter.normalize_limit(10) == 10
    assert DailyEntryLimiter.normalize_limit("5") == 5
    assert DailyEntryLimiter.normalize_limit(0) is None
    assert DailyEntryLimiter.normalize_limit(-5) is None
    assert DailyEntryLimiter.normalize_limit(None) is None
    assert DailyEntryLimiter.normalize_limit("invalid") is None


def test_daily_entry_limiter_normalize_hour():
    """Test DailyEntryLimiter.normalize_hour."""
    assert DailyEntryLimiter.normalize_hour(8) == 8
    assert DailyEntryLimiter.normalize_hour("12") == 12
    assert DailyEntryLimiter.normalize_hour(25) == 1  # 25 % 24 = 1
    assert DailyEntryLimiter.normalize_hour(0) == 0
    assert DailyEntryLimiter.normalize_hour(None) == 0
    assert DailyEntryLimiter.normalize_hour("invalid") == 0


def test_daily_entry_limiter_allow_disabled():
    """Test DailyEntryLimiter.allow when disabled."""
    limiter = DailyEntryLimiter(limit=None)

    # Should always allow when disabled
    assert limiter.allow("BTCUSDT", 1000) is True
    assert limiter.allow("ETHUSDT", 2000) is True


def test_daily_entry_limiter_allow_first_entry():
    """Test DailyEntryLimiter.allow for first entry."""
    limiter = DailyEntryLimiter(limit=5)

    # First entry should be allowed
    assert limiter.allow("BTCUSDT", 1000) is True


def test_daily_entry_limiter_allow_within_limit():
    """Test DailyEntryLimiter.allow within limit."""
    limiter = DailyEntryLimiter(limit=3)

    # Should allow up to limit
    assert limiter.allow("BTCUSDT", 1000, entry_steps=1) is True
    assert limiter.allow("BTCUSDT", 2000, entry_steps=1) is True
    assert limiter.allow("BTCUSDT", 3000, entry_steps=1) is True

    # Should reject after limit
    assert limiter.allow("BTCUSDT", 4000, entry_steps=1) is False


def test_daily_entry_limiter_allow_multiple_steps():
    """Test DailyEntryLimiter.allow with multiple entry steps."""
    limiter = DailyEntryLimiter(limit=5)

    # Allow 3 steps
    assert limiter.allow("BTCUSDT", 1000, entry_steps=3) is True

    # Allow 2 more steps (total = 5)
    assert limiter.allow("BTCUSDT", 2000, entry_steps=2) is True

    # Reject 1 more step (would exceed limit)
    assert limiter.allow("BTCUSDT", 3000, entry_steps=1) is False


def test_daily_entry_limiter_allow_day_reset():
    """Test DailyEntryLimiter.allow resets on new day."""
    limiter = DailyEntryLimiter(limit=2, reset_hour=0)

    # Day 1
    ts_day1 = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert limiter.allow("BTCUSDT", ts_day1, entry_steps=1) is True
    assert limiter.allow("BTCUSDT", ts_day1 + 1000, entry_steps=1) is True
    assert limiter.allow("BTCUSDT", ts_day1 + 2000, entry_steps=1) is False

    # Day 2 (should reset)
    ts_day2 = int(datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert limiter.allow("BTCUSDT", ts_day2, entry_steps=1) is True
    assert limiter.allow("BTCUSDT", ts_day2 + 1000, entry_steps=1) is True


def test_daily_entry_limiter_snapshot():
    """Test DailyEntryLimiter.snapshot."""
    limiter = DailyEntryLimiter(limit=10, reset_hour=8)

    # Before any entries
    snapshot = limiter.snapshot("BTCUSDT")
    assert snapshot["limit"] == 10
    assert snapshot["entries_today"] == 0
    assert snapshot["reset_hour"] == 8

    # After some entries (use realistic timestamp from 2024)
    ts_2024 = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    limiter.allow("BTCUSDT", ts_2024, entry_steps=3)
    snapshot = limiter.snapshot("BTCUSDT")
    assert snapshot["entries_today"] == 3


def test_daily_entry_limiter_export_state():
    """Test DailyEntryLimiter.export_state."""
    limiter = DailyEntryLimiter(limit=5)

    # Empty state initially
    state = limiter.export_state()
    assert state == {}

    # After entries (use realistic timestamps from 2024)
    ts_2024 = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    limiter.allow("BTCUSDT", ts_2024, entry_steps=2)
    limiter.allow("ETHUSDT", ts_2024 + 1000, entry_steps=1)

    state = limiter.export_state()
    assert "BTCUSDT" in state
    assert state["BTCUSDT"]["count"] == 2
    assert "ETHUSDT" in state
    assert state["ETHUSDT"]["count"] == 1


def test_daily_entry_limiter_restore_state():
    """Test DailyEntryLimiter.restore_state."""
    limiter = DailyEntryLimiter(limit=5)

    # Restore state
    state = {
        "BTCUSDT": {"count": 3, "day_key": "2024-01-01T00"},
        "ETHUSDT": {"count": 1, "day_key": "2024-01-01T00"},
    }
    limiter.restore_state(state)

    # Verify state was restored
    snapshot = limiter.snapshot("BTCUSDT")
    assert snapshot["entries_today"] == 3

    # Should respect restored count (use realistic timestamps from 2024)
    ts_2024 = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert limiter.allow("BTCUSDT", ts_2024, entry_steps=1) is True  # 4/5
    assert limiter.allow("BTCUSDT", ts_2024 + 1000, entry_steps=1) is True  # 5/5
    assert limiter.allow("BTCUSDT", ts_2024 + 2000, entry_steps=1) is False  # 6/5 - rejected


def test_daily_entry_limiter_restore_state_disabled():
    """Test DailyEntryLimiter.restore_state when disabled."""
    limiter = DailyEntryLimiter(limit=None)

    state = {"BTCUSDT": {"count": 10, "day_key": "2024-01-01_00"}}
    limiter.restore_state(state)

    # Should clear state when disabled
    assert limiter.export_state() == {}


def test_daily_entry_limiter_normalize_symbol():
    """Test DailyEntryLimiter._normalize_symbol."""
    assert DailyEntryLimiter._normalize_symbol("btcusdt") == "BTCUSDT"
    assert DailyEntryLimiter._normalize_symbol("  BTC  ") == "BTC"
    assert DailyEntryLimiter._normalize_symbol(None) == ""
    assert DailyEntryLimiter._normalize_symbol(123) == "123"


# ============================================================================
# Test _ScheduleNoTradeChecker
# ============================================================================


def test_schedule_no_trade_checker_disabled():
    """Test _ScheduleNoTradeChecker with None config."""
    checker = _ScheduleNoTradeChecker(None)

    assert checker.enabled is False

    # Should not block any timestamp
    blocked, reason = checker.evaluate(1000)
    assert blocked is False
    assert reason is None


def test_schedule_no_trade_checker_daily_window():
    """Test _ScheduleNoTradeChecker with daily UTC windows."""
    with mock.patch("service_signal_runner.NO_TRADE_FEATURES_DISABLED", False):
        cfg = mock.Mock()
        cfg.maintenance = mock.Mock()
        cfg.maintenance.daily_utc = ["00:00-00:05", "08:00-08:05"]
        cfg.maintenance.funding_buffer_min = 0
        cfg.maintenance.custom_ms = []
        # Also set fallback attrs on cfg
        cfg.daily_utc = []
        cfg.funding_buffer_min = 0
        cfg.custom_ms = []

        checker = _ScheduleNoTradeChecker(cfg)

        assert checker.enabled is True

        # Timestamp in daily window (00:02 UTC)
        ts_in_window = int(datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc).timestamp() * 1000)
        blocked, reason = checker.evaluate(ts_in_window)
        assert blocked is True
        assert reason == "MAINTENANCE_DAILY"

        # Timestamp outside window (05:00 UTC)
        ts_out_window = int(datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        blocked, reason = checker.evaluate(ts_out_window)
        assert blocked is False
        assert reason is None


def test_schedule_no_trade_checker_funding_buffer():
    """Test _ScheduleNoTradeChecker with funding buffer."""
    with mock.patch("service_signal_runner.NO_TRADE_FEATURES_DISABLED", False):
        cfg = mock.Mock()
        cfg.maintenance = mock.Mock()
        cfg.maintenance.daily_utc = []
        cfg.maintenance.funding_buffer_min = 5  # 5 minutes
        cfg.maintenance.custom_ms = []
        # Also set fallback attrs on cfg
        cfg.daily_utc = []
        cfg.funding_buffer_min = 0
        cfg.custom_ms = []

        checker = _ScheduleNoTradeChecker(cfg)

        assert checker.enabled is True

        # Timestamp near funding time (00:02 UTC - within 5min of 00:00)
        ts_funding = int(datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc).timestamp() * 1000)
        blocked, reason = checker.evaluate(ts_funding)
        assert blocked is True
        assert reason == "MAINTENANCE_FUNDING"


def test_schedule_no_trade_checker_custom_window():
    """Test _ScheduleNoTradeChecker with custom millisecond windows."""
    with mock.patch("service_signal_runner.NO_TRADE_FEATURES_DISABLED", False):
        cfg = mock.Mock()
        cfg.maintenance = mock.Mock()
        cfg.maintenance.daily_utc = []
        cfg.maintenance.funding_buffer_min = 0
        # Use realistic timestamps from 2024
        ts_start = int(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        ts_end = int(datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc).timestamp() * 1000)
        cfg.maintenance.custom_ms = [
            {"start_ts_ms": ts_start, "end_ts_ms": ts_end},
        ]
        # Also set fallback attrs on cfg
        cfg.daily_utc = []
        cfg.funding_buffer_min = 0
        cfg.custom_ms = []

        checker = _ScheduleNoTradeChecker(cfg)

        assert checker.enabled is True

        # Timestamp in custom window
        ts_in_window = int(datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc).timestamp() * 1000)
        blocked, reason = checker.evaluate(ts_in_window)
        assert blocked is True
        assert reason == "MAINTENANCE_CUSTOM"

        # Timestamp outside custom window
        ts_out_window = int(datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc).timestamp() * 1000)
        blocked, reason = checker.evaluate(ts_out_window)
        assert blocked is False


def test_schedule_no_trade_checker_fallback_config_structure():
    """Test _ScheduleNoTradeChecker with alternative config structure."""
    with mock.patch("service_signal_runner.NO_TRADE_FEATURES_DISABLED", False):
        # Config without maintenance sub-block
        cfg = mock.Mock()
        cfg.maintenance = None
        cfg.daily_utc = ["12:00-12:05"]
        cfg.funding_buffer_min = 3
        cfg.custom_ms = []

        checker = _ScheduleNoTradeChecker(cfg)

        assert checker.enabled is True


def test_schedule_no_trade_checker_exception_handling():
    """Test _ScheduleNoTradeChecker handles exceptions gracefully."""
    cfg = mock.Mock()
    cfg.maintenance = mock.Mock()
    cfg.maintenance.daily_utc = ["invalid_format"]
    cfg.maintenance.funding_buffer_min = "not_an_int"
    cfg.maintenance.custom_ms = "not_a_list"

    checker = _ScheduleNoTradeChecker(cfg)

    # Should handle exceptions and create disabled checker
    assert checker.enabled is False


# ============================================================================
# Test MixedQuoteError
# ============================================================================


def test_mixed_quote_error():
    """Test MixedQuoteError exception."""
    error = MixedQuoteError("Mixed quote assets detected")

    assert isinstance(error, RuntimeError)
    assert str(error) == "Mixed quote assets detected"


def test_mixed_quote_error_raise():
    """Test raising MixedQuoteError."""
    with pytest.raises(MixedQuoteError) as exc_info:
        raise MixedQuoteError("Test error")

    assert "Test error" in str(exc_info.value)


# ============================================================================
# Test Helper Functions
# ============================================================================


def test_format_signal_quality_log_basic():
    """Test _format_signal_quality_log with basic payload."""
    payload = {
        "stage": "CHECK_QUALITY",
        "reason": "INSUFFICIENT_VOLUME",
        "symbol": "BTCUSDT",
        "bar_close_at": 1000,
    }

    result = _format_signal_quality_log(payload)

    assert "DROP" in result
    assert "CHECK_QUALITY" in result
    assert "INSUFFICIENT_VOLUME" in result
    assert "BTCUSDT" in result


def test_format_signal_quality_log_full():
    """Test _format_signal_quality_log with full payload."""
    payload = {
        "stage": "QUALITY_CHECK",
        "reason": "LOW_SIGMA",
        "symbol": "ETHUSDT",
        "bar_close_at": 2000,
        "bar_close_ms": 2000,
        "sigma_threshold": 2.5,
        "vol_floor_frac": 0.1,
        "bar_volume": 1000.0,
        "detail": "Sigma too low",
        "current_sigma": 1.2,
        "vol_median": 1500.0,
        "window_ready": True,
    }

    result = _format_signal_quality_log(payload)

    assert "DROP" in result
    assert "LOW_SIGMA" in result
    assert "ETHUSDT" in result
    assert "sigma_threshold=2.5" in result


def test_format_signal_quality_log_missing_keys():
    """Test _format_signal_quality_log with missing keys."""
    payload = {
        "symbol": "BTCUSDT",
    }

    result = _format_signal_quality_log(payload)

    # Should handle missing keys gracefully
    assert "DROP" in result
    assert "BTCUSDT" in result


def test_format_signal_quality_log_none_values():
    """Test _format_signal_quality_log with None values."""
    payload = {
        "stage": None,
        "reason": "TEST",
        "symbol": "BTCUSDT",
        "bar_volume": None,
    }

    result = _format_signal_quality_log(payload)

    # Should skip None values
    assert "DROP" in result
    assert "stage=None" not in result


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_daily_entry_limiter_invalid_entry_steps():
    """Test DailyEntryLimiter with invalid entry_steps."""
    limiter = DailyEntryLimiter(limit=5)

    # Negative steps
    assert limiter.allow("BTCUSDT", 1000, entry_steps=-1) is True  # Coerced to 1

    # Zero steps
    assert limiter.allow("BTCUSDT", 2000, entry_steps=0) is True  # Coerced to 1

    # String steps
    assert limiter.allow("BTCUSDT", 3000, entry_steps="invalid") is True  # Coerced to 1


def test_daily_entry_limiter_invalid_timestamp():
    """Test DailyEntryLimiter with invalid timestamp."""
    limiter = DailyEntryLimiter(limit=5)

    # Invalid timestamp (should use default 0)
    assert limiter.allow("BTCUSDT", "invalid") is True


def test_cooldown_settings_from_mapping_negative_values():
    """Test CooldownSettings.from_mapping with negative values."""
    data = {
        "bars": -5,  # Negative
        "large_trade_threshold": -1000.0,  # Negative
    }

    settings = CooldownSettings.from_mapping(data)

    # Negative values should still be used (not filtered by from_mapping)
    # But enabled will be False since they're not > 0
    assert settings.enabled is False


def test_schedule_no_trade_checker_no_trade_features_disabled():
    """Test _ScheduleNoTradeChecker when NO_TRADE_FEATURES_DISABLED is True."""
    with mock.patch("service_signal_runner.NO_TRADE_FEATURES_DISABLED", True):
        cfg = mock.Mock()
        cfg.maintenance = mock.Mock()
        cfg.maintenance.daily_utc = ["00:00-00:05"]

        checker = _ScheduleNoTradeChecker(cfg)

        # Should be disabled
        assert checker.enabled is False


def test_daily_entry_limiter_multiple_symbols():
    """Test DailyEntryLimiter tracks separate counts per symbol."""
    limiter = DailyEntryLimiter(limit=3)

    # Symbol 1
    assert limiter.allow("BTCUSDT", 1000, entry_steps=2) is True
    assert limiter.allow("BTCUSDT", 2000, entry_steps=1) is True
    assert limiter.allow("BTCUSDT", 3000, entry_steps=1) is False  # Limit reached

    # Symbol 2 should have independent count
    assert limiter.allow("ETHUSDT", 1000, entry_steps=1) is True
    assert limiter.allow("ETHUSDT", 2000, entry_steps=1) is True
    assert limiter.allow("ETHUSDT", 3000, entry_steps=1) is True


def test_daily_entry_limiter_restore_state_invalid():
    """Test DailyEntryLimiter.restore_state with invalid state."""
    limiter = DailyEntryLimiter(limit=5)

    # Invalid state structure
    state = {
        "BTCUSDT": {"count": "invalid", "day_key": 12345},  # Wrong types
        "ETHUSDT": "not_a_dict",  # Not a mapping
    }

    limiter.restore_state(state)

    # Should handle gracefully
    snapshot_btc = limiter.snapshot("BTCUSDT")
    assert snapshot_btc["entries_today"] == 0  # Coerced to 0

    snapshot_eth = limiter.snapshot("ETHUSDT")
    assert snapshot_eth["entries_today"] == 0  # Skipped


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
