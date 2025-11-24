"""Comprehensive unit tests for seasonality system (utils_time.py)."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from utils_time import (
    bar_start_ms,
    bar_close_ms,
    floor_to_timeframe,
    is_bar_closed,
    next_bar_open_ms,
    interpolate_daily_multipliers,
    daily_from_hourly,
    load_hourly_seasonality,
    load_seasonality,
    watch_seasonality_file,
    get_hourly_multiplier,
    get_liquidity_multiplier,
    get_latency_multiplier,
    parse_time_to_ms,
    HOURS_IN_WEEK,
    SEASONALITY_MULT_MIN,
    SEASONALITY_MULT_MAX,
)


class TestBarTimestampFunctions:
    """Test bar timestamp calculations."""

    def test_bar_start_ms_1min(self):
        """Test bar start for 1-minute timeframe."""
        timeframe_ms = 60_000  # 1 minute
        ts = 1609459200123  # 2021-01-01 00:00:00.123
        expected = 1609459200000  # Floor to minute
        assert bar_start_ms(ts, timeframe_ms) == expected

    def test_bar_start_ms_5min(self):
        """Test bar start for 5-minute timeframe."""
        timeframe_ms = 300_000  # 5 minutes
        ts = 1609459400000  # 2021-01-01 00:03:20
        expected = 1609459200000  # 2021-01-01 00:00:00
        assert bar_start_ms(ts, timeframe_ms) == expected

    def test_bar_close_ms_1min(self):
        """Test bar close for 1-minute timeframe."""
        timeframe_ms = 60_000
        ts = 1609459200123
        expected = 1609459260000  # Next minute
        assert bar_close_ms(ts, timeframe_ms) == expected

    def test_floor_to_timeframe(self):
        """Test floor_to_timeframe matches bar_start_ms."""
        timeframe_ms = 60_000
        ts = 1609459223456
        assert floor_to_timeframe(ts, timeframe_ms) == bar_start_ms(ts, timeframe_ms)

    def test_is_bar_closed_true(self):
        """Test is_bar_closed when bar is closed."""
        close_ms = 1000000
        now_ms = 1000001
        assert is_bar_closed(close_ms, now_ms) is True

    def test_is_bar_closed_false(self):
        """Test is_bar_closed when bar is not closed."""
        close_ms = 1000000
        now_ms = 999999
        assert is_bar_closed(close_ms, now_ms) is False

    def test_is_bar_closed_with_lag(self):
        """Test is_bar_closed with lag parameter."""
        close_ms = 1000000
        lag_ms = 100
        now_ms = 1000050
        assert is_bar_closed(close_ms, now_ms, lag_ms) is False
        now_ms = 1000100
        assert is_bar_closed(close_ms, now_ms, lag_ms) is True

    def test_next_bar_open_ms(self):
        """Test next bar open calculation."""
        timeframe_ms = 60_000
        close_ms = 1609459260000
        expected = 1609459260000  # Same as close for aligned timestamps
        assert next_bar_open_ms(close_ms, timeframe_ms) == expected

    def test_bar_functions_with_invalid_timeframe(self):
        """Test bar functions raise ValueError for invalid timeframe."""
        with pytest.raises(ValueError):
            bar_start_ms(1000000, 0)
        with pytest.raises(ValueError):
            bar_close_ms(1000000, -1)


class TestInterpolationFunctions:
    """Test interpolation between daily/hourly multipliers."""

    def test_interpolate_daily_multipliers_shape(self):
        """Test interpolate_daily_multipliers returns 168 elements."""
        days = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        result = interpolate_daily_multipliers(days)
        assert result.shape == (HOURS_IN_WEEK,)

    def test_interpolate_daily_multipliers_values(self):
        """Test interpolate_daily_multipliers produces smooth transitions."""
        days = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = interpolate_daily_multipliers(days)
        assert np.allclose(result, 1.0)

    def test_interpolate_daily_multipliers_invalid_length(self):
        """Test interpolate_daily_multipliers raises for wrong length."""
        with pytest.raises(ValueError):
            interpolate_daily_multipliers([1.0, 1.0, 1.0])

    def test_daily_from_hourly_shape(self):
        """Test daily_from_hourly returns 7 elements."""
        hours = np.ones(HOURS_IN_WEEK)
        result = daily_from_hourly(hours)
        assert result.shape == (7,)

    def test_daily_from_hourly_averaging(self):
        """Test daily_from_hourly correctly averages 24-hour blocks."""
        hours = np.arange(HOURS_IN_WEEK, dtype=float)
        result = daily_from_hourly(hours)
        expected = np.array([11.5, 35.5, 59.5, 83.5, 107.5, 131.5, 155.5])
        assert np.allclose(result, expected)

    def test_daily_from_hourly_invalid_length(self):
        """Test daily_from_hourly raises for wrong length."""
        with pytest.raises(ValueError):
            daily_from_hourly([1.0] * 100)


class TestLoadHourlySeasonality:
    """Test load_hourly_seasonality function."""

    def test_load_hourly_seasonality_valid_168(self):
        """Test loading valid 168-element array."""
        data = {"liquidity": list(np.ones(HOURS_IN_WEEK) * 1.5)}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_hourly_seasonality(path, "liquidity")
            assert result is not None
            assert result.shape == (HOURS_IN_WEEK,)
            assert np.allclose(result, 1.5)
        finally:
            os.unlink(path)

    def test_load_hourly_seasonality_valid_7(self):
        """Test loading valid 7-element array (daily)."""
        data = {"liquidity": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_hourly_seasonality(path, "liquidity")
            assert result is not None
            assert result.shape == (7,)
        finally:
            os.unlink(path)

    def test_load_hourly_seasonality_clamping(self):
        """Test that liquidity/latency values are clamped."""
        data = {"liquidity": [0.01] * HOURS_IN_WEEK}  # Below SEASONALITY_MULT_MIN
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_hourly_seasonality(path, "liquidity")
            assert result is not None
            assert np.all(result >= SEASONALITY_MULT_MIN)
        finally:
            os.unlink(path)

    def test_load_hourly_seasonality_with_symbol(self):
        """Test loading with symbol nesting."""
        data = {"BTCUSDT": {"latency": list(np.ones(HOURS_IN_WEEK) * 2.0)}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_hourly_seasonality(path, "latency", symbol="BTCUSDT")
            assert result is not None
            assert np.allclose(result, 2.0)
        finally:
            os.unlink(path)

    def test_load_hourly_seasonality_nonexistent_file(self):
        """Test returns None for nonexistent file."""
        result = load_hourly_seasonality("/nonexistent/path.json", "liquidity")
        assert result is None

    def test_load_hourly_seasonality_invalid_values(self):
        """Test raises ValueError for non-positive values."""
        data = {"liquidity": [0.0] * HOURS_IN_WEEK}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            with pytest.raises(ValueError):
                load_hourly_seasonality(path, "liquidity")
        finally:
            os.unlink(path)

    def test_load_hourly_seasonality_hash_mismatch_warning(self):
        """Test hash mismatch produces warning."""
        data = {"liquidity": list(np.ones(HOURS_IN_WEEK))}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            # Should not raise, just warn
            result = load_hourly_seasonality(path, "liquidity", expected_hash="wronghash")
            assert result is not None
        finally:
            os.unlink(path)


class TestLoadSeasonality:
    """Test load_seasonality function."""

    def test_load_seasonality_all_keys(self):
        """Test loading all available seasonality keys."""
        data = {
            "liquidity": list(np.ones(HOURS_IN_WEEK) * 1.5),
            "latency": list(np.ones(HOURS_IN_WEEK) * 2.0),
            "spread": list(np.ones(HOURS_IN_WEEK) * 1.2),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            assert "liquidity" in result
            assert "latency" in result
            assert "spread" in result
            assert np.allclose(result["liquidity"], 1.5)
            assert np.allclose(result["latency"], 2.0)
            assert np.allclose(result["spread"], 1.2)
        finally:
            os.unlink(path)

    def test_load_seasonality_partial_keys(self):
        """Test loading with only some keys present."""
        data = {"liquidity": list(np.ones(HOURS_IN_WEEK))}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            assert "liquidity" in result
            assert "latency" not in result
        finally:
            os.unlink(path)

    def test_load_seasonality_symbol_nested(self):
        """Test loading with symbol nesting (single symbol)."""
        data = {
            "BTCUSDT": {
                "liquidity": list(np.ones(HOURS_IN_WEEK)),
                "latency": list(np.ones(HOURS_IN_WEEK) * 2.0),
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            assert "liquidity" in result
            assert "latency" in result
        finally:
            os.unlink(path)

    def test_load_seasonality_multiple_symbols_error(self):
        """Test raises ValueError for multiple symbols without specification."""
        data = {
            "BTCUSDT": {"liquidity": list(np.ones(HOURS_IN_WEEK))},
            "ETHUSDT": {"liquidity": list(np.ones(HOURS_IN_WEEK))},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="Multiple seasonality"):
                load_seasonality(path)
        finally:
            os.unlink(path)

    def test_load_seasonality_nonexistent_file(self):
        """Test raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_seasonality("/nonexistent/path.json")

    def test_load_seasonality_invalid_json(self):
        """Test raises ValueError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            path = f.name

        try:
            with pytest.raises(ValueError):
                load_seasonality(path)
        finally:
            os.unlink(path)

    def test_load_seasonality_invalid_array_length(self):
        """Test raises ValueError for invalid array length."""
        data = {"liquidity": [1.0] * 100}  # Wrong length
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="length 168 or 7"):
                load_seasonality(path)
        finally:
            os.unlink(path)

    def test_load_seasonality_non_positive_values(self):
        """Test raises ValueError for non-positive values."""
        data = {"liquidity": [0.0] * HOURS_IN_WEEK}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="positive values"):
                load_seasonality(path)
        finally:
            os.unlink(path)


class TestWatchSeasonalityFile:
    """Test watch_seasonality_file function."""

    def test_watch_seasonality_file_basic(self):
        """Test file watching triggers callback on update."""
        data = {"liquidity": list(np.ones(HOURS_IN_WEEK))}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            callback_triggered = threading.Event()
            received_data = []

            def callback(data_dict):
                received_data.append(data_dict)
                callback_triggered.set()

            thread = watch_seasonality_file(path, callback, poll_interval=0.1)

            # Give watcher time to start
            time.sleep(0.2)

            # Update file
            data["liquidity"] = list(np.ones(HOURS_IN_WEEK) * 2.0)
            with open(path, "w") as f:
                json.dump(data, f)

            # Wait for callback
            assert callback_triggered.wait(timeout=2.0), "Callback not triggered"
            assert len(received_data) > 0

        finally:
            # Give thread time to release file handle
            time.sleep(0.3)
            try:
                os.unlink(path)
            except PermissionError:
                pass  # File still locked on Windows, will be cleaned up later


class TestGetHourlyMultiplier:
    """Test get_hourly_multiplier function."""

    def test_get_hourly_multiplier_no_interpolation(self):
        """Test getting multiplier without interpolation."""
        multipliers = np.ones(HOURS_IN_WEEK) * 1.5
        multipliers[0] = 2.0  # Monday 00:00 UTC
        ts_ms = 345600000  # Monday 00:00 UTC (1970-01-05 00:00:00)
        result = get_hourly_multiplier(ts_ms, multipliers, interpolate=False)
        assert result == 2.0

    def test_get_hourly_multiplier_with_interpolation(self):
        """Test getting multiplier with interpolation."""
        multipliers = np.ones(HOURS_IN_WEEK)
        multipliers[0] = 1.0
        multipliers[1] = 2.0
        # Timestamp halfway through first hour
        ts_ms = 345600000 + 30 * 60 * 1000  # Monday 00:30 UTC
        result = get_hourly_multiplier(ts_ms, multipliers, interpolate=True)
        assert 1.4 < result < 1.6  # Should be around 1.5

    def test_get_hourly_multiplier_none_input(self):
        """Test returns 1.0 for None input."""
        result = get_hourly_multiplier(1000000, None, interpolate=False)
        assert result == 1.0

    def test_get_hourly_multiplier_empty_array(self):
        """Test returns 1.0 for empty array."""
        result = get_hourly_multiplier(1000000, [], interpolate=False)
        assert result == 1.0

    def test_get_hourly_multiplier_daily_array(self):
        """Test with 7-element daily array."""
        multipliers = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        ts_ms = 345600000  # Monday (day 0)
        result = get_hourly_multiplier(ts_ms, multipliers, interpolate=False)
        assert result == 1.0

    def test_get_liquidity_multiplier(self):
        """Test get_liquidity_multiplier wrapper."""
        multipliers = np.ones(HOURS_IN_WEEK) * 1.5
        result = get_liquidity_multiplier(345600000, multipliers)
        assert result == 1.5

    def test_get_latency_multiplier(self):
        """Test get_latency_multiplier wrapper."""
        multipliers = np.ones(HOURS_IN_WEEK) * 2.0
        result = get_latency_multiplier(345600000, multipliers)
        assert result == 2.0


class TestParseTimeToMs:
    """Test parse_time_to_ms function."""

    def test_parse_time_to_ms_unix_milliseconds(self):
        """Test parsing Unix milliseconds."""
        result = parse_time_to_ms("1609459200000")
        assert result == 1609459200000

    def test_parse_time_to_ms_unix_seconds(self):
        """Test parsing Unix seconds (auto-converted to ms)."""
        result = parse_time_to_ms("1609459200")
        assert result == 1609459200000

    def test_parse_time_to_ms_iso8601(self):
        """Test parsing ISO 8601 format."""
        result = parse_time_to_ms("2021-01-01T00:00:00+00:00")
        assert result == 1609459200000

    def test_parse_time_to_ms_date_only(self):
        """Test parsing date-only format."""
        result = parse_time_to_ms("2021-01-01")
        assert result == 1609459200000

    def test_parse_time_to_ms_datetime_format(self):
        """Test parsing datetime format."""
        result = parse_time_to_ms("2021-01-01 00:00:00")
        assert result == 1609459200000

    def test_parse_time_to_ms_now(self):
        """Test 'now' keyword."""
        before = int(time.time() * 1000)
        result = parse_time_to_ms("now")
        after = int(time.time() * 1000)
        assert before <= result <= after

    def test_parse_time_to_ms_today(self):
        """Test 'today' keyword."""
        result = parse_time_to_ms("today")
        # Should be midnight UTC today
        assert result > 0
        assert result % 86_400_000 == 0  # Should be at midnight

    def test_parse_time_to_ms_invalid(self):
        """Test raises ValueError for invalid input."""
        with pytest.raises(ValueError):
            parse_time_to_ms("not a valid time")


class TestSeasonalityClamping:
    """Test seasonality multiplier clamping."""

    def test_liquidity_clamping_too_low(self):
        """Test liquidity multipliers are clamped to SEASONALITY_MULT_MIN."""
        data = {"liquidity": [0.01] * HOURS_IN_WEEK}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            assert np.all(result["liquidity"] >= SEASONALITY_MULT_MIN)
        finally:
            os.unlink(path)

    def test_liquidity_clamping_too_high(self):
        """Test liquidity multipliers are clamped to SEASONALITY_MULT_MAX."""
        data = {"liquidity": [100.0] * HOURS_IN_WEEK}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            assert np.all(result["liquidity"] <= SEASONALITY_MULT_MAX)
        finally:
            os.unlink(path)

    def test_latency_clamping(self):
        """Test latency multipliers are also clamped."""
        data = {"latency": [0.01] * HOURS_IN_WEEK}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            assert np.all(result["latency"] >= SEASONALITY_MULT_MIN)
        finally:
            os.unlink(path)

    def test_spread_no_clamping(self):
        """Test spread multipliers are NOT clamped."""
        data = {"spread": [0.01] * HOURS_IN_WEEK}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            result = load_seasonality(path)
            # Spread should NOT be clamped
            assert np.any(result["spread"] < SEASONALITY_MULT_MIN)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
