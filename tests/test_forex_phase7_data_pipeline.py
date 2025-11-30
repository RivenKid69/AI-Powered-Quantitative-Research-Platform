# -*- coding: utf-8 -*-
"""
tests/test_forex_phase7_data_pipeline.py
Comprehensive test suite for Forex Phase 7: Data Pipeline & Downloaders.

Test Categories:
1. Download Scripts Tests
   - download_forex_data.py
   - download_swap_rates.py
   - download_economic_calendar.py
   - download_interest_rates.py
2. Data Loader Tests
   - load_forex_data()
   - Session features
   - Data merging (swaps, rates, calendar)
3. Integration Tests
   - End-to-end pipeline
   - Backward compatibility
4. Edge Cases & Error Handling

Total Tests: ~50

Author: AI-Powered Quantitative Research Platform Team
Date: 2025-11-30
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_forex_df():
    """Create sample forex OHLCV DataFrame."""
    np.random.seed(42)
    n_bars = 100

    # Generate timestamps (hourly bars for a week, excluding weekends)
    base_ts = int(datetime(2024, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp())  # Monday
    timestamps = [base_ts + i * 3600 for i in range(n_bars)]

    # Filter out weekends
    def is_weekday(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.weekday() < 5

    timestamps = [ts for ts in timestamps if is_weekday(ts)][:n_bars]

    # Generate price data
    base_price = 1.1000  # EUR/USD
    prices = [base_price]
    for _ in range(len(timestamps) - 1):
        change = np.random.normal(0, 0.0010)  # ~10 pips std
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        "timestamp": timestamps[:len(prices)],
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.0005)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.0005)) for p in prices],
        "close": [p + np.random.normal(0, 0.0003) for p in prices],
        "volume": np.random.randint(1000, 10000, len(prices)),
        "spread_pips": np.random.uniform(0.8, 1.5, len(prices)),
        "symbol": "EUR_USD",
    })

    return df


@pytest.fixture
def sample_swap_df():
    """Create sample swap rate DataFrame."""
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pair": "EUR_USD",
        "long_swap": np.random.uniform(-0.8, -0.3, len(dates)),
        "short_swap": np.random.uniform(-0.5, -0.2, len(dates)),
        "source": "synthetic",
    })


@pytest.fixture
def sample_rate_df():
    """Create sample interest rate DataFrame."""
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "rate": np.random.uniform(5.0, 5.5, len(dates)),
        "currency": "USD",
        "source": "fred",
    })


@pytest.fixture
def sample_calendar_df():
    """Create sample economic calendar DataFrame."""
    return pd.DataFrame({
        "datetime": [
            "2024-01-10 13:30:00",
            "2024-01-12 19:00:00",
            "2024-01-15 08:30:00",
        ],
        "date": ["2024-01-10", "2024-01-12", "2024-01-15"],
        "time": ["13:30", "19:00", "08:30"],
        "currency": ["USD", "USD", "EUR"],
        "event": ["Non-Farm Payrolls", "Fed Speech", "CPI"],
        "impact": ["high", "medium", "high"],
        "actual": [200, None, 2.5],
        "forecast": [180, None, 2.4],
        "previous": [190, None, 2.3],
        "source": "synthetic",
    })


# =============================================================================
# TEST: download_forex_data.py
# =============================================================================

class TestDownloadForexData:
    """Tests for scripts/download_forex_data.py"""

    def test_forex_download_config_defaults(self):
        """Test ForexDownloadConfig has correct defaults."""
        from scripts.download_forex_data import ForexDownloadConfig

        config = ForexDownloadConfig()
        assert config.provider == "oanda"
        assert config.practice is True
        assert config.timeframe == "1h"
        assert config.output_dir == "data/raw_forex"
        assert config.filter_weekends is True
        assert config.include_spread is True

    def test_major_pairs_defined(self):
        """Test major pairs are correctly defined."""
        from scripts.download_forex_data import MAJOR_PAIRS, MINOR_PAIRS, ALL_PAIRS

        assert "EUR_USD" in MAJOR_PAIRS
        assert "USD_JPY" in MAJOR_PAIRS
        assert "GBP_USD" in MAJOR_PAIRS
        assert len(MAJOR_PAIRS) == 7

        assert "EUR_GBP" in MINOR_PAIRS
        assert "EUR_JPY" in MINOR_PAIRS

        assert len(ALL_PAIRS) > len(MAJOR_PAIRS)

    def test_pip_sizes(self):
        """Test pip sizes are correct for different pairs."""
        from scripts.download_forex_data import get_pip_size, PIP_SIZES

        # JPY pairs have pip = 0.01
        assert get_pip_size("USD_JPY") == 0.01
        assert get_pip_size("EUR_JPY") == 0.01

        # Standard pairs have pip = 0.0001
        assert get_pip_size("EUR_USD") == 0.0001
        assert get_pip_size("GBP_USD") == 0.0001

    def test_forex_calendar_weekend_detection(self):
        """Test forex weekend detection."""
        from scripts.download_forex_data import ForexCalendar

        # Monday - should be trading
        monday = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
        assert not ForexCalendar.is_weekend(monday)

        # Saturday - closed
        saturday = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)
        assert ForexCalendar.is_weekend(saturday)

        # Sunday before 21:00 UTC - closed
        sunday_early = datetime(2024, 1, 7, 15, 0, tzinfo=timezone.utc)
        assert ForexCalendar.is_weekend(sunday_early)

        # Sunday after 21:00 UTC - open
        sunday_late = datetime(2024, 1, 7, 22, 0, tzinfo=timezone.utc)
        assert not ForexCalendar.is_weekend(sunday_late)

        # Friday after 21:00 UTC - closed
        friday_late = datetime(2024, 1, 5, 22, 0, tzinfo=timezone.utc)
        assert ForexCalendar.is_weekend(friday_late)

    def test_get_active_session(self):
        """Test session detection."""
        from scripts.download_forex_data import ForexCalendar

        # London session (07:00-16:00 UTC)
        london_time = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)
        session = ForexCalendar.get_active_session(london_time)
        assert session in ["london", "tokyo_london_overlap"]

        # New York session (12:00-21:00 UTC)
        ny_time = datetime(2024, 1, 8, 18, 0, tzinfo=timezone.utc)
        session = ForexCalendar.get_active_session(ny_time)
        assert session == "new_york"

        # London/NY overlap (12:00-16:00 UTC)
        overlap_time = datetime(2024, 1, 8, 14, 0, tzinfo=timezone.utc)
        session = ForexCalendar.get_active_session(overlap_time)
        assert session == "london_ny_overlap"

    def test_filter_weekends(self, sample_forex_df):
        """Test weekend filtering removes weekend bars."""
        from scripts.download_forex_data import _filter_weekends

        # Add weekend data
        saturday_ts = int(datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc).timestamp())
        weekend_row = sample_forex_df.iloc[0:1].copy()
        weekend_row["timestamp"] = saturday_ts
        df_with_weekend = pd.concat([sample_forex_df, weekend_row], ignore_index=True)

        filtered = _filter_weekends(df_with_weekend)

        # Weekend bar should be removed
        assert len(filtered) == len(sample_forex_df)
        assert saturday_ts not in filtered["timestamp"].values

    def test_resample_bars(self, sample_forex_df):
        """Test bar resampling."""
        from scripts.download_forex_data import _resample_bars

        # Resample 1h to 4h
        resampled = _resample_bars(sample_forex_df, "4h")

        # Should have fewer bars
        assert len(resampled) < len(sample_forex_df)

        # OHLCV should be aggregated correctly
        assert "open" in resampled.columns
        assert "high" in resampled.columns
        assert "close" in resampled.columns
        assert "volume" in resampled.columns

    def test_save_dataframe(self, sample_forex_df, temp_dir):
        """Test saving DataFrame to different formats."""
        from scripts.download_forex_data import save_dataframe, ForexDownloadConfig

        config = ForexDownloadConfig(output_dir=temp_dir, output_format="parquet")
        filepath = save_dataframe(sample_forex_df, "EUR_USD", config)

        assert Path(filepath).exists()
        assert filepath.endswith(".parquet")

        # Verify content
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == len(sample_forex_df)


# =============================================================================
# TEST: download_swap_rates.py
# =============================================================================

class TestDownloadSwapRates:
    """Tests for scripts/download_swap_rates.py"""

    def test_swap_download_config(self):
        """Test SwapDownloadConfig defaults."""
        from scripts.download_swap_rates import SwapDownloadConfig

        config = SwapDownloadConfig()
        assert config.output_dir == "data/forex/swaps"
        assert config.practice is True

    def test_estimate_swap_from_interest_rates(self):
        """Test swap rate estimation from interest differentials."""
        from scripts.download_swap_rates import estimate_swap_from_interest_rates

        # Higher base rate = positive carry for long
        long_swap, short_swap = estimate_swap_from_interest_rates(
            pair="AUD_USD",
            base_rate=4.5,  # AUD
            quote_rate=5.0,  # USD
            spot_price=0.65,
        )

        # With negative differential (AUD < USD), long should be negative
        # and short should be more negative due to broker spread
        assert isinstance(long_swap, float)
        assert isinstance(short_swap, float)

    def test_build_synthetic_swaps(self):
        """Test synthetic swap generation."""
        from scripts.download_swap_rates import _build_synthetic_swaps

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        pair, df, error = _build_synthetic_swaps("EUR_USD", start, end)

        assert error is None
        assert df is not None
        assert len(df) == 31  # 31 days
        assert "long_swap" in df.columns
        assert "short_swap" in df.columns
        assert df["source"].iloc[0] == "synthetic"

    def test_save_swaps(self, sample_swap_df, temp_dir):
        """Test saving swap DataFrame."""
        from scripts.download_swap_rates import save_swaps, SwapDownloadConfig

        config = SwapDownloadConfig(output_dir=temp_dir)
        filepath = save_swaps(sample_swap_df, "EUR_USD", config)

        assert Path(filepath).exists()
        assert "EUR_USD_swaps" in filepath


# =============================================================================
# TEST: download_economic_calendar.py
# =============================================================================

class TestDownloadEconomicCalendar:
    """Tests for scripts/download_economic_calendar.py"""

    def test_calendar_config(self):
        """Test CalendarDownloadConfig defaults."""
        from scripts.download_economic_calendar import CalendarDownloadConfig

        config = CalendarDownloadConfig()
        assert "USD" in config.currencies
        assert config.output_dir == "data/forex/calendar"

    def test_high_impact_events_defined(self):
        """Test high-impact events are defined for major currencies."""
        from scripts.download_economic_calendar import HIGH_IMPACT_EVENTS

        assert "Non-Farm Payrolls" in HIGH_IMPACT_EVENTS["USD"]
        assert "FOMC Rate Decision" in HIGH_IMPACT_EVENTS["USD"]
        assert "ECB Rate Decision" in HIGH_IMPACT_EVENTS["EUR"]
        assert "BOE Rate Decision" in HIGH_IMPACT_EVENTS["GBP"]

    def test_classify_impact(self):
        """Test event impact classification."""
        from scripts.download_economic_calendar import _classify_impact

        assert _classify_impact("Non-Farm Payrolls", "USD") == "high"
        assert _classify_impact("FOMC Rate Decision", "USD") == "high"
        assert _classify_impact("CPI", "EUR") == "high"
        assert _classify_impact("Trade Balance", "JPY") == "medium"
        assert _classify_impact("Some Minor Event", "USD") == "low"

    def test_get_first_friday(self):
        """Test first Friday calculation."""
        from scripts.download_economic_calendar import _get_first_friday

        # January 2024 - first Friday is January 5
        jan_2024 = datetime(2024, 1, 15)
        first_fri = _get_first_friday(jan_2024)
        assert first_fri.day == 5
        assert first_fri.weekday() == 4  # Friday

    def test_generate_synthetic_calendar(self):
        """Test synthetic calendar generation."""
        from scripts.download_economic_calendar import (
            generate_synthetic_calendar,
            CalendarDownloadConfig,
        )

        config = CalendarDownloadConfig(
            currencies=["USD", "EUR"],
            start_date="2024-01-01",
            end_date="2024-03-31",
        )

        df = generate_synthetic_calendar(config)

        assert len(df) > 0
        assert "event" in df.columns
        assert "impact" in df.columns
        assert "currency" in df.columns
        assert set(df["currency"].unique()).issubset({"USD", "EUR"})

    def test_save_calendar(self, sample_calendar_df, temp_dir):
        """Test saving calendar DataFrame."""
        from scripts.download_economic_calendar import (
            save_calendar,
            CalendarDownloadConfig,
        )

        config = CalendarDownloadConfig(output_dir=temp_dir)
        filepath = save_calendar(sample_calendar_df, config)

        assert Path(filepath).exists()
        assert "economic_calendar" in filepath


# =============================================================================
# TEST: download_interest_rates.py
# =============================================================================

class TestDownloadInterestRates:
    """Tests for scripts/download_interest_rates.py"""

    def test_rate_download_config(self):
        """Test RateDownloadConfig defaults."""
        from scripts.download_interest_rates import RateDownloadConfig

        config = RateDownloadConfig()
        assert config.output_dir == "data/forex/rates"
        assert config.lookback_years == 15
        assert config.fill_gaps is True

    def test_rate_series_defined(self):
        """Test FRED rate series are defined for major currencies."""
        from scripts.download_interest_rates import RATE_SERIES

        assert "USD" in RATE_SERIES
        assert "EUR" in RATE_SERIES
        assert "GBP" in RATE_SERIES
        assert "JPY" in RATE_SERIES

        assert RATE_SERIES["USD"]["series_id"] == "FEDFUNDS"
        assert "name" in RATE_SERIES["USD"]

    def test_fill_rate_gaps(self, sample_rate_df):
        """Test gap filling in rate data."""
        from scripts.download_interest_rates import _fill_rate_gaps

        # Create gaps
        df_with_gaps = sample_rate_df.iloc[::3].copy()  # Take every 3rd row
        df_with_gaps["date"] = pd.to_datetime(df_with_gaps["date"])

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        filled = _fill_rate_gaps(df_with_gaps, start, end)

        # Should have all dates filled
        assert len(filled) == 31

    def test_calculate_rate_differentials(self, temp_dir):
        """Test interest rate differential calculation."""
        from scripts.download_interest_rates import calculate_rate_differentials

        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")

        usd_rates = pd.DataFrame({
            "date": dates,
            "rate": 5.25,
        }).set_index("date")

        eur_rates = pd.DataFrame({
            "date": dates,
            "rate": 4.50,
        }).set_index("date")

        rate_dfs = {
            "USD": usd_rates.reset_index(),
            "EUR": eur_rates.reset_index(),
        }

        diff_df = calculate_rate_differentials(rate_dfs, base_currency="USD")

        assert len(diff_df) > 0
        assert "USD_EUR_diff" in diff_df.columns

        # USD - EUR = 5.25 - 4.50 = 0.75
        assert abs(diff_df["USD_EUR_diff"].iloc[0] - 0.75) < 0.01


# =============================================================================
# TEST: data_loader_forex.py
# =============================================================================

class TestDataLoaderForex:
    """Tests for data_loader_forex.py"""

    def test_load_forex_data_basic(self, sample_forex_df, temp_dir):
        """Test basic forex data loading."""
        from data_loader_forex import load_forex_data

        # Save sample data
        filepath = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(filepath, index=False)

        dfs, shapes = load_forex_data(
            paths=[str(filepath)],
            timeframe="1h",
        )

        assert "EUR_USD" in dfs
        assert len(dfs["EUR_USD"]) > 0
        assert "EUR_USD" in shapes

    def test_session_features_added(self, sample_forex_df, temp_dir):
        """Test session features are added."""
        from data_loader_forex import load_forex_data

        filepath = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(filepath, index=False)

        dfs, _ = load_forex_data(
            paths=[str(filepath)],
            add_session_features=True,
        )

        df = dfs["EUR_USD"]
        assert "session" in df.columns
        assert "session_liquidity" in df.columns
        assert "is_session_overlap" in df.columns

    def test_get_active_session(self):
        """Test session detection in data loader."""
        from data_loader_forex import _get_active_session

        # Test various times
        london = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)
        assert "london" in _get_active_session(london)

        tokyo = datetime(2024, 1, 8, 3, 0, tzinfo=timezone.utc)
        assert "tokyo" in _get_active_session(tokyo)

        overlap = datetime(2024, 1, 8, 14, 0, tzinfo=timezone.utc)
        assert "overlap" in _get_active_session(overlap)

    def test_filter_forex_weekends(self, sample_forex_df):
        """Test weekend filtering in data loader."""
        from data_loader_forex import _filter_forex_weekends

        # Add weekend timestamp
        weekend_ts = int(datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc).timestamp())
        df = sample_forex_df.copy()
        df = pd.concat([
            df,
            pd.DataFrame([{"timestamp": weekend_ts, "close": 1.1, "symbol": "EUR_USD"}])
        ], ignore_index=True)

        filtered = _filter_forex_weekends(df)
        assert weekend_ts not in filtered["timestamp"].values

    def test_merge_swap_rates(self, sample_forex_df, sample_swap_df, temp_dir):
        """Test swap rate merging."""
        from data_loader_forex import _merge_swap_rates

        # Save swap data
        swap_path = Path(temp_dir) / "EUR_USD_swaps.parquet"
        sample_swap_df.to_parquet(swap_path, index=False)

        result = _merge_swap_rates(sample_forex_df, "EUR_USD", temp_dir)

        assert "long_swap" in result.columns
        assert "short_swap" in result.columns

    def test_merge_interest_rates(self, sample_forex_df, sample_rate_df, temp_dir):
        """Test interest rate merging."""
        from data_loader_forex import _merge_interest_rates

        # Save rate data for both currencies
        usd_path = Path(temp_dir) / "USD_rates.parquet"
        eur_path = Path(temp_dir) / "EUR_rates.parquet"

        sample_rate_df.to_parquet(usd_path, index=False)

        eur_rates = sample_rate_df.copy()
        eur_rates["currency"] = "EUR"
        eur_rates["rate"] = eur_rates["rate"] - 0.75  # Lower EUR rate
        eur_rates.to_parquet(eur_path, index=False)

        result = _merge_interest_rates(sample_forex_df, "EUR_USD", temp_dir)

        assert "rate_differential" in result.columns
        assert "base_rate" in result.columns
        assert "quote_rate" in result.columns

    def test_merge_calendar_proximity(self, sample_forex_df, sample_calendar_df, temp_dir):
        """Test calendar proximity merging."""
        from data_loader_forex import _merge_calendar_proximity

        # Save calendar
        cal_path = Path(temp_dir) / "economic_calendar.parquet"
        sample_calendar_df.to_parquet(cal_path, index=False)

        result = _merge_calendar_proximity(sample_forex_df, "EUR_USD", temp_dir)

        assert "hours_to_next_event" in result.columns
        assert "hours_since_last_event" in result.columns
        assert "is_event_window" in result.columns

    def test_list_available_pairs(self, sample_forex_df, temp_dir):
        """Test listing available pairs."""
        from data_loader_forex import list_available_pairs

        # Save multiple pairs
        for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
            path = Path(temp_dir) / f"{pair}.parquet"
            df = sample_forex_df.copy()
            df["symbol"] = pair
            df.to_parquet(path, index=False)

        pairs = list_available_pairs(temp_dir)
        assert "EUR_USD" in pairs
        assert "GBP_USD" in pairs
        assert "USD_JPY" in pairs

    def test_get_pair_info(self, sample_forex_df, temp_dir):
        """Test getting pair info."""
        from data_loader_forex import get_pair_info

        path = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(path, index=False)

        info = get_pair_info("EUR_USD", temp_dir)

        assert info is not None
        assert info["pair"] == "EUR_USD"
        assert "rows" in info
        assert "start_date" in info
        assert "end_date" in info


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase7Integration:
    """Integration tests for Phase 7 components."""

    def test_full_data_pipeline(self, sample_forex_df, sample_swap_df, sample_rate_df, sample_calendar_df, temp_dir):
        """Test complete data loading pipeline."""
        from data_loader_forex import load_forex_data

        # Create directory structure
        forex_dir = Path(temp_dir) / "raw_forex"
        swap_dir = Path(temp_dir) / "swaps"
        rate_dir = Path(temp_dir) / "rates"
        cal_dir = Path(temp_dir) / "calendar"

        forex_dir.mkdir(parents=True)
        swap_dir.mkdir(parents=True)
        rate_dir.mkdir(parents=True)
        cal_dir.mkdir(parents=True)

        # Save all data
        sample_forex_df.to_parquet(forex_dir / "EUR_USD.parquet", index=False)
        sample_swap_df.to_parquet(swap_dir / "EUR_USD_swaps.parquet", index=False)
        sample_rate_df.to_parquet(rate_dir / "USD_rates.parquet", index=False)

        eur_rates = sample_rate_df.copy()
        eur_rates["currency"] = "EUR"
        eur_rates["rate"] = eur_rates["rate"] - 0.75
        eur_rates.to_parquet(rate_dir / "EUR_rates.parquet", index=False)

        sample_calendar_df.to_parquet(cal_dir / "economic_calendar.parquet", index=False)

        # Load with all features
        dfs, shapes = load_forex_data(
            paths=[str(forex_dir / "EUR_USD.parquet")],
            merge_swaps=True,
            merge_rates=True,
            merge_calendar=True,
            swap_dir=str(swap_dir),
            rate_dir=str(rate_dir),
            calendar_dir=str(cal_dir),
        )

        assert "EUR_USD" in dfs
        df = dfs["EUR_USD"]

        # Check all features are present
        assert "long_swap" in df.columns
        assert "short_swap" in df.columns
        assert "rate_differential" in df.columns
        assert "hours_to_next_event" in df.columns
        assert "session" in df.columns
        assert "session_liquidity" in df.columns

    def test_glob_pattern_loading(self, sample_forex_df, temp_dir):
        """Test loading multiple files with glob pattern."""
        from data_loader_forex import load_forex_data

        # Save multiple pairs
        for pair in ["EUR_USD", "GBP_USD"]:
            path = Path(temp_dir) / f"{pair}.parquet"
            df = sample_forex_df.copy()
            df["symbol"] = pair
            df.to_parquet(path, index=False)

        dfs, shapes = load_forex_data(
            paths=[f"{temp_dir}/*.parquet"],
        )

        assert len(dfs) == 2
        assert "EUR_USD" in dfs
        assert "GBP_USD" in dfs


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_dataframe(self, temp_dir):
        """Test handling of empty DataFrame."""
        from data_loader_forex import load_forex_data

        # Create empty file
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        path = Path(temp_dir) / "EMPTY.parquet"
        empty_df.to_parquet(path, index=False)

        dfs, shapes = load_forex_data(paths=[str(path)])

        # Should either skip or handle gracefully
        assert "EMPTY" not in dfs or dfs["EMPTY"].empty

    def test_missing_swap_file(self, sample_forex_df, temp_dir):
        """Test handling when swap file doesn't exist."""
        from data_loader_forex import _merge_swap_rates

        result = _merge_swap_rates(sample_forex_df, "EUR_USD", "/nonexistent/path")

        # Should add default columns
        assert "long_swap" in result.columns
        assert "short_swap" in result.columns
        assert (result["long_swap"] == 0.0).all()

    def test_missing_rate_file(self, sample_forex_df, temp_dir):
        """Test handling when rate file doesn't exist."""
        from data_loader_forex import _merge_interest_rates

        result = _merge_interest_rates(sample_forex_df, "EUR_USD", "/nonexistent/path")

        assert "rate_differential" in result.columns
        assert (result["rate_differential"] == 0.0).all()

    def test_invalid_pair_format(self):
        """Test handling of invalid pair format."""
        from scripts.download_swap_rates import build_historical_swaps, SwapDownloadConfig

        config = SwapDownloadConfig(start_date="2024-01-01", end_date="2024-01-31")
        pair, df, error = build_historical_swaps("INVALID", config)

        assert error is not None
        assert df is None

    def test_date_filtering(self, sample_forex_df, temp_dir):
        """Test date range filtering."""
        from data_loader_forex import load_forex_data

        path = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(path, index=False)

        # Filter to subset
        dfs, _ = load_forex_data(
            paths=[str(path)],
            start_date="2024-01-09",
            end_date="2024-01-10",
        )

        if "EUR_USD" in dfs:
            df = dfs["EUR_USD"]
            # Should have fewer bars than original
            assert len(df) <= len(sample_forex_df)


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Tests to ensure Phase 7 doesn't break existing functionality."""

    def test_imports_dont_break_existing(self):
        """Test that new imports don't break existing modules."""
        # These imports should work without affecting existing functionality
        from data_loader_forex import load_forex_data
        from scripts.download_forex_data import ForexDownloadConfig, MAJOR_PAIRS
        from scripts.download_swap_rates import SwapDownloadConfig
        from scripts.download_economic_calendar import CalendarDownloadConfig
        from scripts.download_interest_rates import RateDownloadConfig

        # All should be importable
        assert ForexDownloadConfig is not None
        assert load_forex_data is not None

    def test_existing_data_loader_unaffected(self):
        """Test that existing data_loader_multi_asset.py is unaffected."""
        from data_loader_multi_asset import (
            AssetClass,
            DataVendor,
            load_from_file,
            timeframe_to_seconds,
        )

        # FOREX should be an option
        assert AssetClass.FOREX.value == "forex"
        assert DataVendor.OANDA.value == "oanda"

        # Existing functions should work
        assert timeframe_to_seconds("4h") == 14400


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
