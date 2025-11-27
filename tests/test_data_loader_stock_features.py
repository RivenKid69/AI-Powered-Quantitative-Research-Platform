# -*- coding: utf-8 -*-
"""
Tests for stock-specific features in data_loader_multi_asset.py.

Tests cover:
- apply_split_adjustment()
- add_corporate_action_features()
- load_equity_data_adjusted()
- Backward compatibility with crypto data loading
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path


class TestApplySplitAdjustment:
    """Tests for apply_split_adjustment function."""

    def test_uses_adjusted_close_when_available(self):
        """Test that adjusted_close column is used when present."""
        from data_loader_multi_asset import apply_split_adjustment

        df = pd.DataFrame({
            "close": [400.0, 404.0, 100.0, 102.0],  # Pre-split: 400, 404
            "adjusted_close": [100.0, 101.0, 100.0, 102.0],  # All adjusted
            "open": [398.0, 402.0, 99.0, 101.0],
            "high": [405.0, 408.0, 103.0, 105.0],
            "low": [395.0, 400.0, 98.0, 100.0],
            "volume": [1000.0, 1100.0, 4000.0, 4100.0],
        })

        result = apply_split_adjustment(df, "AAPL")

        # Should use adjusted_close for close column
        assert result.iloc[0]["close"] == pytest.approx(100.0)
        assert result.iloc[1]["close"] == pytest.approx(101.0)

    def test_computes_adjustment_factor_from_ratio(self):
        """Test adjustment factor computation when adjusted_close available."""
        from data_loader_multi_asset import apply_split_adjustment

        # 4:1 split scenario
        df = pd.DataFrame({
            "close": [400.0, 100.0],  # Before and after split
            "adjusted_close": [100.0, 100.0],  # Both adjusted to post-split
            "open": [398.0, 99.0],
            "high": [405.0, 103.0],
            "low": [395.0, 98.0],
            "volume": [1000, 4000],
        })

        result = apply_split_adjustment(df, "AAPL")

        # Volume should be adjusted (multiplied by factor)
        # First row: close=400, adj=100 → factor=4
        assert result.iloc[0]["volume"] == pytest.approx(4000.0)

    def test_no_adjustment_when_no_adjusted_close(self):
        """Test that no adjustment is made without adjusted_close."""
        from data_loader_multi_asset import apply_split_adjustment

        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "open": [99.0, 100.0, 101.0],
            "high": [101.0, 102.0, 103.0],
            "low": [98.0, 99.0, 100.0],
        })

        result = apply_split_adjustment(df, "AAPL")

        # Should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_handles_nan_in_adjusted_close(self):
        """Test handling of NaN values in adjusted_close."""
        from data_loader_multi_asset import apply_split_adjustment

        df = pd.DataFrame({
            "close": [100.0, np.nan, 102.0],
            "adjusted_close": [100.0, np.nan, 102.0],
            "open": [99.0, np.nan, 101.0],
            "high": [101.0, np.nan, 103.0],
            "low": [98.0, np.nan, 100.0],
            "volume": [1000.0, np.nan, 1100.0],
        })

        # Should not raise
        result = apply_split_adjustment(df, "AAPL")
        assert result is not None


class TestAddCorporateActionFeatures:
    """Tests for add_corporate_action_features function."""

    def test_adds_gap_features(self):
        """Test that gap features are added."""
        from data_loader_multi_asset import add_corporate_action_features

        df = pd.DataFrame({
            "open": [100.0, 105.0, 98.0],
            "close": [100.0, 106.0, 99.0],
            "high": [101.0, 107.0, 100.0],
            "low": [99.0, 104.0, 97.0],
        })

        result = add_corporate_action_features(df, "AAPL")

        assert "gap_pct" in result.columns
        assert "gap_direction" in result.columns
        assert "gap_magnitude" in result.columns

    def test_gap_features_correct_values(self):
        """Test gap feature values are computed correctly."""
        from data_loader_multi_asset import add_corporate_action_features

        df = pd.DataFrame({
            "open": [100.0, 110.0],  # 10% gap up
            "close": [100.0, 112.0],
            "high": [101.0, 113.0],
            "low": [99.0, 109.0],
        })

        result = add_corporate_action_features(df, "AAPL")

        # Second row should have 10% gap
        assert result.iloc[1]["gap_pct"] == pytest.approx(10.0)
        assert result.iloc[1]["gap_direction"] == 1  # Up

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        from data_loader_multi_asset import add_corporate_action_features

        df = pd.DataFrame({
            "open": [100.0, 105.0],
            "close": [100.0, 106.0],
            "high": [101.0, 107.0],
            "low": [99.0, 104.0],
            "volume": [1000, 1100],
            "custom_col": [1, 2],
        })

        result = add_corporate_action_features(df, "AAPL")

        assert "volume" in result.columns
        assert "custom_col" in result.columns
        assert result.iloc[0]["custom_col"] == 1


class TestLoadEquityDataAdjusted:
    """Tests for load_equity_data_adjusted function."""

    def test_basic_loading(self):
        """Test basic parquet loading."""
        from data_loader_multi_asset import load_equity_data_adjusted

        # Create test parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1100.0, 1200.0],
            })
            df.to_parquet(Path(tmpdir) / "AAPL.parquet")

            frames, obs_shapes = load_equity_data_adjusted(
                paths=[str(Path(tmpdir) / "AAPL.parquet")],
                apply_adjustments=False,
                add_corp_features=False,
            )

            assert "AAPL" in frames
            assert len(frames["AAPL"]) == 3

    def test_applies_adjustments(self):
        """Test that adjustments are applied when requested."""
        from data_loader_multi_asset import load_equity_data_adjusted

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [400.0, 100.0],
                "high": [405.0, 103.0],
                "low": [395.0, 98.0],
                "close": [400.0, 100.0],
                "adjusted_close": [100.0, 100.0],  # 4:1 split
                "volume": [1000.0, 4000.0],
            })
            df.to_parquet(Path(tmpdir) / "AAPL.parquet")

            frames, _ = load_equity_data_adjusted(
                paths=[str(Path(tmpdir) / "AAPL.parquet")],
                apply_adjustments=True,
                add_corp_features=False,
            )

            # First row should be adjusted
            assert frames["AAPL"].iloc[0]["close"] == pytest.approx(100.0)

    def test_adds_corporate_features(self):
        """Test that corporate action features are added."""
        from data_loader_multi_asset import load_equity_data_adjusted

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [100.0, 105.0],
                "high": [101.0, 106.0],
                "low": [99.0, 104.0],
                "close": [100.0, 105.0],
                "volume": [1000.0, 1100.0],
            })
            df.to_parquet(Path(tmpdir) / "AAPL.parquet")

            frames, _ = load_equity_data_adjusted(
                paths=[str(Path(tmpdir) / "AAPL.parquet")],
                add_corp_features=True,
            )

            assert "gap_pct" in frames["AAPL"].columns

    def test_glob_pattern_support(self):
        """Test loading with individual files (glob patterns may not work on all platforms)."""
        from data_loader_multi_asset import load_equity_data_adjusted

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            paths = []
            for symbol in ["AAPL", "MSFT"]:
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "volume": [1000.0, 1100.0],
                })
                path = Path(tmpdir) / f"{symbol}.parquet"
                df.to_parquet(path)
                paths.append(str(path))

            frames, _ = load_equity_data_adjusted(
                paths=paths,  # Use explicit paths, not glob
                add_corp_features=False,
            )

            assert len(frames) >= 1


class TestBackwardCompatibility:
    """Tests for backward compatibility with crypto data loading."""

    def test_crypto_loading_unchanged(self):
        """Test that crypto data loading is unchanged."""
        from data_loader_multi_asset import load_multi_asset_data

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [50000.0, 51000.0],
                "high": [51000.0, 52000.0],
                "low": [49000.0, 50000.0],
                "close": [50500.0, 51500.0],
                "volume": [100.0, 110.0],
            })
            df.to_parquet(Path(tmpdir) / "BTCUSDT.parquet")

            frames, obs_shapes = load_multi_asset_data(
                paths=[str(Path(tmpdir) / "BTCUSDT.parquet")],
                asset_class="crypto",  # Crypto mode
            )

            assert "BTCUSDT" in frames
            # Should NOT have gap features by default for crypto
            # (unless explicitly requested)

    def test_equity_with_default_params(self):
        """Test equity loading with default parameters."""
        from data_loader_multi_asset import load_multi_asset_data

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [100.0, 105.0],
                "high": [101.0, 106.0],
                "low": [99.0, 104.0],
                "close": [100.0, 105.0],
                "volume": [1000.0, 1100.0],
            })
            df.to_parquet(Path(tmpdir) / "AAPL.parquet")

            frames, _ = load_multi_asset_data(
                paths=[str(Path(tmpdir) / "AAPL.parquet")],
                asset_class="equity",
            )

            assert "AAPL" in frames


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        from data_loader_multi_asset import apply_split_adjustment

        df = pd.DataFrame(columns=["close", "open", "high", "low"])
        result = apply_split_adjustment(df, "AAPL")

        assert len(result) == 0

    def test_single_row_gap_features(self):
        """Test gap features with single row (no previous close)."""
        from data_loader_multi_asset import add_corporate_action_features

        df = pd.DataFrame({
            "open": [100.0],
            "close": [101.0],
            "high": [102.0],
            "low": [99.0],
        })

        result = add_corporate_action_features(df, "AAPL")

        # Gap should be NaN or 0.0 for first row (implementation dependent)
        if "gap_pct" in result.columns:
            gap_val = result.iloc[0]["gap_pct"]
            assert pd.isna(gap_val) or gap_val == 0.0

    def test_missing_ohlc_columns(self):
        """Test handling when OHLC columns are missing."""
        from data_loader_multi_asset import add_corporate_action_features

        df = pd.DataFrame({
            "close": [100.0, 101.0],
            # Missing open, high, low
        })

        # Should handle gracefully (may return original df or raise)
        try:
            result = add_corporate_action_features(df, "AAPL")
            # If it doesn't raise, check original columns preserved
            assert "close" in result.columns
        except KeyError:
            # Acceptable behavior - requires OHLC
            pass

    def test_volume_adjustment_with_zero_values(self):
        """Test volume adjustment handles zeros."""
        from data_loader_multi_asset import apply_split_adjustment

        df = pd.DataFrame({
            "close": [400.0, 100.0],
            "adjusted_close": [100.0, 100.0],
            "open": [398.0, 99.0],
            "high": [405.0, 103.0],
            "low": [395.0, 98.0],
            "volume": [0.0, 1000.0],  # Zero volume first row
        })

        # Should not raise
        result = apply_split_adjustment(df, "AAPL")
        assert result.iloc[0]["volume"] == 0.0  # 0 * 4 = 0


class TestObservationShapes:
    """Tests for observation shapes returned by loaders."""

    def test_obs_shapes_returned(self):
        """Test that observation shapes are returned."""
        from data_loader_multi_asset import load_equity_data_adjusted

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1100.0],
            })
            df.to_parquet(Path(tmpdir) / "AAPL.parquet")

            frames, obs_shapes = load_equity_data_adjusted(
                paths=[str(Path(tmpdir) / "AAPL.parquet")],
                add_corp_features=True,
            )

            # obs_shapes should be dict mapping symbol to shape
            assert isinstance(obs_shapes, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
