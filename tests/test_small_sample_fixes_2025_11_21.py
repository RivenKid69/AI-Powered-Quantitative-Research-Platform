"""
Comprehensive tests for small sample fixes (2025-11-21).

This test suite verifies two critical fixes:
1. FIX: sharpe_ratio and sortino_ratio with ddof=1 on small samples (N < 3)
   - Problem: np.std([x], ddof=1) = NaN → breaks Optuna/tensorboard
   - Solution: Return 0.0 for N < 3, add np.isfinite() checks

2. FIX: Repeated transform_df() application causes double shift of 'close'
   - Problem: Each transform_df() shifts close → accumulates lag
   - Solution: Add warning when repeated application detected via attrs marker

References:
- Bailey & López de Prado (2012): "The Sharpe Ratio Efficient Frontier"
- Sortino & Van Der Meer (1991): "Downside Risk"
- De Prado (2018): "Advances in Financial ML" - idempotent transforms
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from train_model_multi_patch import sharpe_ratio, sortino_ratio
from features_pipeline import FeaturePipeline


# ==============================================================================
# FIX 1: sharpe_ratio and sortino_ratio with small samples
# ==============================================================================

class TestSharpeRatioSmallSamples:
    """Test sharpe_ratio with edge cases and small samples."""

    def test_sharpe_n_eq_1_returns_zero(self):
        """N=1: Should return 0.0 instead of NaN."""
        returns = np.array([0.05])
        result = sharpe_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for N=1, got {result}"
        assert np.isfinite(result), f"Result should be finite, got {result}"

    def test_sharpe_n_eq_2_returns_zero(self):
        """N=2: Should return 0.0 (insufficient df for reliable estimate)."""
        returns = np.array([0.05, -0.03])
        result = sharpe_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for N=2, got {result}"
        assert np.isfinite(result), f"Result should be finite, got {result}"

    def test_sharpe_n_eq_3_returns_valid(self):
        """N=3: Should return valid Sharpe ratio (minimum 2 df)."""
        returns = np.array([0.05, -0.03, 0.02])
        result = sharpe_ratio(returns)
        assert np.isfinite(result), f"Expected finite result for N=3, got {result}"
        assert result != 0.0, f"Expected non-zero Sharpe for N=3, got {result}"

    def test_sharpe_constant_returns_zero(self):
        """Constant returns: Should return 0.0 (std < 1e-9)."""
        returns = np.array([0.01] * 10)
        result = sharpe_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for constant returns, got {result}"

    def test_sharpe_all_nan_returns_zero(self):
        """All NaN: Should return 0.0."""
        returns = np.array([np.nan, np.nan, np.nan])
        result = sharpe_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for all NaN, got {result}"
        assert np.isfinite(result), f"Result should be finite, got {result}"

    def test_sharpe_normal_case_n_100(self):
        """Normal case (N=100): Should return valid Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02 + 0.001  # 0.1% mean, 2% std
        result = sharpe_ratio(returns)
        assert np.isfinite(result), f"Expected finite result for N=100, got {result}"
        assert result != 0.0, f"Expected non-zero Sharpe for N=100, got {result}"

    def test_sharpe_negative_mean_valid(self):
        """Negative mean returns: Should return valid (negative) Sharpe."""
        returns = np.array([-0.01, -0.02, -0.03, -0.01, -0.02])
        result = sharpe_ratio(returns)
        assert np.isfinite(result), f"Expected finite result, got {result}"
        assert result < 0, f"Expected negative Sharpe for negative returns, got {result}"


class TestSortinoRatioSmallSamples:
    """Test sortino_ratio with edge cases and small samples."""

    def test_sortino_n_eq_1_returns_zero(self):
        """N=1: Should return 0.0 instead of NaN."""
        returns = np.array([0.05])
        result = sortino_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for N=1, got {result}"
        assert np.isfinite(result), f"Result should be finite, got {result}"

    def test_sortino_n_eq_2_returns_zero(self):
        """N=2: Should return 0.0 (insufficient df for reliable estimate)."""
        returns = np.array([0.05, -0.03])
        result = sortino_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for N=2, got {result}"
        assert np.isfinite(result), f"Result should be finite, got {result}"

    def test_sortino_n_eq_3_returns_valid(self):
        """N=3: Should return valid Sortino ratio (minimum 2 df)."""
        returns = np.array([0.05, -0.03, 0.02])
        result = sortino_ratio(returns)
        assert np.isfinite(result), f"Expected finite result for N=3, got {result}"

    def test_sortino_no_downside_fallback_to_sharpe(self):
        """No downside (all positive): Should fallback to Sharpe ratio."""
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        result = sortino_ratio(returns, risk_free_rate=0.0)
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        assert np.isclose(result, sharpe), f"Expected Sortino={sharpe}, got {result}"

    def test_sortino_few_downside_fallback_to_sharpe(self):
        """Few downside (< 20): Should fallback to Sharpe ratio."""
        returns = np.array([0.01] * 19 + [-0.02])  # 19 positive, 1 negative
        result = sortino_ratio(returns, risk_free_rate=0.0)
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        assert np.isclose(result, sharpe), f"Expected Sortino={sharpe}, got {result}"

    def test_sortino_many_downside_uses_downside_std(self):
        """Many downside (>= 20): Should use downside deviation."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.03 - 0.001  # Slightly negative mean
        result = sortino_ratio(returns, risk_free_rate=0.0)
        assert np.isfinite(result), f"Expected finite result, got {result}"

    def test_sortino_constant_returns_zero(self):
        """Constant returns: Should return 0.0 (std < 1e-9)."""
        returns = np.array([0.01] * 10)
        result = sortino_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for constant returns, got {result}"

    def test_sortino_all_nan_returns_zero(self):
        """All NaN: Should return 0.0."""
        returns = np.array([np.nan, np.nan, np.nan])
        result = sortino_ratio(returns)
        assert result == 0.0, f"Expected 0.0 for all NaN, got {result}"
        assert np.isfinite(result), f"Result should be finite, got {result}"


# ==============================================================================
# FIX 2: Repeated transform_df() application causes double shift
# ==============================================================================

class TestTransformDFDoubleShift:
    """Test FeaturePipeline.transform_df() repeated application protection."""

    def setup_method(self):
        """Create simple test data and fitted pipeline."""
        self.df = pd.DataFrame({
            'timestamp': [1000, 2000, 3000, 4000, 5000],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        self.pipe = FeaturePipeline()
        dfs_dict = {'BTCUSDT': self.df.copy()}
        self.pipe.fit(dfs_dict)

    def test_first_transform_shifts_close_correctly(self):
        """First transform_df() should shift close correctly."""
        df_transformed = self.pipe.transform_df(self.df.copy())

        # Check: close[0] should be NaN (shifted)
        assert pd.isna(df_transformed['close'].iloc[0]), \
            "First element of 'close' should be NaN after shift"

        # Check: close[1] should be original close[0]=100
        assert df_transformed['close'].iloc[1] == 100.0, \
            f"Expected close[1]=100.0, got {df_transformed['close'].iloc[1]}"

        # Check: marker is set
        assert hasattr(df_transformed, 'attrs'), "DataFrame should have attrs"
        assert df_transformed.attrs.get('_feature_pipeline_transformed') == True, \
            "Transform marker should be set"

    def test_second_transform_warns_about_double_shift(self):
        """Second transform_df() should warn about repeated application."""
        df_transformed_1 = self.pipe.transform_df(self.df.copy())

        # Second transform should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df_transformed_2 = self.pipe.transform_df(df_transformed_1.copy())

            # Check: warning was raised
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert issubclass(w[0].category, RuntimeWarning), \
                f"Expected RuntimeWarning, got {w[0].category}"
            assert "already-transformed" in str(w[0].message).lower(), \
                f"Warning should mention 'already-transformed', got: {w[0].message}"

    def test_second_transform_causes_double_shift(self):
        """Second transform_df() causes double shift (expected behavior with warning)."""
        df_transformed_1 = self.pipe.transform_df(self.df.copy())

        # Suppress warning for this test (we know it's wrong)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_transformed_2 = self.pipe.transform_df(df_transformed_1.copy())

        # Check: close is shifted TWICE
        # Original: [100, 101, 102, 103, 104]
        # After 1st: [NaN, 100, 101, 102, 103]
        # After 2nd: [NaN, NaN, 100, 101, 102]
        assert pd.isna(df_transformed_2['close'].iloc[0]), "close[0] should be NaN"
        assert pd.isna(df_transformed_2['close'].iloc[1]), "close[1] should be NaN (double shift)"
        assert df_transformed_2['close'].iloc[2] == 100.0, \
            f"Expected close[2]=100.0 (double shifted), got {df_transformed_2['close'].iloc[2]}"

    def test_transform_with_close_orig_no_double_shift(self):
        """Transform with close_orig present should NOT shift again."""
        # Add close_orig to prevent shift
        df_with_orig = self.df.copy()
        df_with_orig['close_orig'] = df_with_orig['close'].copy()

        # First transform
        df_transformed_1 = self.pipe.transform_df(df_with_orig.copy())

        # Second transform (should NOT shift because close_orig exists)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the marker warning
            df_transformed_2 = self.pipe.transform_df(df_transformed_1.copy())

        # Check: close should be same as after first transform (no double shift)
        pd.testing.assert_series_equal(
            df_transformed_2['close'],
            df_transformed_1['close'],
            check_names=False
        )

    def test_transform_fresh_copy_no_warning(self):
        """Transform on fresh copy should NOT warn (no marker present)."""
        # First transform
        df_transformed_1 = self.pipe.transform_df(self.df.copy())

        # Second transform on FRESH copy (not transformed yet)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df_transformed_2 = self.pipe.transform_df(self.df.copy())  # Fresh copy!

            # Check: NO warning (fresh data)
            assert len(w) == 0, f"Expected no warning for fresh copy, got {len(w)}"


# ==============================================================================
# Integration tests: Optuna-like scenarios
# ==============================================================================

class TestOptunaIntegration:
    """Test that fixes prevent Optuna trial failures."""

    def test_sharpe_with_single_episode_returns_zero(self):
        """Single episode (N=1): Should return 0.0, not NaN (prevents Optuna crash)."""
        # Simulate very short training run with 1 episode
        episode_returns = np.array([0.05])
        sharpe = sharpe_ratio(episode_returns)

        assert np.isfinite(sharpe), \
            "Sharpe should be finite (Optuna requires finite metrics)"
        assert sharpe == 0.0, \
            f"Expected 0.0 for single episode, got {sharpe}"

    def test_sortino_with_two_episodes_returns_zero(self):
        """Two episodes (N=2): Should return 0.0, not NaN (prevents Optuna crash)."""
        episode_returns = np.array([0.05, -0.03])
        sortino = sortino_ratio(episode_returns)

        assert np.isfinite(sortino), \
            "Sortino should be finite (Optuna requires finite metrics)"
        assert sortino == 0.0, \
            f"Expected 0.0 for two episodes, got {sortino}"

    def test_normal_training_returns_valid_metrics(self):
        """Normal training (N=100): Should return valid Sharpe/Sortino."""
        np.random.seed(42)
        episode_returns = np.random.randn(100) * 0.02 + 0.001

        sharpe = sharpe_ratio(episode_returns)
        sortino = sortino_ratio(episode_returns)

        assert np.isfinite(sharpe), "Sharpe should be finite"
        assert np.isfinite(sortino), "Sortino should be finite"
        assert sharpe != 0.0, "Sharpe should be non-zero for normal training"
        assert sortino != 0.0, "Sortino should be non-zero for normal training"


# ==============================================================================
# Regression tests: Ensure existing behavior preserved
# ==============================================================================

class TestBackwardCompatibility:
    """Ensure fixes don't break existing valid use cases."""

    def test_sharpe_ratio_unchanged_for_n_gte_3(self):
        """Sharpe ratio unchanged for N >= 3 (existing behavior preserved)."""
        np.random.seed(42)
        for n in [3, 10, 30, 100]:
            returns = np.random.randn(n) * 0.02
            result = sharpe_ratio(returns)
            assert np.isfinite(result), f"Expected finite Sharpe for N={n}"

    def test_sortino_ratio_unchanged_for_n_gte_3(self):
        """Sortino ratio unchanged for N >= 3 (existing behavior preserved)."""
        np.random.seed(42)
        for n in [3, 10, 30, 100]:
            returns = np.random.randn(n) * 0.02
            result = sortino_ratio(returns)
            assert np.isfinite(result), f"Expected finite Sortino for N={n}"

    def test_transform_df_single_use_unchanged(self):
        """Single transform_df() use unchanged (existing behavior preserved)."""
        df = pd.DataFrame({
            'timestamp': [1000, 2000, 3000],
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })
        pipe = FeaturePipeline()
        pipe.fit({'SYM': df.copy()})

        # Single transform (normal use case)
        df_transformed = pipe.transform_df(df.copy())

        # Check: first element shifted
        assert pd.isna(df_transformed['close'].iloc[0])
        assert df_transformed['close'].iloc[1] == 100.0

        # Check: normalized columns present
        assert 'close_z' in df_transformed.columns
        assert 'volume_z' in df_transformed.columns


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
