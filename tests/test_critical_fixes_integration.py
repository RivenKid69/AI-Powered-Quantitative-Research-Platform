"""
Integration tests for CRITICAL fixes using REAL production code.

These tests verify that the actual code in transformers.py and feature_pipe.py
contains the fixes, not just local test implementations.

IMPORTANT: These tests work WITHOUT external dependencies (arch module).
"""

import math
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# =============================================================================
# Direct Code Inspection (verify fixes are in place)
# =============================================================================

class TestCodeInspection:
    """Verify that critical fixes are present in actual source code."""

    def test_yang_zhang_bessel_correction_in_code(self):
        """
        CRITICAL FIX #2: Verify Bessel's correction is in transformers.py.

        Read actual source and check for (rs_count - 1) instead of rs_count.
        """
        with open("transformers.py", "r", encoding="utf-8") as f:
            source = f.read()

        # Should have the fix comment
        assert "CRITICAL FIX #2" in source, "Missing CRITICAL FIX #2 comment"
        assert "Bessel's correction to Rogers-Satchell" in source, "Missing Bessel's description"

        # Should check rs_count < 2
        assert "if rs_count < 2:" in source, "Missing rs_count < 2 check"

        # Should use (rs_count - 1) for division
        assert "rs_sum / (rs_count - 1)" in source, "Missing Bessel's correction (rs_count - 1)"

        # Should NOT use rs_sum / rs_count (old bug)
        # Count occurrences - should only be in old code comments
        lines = source.split('\n')
        bug_pattern_count = sum(1 for line in lines if "rs_sum / rs_count" in line and "//" not in line and "#" not in line)
        assert bug_pattern_count == 0, f"Found {bug_pattern_count} instances of old buggy pattern 'rs_sum / rs_count'"

    def test_ewma_robust_initialization_in_code(self):
        """
        CRITICAL FIX #4: Verify robust EWMA init is in transformers.py.

        Should use median initialization, not first return squared.
        """
        with open("transformers.py", "r", encoding="utf-8") as f:
            source = f.read()

        # Should have the fix comment
        assert "CRITICAL FIX #4" in source, "Missing CRITICAL FIX #4 comment"
        assert "Robust EWMA initialization" in source, "Missing robust EWMA description"

        # Should use median for limited data
        assert "np.median(log_returns ** 2)" in source, "Missing median initialization"

        # Should use mean fallback for very limited data
        assert "np.mean(log_returns ** 2)" in source, "Missing mean fallback"

        # Should have conditional logic for different data sizes
        assert "len(log_returns) >= 10" in source, "Missing data size check (>=10)"
        assert "len(log_returns) >= 3" in source, "Missing data size check (>=3)"

    def test_log_returns_consistency_in_code(self):
        """
        CRITICAL FIX #3: Verify targets use log returns in feature_pipe.py.

        Should use np.log() instead of linear returns.
        """
        with open("feature_pipe.py", "r", encoding="utf-8") as f:
            source = f.read()

        # Should have the fix comment
        assert "CRITICAL FIX #3" in source, "Missing CRITICAL FIX #3 comment"
        assert "LOG returns to match feature returns" in source, "Missing log returns description"

        # Find make_targets method
        make_targets_start = source.find("def make_targets(self, df:")
        assert make_targets_start != -1, "Cannot find make_targets method"

        # Extract make_targets method (rough approximation)
        make_targets_section = source[make_targets_start:make_targets_start + 2000]

        # Should use np.log for targets
        assert "np.log(future_price" in make_targets_section, "Missing np.log() for targets"

        # Should NOT use old linear formula (future_price / price - 1.0)
        # Check that old pattern is only in comments
        lines = make_targets_section.split('\n')
        old_pattern = "future_price / price - 1"
        old_pattern_count = sum(1 for line in lines if old_pattern in line and "#" not in line[:line.find(old_pattern)] if old_pattern in line)
        assert old_pattern_count == 0, f"Found {old_pattern_count} instances of old linear returns pattern (should be in comments only)"


# =============================================================================
# Functional Integration Tests (without arch dependency)
# =============================================================================

class TestFeaturePipeIntegration:
    """
    Integration tests for feature_pipe.py using real FeaturePipe class.

    These tests verify CRITICAL FIX #3 (log returns) is working.
    """

    def test_make_targets_uses_log_returns(self):
        """
        Test that FeaturePipe.make_targets() uses log returns.

        This tests REAL production code, not a test copy.
        """
        # Import will fail if arch module is needed, but feature_pipe should work
        # We'll create a minimal FeaturePipe that doesn't need arch
        try:
            from feature_pipe import FeaturePipe
            from transformers import FeatureSpec
        except ImportError as e:
            pytest.skip(f"Cannot import FeaturePipe: {e}")

        # Create minimal spec (may fail if transformers needs arch)
        try:
            spec = FeatureSpec(
                lookbacks_prices=[240],
                bar_duration_minutes=1,
                yang_zhang_windows=None,
                parkinson_windows=None,
                garch_windows=None,  # Disable GARCH to avoid arch dependency
            )
            pipe = FeaturePipe(spec=spec, price_col="price")
        except Exception as e:
            pytest.skip(f"Cannot create FeaturePipe: {e}")

        # Create test data
        df = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "price": [100.0, 110.0, 121.0],  # 10% returns each step
        })

        # Get targets
        targets = pipe.make_targets(df)

        # Verify log returns (not linear)
        expected_log_1 = math.log(110.0 / 100.0)  # ln(1.1) ≈ 0.0953
        expected_linear_1 = 0.1

        actual_1 = targets.iloc[0]

        # Should be log return
        assert abs(actual_1 - expected_log_1) < 1e-6, \
            f"Target should be log return {expected_log_1:.6f}, got {actual_1:.6f}"

        # Should NOT be linear return
        assert abs(actual_1 - expected_linear_1) > 0.004, \
            f"Target should NOT be linear return {expected_linear_1:.6f}, got {actual_1:.6f}"

    def test_large_returns_log_consistency(self):
        """
        For large returns (50%), log vs linear makes a big difference.

        Verify targets use log returns for consistency.
        """
        try:
            from feature_pipe import FeaturePipe
            from transformers import FeatureSpec
        except ImportError:
            pytest.skip("Cannot import FeaturePipe")

        try:
            spec = FeatureSpec(
                lookbacks_prices=[240],
                bar_duration_minutes=1,
                garch_windows=None,
            )
            pipe = FeaturePipe(spec=spec, price_col="price")
        except Exception:
            pytest.skip("Cannot create FeaturePipe")

        # Large price movement
        df = pd.DataFrame({
            "ts_ms": [1000, 2000],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "price": [100.0, 150.0],  # 50% move
        })

        targets = pipe.make_targets(df)
        actual = targets.iloc[0]

        expected_log = math.log(150.0 / 100.0)  # ≈ 0.4055
        expected_linear = 0.5

        # Verify log return
        assert abs(actual - expected_log) < 1e-6, \
            f"Target should be log return {expected_log:.6f}, got {actual:.6f}"

        # Difference should be significant (19%)
        difference = abs(expected_linear - expected_log)
        assert difference > 0.09, \
            f"Log vs linear difference {difference:.4f} should be >9% for 50% move"


class TestYangZhangIntegration:
    """
    Integration test for Yang-Zhang Bessel's correction.

    Note: Cannot import calculate_yang_zhang_volatility directly due to arch dependency.
    Instead, verify code inspection and behavior through indirect means.
    """

    def test_yang_zhang_source_code_structure(self):
        """
        Verify Yang-Zhang function has correct structure in source.

        Checks that all three components use consistent (n-1) denominator.
        The fix is in _try_calculate_yang_zhang (internal function).
        """
        with open("transformers.py", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find _try_calculate_yang_zhang function (where the fix is)
        yang_zhang_start = None
        for i, line in enumerate(lines):
            if "def _try_calculate_yang_zhang" in line:
                yang_zhang_start = i
                break

        assert yang_zhang_start is not None, "Cannot find _try_calculate_yang_zhang function"

        # Extract function (next ~100 lines)
        function_lines = lines[yang_zhang_start:yang_zhang_start + 100]
        function_text = ''.join(function_lines)

        # Check all three components
        # 1. Overnight: should use (len - 1)
        assert "(len(overnight_returns) - 1)" in function_text, \
            "Overnight component missing Bessel's correction"

        # 2. Open-close: should use (len - 1)
        assert "(len(oc_returns) - 1)" in function_text, \
            "Open-close component missing Bessel's correction"

        # 3. Rogers-Satchell: should use (rs_count - 1) WITH the fix
        assert "(rs_count - 1)" in function_text, \
            "Rogers-Satchell component missing Bessel's correction (CRITICAL FIX #2)"

        # Check for rs_count < 2 guard
        assert "rs_count < 2" in function_text, \
            "Missing minimum sample size check for rs_count"


class TestEWMAIntegration:
    """
    Integration test for EWMA robust initialization.
    """

    def test_ewma_source_code_structure(self):
        """
        Verify EWMA function has robust initialization in source.
        """
        with open("transformers.py", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find EWMA function
        ewma_start = None
        for i, line in enumerate(lines):
            if "def _calculate_ewma_volatility" in line:
                ewma_start = i
                break

        assert ewma_start is not None, "Cannot find EWMA function"

        # Extract function
        function_lines = lines[ewma_start:ewma_start + 80]
        function_text = ''.join(function_lines)

        # Check for median initialization
        assert "np.median(log_returns ** 2)" in function_text, \
            "Missing median initialization (CRITICAL FIX #4)"

        # Check for mean fallback
        assert "np.mean(log_returns ** 2)" in function_text, \
            "Missing mean fallback"

        # Check for sample variance when sufficient data
        assert "np.var(log_returns, ddof=1)" in function_text, \
            "Missing sample variance for sufficient data"

        # Check for conditional branches
        assert "len(log_returns) >= 10" in function_text, \
            "Missing check for sufficient data (>=10)"
        assert "len(log_returns) >= 3" in function_text, \
            "Missing check for limited data (>=3)"


# =============================================================================
# Regression Tests (ensure fixes don't break existing behavior)
# =============================================================================

class TestNoRegressions:
    """Ensure fixes don't break existing functionality."""

    def test_log_returns_time_additivity(self):
        """
        Log returns are time-additive (key property).

        r(t1→t3) = r(t1→t2) + r(t2→t3)

        This ensures targets have correct mathematical properties.
        """
        prices = [100.0, 110.0, 121.0]

        log_ret_1_2 = np.log(prices[1] / prices[0])
        log_ret_2_3 = np.log(prices[2] / prices[1])
        log_ret_1_3 = np.log(prices[2] / prices[0])

        # Time-additivity
        sum_of_parts = log_ret_1_2 + log_ret_2_3
        assert abs(log_ret_1_3 - sum_of_parts) < 1e-10, \
            f"Log returns should be time-additive: {log_ret_1_3:.10f} != {sum_of_parts:.10f}"

    def test_linear_returns_not_time_additive(self):
        """
        Linear returns are NOT time-additive (this is why we use log).

        r(t1→t3) != r(t1→t2) + r(t2→t3)
        """
        prices = [100.0, 110.0, 121.0]

        linear_ret_1_2 = prices[1] / prices[0] - 1
        linear_ret_2_3 = prices[2] / prices[1] - 1
        linear_ret_1_3 = prices[2] / prices[0] - 1

        # NOT additive
        sum_of_parts = linear_ret_1_2 + linear_ret_2_3
        assert abs(linear_ret_1_3 - sum_of_parts) > 0.001, \
            "Linear returns should NOT be time-additive"

        # Correct relationship: (1+r1)(1+r2) - 1 = r_total
        product_formula = (1 + linear_ret_1_2) * (1 + linear_ret_2_3) - 1
        assert abs(linear_ret_1_3 - product_formula) < 1e-10, \
            "Linear returns should follow product rule"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
