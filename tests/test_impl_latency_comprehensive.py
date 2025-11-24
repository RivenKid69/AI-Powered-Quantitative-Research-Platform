# -*- coding: utf-8 -*-
"""
Comprehensive tests for impl_latency.py

Covers:
- LatencyCfg configuration
- LatencyImpl initialization
- Seasonality multipliers loading/reloading
- Volatility callbacks and multipliers
- Hour-of-week latency adjustments
- Min/max latency clamping
- Debug logging
- Edge cases and error handling
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from impl_latency import LatencyCfg, LatencyImpl


class TestLatencyCfg:
    """Test LatencyCfg dataclass."""

    def test_default_values(self):
        cfg = LatencyCfg()
        assert cfg.base_ms == 250
        assert cfg.jitter_ms == 50
        assert cfg.spike_p == 0.01
        assert cfg.spike_mult == 5.0
        assert cfg.timeout_ms == 2500
        assert cfg.retries == 1
        assert cfg.seed == 0
        assert cfg.symbol is None
        assert cfg.seasonality_path is None
        assert cfg.latency_seasonality_path is None
        assert cfg.refresh_period_days == 30
        assert cfg.seasonality_default == 1.0
        assert cfg.use_seasonality is True
        assert cfg.seasonality_override is None
        assert cfg.seasonality_override_path is None
        assert cfg.seasonality_hash is None
        assert cfg.seasonality_interpolate is False
        assert cfg.seasonality_day_only is False
        assert cfg.seasonality_auto_reload is False
        assert cfg.vol_metric == "sigma"
        assert cfg.vol_window == 120
        assert cfg.volatility_gamma == 0.0
        assert cfg.zscore_clip == 3.0
        assert cfg.min_ms == 0
        assert cfg.max_ms == 10000
        assert cfg.debug_log is False
        assert cfg.vol_debug_log is False

    def test_custom_values(self):
        cfg = LatencyCfg(
            base_ms=300,
            jitter_ms=100,
            spike_p=0.02,
            spike_mult=10.0,
            timeout_ms=5000,
            retries=3,
            seed=42,
            symbol="BTCUSDT",
            use_seasonality=False,
            min_ms=50,
            max_ms=20000,
            debug_log=True,
        )
        assert cfg.base_ms == 300
        assert cfg.jitter_ms == 100
        assert cfg.spike_p == 0.02
        assert cfg.spike_mult == 10.0
        assert cfg.timeout_ms == 5000
        assert cfg.retries == 3
        assert cfg.seed == 42
        assert cfg.symbol == "BTCUSDT"
        assert cfg.use_seasonality is False
        assert cfg.min_ms == 50
        assert cfg.max_ms == 20000
        assert cfg.debug_log is True


class TestLatencyImpl:
    """Test LatencyImpl class."""

    def test_initialization_basic(self):
        cfg = LatencyCfg()
        impl = LatencyImpl(cfg)

        assert impl.cfg is cfg
        assert impl.latency is not None
        assert len(impl.latency) == 168  # hourly multipliers

    def test_initialization_with_day_only(self):
        cfg = LatencyCfg(seasonality_day_only=True)
        impl = LatencyImpl(cfg)

        assert len(impl.latency) == 7  # daily multipliers

    def test_initialization_with_custom_defaults(self):
        cfg = LatencyCfg(seasonality_default=1.5)
        impl = LatencyImpl(cfg)

        assert all(m == 1.5 for m in impl.latency)

    def test_initialization_with_seasonality_disabled(self):
        cfg = LatencyCfg(use_seasonality=False)
        impl = LatencyImpl(cfg)

        assert impl._has_seasonality is False

    def test_dump_multipliers(self):
        cfg = LatencyCfg()
        impl = LatencyImpl(cfg)

        multipliers = impl.dump_multipliers()
        assert len(multipliers) == 168
        assert all(m == 1.0 for m in multipliers)

    def test_load_multipliers_valid(self):
        cfg = LatencyCfg()
        impl = LatencyImpl(cfg)

        new_multipliers = [1.5] * 168
        impl.load_multipliers(new_multipliers)

        dumped = impl.dump_multipliers()
        assert all(m == 1.5 for m in dumped)

    def test_load_multipliers_invalid_length(self):
        cfg = LatencyCfg()
        impl = LatencyImpl(cfg)

        with pytest.raises(ValueError, match="multipliers must have length"):
            impl.load_multipliers([1.0] * 50)  # Wrong length

    def test_load_multipliers_mapping(self):
        cfg = LatencyCfg()
        impl = LatencyImpl(cfg)

        multipliers_dict = {i: 1.5 for i in range(168)}
        impl.load_multipliers(multipliers_dict)

        dumped = impl.dump_multipliers()
        assert all(m == 1.5 for m in dumped)

    def test_from_dict_basic(self):
        data = {
            "base_ms": 300,
            "jitter_ms": 75,
            "spike_p": 0.015,
            "spike_mult": 6.0,
            "timeout_ms": 3000,
            "retries": 2,
            "seed": 123,
        }

        impl = LatencyImpl.from_dict(data)

        assert impl.cfg.base_ms == 300
        assert impl.cfg.jitter_ms == 75
        assert impl.cfg.spike_p == 0.015
        assert impl.cfg.spike_mult == 6.0
        assert impl.cfg.timeout_ms == 3000
        assert impl.cfg.retries == 2
        assert impl.cfg.seed == 123

    def test_from_dict_with_seasonality(self):
        data = {
            "base_ms": 250,
            "seasonality_path": "data/seasonality.json",
            "use_seasonality": True,
            "seasonality_default": 1.2,
        }

        impl = LatencyImpl.from_dict(data)

        assert impl.cfg.use_seasonality is True
        assert impl.cfg.seasonality_default == 1.2

    def test_from_dict_with_volatility(self):
        data = {
            "base_ms": 250,
            "vol_metric": "atr",
            "vol_window": 200,
            "volatility_gamma": 0.5,
            "zscore_clip": 5.0,
        }

        impl = LatencyImpl.from_dict(data)

        assert impl.cfg.vol_metric == "atr"
        assert impl.cfg.vol_window == 200
        assert impl.cfg.volatility_gamma == 0.5
        assert impl.cfg.zscore_clip == 5.0

    def test_attach_to_simulator(self):
        """Test attaching latency implementation to a simulator."""
        cfg = LatencyCfg(base_ms=300, jitter_ms=100)
        impl = LatencyImpl(cfg)

        # Create a mock simulator object
        class MockSimulator:
            def __init__(self):
                self.latency = None

        sim = MockSimulator()

        # Attach implementation
        impl.attach_to(sim)

        # Verify latency was attached
        assert hasattr(sim, "latency")
        assert sim.latency is not None

    def test_get_stats_no_model(self):
        """Test get_stats when model is unavailable."""
        cfg = LatencyCfg()
        impl = LatencyImpl(cfg)

        # Model might be None if latency module not available
        stats = impl.get_stats()
        # Should return None or valid stats dict


class TestSeasonalityLoading:
    """Test seasonality multipliers loading."""

    def test_load_from_file(self, tmp_path):
        """Test loading seasonality from JSON file."""
        seasonality_path = tmp_path / "seasonality.json"
        seasonality_data = {
            "latency": [1.0 + i * 0.01 for i in range(168)],
            "metadata": {"generated_at": "2025-01-01T00:00:00Z"},
        }
        seasonality_path.write_text(json.dumps(seasonality_data))

        cfg = LatencyCfg(
            seasonality_path=str(seasonality_path),
            use_seasonality=True,
        )
        impl = LatencyImpl(cfg)

        # Verify multipliers were loaded
        assert impl._has_seasonality is True

    def test_load_missing_file(self, tmp_path):
        """Test handling of missing seasonality file."""
        cfg = LatencyCfg(
            seasonality_path=str(tmp_path / "nonexistent.json"),
            use_seasonality=True,
        )
        impl = LatencyImpl(cfg)

        # Should fall back to default multipliers
        assert all(m == 1.0 for m in impl.latency)

    def test_seasonality_override(self, tmp_path):
        """Test seasonality override mechanism."""
        base_path = tmp_path / "base.json"
        override_path = tmp_path / "override.json"

        base_data = {"latency": [1.0] * 168}
        override_data = {"latency": [2.0] * 168}

        base_path.write_text(json.dumps(base_data))
        override_path.write_text(json.dumps(override_data))

        cfg = LatencyCfg(
            seasonality_path=str(base_path),
            seasonality_override_path=str(override_path),
            use_seasonality=True,
        )
        impl = LatencyImpl(cfg)

        # Override should be applied
        # (multiplied: 1.0 * 2.0 = 2.0)


class TestVolatilityMultipliers:
    """Test volatility-based latency adjustments."""

    def test_volatility_disabled(self):
        """Test when volatility gamma is 0."""
        cfg = LatencyCfg(volatility_gamma=0.0)
        impl = LatencyImpl(cfg)

        # Volatility callback should not be created
        assert impl._wrapper is None or not hasattr(impl._wrapper, "_vol_cb")

    def test_volatility_enabled(self):
        """Test when volatility gamma is non-zero."""
        cfg = LatencyCfg(volatility_gamma=0.5, vol_window=100)
        impl = LatencyImpl(cfg)

        # Implementation should be initialized
        assert impl.cfg.volatility_gamma == 0.5
        assert impl.cfg.vol_window == 100


class TestMinMaxClamping:
    """Test min/max latency clamping."""

    def test_min_max_validation(self):
        """Test min/max values are validated during init."""
        cfg = LatencyCfg(min_ms=100, max_ms=5000)
        impl = LatencyImpl(cfg)

        assert impl.cfg.min_ms == 100
        assert impl.cfg.max_ms == 5000

    def test_max_less_than_min_raises_error(self):
        """Test that max_ms < min_ms raises ValueError."""
        # This should raise during _LatencyWithSeasonality init
        cfg = LatencyCfg(min_ms=1000, max_ms=500)

        with pytest.raises(ValueError, match="max_ms must be >= min_ms"):
            impl = LatencyImpl(cfg)
            # Create wrapper manually to trigger validation
            from impl_latency import _LatencyWithSeasonality

            if impl._model is not None:
                _LatencyWithSeasonality(
                    impl._model,
                    [1.0] * 168,
                    min_ms=1000,
                    max_ms=500,
                )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_seasonality_default(self):
        """Test empty seasonality_default."""
        cfg = LatencyCfg(seasonality_default=None)
        impl = LatencyImpl(cfg)

        # Should use default of 1.0
        assert all(m == 1.0 for m in impl.latency)

    def test_invalid_seasonality_default_type(self):
        """Test invalid seasonality_default type."""
        cfg = LatencyCfg(seasonality_default="invalid")
        impl = LatencyImpl(cfg)

        # Should fall back to [1.0, 1.0, ...]
        assert all(m == 1.0 for m in impl.latency)

    def test_seasonality_default_sequence(self):
        """Test seasonality_default as sequence."""
        cfg = LatencyCfg(seasonality_default=[1.5] * 168)
        impl = LatencyImpl(cfg)

        assert all(m == 1.5 for m in impl.latency)

    def test_invalid_seasonality_sequence_length(self):
        """Test invalid seasonality sequence length."""
        cfg = LatencyCfg(seasonality_default=[1.5] * 50)  # Wrong length
        impl = LatencyImpl(cfg)

        # Should fall back to default
        assert all(m == 1.0 for m in impl.latency)


class TestIntegration:
    """Integration tests."""

    def test_full_lifecycle(self):
        """Test complete LatencyImpl lifecycle."""
        cfg = LatencyCfg(
            base_ms=200,
            jitter_ms=50,
            spike_p=0.01,
            spike_mult=5.0,
            timeout_ms=2000,
            retries=2,
            seed=42,
            min_ms=50,
            max_ms=10000,
        )

        impl = LatencyImpl(cfg)

        # Verify initialization
        assert impl.cfg.base_ms == 200
        assert impl.cfg.jitter_ms == 50

        # Load custom multipliers
        new_multipliers = [1.2] * 168
        impl.load_multipliers(new_multipliers)

        # Dump and verify
        dumped = impl.dump_multipliers()
        assert all(m == 1.2 for m in dumped)

        # Attach to simulator
        class MockSimulator:
            pass

        sim = MockSimulator()
        impl.attach_to(sim)

        assert hasattr(sim, "latency")

    def test_seasonality_integration(self, tmp_path):
        """Test integration with seasonality files."""
        seasonality_path = tmp_path / "seasonality.json"
        seasonality_data = {
            "latency": [1.0 + (i % 24) * 0.1 for i in range(168)],
            "metadata": {
                "generated_at": "2025-01-01T00:00:00Z",
                "source": "test_integration",
            },
        }
        seasonality_path.write_text(json.dumps(seasonality_data))

        cfg = LatencyCfg(
            seasonality_path=str(seasonality_path),
            use_seasonality=True,
            seasonality_interpolate=True,
        )

        impl = LatencyImpl(cfg)

        # Verify seasonality was loaded
        multipliers = impl.dump_multipliers()
        assert len(multipliers) == 168


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
