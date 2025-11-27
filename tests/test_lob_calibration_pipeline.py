# -*- coding: utf-8 -*-
"""
Tests for L3 Calibration Pipeline (Stage 8).

Tests cover:
1. L3CalibrationPipeline - unified calibration
2. LatencyCalibrator - latency distribution fitting
3. QueueDynamicsCalibrator - queue dynamics estimation
4. LatencyDistributionParams - parameter fitting
5. Factory functions and data loading

Total: 25+ tests
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lob.calibration_pipeline import (
    L3CalibrationPipeline,
    L3CalibrationResult,
    LatencyCalibrationResult,
    LatencyCalibrator,
    LatencyDistributionParams,
    LatencyObservation,
    QueueDynamicsCalibrator,
    QueueDynamicsResult,
    QuoteObservation,
    calibrate_from_dataframe,
    create_calibration_pipeline,
)
from lob.config import (
    FillProbabilityModelType,
    ImpactModelType,
    L3ExecutionConfig,
    LatencyDistributionType,
)
from lob.data_structures import Side


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_trades() -> List[Dict[str, Any]]:
    """Create sample trade data for calibration."""
    base_ts = 1705315800000  # 2024-01-15 10:30:00

    trades = []
    price = 100.0

    for i in range(100):
        # Simulate price movement
        price += (i % 3 - 1) * 0.01  # Small random walk

        trade = {
            "timestamp_ms": base_ts + i * 1000,
            "price": price,
            "qty": 100 + i % 50,
            "side": 1 if i % 2 == 0 else -1,
            "pre_trade_mid": price - 0.005,
            "post_trade_mid": price + 0.005 * (1 if i % 2 == 0 else -1),
        }
        trades.append(trade)

    return trades


@pytest.fixture
def sample_latencies() -> List[Dict[str, Any]]:
    """Create sample latency observations."""
    import random

    random.seed(42)
    latencies = []

    base_ts = 1705315800000000000  # nanoseconds

    for i in range(200):
        # Log-normal distributed latencies
        latency_us = max(10.0, random.lognormvariate(4.6, 0.5))  # ~100us mean

        obs = {
            "timestamp_ns": base_ts + i * 1000000,
            "type": ["feed", "order", "exchange", "fill"][i % 4],
            "latency_us": latency_us,
        }
        latencies.append(obs)

    return latencies


@pytest.fixture
def sample_orders() -> List[Dict[str, Any]]:
    """Create sample order data."""
    base_ts = 1705315800000000000  # nanoseconds

    orders = []
    for i in range(50):
        order = {
            "timestamp_ns": base_ts + i * 1000000000,
            "price": 100.0 + (i % 10) * 0.01,
            "qty": 100.0,
            "side": 1 if i % 2 == 0 else -1,
            "event_type": ["ADD", "CANCEL", "FILL"][i % 3],
            "fill_qty": 50.0 if i % 3 == 2 else 0.0,
        }
        orders.append(order)

    return orders


# ==============================================================================
# Test LatencyDistributionParams
# ==============================================================================


class TestLatencyDistributionParams:
    """Tests for LatencyDistributionParams."""

    def test_fit_lognormal(self) -> None:
        """Test fitting log-normal distribution."""
        import random

        random.seed(42)
        # Generate log-normal samples
        samples = [random.lognormvariate(4.6, 0.5) for _ in range(1000)]

        params = LatencyDistributionParams.fit_lognormal(samples)

        assert params.distribution == LatencyDistributionType.LOGNORMAL
        assert params.mean_us > 0
        assert params.std_us > 0
        assert params.mu > 0
        assert params.sigma > 0

        # Mean should be around exp(4.6) ≈ 100
        assert 50 < params.mean_us < 200

    def test_fit_lognormal_empty(self) -> None:
        """Test fitting with empty samples."""
        params = LatencyDistributionParams.fit_lognormal([])
        assert params.mean_us == 100.0  # Default

    def test_fit_lognormal_single_sample(self) -> None:
        """Test fitting with single sample."""
        params = LatencyDistributionParams.fit_lognormal([50.0])
        assert params.distribution == LatencyDistributionType.LOGNORMAL

    def test_fit_pareto(self) -> None:
        """Test fitting Pareto distribution."""
        import random

        random.seed(42)
        # Generate Pareto-like samples
        samples = [random.paretovariate(2.0) * 50 for _ in range(1000)]

        params = LatencyDistributionParams.fit_pareto(samples)

        assert params.distribution == LatencyDistributionType.PARETO
        assert params.pareto_alpha > 1.0
        assert params.pareto_xmin_us > 0

    def test_fit_pareto_empty(self) -> None:
        """Test Pareto fitting with empty samples."""
        params = LatencyDistributionParams.fit_pareto([])
        assert params.distribution == LatencyDistributionType.PARETO


# ==============================================================================
# Test LatencyCalibrator
# ==============================================================================


class TestLatencyCalibrator:
    """Tests for LatencyCalibrator."""

    def test_init(self) -> None:
        """Test calibrator initialization."""
        calibrator = LatencyCalibrator()
        assert calibrator._distribution_type == LatencyDistributionType.LOGNORMAL

    def test_add_observation(self) -> None:
        """Test adding latency observations."""
        calibrator = LatencyCalibrator()

        obs = LatencyObservation(
            timestamp_ns=1000000000,
            latency_type="order",
            latency_us=100.0,
        )
        calibrator.add_observation(obs)

        assert len(calibrator._observations["order"]) == 1

    def test_add_multiple_observations(self, sample_latencies: List[Dict]) -> None:
        """Test adding multiple observations."""
        calibrator = LatencyCalibrator()

        for lat in sample_latencies:
            obs = LatencyObservation(
                timestamp_ns=lat["timestamp_ns"],
                latency_type=lat["type"],
                latency_us=lat["latency_us"],
            )
            calibrator.add_observation(obs)

        # Each type should have ~50 observations
        assert len(calibrator._observations["feed"]) > 0
        assert len(calibrator._observations["order"]) > 0

    def test_fit(self, sample_latencies: List[Dict]) -> None:
        """Test fitting latency distributions."""
        calibrator = LatencyCalibrator()

        for lat in sample_latencies:
            obs = LatencyObservation(
                timestamp_ns=lat["timestamp_ns"],
                latency_type=lat["type"],
                latency_us=lat["latency_us"],
            )
            calibrator.add_observation(obs)

        result = calibrator.fit()

        assert isinstance(result, LatencyCalibrationResult)
        assert result.n_samples > 0
        assert result.feed_latency.mean_us > 0
        assert result.order_latency.mean_us > 0

    def test_clear(self, sample_latencies: List[Dict]) -> None:
        """Test clearing observations."""
        calibrator = LatencyCalibrator()

        for lat in sample_latencies[:10]:
            obs = LatencyObservation(
                timestamp_ns=lat["timestamp_ns"],
                latency_type=lat["type"],
                latency_us=lat["latency_us"],
            )
            calibrator.add_observation(obs)

        assert len(calibrator._observations["feed"]) > 0

        calibrator.clear()

        for key in calibrator._observations:
            assert len(calibrator._observations[key]) == 0


# ==============================================================================
# Test QueueDynamicsCalibrator
# ==============================================================================


class TestQueueDynamicsCalibrator:
    """Tests for QueueDynamicsCalibrator."""

    def test_init(self) -> None:
        """Test calibrator initialization."""
        calibrator = QueueDynamicsCalibrator()
        assert len(calibrator._orders) == 0
        assert len(calibrator._trades) == 0

    def test_add_order(self) -> None:
        """Test adding order records."""
        from lob.calibration import OrderRecord

        calibrator = QueueDynamicsCalibrator()

        order = OrderRecord(
            timestamp_ns=1000000000,
            price=100.0,
            qty=100.0,
            side=Side.BUY,
            event_type="ADD",
        )
        calibrator.add_order(order)

        assert len(calibrator._orders) == 1

    def test_add_queue_fill(self) -> None:
        """Test adding queue fill observations."""
        calibrator = QueueDynamicsCalibrator()

        calibrator.add_queue_fill(
            queue_position=5,
            fill_qty=100.0,
            time_in_queue_sec=30.0,
        )

        assert len(calibrator._queue_fills) == 1

    def test_fit(self) -> None:
        """Test fitting queue dynamics."""
        from lob.calibration import OrderRecord

        calibrator = QueueDynamicsCalibrator()

        base_ts = 1000000000
        for i in range(100):
            order = OrderRecord(
                timestamp_ns=base_ts + i * 1000000000,
                price=100.0,
                qty=100.0,
                side=Side.BUY,
                event_type=["ADD", "CANCEL", "FILL"][i % 3],
                fill_qty=50.0 if i % 3 == 2 else 0.0,
            )
            calibrator.add_order(order)

        # Add queue fills
        for i in range(20):
            calibrator.add_queue_fill(
                queue_position=i % 10,
                fill_qty=100.0,
                time_in_queue_sec=20.0 + i,
            )

        result = calibrator.fit()

        assert isinstance(result, QueueDynamicsResult)
        assert result.avg_arrival_rate > 0
        assert result.avg_cancel_rate >= 0
        assert result.avg_time_in_queue_sec > 0

    def test_clear(self) -> None:
        """Test clearing calibrator data."""
        from lob.calibration import OrderRecord

        calibrator = QueueDynamicsCalibrator()

        order = OrderRecord(
            timestamp_ns=1000000000,
            price=100.0,
            qty=100.0,
            side=Side.BUY,
            event_type="ADD",
        )
        calibrator.add_order(order)
        calibrator.add_queue_fill(5, 100.0, 30.0)

        assert len(calibrator._orders) == 1
        assert len(calibrator._queue_fills) == 1

        calibrator.clear()

        assert len(calibrator._orders) == 0
        assert len(calibrator._queue_fills) == 0


# ==============================================================================
# Test L3CalibrationPipeline
# ==============================================================================


class TestL3CalibrationPipeline:
    """Tests for L3CalibrationPipeline."""

    def test_init(self) -> None:
        """Test pipeline initialization."""
        pipeline = L3CalibrationPipeline(symbol="AAPL", asset_class="equity")

        assert pipeline.symbol == "AAPL"
        assert pipeline.n_trades == 0
        assert pipeline.n_orders == 0

    def test_add_trade(self) -> None:
        """Test adding trade observations."""
        pipeline = L3CalibrationPipeline()

        pipeline.add_trade(
            timestamp_ms=1705315800000,
            price=100.0,
            qty=100.0,
            side=1,
            pre_trade_mid=99.99,
            post_trade_mid=100.01,
        )

        assert pipeline.n_trades == 1

    def test_add_trades_batch(self, sample_trades: List[Dict]) -> None:
        """Test adding multiple trades."""
        pipeline = L3CalibrationPipeline()

        for trade in sample_trades:
            pipeline.add_trade(**trade)

        assert pipeline.n_trades == len(sample_trades)

    def test_add_order(self) -> None:
        """Test adding order observations."""
        pipeline = L3CalibrationPipeline()

        pipeline.add_order(
            timestamp_ns=1705315800000000000,
            price=100.0,
            qty=100.0,
            side=Side.BUY,
            event_type="ADD",
        )

        assert pipeline.n_orders == 1

    def test_add_latency_observation(self) -> None:
        """Test adding latency observations."""
        pipeline = L3CalibrationPipeline()

        pipeline.add_latency_observation(
            timestamp_ns=1705315800000000000,
            latency_type="order",
            latency_us=150.0,
        )

        assert len(pipeline._latency_observations) == 1

    def test_add_quote_observation(self) -> None:
        """Test adding quote observations."""
        pipeline = L3CalibrationPipeline()

        pipeline.add_quote_observation(
            timestamp_ns=1705315800000000000,
            bid_price=100.0,
            ask_price=100.05,
            bid_size=1000.0,
            ask_size=800.0,
        )

        assert len(pipeline._quote_observations) == 1

    def test_set_market_params(self) -> None:
        """Test setting market parameters."""
        pipeline = L3CalibrationPipeline()

        pipeline.set_market_params(
            avg_adv=5_000_000.0,
            avg_volatility=0.03,
        )

        assert pipeline._avg_adv == 5_000_000.0
        assert pipeline._avg_volatility == 0.03

    def test_calibrate_impact(self, sample_trades: List[Dict]) -> None:
        """Test impact calibration."""
        pipeline = L3CalibrationPipeline()
        pipeline.set_market_params(avg_adv=10_000_000.0, avg_volatility=0.02)

        for trade in sample_trades:
            pipeline.add_trade(**trade)

        result = pipeline.calibrate_impact()

        assert result is not None
        params = result.parameters
        assert "eta" in params or "gamma" in params or "delta" in params

    def test_calibrate_fill_probability(self, sample_trades: List[Dict]) -> None:
        """Test fill probability calibration."""
        pipeline = L3CalibrationPipeline()

        for trade in sample_trades:
            pipeline.add_trade(**trade)

        result = pipeline.calibrate_fill_probability()

        assert result is not None

    def test_calibrate_latency(self, sample_latencies: List[Dict]) -> None:
        """Test latency calibration."""
        pipeline = L3CalibrationPipeline()

        for lat in sample_latencies:
            pipeline.add_latency_observation(
                timestamp_ns=lat["timestamp_ns"],
                latency_type=lat["type"],
                latency_us=lat["latency_us"],
            )

        result = pipeline.calibrate_latency()

        assert isinstance(result, LatencyCalibrationResult)
        assert result.n_samples > 0

    def test_calibrate_queue_dynamics(self, sample_orders: List[Dict]) -> None:
        """Test queue dynamics calibration."""
        pipeline = L3CalibrationPipeline()

        for order in sample_orders:
            pipeline.add_order(
                timestamp_ns=order["timestamp_ns"],
                price=order["price"],
                qty=order["qty"],
                side=Side.BUY if order["side"] > 0 else Side.SELL,
                event_type=order["event_type"],
                fill_qty=order["fill_qty"],
            )

        result = pipeline.calibrate_queue_dynamics()

        assert isinstance(result, QueueDynamicsResult)

    def test_run_full_calibration(
        self, sample_trades: List[Dict], sample_latencies: List[Dict]
    ) -> None:
        """Test full calibration pipeline."""
        pipeline = L3CalibrationPipeline(symbol="TEST", asset_class="equity")
        pipeline.set_market_params(avg_adv=10_000_000.0, avg_volatility=0.02)

        # Add trades
        for trade in sample_trades:
            pipeline.add_trade(**trade)

        # Add latencies
        for lat in sample_latencies:
            pipeline.add_latency_observation(
                timestamp_ns=lat["timestamp_ns"],
                latency_type=lat["type"],
                latency_us=lat["latency_us"],
            )

        config = pipeline.run_full_calibration()

        assert isinstance(config, L3ExecutionConfig)
        assert config.market_impact.enabled
        assert config.fill_probability.enabled

    def test_get_calibration_result(self, sample_trades: List[Dict]) -> None:
        """Test getting detailed calibration result."""
        pipeline = L3CalibrationPipeline()

        for trade in sample_trades:
            pipeline.add_trade(**trade)

        result = pipeline.get_calibration_result()

        assert isinstance(result, L3CalibrationResult)
        assert result.n_trades == len(sample_trades)
        assert result.calibration_quality in ("low", "medium", "high")

    def test_clear(self, sample_trades: List[Dict]) -> None:
        """Test clearing pipeline data."""
        pipeline = L3CalibrationPipeline()

        for trade in sample_trades[:10]:
            pipeline.add_trade(**trade)

        assert pipeline.n_trades > 0

        pipeline.clear()

        assert pipeline.n_trades == 0
        assert pipeline.n_orders == 0

    def test_crypto_asset_class(self, sample_trades: List[Dict]) -> None:
        """Test calibration for crypto asset class."""
        pipeline = L3CalibrationPipeline(symbol="BTCUSDT", asset_class="crypto")

        for trade in sample_trades:
            pipeline.add_trade(**trade)

        config = pipeline.run_full_calibration()

        assert isinstance(config, L3ExecutionConfig)


# ==============================================================================
# Test Factory Functions
# ==============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_calibration_pipeline(self) -> None:
        """Test creating pipeline via factory."""
        pipeline = create_calibration_pipeline(symbol="AAPL", asset_class="equity")

        assert isinstance(pipeline, L3CalibrationPipeline)
        assert pipeline.symbol == "AAPL"

    def test_calibrate_from_dataframe(self) -> None:
        """Test calibration from pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        # Create sample DataFrame
        data = {
            "timestamp": [1000 + i * 1000 for i in range(100)],
            "price": [100.0 + i * 0.01 for i in range(100)],
            "qty": [100.0 + i for i in range(100)],
            "side": [1 if i % 2 == 0 else -1 for i in range(100)],
        }
        df = pd.DataFrame(data)

        config = calibrate_from_dataframe(
            df,
            symbol="TEST",
            asset_class="equity",
            price_col="price",
            qty_col="qty",
            side_col="side",
            timestamp_col="timestamp",
        )

        assert isinstance(config, L3ExecutionConfig)


# ==============================================================================
# Test Data Loading
# ==============================================================================


class TestDataLoading:
    """Tests for loading calibration data from files."""

    def test_load_from_json_file(self, tmp_path: Path, sample_trades: List[Dict]) -> None:
        """Test loading data from JSON file."""
        # Create calibration data file
        data = {
            "trades": sample_trades,
            "orders": [],
            "latencies": [],
            "market_params": {
                "avg_adv": 10_000_000,
                "avg_volatility": 0.02,
            },
        }

        file_path = tmp_path / "calibration_data.json"
        file_path.write_text(json.dumps(data))

        pipeline = L3CalibrationPipeline()
        pipeline._load_calibration_data(str(file_path))

        assert pipeline.n_trades == len(sample_trades)

    def test_load_missing_file(self) -> None:
        """Test handling missing calibration file."""
        pipeline = L3CalibrationPipeline()
        pipeline._load_calibration_data("/nonexistent/path/data.json")

        # Should not raise, just log warning
        assert pipeline.n_trades == 0


# ==============================================================================
# Test Config Building
# ==============================================================================


class TestConfigBuilding:
    """Tests for building L3ExecutionConfig from calibration results."""

    def test_build_config_with_all_results(
        self, sample_trades: List[Dict], sample_latencies: List[Dict]
    ) -> None:
        """Test building config with all calibration results."""
        pipeline = L3CalibrationPipeline(asset_class="equity")

        for trade in sample_trades:
            pipeline.add_trade(**trade)

        for lat in sample_latencies:
            pipeline.add_latency_observation(
                timestamp_ns=lat["timestamp_ns"],
                latency_type=lat["type"],
                latency_us=lat["latency_us"],
            )

        config = pipeline.run_full_calibration()

        # Check impact config
        assert config.market_impact.enabled
        assert config.market_impact.model == ImpactModelType.ALMGREN_CHRISS

        # Check fill probability config
        assert config.fill_probability.enabled
        assert config.fill_probability.model == FillProbabilityModelType.QUEUE_REACTIVE

        # Check latency config (enabled since we have latency data)
        assert config.latency.enabled

    def test_build_config_minimal_data(self) -> None:
        """Test building config with minimal data."""
        pipeline = L3CalibrationPipeline()

        # Add just one trade
        pipeline.add_trade(
            timestamp_ms=1705315800000,
            price=100.0,
            qty=100.0,
            side=1,
        )

        config = pipeline.run_full_calibration()

        # Should still produce valid config with defaults
        assert isinstance(config, L3ExecutionConfig)


# ==============================================================================
# Test Quality Assessment
# ==============================================================================


class TestQualityAssessment:
    """Tests for calibration quality assessment."""

    def test_low_quality(self) -> None:
        """Test low quality assessment with few samples."""
        pipeline = L3CalibrationPipeline()

        for i in range(10):
            pipeline.add_trade(
                timestamp_ms=1000 + i * 1000,
                price=100.0,
                qty=100.0,
                side=1,
            )

        result = pipeline.get_calibration_result()
        assert result.calibration_quality == "low"

    def test_medium_quality(self) -> None:
        """Test medium quality assessment."""
        pipeline = L3CalibrationPipeline()

        for i in range(150):
            pipeline.add_trade(
                timestamp_ms=1000 + i * 1000,
                price=100.0,
                qty=100.0,
                side=1,
            )

        result = pipeline.get_calibration_result()
        assert result.calibration_quality == "medium"

    def test_high_quality(self) -> None:
        """Test high quality assessment with many samples."""
        pipeline = L3CalibrationPipeline()

        for i in range(1500):
            pipeline.add_trade(
                timestamp_ms=1000 + i * 1000,
                price=100.0,
                qty=100.0,
                side=1,
            )

        for i in range(600):
            pipeline.add_order(
                timestamp_ns=1000000000 + i * 1000000000,
                price=100.0,
                qty=100.0,
                side=Side.BUY,
                event_type="ADD",
            )

        result = pipeline.get_calibration_result()
        assert result.calibration_quality == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
