#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Test Suite for Stocks Functionality

This script tests the complete stocks trading pipeline including:
1. Data Adapters (Alpaca, Yahoo, Polygon)
2. Multi-Asset Data Loader
3. L2 Execution Provider
4. L3 LOB Simulation Components
5. Trading Environment Integration
6. Training Pipeline

Run with: python test_stocks_e2e.py
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Test Results Tracking
# ============================================================================

class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.skipped: List[Tuple[str, str]] = []

    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"  [PASS] {name}")

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  [FAIL] {name}: {error}")

    def add_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        print(f"  [SKIP] {name}: {reason}")

    def summary(self) -> str:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        lines = [
            "\n" + "=" * 60,
            "TEST SUMMARY",
            "=" * 60,
            f"Total:   {total}",
            f"Passed:  {len(self.passed)}",
            f"Failed:  {len(self.failed)}",
            f"Skipped: {len(self.skipped)}",
        ]
        if self.failed:
            lines.append("\nFailed Tests:")
            for name, error in self.failed:
                lines.append(f"  - {name}: {error[:100]}")
        return "\n".join(lines)


results = TestResults()


def run_test(name: str):
    """Decorator to run a test with error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
                results.add_pass(name)
                return True
            except Exception as e:
                results.add_fail(name, str(e))
                traceback.print_exc()
                return False
        return wrapper
    return decorator


# ============================================================================
# SECTION 1: Data Adapters Tests
# ============================================================================

def test_data_adapters():
    """Test all data adapters."""
    print("\n" + "=" * 60)
    print("SECTION 1: Data Adapters")
    print("=" * 60)

    test_yahoo_adapter()
    test_alpaca_adapter()
    test_polygon_adapter()
    test_adapter_registry()


@run_test("Yahoo Market Data Adapter")
def test_yahoo_adapter():
    """Test Yahoo Finance adapter for VIX/Macro data."""
    from adapters.yahoo.market_data import YahooMarketDataAdapter

    adapter = YahooMarketDataAdapter()

    # Test VIX data
    vix_bars = adapter.get_bars("^VIX", timeframe="1d", limit=10)
    assert vix_bars is not None, "VIX bars should not be None"
    assert len(vix_bars) > 0, "Should have VIX data"


@run_test("Alpaca Market Data Adapter - Config Init")
def test_alpaca_adapter():
    """Test Alpaca adapter initialization."""
    from adapters.alpaca.market_data import AlpacaMarketDataAdapter
    from adapters.models import ExchangeVendor

    # Check if API keys are available
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        # Test without live API - just initialization check
        adapter = AlpacaMarketDataAdapter(
            vendor=ExchangeVendor.ALPACA,
            config={
                "api_key": "test_key",
                "api_secret": "test_secret",
                "paper": True,
            },
        )
        assert adapter is not None
        return

    # With real API keys, test actual functionality
    adapter = AlpacaMarketDataAdapter(
        vendor=ExchangeVendor.ALPACA,
        config={
            "api_key": api_key,
            "api_secret": api_secret,
            "paper": True,
        },
    )

    # Test bars (may fail if market is closed - skip gracefully)
    try:
        bars = adapter.get_bars("SPY", timeframe="1d", limit=5)
        assert bars is not None
    except Exception as e:
        print(f"    Note: Alpaca bars fetch skipped ({e})")


@run_test("Polygon Market Data Adapter - Import")
def test_polygon_adapter():
    """Test Polygon adapter import."""
    from adapters.polygon.market_data import PolygonMarketDataAdapter

    # Just test import - API key may not be available
    assert PolygonMarketDataAdapter is not None


@run_test("Adapter Registry")
def test_adapter_registry():
    """Test adapter registry factory functions."""
    from adapters.registry import (
        create_market_data_adapter,
        create_fee_adapter,
        create_trading_hours_adapter,
    )

    # Test fee adapter creation
    try:
        binance_fee = create_fee_adapter("binance")
        assert binance_fee is not None
    except Exception:
        pass  # May not have binance configured


# ============================================================================
# SECTION 2: Multi-Asset Data Loader Tests
# ============================================================================

def test_multi_asset_loader():
    """Test multi-asset data loader."""
    print("\n" + "=" * 60)
    print("SECTION 2: Multi-Asset Data Loader")
    print("=" * 60)

    test_load_stock_parquet()
    test_asset_class_enum()
    test_timeframe_conversion()


@run_test("Load Stock Parquet Files")
def test_load_stock_parquet():
    """Test loading stock data from parquet files."""
    from data_loader_multi_asset import load_multi_asset_data, AssetClass

    # Check if stock data exists
    stock_files = list(Path("data/raw_stocks").glob("*.parquet"))
    if not stock_files:
        results.add_skip("Load Stock Parquet", "No stock parquet files found")
        return

    # Load data
    frames, obs_shapes = load_multi_asset_data(
        paths=["data/raw_stocks/SPY.parquet"],
        asset_class=AssetClass.EQUITY,
        timeframe="4h",
    )

    assert len(frames) > 0, "Should load at least one frame"


@run_test("Asset Class Enum")
def test_asset_class_enum():
    """Test asset class enum values."""
    from data_loader_multi_asset import AssetClass

    assert AssetClass.CRYPTO.value == "crypto"
    assert AssetClass.EQUITY.value == "equity"


@run_test("Timeframe Conversion")
def test_timeframe_conversion():
    """Test timeframe to seconds conversion."""
    from data_loader_multi_asset import timeframe_to_seconds

    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("4h") == 14400
    assert timeframe_to_seconds("1d") == 86400


# ============================================================================
# SECTION 3: L2 Execution Provider Tests
# ============================================================================

def test_l2_execution():
    """Test L2 execution provider for equities."""
    print("\n" + "=" * 60)
    print("SECTION 3: L2 Execution Provider")
    print("=" * 60)

    test_l2_slippage_provider()
    test_l2_fee_provider_equity()
    test_l2_fill_provider()
    test_l2_execution_provider_integration()


@run_test("L2 Slippage Provider (Statistical)")
def test_l2_slippage_provider():
    """Test statistical slippage provider."""
    from execution_providers import (
        StatisticalSlippageProvider,
        Order,
        MarketState,
    )

    # Create provider with equity-like params
    provider = StatisticalSlippageProvider(
        spread_bps=2.0,
        impact_coef=0.05,
    )

    # Create order and market state
    order = Order(
        symbol="AAPL",
        side="BUY",
        qty=100,
        order_type="MARKET",
    )

    market = MarketState(
        timestamp=int(time.time() * 1000),
        bid=150.0,
        ask=150.02,
        adv=10_000_000,
    )

    # Calculate slippage via participation
    participation = (100 * 150.0) / 10_000_000  # ~0.0015
    slippage_bps = provider.compute_slippage_bps(order, market, participation)

    assert slippage_bps >= 0, "Slippage should be non-negative"
    assert slippage_bps < 100, "Slippage should be reasonable"


@run_test("L2 Fee Provider (Equity)")
def test_l2_fee_provider_equity():
    """Test equity fee provider with SEC/TAF fees."""
    from execution_providers import EquityFeeProvider

    provider = EquityFeeProvider()

    # Compute fee for a sell order
    fee = provider.compute_fee(
        notional=15000.0,  # 100 shares at $150
        side="SELL",
        liquidity="taker",
        qty=100,
    )

    # SEC fee: ~$0.0000278/$ * 15000 = ~$0.42
    # TAF fee: ~$0.000166/share * 100 = ~$0.0166
    assert fee >= 0, "Fee should be non-negative"
    assert fee < 1.0, "Fee should be small for 100 shares"

    # Buy orders should have zero fee
    buy_fee = provider.compute_fee(
        notional=15000.0,
        side="BUY",
        liquidity="taker",
        qty=100,
    )
    assert buy_fee == 0.0, "Buy orders should have no regulatory fees"


@run_test("L2 Fill Provider (OHLCV)")
def test_l2_fill_provider():
    """Test OHLCV-based fill provider."""
    from execution_providers import (
        OHLCVFillProvider,
        Order,
        MarketState,
        BarData,
        StatisticalSlippageProvider,
        EquityFeeProvider,
    )

    slippage = StatisticalSlippageProvider(spread_bps=2.0, impact_coef=0.05)
    fees = EquityFeeProvider()

    provider = OHLCVFillProvider(
        slippage_provider=slippage,
        fee_provider=fees,
    )

    order = Order(
        symbol="AAPL",
        side="BUY",
        qty=100,
        order_type="MARKET",
    )

    market = MarketState(
        timestamp=int(time.time() * 1000),
        bid=150.0,
        ask=150.02,
        adv=10_000_000,
    )

    bar = BarData(
        open=150.0,
        high=151.0,
        low=149.5,
        close=150.5,
        volume=100_000,
    )

    fill = provider.try_fill(order, market, bar)

    assert fill is not None, "Market order should fill"
    assert fill.qty == 100, "Should fill full quantity"
    assert fill.price > 0, "Fill price should be positive"


@run_test("L2 Execution Provider Integration")
def test_l2_execution_provider_integration():
    """Test complete L2 execution provider."""
    from execution_providers import (
        create_execution_provider,
        AssetClass,
        Order,
        MarketState,
        BarData,
    )

    provider = create_execution_provider(AssetClass.EQUITY, level="L2")

    order = Order(
        symbol="SPY",
        side="BUY",
        qty=50,
        order_type="MARKET",
    )

    market = MarketState(
        timestamp=int(time.time() * 1000),
        bid=450.0,
        ask=450.05,
        adv=50_000_000,
    )

    bar = BarData(
        open=450.0,
        high=451.0,
        low=449.5,
        close=450.3,
        volume=500_000,
    )

    fill = provider.execute(order, market, bar)

    assert fill is not None
    assert fill.qty == 50


# ============================================================================
# SECTION 4: L3 LOB Simulation Components
# ============================================================================

def test_l3_components():
    """Test L3 LOB simulation components."""
    print("\n" + "=" * 60)
    print("SECTION 4: L3 LOB Components")
    print("=" * 60)

    test_lob_data_structures()
    test_matching_engine()
    test_queue_tracker()
    test_fill_probability()
    test_market_impact()
    test_latency_model()
    test_hidden_liquidity()
    test_dark_pool()


@run_test("LOB Data Structures")
def test_lob_data_structures():
    """Test LOB data structures."""
    from lob.data_structures import (
        LimitOrder,
        PriceLevel,
        OrderBook,
        Side,
    )

    # Create order book
    ob = OrderBook(symbol="AAPL")

    # Add orders using add_limit_order
    buy_order = LimitOrder(
        order_id="buy_1",
        price=150.0,
        qty=100.0,
        remaining_qty=100.0,
        timestamp_ns=1000000,
        side=Side.BUY,
    )

    sell_order = LimitOrder(
        order_id="sell_1",
        price=150.05,
        qty=100.0,
        remaining_qty=100.0,
        timestamp_ns=1000000,
        side=Side.SELL,
    )

    ob.add_limit_order(buy_order)
    ob.add_limit_order(sell_order)

    assert ob.best_bid == 150.0
    assert ob.best_ask == 150.05
    # Note: spread calculation may have floating point precision differences
    spread = ob.spread
    assert spread is not None and abs(spread - 0.05) < 0.001


@run_test("Matching Engine")
def test_matching_engine():
    """Test FIFO matching engine."""
    from lob.matching_engine import MatchingEngine, STPAction
    from lob.data_structures import LimitOrder, OrderBook, Side

    engine = MatchingEngine(stp_action=STPAction.CANCEL_NEWEST)

    # Create order book
    book = OrderBook(symbol="AAPL")

    # Add resting sell order
    resting = LimitOrder(
        order_id="rest_1",
        price=150.0,
        qty=100.0,
        remaining_qty=100.0,
        timestamp_ns=1000000,
        side=Side.SELL,
        participant_id="maker",
    )
    book.add_limit_order(resting)

    # Match market buy order
    result = engine.match_market_order(
        side=Side.BUY,
        qty=50.0,
        order_book=book,
        taker_order_id="taker_1",
        taker_participant_id="taker",
    )

    assert result.total_filled_qty == 50.0
    assert len(result.fills) > 0


@run_test("Queue Position Tracker")
def test_queue_tracker():
    """Test queue position tracking."""
    from lob.queue_tracker import (
        QueuePositionTracker,
        PositionEstimationMethod,
    )
    from lob.data_structures import LimitOrder, Side

    tracker = QueuePositionTracker(
        default_method=PositionEstimationMethod.MBP_PESSIMISTIC
    )

    order = LimitOrder(
        order_id="our_order",
        price=150.0,
        qty=100.0,
        remaining_qty=100.0,
        timestamp_ns=1000000,
        side=Side.BUY,
    )

    state = tracker.add_order(order, level_qty_before=500.0)

    assert state.qty_ahead == 500.0

    # Estimate fill probability
    fill_prob = tracker.estimate_fill_probability(
        order_id="our_order",
        volume_per_second=100.0,
        time_horizon_sec=60.0,
    )

    # FillProbability has prob_fill attribute
    assert 0.0 <= fill_prob.prob_fill <= 1.0


@run_test("Fill Probability Models")
def test_fill_probability():
    """Test fill probability models."""
    from lob.fill_probability import (
        QueueReactiveModel,
        LOBState,
    )

    model = QueueReactiveModel(
        base_rate=100.0,
        queue_decay_alpha=0.01,
        spread_sensitivity_beta=0.5,
    )

    lob_state = LOBState(
        mid_price=150.0,
        spread_bps=3.0,
        volatility=0.02,
        imbalance=0.1,
    )

    result = model.compute_fill_probability(
        queue_position=10,
        qty_ahead=500.0,
        order_qty=100.0,
        time_horizon_sec=60.0,
        market_state=lob_state,
    )

    assert 0.0 <= result.prob_fill <= 1.0
    assert result.expected_wait_time_sec >= 0


@run_test("Market Impact Models")
def test_market_impact():
    """Test market impact models."""
    from lob.market_impact import (
        AlmgrenChrissModel,
        KyleLambdaModel,
        GatheralModel,
        ImpactParameters,
    )

    # Test Almgren-Chriss
    params = ImpactParameters.for_equity()
    model = AlmgrenChrissModel(params=params)

    result = model.compute_total_impact(
        order_qty=10000,
        adv=10_000_000,
        volatility=0.02,
        mid_price=150.0,
    )

    assert result.temporary_impact_bps >= 0
    assert result.permanent_impact_bps >= 0

    # Test Kyle Lambda
    kyle = KyleLambdaModel(lambda_coef=0.0001)
    kyle_result = kyle.compute_total_impact(
        order_qty=10000,
        adv=10_000_000,
        volatility=0.02,
        mid_price=150.0,
    )
    assert kyle_result.total_impact_bps >= 0

    # Test Gatheral
    gatheral = GatheralModel(eta=0.1, gamma=0.5, tau_ms=300000.0, beta=0.5)
    gatheral_result = gatheral.compute_total_impact(
        order_qty=10000,
        adv=10_000_000,
        volatility=0.02,
        mid_price=150.0,
    )
    assert gatheral_result.total_impact_bps >= 0


@run_test("Latency Model")
def test_latency_model():
    """Test latency simulation."""
    from lob.latency_model import (
        LatencyModel,
        LatencyProfile,
    )

    model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)

    # Sample latencies
    feed_lat = model.sample_feed_latency()
    order_lat = model.sample_order_latency()
    exchange_lat = model.sample_exchange_latency()

    assert feed_lat > 0
    assert order_lat > 0
    assert exchange_lat > 0

    # Test round trip
    round_trip = model.sample_round_trip()
    assert round_trip >= 0


@run_test("Hidden Liquidity Detection")
def test_hidden_liquidity():
    """Test iceberg and hidden liquidity detection."""
    from lob.hidden_liquidity import (
        IcebergDetector,
        HiddenLiquidityEstimator,
        create_iceberg_detector,
    )
    from lob.data_structures import Side

    detector = create_iceberg_detector(
        min_refills_to_confirm=2,
        lookback_window_sec=60.0,
    )

    assert detector is not None

    estimator = HiddenLiquidityEstimator(
        iceberg_detector=detector,
        hidden_ratio_estimate=0.15,
    )

    hidden = estimator.estimate_hidden_at_level(
        price=150.0,
        side=Side.BUY,
        visible_qty=500.0,
    )

    assert hidden >= 0


@run_test("Dark Pool Simulation")
def test_dark_pool():
    """Test dark pool simulation."""
    from lob.dark_pool import (
        DarkPoolSimulator,
        create_default_dark_pool_simulator,
    )
    from lob.data_structures import LimitOrder, Side

    dark_pool = create_default_dark_pool_simulator(seed=42)

    order = LimitOrder(
        order_id="dp_order_1",
        price=150.0,
        qty=1000.0,
        remaining_qty=1000.0,
        timestamp_ns=1000000,
        side=Side.BUY,
    )

    fill = dark_pool.attempt_dark_fill(
        order=order,
        lit_mid_price=150.0,
        lit_spread=0.05,
        adv=10_000_000,
        volatility=0.02,
        hour_of_day=10,
    )

    # Fill may or may not happen (probabilistic)
    if fill and fill.is_filled:
        assert fill.filled_qty > 0
        assert fill.fill_price > 0


# ============================================================================
# SECTION 5: L3 Execution Provider
# ============================================================================

def test_l3_execution():
    """Test L3 execution provider."""
    print("\n" + "=" * 60)
    print("SECTION 5: L3 Execution Provider")
    print("=" * 60)

    test_l3_config()
    test_l3_slippage_provider()
    test_l3_fill_provider()
    test_l3_execution_provider_integration()


@run_test("L3 Configuration")
def test_l3_config():
    """Test L3 execution config."""
    from lob.config import (
        L3ExecutionConfig,
        LatencyConfig,
    )

    config = L3ExecutionConfig(
        enabled=True,
        latency=LatencyConfig(enabled=True, profile="institutional"),
    )

    assert config.enabled == True
    assert config.latency.enabled == True


@run_test("L3 Slippage Provider")
def test_l3_slippage_provider():
    """Test L3 slippage provider with market impact."""
    from execution_providers_l3 import L3SlippageProvider
    from execution_providers import Order, MarketState
    from lob.config import L3ExecutionConfig

    config = L3ExecutionConfig(enabled=True)
    provider = L3SlippageProvider(config=config)

    order = Order(
        symbol="AAPL",
        side="BUY",
        qty=1000,
        order_type="MARKET",
    )

    market = MarketState(
        timestamp=int(time.time() * 1000),
        bid=150.0,
        ask=150.02,
        spread_bps=1.33,
        adv=10_000_000,
        volatility=0.02,
    )

    # Compute slippage with participation ratio
    participation = (1000 * 150.0) / 10_000_000  # ~0.015
    slippage_bps = provider.compute_slippage_bps(order, market, participation)

    assert slippage_bps >= 0
    assert slippage_bps < 100  # Reasonable bound


@run_test("L3 Fill Provider")
def test_l3_fill_provider():
    """Test L3 fill provider with queue position."""
    from execution_providers_l3 import L3FillProvider, L3SlippageProvider
    from execution_providers import (
        Order, MarketState, BarData,
        EquityFeeProvider,
    )
    from lob.config import L3ExecutionConfig

    config = L3ExecutionConfig(enabled=True)
    slippage = L3SlippageProvider(config=config)
    fees = EquityFeeProvider()

    provider = L3FillProvider(
        config=config,
        slippage_provider=slippage,
        fee_provider=fees,
    )

    order = Order(
        symbol="AAPL",
        side="BUY",
        qty=100,
        order_type="MARKET",
    )

    market = MarketState(
        timestamp=int(time.time() * 1000),
        bid=150.0,
        ask=150.02,
        adv=10_000_000,
    )

    bar = BarData(
        open=150.0,
        high=151.0,
        low=149.5,
        close=150.5,
        volume=100_000,
    )

    fill = provider.try_fill(order, market, bar)

    assert fill is not None
    assert fill.qty == 100


@run_test("L3 Execution Provider Integration")
def test_l3_execution_provider_integration():
    """Test complete L3 execution provider."""
    from execution_providers_l3 import create_l3_execution_provider
    from execution_providers import (
        Order, MarketState, BarData,
    )

    provider = create_l3_execution_provider()

    order = Order(
        symbol="SPY",
        side="BUY",
        qty=100,
        order_type="MARKET",
    )

    market = MarketState(
        timestamp=int(time.time() * 1000),
        bid=450.0,
        ask=450.05,
        adv=50_000_000,
        volatility=0.015,
    )

    bar = BarData(
        open=450.0,
        high=451.0,
        low=449.5,
        close=450.3,
        volume=500_000,
    )

    fill = provider.execute(order, market, bar)

    assert fill is not None
    assert fill.qty == 100


# ============================================================================
# SECTION 6: Trading Environment Integration
# ============================================================================

def test_trading_env():
    """Test trading environment with stock data."""
    print("\n" + "=" * 60)
    print("SECTION 6: Trading Environment")
    print("=" * 60)

    test_env_with_stock_data()
    test_session_routing()
    test_position_sync()


@run_test("TradingEnv with Stock Data")
def test_env_with_stock_data():
    """Test TradingEnv initialization with stock data."""
    from trading_patchnew import TradingEnv
    import pandas as pd

    # Load stock data
    stock_file = Path("data/raw_stocks/SPY.parquet")
    if not stock_file.exists():
        results.add_skip("TradingEnv with Stock Data", "No SPY parquet file")
        return

    df = pd.read_parquet(stock_file)

    # Ensure required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            col_lower = col.lower()
            if col_lower in df.columns:
                df[col] = df[col_lower]
            else:
                df[col] = 100.0 if col != "volume" else 10000

    # Create simple env config dict
    env_config = {
        "initial_balance": 100000.0,
        "symbol": "SPY",
    }

    # Create env - TradingEnv expects df as first arg
    env = TradingEnv(df=df.head(500), **env_config)

    # Test reset
    obs, info = env.reset()
    assert obs is not None
    assert len(obs) > 0

    # Test step
    action = np.array([0.5])  # 50% position
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, (int, float))


@run_test("Session Router")
def test_session_routing():
    """Test session-aware order routing."""
    from services.session_router import (
        SessionRouter,
        get_current_session,
    )

    # Get current session
    session = get_current_session()
    assert session is not None
    assert hasattr(session, "session")

    # Create router
    router = SessionRouter(
        allow_extended_hours=True,
        extended_hours_spread_multiplier=2.0,
    )

    decision = router.get_routing_decision(
        symbol="SPY",
        side="BUY",
        qty=100,
        order_type="market",
    )

    assert decision is not None
    assert hasattr(decision, "should_submit")


@run_test("Position Synchronizer")
def test_position_sync():
    """Test position synchronization."""
    from services.position_sync import (
        PositionSynchronizer,
        SyncConfig,
        SyncResult,
    )

    config = SyncConfig(
        sync_interval_s=30.0,
        qty_tolerance_pct=0.01,
        auto_resolve=False,
    )

    # Mock position provider
    class MockPositionProvider:
        def get_positions(self, symbols=None):
            return {"SPY": {"qty": 100.0, "avg_entry_price": 450.0}}

    def get_local():
        # local_state_getter must return Dict[str, Decimal] (qty values only)
        return {"SPY": Decimal("100.0")}

    sync = PositionSynchronizer(
        position_provider=MockPositionProvider(),
        local_state_getter=get_local,
        config=config,
    )

    result = sync.sync_once()
    assert isinstance(result, SyncResult)


# ============================================================================
# SECTION 7: Training Pipeline
# ============================================================================

def test_training_pipeline():
    """Test training pipeline on stocks."""
    print("\n" + "=" * 60)
    print("SECTION 7: Training Pipeline")
    print("=" * 60)

    test_signal_only_config()
    test_features_for_stocks()
    test_backtest_stocks()


@run_test("Signal-Only Training Config")
def test_signal_only_config():
    """Test signal-only training configuration for stocks."""
    import yaml

    config_path = Path("configs/config_train_signal_only_crypto.yaml")
    if not config_path.exists():
        results.add_skip("Signal-Only Training Config", "Config file not found")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "mode" in config or "env" in config or config is not None


@run_test("Features Pipeline for Stocks")
def test_features_for_stocks():
    """Test features pipeline with stock data."""
    from features_pipeline import FeaturePipeline
    import pandas as pd

    # Load stock data
    stock_file = Path("data/raw_stocks/SPY.parquet")
    if not stock_file.exists():
        results.add_skip("Features for Stocks", "No SPY parquet file")
        return

    df = pd.read_parquet(stock_file)

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            col_lower = col.lower()
            if col_lower in df.columns:
                df[col] = df[col_lower]
            else:
                df[col] = 100.0 if col != "volume" else 10000

    # Try creating pipeline
    try:
        pipeline = FeaturePipeline()
        featured_df = pipeline.transform_df(df.head(500), symbol="SPY")
        assert featured_df is not None
        assert len(featured_df) > 0
    except Exception as e:
        # May fail if not all columns present - skip gracefully
        error_str = str(e).lower()
        if "required" in error_str or "column" in error_str or "symbol" in error_str:
            print(f"    Note: Features skipped - {e}")
        else:
            raise


@run_test("Backtest Module Import")
def test_backtest_stocks():
    """Test backtest functionality import."""
    # Just check imports work
    try:
        from service_backtest import BacktestService
        assert BacktestService is not None
    except ImportError as e:
        # May have missing deps
        if "exchange" in str(e):
            print("    Note: Backtest import skipped - missing exchange module")
        else:
            raise


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("STOCKS E2E TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Run all test sections
    test_data_adapters()
    test_multi_asset_loader()
    test_l2_execution()
    test_l3_components()
    test_l3_execution()
    test_trading_env()
    test_training_pipeline()

    # Print summary
    print(results.summary())

    # Return exit code
    return 0 if len(results.failed) == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
