# Futures API Reference

## Core Models

### `core_futures.py`

#### FuturesType

```python
from core_futures import FuturesType

class FuturesType(Enum):
    CRYPTO_PERPETUAL = "crypto_perpetual"
    CRYPTO_QUARTERLY = "crypto_quarterly"
    CME_EQUITY_INDEX = "cme_equity_index"
    CME_METAL = "cme_metal"
    CME_ENERGY = "cme_energy"
    CME_CURRENCY = "cme_currency"
    CME_BOND = "cme_bond"
    CME_AGRICULTURAL = "cme_agricultural"
```

#### MarginMode

```python
from core_futures import MarginMode

class MarginMode(Enum):
    CROSS = "cross"       # Shared margin across positions
    ISOLATED = "isolated" # Margin isolated per position
    SPAN = "span"         # CME SPAN margin
```

#### PositionSide

```python
from core_futures import PositionSide

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
```

#### FuturesContractSpec

```python
from core_futures import FuturesContractSpec
from decimal import Decimal

@dataclass
class FuturesContractSpec:
    symbol: str
    futures_type: FuturesType
    base_asset: str
    quote_asset: str
    contract_multiplier: Decimal
    tick_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    initial_margin_rate: Decimal
    maintenance_margin_rate: Decimal
    max_leverage: int = 125
    funding_interval_hours: int = 8  # Crypto perpetual only
    expiration_date: Optional[date] = None  # Quarterly/CME only
    settlement_time_utc: Optional[time] = None  # CME only
```

#### FuturesPosition

```python
from core_futures import FuturesPosition
from decimal import Decimal

@dataclass
class FuturesPosition:
    symbol: str
    qty: Decimal
    entry_price: Decimal
    side: PositionSide
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    liquidation_price: Optional[Decimal] = None
    margin_used: Optional[Decimal] = None
    timestamp_ms: int = 0

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL."""
        ...

    def get_notional(self, price: Optional[Decimal] = None) -> Decimal:
        """Get position notional value."""
        ...
```

---

## Margin Calculation

### TieredMarginCalculator (`impl_futures_margin.py`)

Binance-style tiered margin calculation.

```python
from impl_futures_margin import TieredMarginCalculator
from core_futures import FuturesPosition
from decimal import Decimal

calculator = TieredMarginCalculator()

# Calculate initial margin
initial_margin = calculator.calculate_initial_margin(
    position=position,
    current_price=Decimal("50000"),
)

# Calculate maintenance margin
maint_margin = calculator.calculate_maintenance_margin(
    position=position,
    current_price=Decimal("50000"),
)

# Calculate liquidation price
liq_price = calculator.calculate_liquidation_price(
    position=position,
    wallet_balance=Decimal("2500"),
)
```

#### Leverage Brackets (BTC)

| Notional (USD) | Max Leverage | Maint. Margin Rate |
|----------------|--------------|-------------------|
| < $50,000 | 125x | 0.4% |
| $50K - $250K | 100x | 0.5% |
| $250K - $1M | 50x | 1.0% |
| $1M - $5M | 20x | 2.5% |
| $5M - $20M | 10x | 5.0% |
| > $20M | 5x | 10.0% |

### SPANMarginCalculator (`impl_span_margin.py`)

CME SPAN margin calculation.

```python
from impl_span_margin import SPANMarginCalculator
from core_futures import FuturesPosition
from decimal import Decimal

calculator = SPANMarginCalculator()

# Calculate margin for single position
result = calculator.calculate_margin(
    position=position,
    current_price=Decimal("4500"),
)
print(f"Scanning Risk: ${result.scanning_risk}")
print(f"Initial Margin: ${result.initial_margin}")
print(f"Maintenance Margin: ${result.maintenance_margin}")

# Calculate portfolio margin with spread credits
portfolio_result = calculator.calculate_portfolio_margin(
    positions=[es_position, nq_position],
    prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
)
print(f"Inter-commodity Credit: ${portfolio_result.inter_commodity_credit}")
print(f"Net Portfolio Margin: ${portfolio_result.net_portfolio_margin}")
```

---

## Execution Providers

### L2 Crypto Futures Execution

```python
from execution_providers_futures import (
    FuturesSlippageProvider,
    FuturesSlippageConfig,
    FuturesFeeProvider,
    FuturesL2ExecutionProvider,
    create_futures_execution_provider,
)
from execution_providers import Order, MarketState, BarData

# Create provider
provider = create_futures_execution_provider(
    use_mark_price=True,
)

# Execute order
fill = provider.execute(
    order=Order("BTCUSDT", "BUY", 0.1, "MARKET"),
    market=MarketState(timestamp=0, bid=50000.0, ask=50001.0, adv=1e9),
    bar=BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=10000.0),
    funding_rate=0.0001,
    open_interest=2e9,
    recent_liquidations=1e7,
)

print(f"Fill price: {fill.price}")
print(f"Slippage: {fill.slippage_bps} bps")
print(f"Fee: ${fill.fee}")
```

### L2 CME Futures Execution

```python
from execution_providers_cme import (
    CMESlippageProvider,
    CMEFeeProvider,
    create_cme_execution_provider,
)

# Create from profile
provider = create_cme_execution_provider(profile="equity_index")

# Execute order
fill = provider.execute(
    order=Order("ES", "BUY", 5.0, "MARKET"),
    market=MarketState(timestamp=0, bid=4500.0, ask=4500.25, adv=2e9),
    bar=BarData(open=4500.0, high=4510.0, low=4490.0, close=4505.0, volume=100000.0),
    is_eth_session=False,
    is_settlement_period=False,
)
```

### L3 Crypto Futures Execution

```python
from execution_providers_futures_l3 import (
    FuturesL3ExecutionProvider,
    FuturesL3Config,
    create_futures_l3_execution_provider,
)

# Create from preset
provider = create_futures_l3_execution_provider(preset="default")

# Execute with full LOB simulation
fill = provider.execute(
    order=order,
    market=market,
    bar=bar,
    order_book=lob_order_book,
    matching_engine=matching_engine,
    funding_rate=0.0001,
    open_interest=2e9,
    recent_liquidations=1e7,
    positions=current_positions,
)
```

### L3 CME Futures Execution

```python
from execution_providers_cme_l3 import (
    CMEL3ExecutionProvider,
    create_cme_l3_execution_provider,
)

# Create provider
provider = create_cme_l3_execution_provider(symbol="ES", profile="default")

# Execute with Globex matching
fill = provider.execute(
    order=order,
    market=market,
    bar=bar,
    timestamp_ms=current_time_ms,
)
```

---

## Risk Guards

### Crypto Futures Risk Guards

```python
from services.futures_risk_guards import (
    FuturesLeverageGuard,
    FuturesMarginGuard,
    FundingExposureGuard,
    ConcentrationGuard,
    ADLRiskGuard,
)

# Leverage guard
leverage_guard = FuturesLeverageGuard(
    max_account_leverage=20,
    max_symbol_leverage=125,
    concentration_limit=0.5,
)
result = leverage_guard.validate_new_position(
    proposed_position=position,
    current_positions=existing_positions,
    account_balance=Decimal("10000"),
)

# Margin guard
margin_guard = FuturesMarginGuard(
    warning_level=Decimal("1.5"),
    danger_level=Decimal("1.2"),
    critical_level=Decimal("1.05"),
)
result = margin_guard.check_margin_ratio(
    margin_ratio=1.35,
    account_equity=10000.0,
    total_margin_used=7407.0,
)

# Funding exposure guard
funding_guard = FundingExposureGuard(
    warning_threshold=Decimal("0.0001"),  # 0.01% per 8h
)
result = funding_guard.check_funding_exposure(
    funding_rate=Decimal("0.0005"),
    position_side="LONG",
    position_notional=Decimal("100000"),
)

# ADL risk guard
adl_guard = ADLRiskGuard(
    warning_percentile=75.0,
    critical_percentile=90.0,
)
result = adl_guard.check_adl_risk(
    position_pnl_percentile=85.0,
    position_leverage_percentile=80.0,
)
```

### CME Futures Risk Guards

```python
from services.cme_risk_guards import (
    SPANMarginGuard,
    CMEPositionLimitGuard,
    CircuitBreakerAwareGuard,
    SettlementRiskGuard,
    RolloverGuard,
    CMEFuturesRiskGuard,
)

# Unified CME risk guard
guard = CMEFuturesRiskGuard(strict_mode=True)
guard.add_symbol_to_monitor("ES", reference_price=Decimal("4500"))

event = guard.check_trade(
    symbol="ES",
    side="LONG",
    quantity=5,
    account_equity=Decimal("500000"),
    positions=current_positions,
    prices={"ES": Decimal("4500")},
    timestamp_ms=int(time.time() * 1000),
)
```

### Unified Risk Guard

```python
from services.unified_futures_risk import (
    UnifiedFuturesRiskGuard,
    UnifiedRiskConfig,
    create_unified_risk_guard,
)

# Create from config
config = UnifiedRiskConfig.from_yaml("configs/unified_futures_risk.yaml")
guard = create_unified_risk_guard(config)

# Auto-delegates based on symbol
event = guard.check_trade(
    symbol="BTCUSDT",  # Crypto → crypto guards
    side="BUY",
    quantity=0.1,
    leverage=10,
    account_equity=Decimal("10000"),
    mark_price=Decimal("50000"),
)

event = guard.check_trade(
    symbol="ES",  # CME → CME guards
    side="BUY",
    quantity=5,
    account_equity=Decimal("500000"),
)
```

---

## LOB Extensions

### Liquidation Cascade Simulator

```python
from lob.futures_extensions import (
    LiquidationCascadeSimulator,
    create_cascade_simulator,
)

simulator = create_cascade_simulator(
    price_impact_coef=0.5,
    cascade_decay=0.7,
    max_waves=5,
)

result = simulator.simulate_cascade(
    initial_liquidation_volume=1_000_000,
    market_price=50000.0,
    adv=500_000_000,
)

print(f"Total waves: {len(result.waves)}")
print(f"Total liquidated: ${result.total_liquidated_volume:,.0f}")
print(f"Price impact: {result.total_price_impact_bps:.2f} bps")
```

### Insurance Fund Manager

```python
from lob.futures_extensions import (
    InsuranceFundManager,
    create_insurance_fund,
)

fund = create_insurance_fund(initial_balance=10_000_000)

result = fund.process_liquidation(
    liquidation_info=liq_order,
    fill_price=49500.0,
)

print(f"Contribution: ${result.contribution}")
print(f"Payout: ${result.payout}")
print(f"Balance: ${fund.get_state().current_balance}")
```

### CME Globex Matching Engine

```python
from lob.cme_matching import GlobexMatchingEngine, StopOrder
from lob.data_structures import LimitOrder, Side

engine = GlobexMatchingEngine(
    symbol="ES",
    tick_size=0.25,
    protection_points=6,
)

# Add resting order
engine.add_resting_order(LimitOrder(
    order_id="rest_1",
    price=4500.0,
    qty=10.0,
    remaining_qty=10.0,
    timestamp_ns=0,
    side=Side.BUY,
))

# Match aggressive order
result = engine.match(aggressive_order)

# Match with protection (MWP)
result = engine.match_with_protection(market_order, protection_points=6)

# Submit stop order
stop = StopOrder(
    order_id="stop_1",
    symbol="ES",
    side=Side.SELL,
    qty=5.0,
    stop_price=4490.0,
)
engine.submit_stop_order(stop)

# Check stop triggers
results = engine.check_stop_triggers(
    last_trade_price=4489.0,
    bid=4488.5,
    ask=4489.5,
    timestamp_ns=int(time.time() * 1e9),
)
```

---

## Live Trading Components

### Futures Live Runner

```python
from services.futures_live_runner import (
    FuturesLiveRunner,
    FuturesLiveConfig,
    create_futures_live_runner,
)

# Load from YAML
config = FuturesLiveConfig.from_yaml("configs/config_live_futures.yaml")

# Create runner
runner = create_futures_live_runner(config)

# Start live trading
await runner.start()
```

### Position Synchronizer

```python
from services.futures_position_sync import (
    FuturesPositionSynchronizer,
    FuturesSyncConfig,
)

config = FuturesSyncConfig(
    exchange=Exchange.BINANCE,
    futures_type=FuturesType.CRYPTO_PERPETUAL,
    sync_interval_sec=10.0,
    qty_tolerance_pct=0.001,
    auto_reconcile=False,
)

sync = FuturesPositionSynchronizer(
    position_provider=position_provider,
    account_provider=account_provider,
    local_state_getter=get_local_positions,
    config=config,
    on_event=handle_sync_event,
)

await sync.start_async()
```

### Margin Monitor

```python
from services.futures_margin_monitor import (
    FuturesMarginMonitor,
    MarginMonitorConfig,
)

config = MarginMonitorConfig(
    check_interval_sec=5.0,
    warning_ratio=1.5,
    danger_ratio=1.2,
    critical_ratio=1.05,
)

monitor = FuturesMarginMonitor(
    account_provider=account_provider,
    position_provider=position_provider,
    config=config,
    on_status_change=handle_margin_alert,
)

status = await monitor.check_margin()
```

### Funding Tracker

```python
from services.futures_funding_tracker import (
    FuturesFundingTracker,
    FundingTrackerConfig,
)

config = FundingTrackerConfig(
    data_dir="data/futures",
    prediction_method="ewma",
    cache_ttl_sec=300,
)

tracker = FuturesFundingTracker(
    funding_provider=funding_provider,
    config=config,
)

info = await tracker.get_funding_info("BTCUSDT")
print(f"Current rate: {info.funding_rate:.4%}")
print(f"Predicted: {info.predicted_rate:.4%}")
```

---

## Feature Flags

```python
from services.futures_feature_flags import (
    FuturesFeatureFlags,
    RolloutStage,
)

flags = FuturesFeatureFlags.from_yaml("configs/feature_flags_futures.yaml")

# Check feature status
if flags.is_enabled("futures_l3_execution"):
    provider = create_futures_l3_execution_provider()
else:
    provider = create_futures_execution_provider()

# Get rollout stage
stage = flags.get_stage("unified_risk_guard")
if stage == RolloutStage.PRODUCTION:
    # Full production mode
    pass
elif stage == RolloutStage.CANARY:
    # Canary mode - limited symbols
    allowed = flags.get_canary_symbols("unified_risk_guard")
    if symbol in allowed:
        pass
```

---

## Training Integration

### Futures Environment Wrapper

```python
from wrappers.futures_env import FuturesTradingEnvWrapper

env = FuturesTradingEnvWrapper(
    base_env=trading_env,
    futures_type=FuturesType.CRYPTO_PERPETUAL,
    max_leverage=10,
    margin_mode=MarginMode.ISOLATED,
    initial_margin=10000.0,
)

obs = env.reset()
action = agent.predict(obs)
obs, reward, terminated, truncated, info = env.step(action)

# Info contains futures-specific data
print(f"Margin ratio: {info['margin_ratio']}")
print(f"Leverage: {info['current_leverage']}")
print(f"Unrealized PnL: {info['unrealized_pnl']}")
```
