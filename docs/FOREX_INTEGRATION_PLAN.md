# 📊 FOREX INTEGRATION PLAN - L3 Level Implementation

## Executive Summary

Комплексный план интеграции Forex в TradingBot2 на уровне L3 (95%+ реализма симуляции), полностью параллельно crypto и equity веткам без нарушения существующего функционала.

**Статус**: 📋 Plan Draft v1.0
**Дата**: 2025-11-29
**Estimated LOC**: 8,500-10,000
**Estimated Tests**: 400+
**Timeline**: 10-12 недель (1 инженер)

---

## 🎯 Ключевые требования

### Функциональные
1. **Полный pipeline** от загрузки данных до live trading
2. **L3 уровень симуляции** (95%+ реализм)
3. **Параллельность** с crypto/equity (никаких регрессий)
4. **Гибкая конфигурация** через YAML
5. **100% покрытие тестами** нового функционала

### Архитектурные
1. Использование существующего adapter registry pattern
2. Совместимость с feature pipeline
3. Интеграция с risk management system
4. Поддержка PBT/SA-PPO training

---

## 📐 Forex Market Microstructure Analysis

### Ключевые отличия от Crypto/Equity

| Аспект | Crypto | Equity | **Forex** |
|--------|--------|--------|-----------|
| **Market structure** | Central LOB | Central LOB | **OTC Dealer Network** |
| **Trading hours** | 24/7 | NYSE 9:30-16:00 ET | **Sun 5pm - Fri 4pm ET** |
| **Fees** | Maker/Taker % | $0 + regulatory | **Spread-based (0 commission)** |
| **Order book** | Real LOB | Real LOB | **Dealer quotes (synthetic)** |
| **Liquidity** | Varies by coin | Market cap based | **Session-dependent** |
| **Leverage** | 1x-125x | 1x-4x (margin) | **50:1 - 500:1** |
| **Tick size** | Varies | $0.01 | **Pip (0.0001/0.01 for JPY)** |
| **Settlement** | T+0 | T+2 | **T+2 (spot), T+0 (rolling)** |

### Forex Sessions (Критично для моделирования)

| Session | Время (UTC) | Время (ET) | Liquidity Factor | Major Pairs |
|---------|-------------|------------|------------------|-------------|
| **Sydney** | 21:00-06:00 | 4pm-1am | 0.6-0.7 | AUD, NZD |
| **Tokyo** | 00:00-09:00 | 7pm-4am | 0.7-0.85 | JPY crosses |
| **London** | 07:00-16:00 | 2am-11am | **1.0-1.2** | EUR, GBP |
| **New York** | 12:00-21:00 | 7am-4pm | **1.0-1.15** | USD pairs |
| **London/NY overlap** | 12:00-16:00 | 7am-11am | **1.3-1.5** | ALL MAJORS |

### Currency Pair Classification

| Category | Examples | Typical Spread | Daily Range | ADV (est.) |
|----------|----------|----------------|-------------|------------|
| **Majors** | EUR/USD, USD/JPY | 0.5-1.5 pips | 50-100 pips | $500B+ |
| **Minors** | EUR/GBP, GBP/JPY | 1.5-3 pips | 60-120 pips | $50-100B |
| **Crosses** | EUR/JPY, GBP/CHF | 2-5 pips | 70-150 pips | $20-50B |
| **Exotics** | USD/TRY, USD/ZAR | 10-50 pips | 200-500 pips | $1-10B |

---

## 🏗️ Architecture Design

### Layer Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                        FOREX INTEGRATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   CRYPTO    │   │   EQUITY    │   │   FOREX     │           │
│  │   (Binance) │   │  (Alpaca)   │   │   (OANDA)   │  ← NEW    │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│  ┌──────┴─────────────────┴─────────────────┴──────┐           │
│  │              Adapter Registry (adapters/)        │           │
│  └──────┬─────────────────┬─────────────────┬──────┘           │
│         │                 │                 │                   │
│  ┌──────┴─────────────────┴─────────────────┴──────┐           │
│  │        Execution Providers (L2/L2+/L3)          │           │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │           │
│  │  │ CryptoParam │  │ EquityParam │  │ForexParam│ │  ← NEW    │
│  │  │ Slippage L2+│  │ Slippage L2+│  │Slippage  │ │           │
│  │  └─────────────┘  └─────────────┘  └──────────┘ │           │
│  └──────────────────────────────────────────────────┘           │
│                             │                                   │
│  ┌──────────────────────────┴──────────────────────┐           │
│  │                 LOB Simulation (lob/)            │           │
│  │  ┌─────────────────────────────────────────────┐│           │
│  │  │     ForexDealerSimulator (Synthetic LOB)    ││  ← NEW    │
│  │  │  - Multi-dealer quote aggregation           ││           │
│  │  │  - Last-look rejection simulation           ││           │
│  │  │  - Latency arbitrage protection             ││           │
│  │  └─────────────────────────────────────────────┘│           │
│  └──────────────────────────────────────────────────┘           │
│                             │                                   │
│  ┌──────────────────────────┴──────────────────────┐           │
│  │            Feature Pipeline / Training           │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │           │
│  │  │ FG Index │  │VIX/RS SPY│  │Carry/IR  │      │  ← NEW    │
│  │  │ (crypto) │  │ (equity) │  │ (forex)  │      │           │
│  │  └──────────┘  └──────────┘  └──────────┘      │           │
│  └──────────────────────────────────────────────────┘           │
│                             │                                   │
│  ┌──────────────────────────┴──────────────────────┐           │
│  │              Risk Management System              │           │
│  │  - Position limits                               │           │
│  │  - Drawdown guards                               │           │
│  │  - Leverage monitoring (NEW: Forex margin)       │  ← UPDATE │
│  │  - Kill switch                                   │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### New Files Structure

```
TradingBot2/
├── adapters/
│   └── oanda/                          # NEW: OANDA adapter
│       ├── __init__.py
│       ├── market_data.py              # Historical + streaming data
│       ├── fees.py                     # Spread-based fee model
│       ├── trading_hours.py            # Forex sessions calendar
│       ├── exchange_info.py            # Currency pairs info
│       └── order_execution.py          # Order placement API
│
├── lob/
│   └── forex_dealer.py                 # NEW: Dealer quote simulation
│
├── execution_providers_forex.py        # NEW: ForexParametricSlippageProvider
│
├── forex_features.py                   # NEW: Forex-specific features
│
├── services/
│   └── forex_risk_guards.py            # NEW: Margin/leverage guards
│
├── scripts/
│   └── download_forex_data.py          # NEW: Data downloader
│
├── configs/
│   ├── config_train_forex.yaml         # NEW
│   ├── config_backtest_forex.yaml      # NEW
│   ├── config_live_oanda.yaml          # NEW
│   └── forex_defaults.yaml             # NEW: Forex-specific defaults
│
└── tests/
    ├── test_oanda_adapters.py          # NEW
    ├── test_forex_execution_providers.py
    ├── test_forex_features.py
    ├── test_forex_risk_guards.py
    ├── test_forex_dealer_simulation.py
    └── test_forex_integration.py
```

---

## 📅 Implementation Phases

### Phase 0: Foundation & Research (Week 1)

**Цель**: Исследование и подготовка инфраструктуры

#### 0.1 Market Research
- [ ] Изучить OANDA v20 API documentation
- [ ] Изучить альтернативы (IG, Dukascopy, FXCM)
- [ ] Собрать reference spreads по currency pairs
- [ ] Проанализировать session liquidity patterns

#### 0.2 Architecture Design
- [ ] Финализировать file structure
- [ ] Определить interfaces для forex adapters
- [ ] Спланировать backward compatibility checks

#### 0.3 Test Infrastructure
- [ ] Создать test fixtures для forex data
- [ ] Настроить mock OANDA API для тестов
- [ ] Добавить forex в CI/CD pipeline

**Deliverables**:
- Research document
- Finalized architecture diagram
- Test infrastructure

**Tests**: ~20 (infrastructure)

---

### Phase 1: Core Enums & Models (Week 2)

**Цель**: Расширение базовых моделей для поддержки Forex

#### 1.1 Update adapters/models.py

```python
# Добавить в ExchangeVendor
class ExchangeVendor(str, Enum):
    BINANCE = "binance"
    BINANCE_US = "binance_us"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    OANDA = "oanda"          # NEW
    IG = "ig"                # NEW (alternative)
    DUKASCOPY = "dukascopy"  # NEW (historical data)
    UNKNOWN = "unknown"

# Добавить forex sessions
FOREX_SESSIONS = [
    TradingSession(
        session_type=SessionType.REGULAR,  # Sydney
        start_minutes=21 * 60,  # 21:00 UTC
        end_minutes=6 * 60,     # 06:00 UTC (next day)
        timezone="UTC",
        days_of_week=(0, 1, 2, 3, 4),  # Mon-Fri
    ),
    TradingSession(
        session_type=SessionType.REGULAR,  # Tokyo
        start_minutes=0,
        end_minutes=9 * 60,
        timezone="UTC",
        days_of_week=(0, 1, 2, 3, 4),
    ),
    # ... London, New York
]
```

#### 1.2 Update execution_providers.py

```python
class AssetClass(enum.Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"  # NEW
```

#### 1.3 Update data_loader_multi_asset.py

```python
class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"  # Already exists! ✓
```

#### 1.4 Update adapters/registry.py

```python
self._lazy_modules: Dict[ExchangeVendor, str] = {
    ExchangeVendor.BINANCE: "adapters.binance",
    ExchangeVendor.ALPACA: "adapters.alpaca",
    ExchangeVendor.YAHOO: "adapters.yahoo",
    ExchangeVendor.OANDA: "adapters.oanda",  # NEW
}
```

**Deliverables**:
- Updated enums in 4 files
- No breaking changes to existing code
- Unit tests for new enums

**Tests**: ~30

---

### Phase 2: OANDA Adapter Implementation (Weeks 3-4)

**Цель**: Полная реализация OANDA adapter по паттерну Binance/Alpaca

#### 2.1 adapters/oanda/__init__.py

```python
"""
OANDA Forex Adapter Package
Supports:
- OANDA v20 REST API
- Streaming prices
- Historical candles
- Order execution
"""
from .market_data import OandaMarketDataAdapter
from .fees import OandaFeeAdapter
from .trading_hours import OandaTradingHoursAdapter
from .exchange_info import OandaExchangeInfoAdapter
from .order_execution import OandaOrderExecutionAdapter

__all__ = [
    "OandaMarketDataAdapter",
    "OandaFeeAdapter",
    "OandaTradingHoursAdapter",
    "OandaExchangeInfoAdapter",
    "OandaOrderExecutionAdapter",
]
```

#### 2.2 adapters/oanda/market_data.py (~400 LOC)

```python
class OandaMarketDataAdapter(MarketDataAdapter):
    """
    OANDA v20 API market data adapter.

    Features:
    - Historical candles (M1 to M)
    - Real-time streaming prices
    - Bid/ask spreads
    - Multiple granularities

    API Endpoints:
    - GET /v3/instruments/{instrument}/candles
    - GET /v3/accounts/{account}/pricing/stream
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._api_key = config.get("api_key") or os.getenv("OANDA_API_KEY")
        self._account_id = config.get("account_id") or os.getenv("OANDA_ACCOUNT_ID")
        self._practice = config.get("practice", True)  # Demo vs Live
        self._base_url = self._get_base_url()

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Fetch historical candles from OANDA.

        Symbol format: "EUR_USD", "GBP_JPY"
        Timeframe: "M1", "M5", "M15", "H1", "H4", "D"
        """
        granularity = self._map_timeframe(timeframe)
        instrument = self._normalize_symbol(symbol)

        params = {
            "granularity": granularity,
            "count": min(limit, 5000),  # OANDA limit
            "price": "MBA",  # Mid, Bid, Ask
        }
        # ... implementation
```

#### 2.3 adapters/oanda/fees.py (~150 LOC)

```python
class OandaFeeAdapter(FeeAdapter):
    """
    OANDA spread-based fee model.

    Forex характеристики:
    - No commission (spread only)
    - Variable spreads by session
    - Spread depends on pair type (major/minor/exotic)

    Note: Fee = 0, cost is in slippage via spread
    """

    # Typical spreads in pips (1 pip = 0.0001 for most pairs)
    TYPICAL_SPREADS: Dict[str, float] = {
        # Majors
        "EUR_USD": 1.0,
        "GBP_USD": 1.2,
        "USD_JPY": 1.0,
        "USD_CHF": 1.5,
        "AUD_USD": 1.3,
        "USD_CAD": 1.5,
        "NZD_USD": 1.8,
        # Minors
        "EUR_GBP": 1.5,
        "EUR_JPY": 1.8,
        # Crosses
        "GBP_JPY": 3.0,
        # Exotics
        "USD_TRY": 30.0,
        "USD_ZAR": 50.0,
    }

    def compute_fee(
        self,
        notional: float,
        side: Side,
        liquidity: Union[str, Liquidity],
        *,
        symbol: Optional[str] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        """
        Forex fee = 0 (spread-based).
        Actual cost captured in slippage model.
        """
        return 0.0  # Commission-free
```

#### 2.4 adapters/oanda/trading_hours.py (~250 LOC)

```python
class OandaTradingHoursAdapter(TradingHoursAdapter):
    """
    Forex trading hours adapter.

    Forex Market Hours (ET):
    - Opens: Sunday 5:00 PM ET
    - Closes: Friday 4:00 PM ET
    - Daily rollover: 5:00 PM ET

    Sessions with varying liquidity:
    - Sydney: 5pm-2am ET
    - Tokyo: 7pm-4am ET
    - London: 3am-12pm ET
    - New York: 8am-5pm ET
    """

    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """Check if forex market is open."""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        weekday = dt.weekday()

        # Closed Saturday all day
        if weekday == 5:
            return False

        # Sunday: opens at 21:00 UTC (5pm ET)
        if weekday == 6:
            return dt.hour >= 21

        # Friday: closes at 21:00 UTC (5pm ET)
        if weekday == 4:
            return dt.hour < 21

        # Mon-Thu: 24 hours
        return True

    def get_current_session(self, ts: int) -> ForexSession:
        """
        Determine current forex session for liquidity modeling.

        Returns:
            ForexSession enum: SYDNEY, TOKYO, LONDON, NEW_YORK, OVERLAP
        """
        # ... implementation
```

#### 2.5 adapters/oanda/exchange_info.py (~200 LOC)

```python
class OandaExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    OANDA instruments information.

    Provides:
    - Available currency pairs
    - Pip values and sizes
    - Min/max trade sizes
    - Margin requirements
    """

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get forex pair information."""
        instrument = self._normalize_symbol(symbol)

        # API call to /v3/accounts/{account}/instruments
        # ...

        return SymbolInfo(
            symbol=symbol,
            vendor=ExchangeVendor.OANDA,
            market_type=MarketType.FOREX,
            exchange_rule=ExchangeRule(
                symbol=symbol,
                tick_size=self._get_pip_size(symbol),  # 0.0001 or 0.01
                step_size=Decimal("1"),  # 1 unit
                min_notional=Decimal("1"),  # $1 minimum
                min_qty=Decimal("1"),
                max_qty=Decimal("100000000"),  # 100M units
                market_type=MarketType.FOREX,
                base_asset=symbol[:3],
                quote_asset=symbol[4:],
            ),
        )
```

#### 2.6 adapters/oanda/order_execution.py (~350 LOC)

```python
class OandaOrderExecutionAdapter(OrderExecutionAdapter):
    """
    OANDA order execution adapter.

    Order Types:
    - Market: Execute at current price
    - Limit: Execute at specified price or better
    - Stop: Trigger market order at price
    - Stop-Limit: Trigger limit order at price

    Special Features:
    - Take Profit / Stop Loss
    - Trailing Stop
    - Guaranteed Stop Loss (premium)
    """

    def submit_order(self, order: Order) -> OrderResult:
        """Submit order to OANDA."""
        endpoint = f"/v3/accounts/{self._account_id}/orders"

        body = {
            "order": {
                "instrument": self._normalize_symbol(order.symbol),
                "units": str(int(order.qty * (1 if order.is_buy else -1))),
                "type": self._map_order_type(order.order_type),
                "timeInForce": order.time_in_force,
            }
        }

        if order.order_type == "LIMIT":
            body["order"]["price"] = str(order.limit_price)

        # ... API call
```

**Deliverables**:
- Complete OANDA adapter package (5 files)
- Registration in adapter registry
- Environment variable support

**Tests**: ~120

---

### Phase 3: ForexParametricSlippageProvider (L2+) (Weeks 5-6)

**Цель**: Research-backed parametric TCA model для Forex

#### 3.1 execution_providers_forex.py (~600 LOC)

```python
"""
Forex Parametric TCA Model (L2+)

Extends Almgren-Chriss √participation with forex-specific factors:
1. √Participation impact (base)
2. Session liquidity curve (Tokyo/London/NY)
3. Spread regime (tight/normal/wide)
4. Interest rate differential (carry)
5. Volatility regime
6. News event proximity
7. Correlation with DXY
8. Pair type multiplier (major/minor/exotic)

References:
- Lyons (2001): "The Microstructure Approach to Exchange Rates"
- Evans & Lyons (2002): "Order Flow and Exchange Rate Dynamics"
- Berger et al. (2008): "The Development of the Global FX Market"
"""

class ForexSession(enum.Enum):
    """Forex trading session."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    TOKYO_LONDON_OVERLAP = "tokyo_london_overlap"
    OFF_HOURS = "off_hours"


class PairType(enum.Enum):
    """Currency pair classification."""
    MAJOR = "major"      # EUR/USD, USD/JPY, GBP/USD, etc.
    MINOR = "minor"      # EUR/GBP, EUR/CHF, etc.
    CROSS = "cross"      # GBP/JPY, EUR/JPY, etc.
    EXOTIC = "exotic"    # USD/TRY, USD/ZAR, etc.


@dataclass
class ForexParametricConfig:
    """
    Configuration for ForexParametricSlippageProvider.

    All spreads in pips (1 pip = 0.0001 for most pairs).
    """
    # Base impact (Almgren-Chriss)
    impact_coef_base: float = 0.03  # Lower than crypto (more liquid)
    impact_coef_range: Tuple[float, float] = (0.02, 0.05)

    # Default spreads by pair type (pips)
    default_spreads: Dict[str, float] = field(default_factory=lambda: {
        "major": 1.0,
        "minor": 2.0,
        "cross": 3.0,
        "exotic": 20.0,
    })

    # Session liquidity multipliers
    session_liquidity: Dict[str, float] = field(default_factory=lambda: {
        "sydney": 0.65,
        "tokyo": 0.75,
        "london": 1.10,
        "new_york": 1.05,
        "london_ny_overlap": 1.30,
        "tokyo_london_overlap": 0.90,
        "off_hours": 0.50,
    })

    # Interest rate differential sensitivity
    carry_sensitivity: float = 0.05  # 5% adjustment per 1% rate diff

    # DXY correlation decay (for non-USD pairs)
    dxy_correlation_decay: float = 0.3

    # News event impact
    news_event_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "nfp": 2.5,           # Non-Farm Payrolls
        "fomc": 2.0,          # Fed decisions
        "ecb": 1.8,           # ECB decisions
        "boe": 1.5,           # Bank of England
        "cpi": 1.5,           # Inflation data
        "gdp": 1.3,           # GDP releases
        "other": 1.2,
    })

    # Pair type multipliers (base slippage adjustment)
    pair_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "major": 1.0,
        "minor": 1.3,
        "cross": 1.6,
        "exotic": 3.0,
    })

    # Volatility regime
    vol_regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.85,
        "normal": 1.0,
        "high": 1.4,
        "extreme": 2.0,  # Flash crash scenarios
    })

    # Bounds
    min_slippage_pips: float = 0.1
    max_slippage_pips: float = 100.0


class ForexParametricSlippageProvider:
    """
    L2+: Smart parametric TCA model for forex markets.

    Formula:
        slippage_pips = half_spread
            × (1 + k × √participation)
            × session_liquidity_factor
            × volatility_regime
            × (1 + carry_adjustment)
            × dxy_correlation_factor
            × news_event_factor
            × pair_type_multiplier

    Key differences from Crypto/Equity:
    - Spreads in pips, not basis points
    - Session-based liquidity (not hour-of-day)
    - Carry trade considerations
    - No central LOB (dealer quotes)
    - Different market structure (OTC vs exchange)
    """

    def __init__(
        self,
        config: Optional[ForexParametricConfig] = None,
    ) -> None:
        self.config = config or ForexParametricConfig()
        self._adaptive_k: float = self.config.impact_coef_base

    def compute_slippage_pips(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        *,
        session: Optional[ForexSession] = None,
        pair_type: Optional[PairType] = None,
        interest_rate_diff: Optional[float] = None,  # Base - Quote rate
        dxy_correlation: Optional[float] = None,
        upcoming_news: Optional[str] = None,
        recent_returns: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Compute expected slippage in pips.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / estimated session volume
            session: Current forex session
            pair_type: Currency pair classification
            interest_rate_diff: Interest rate differential (base - quote)
            dxy_correlation: Correlation with Dollar Index
            upcoming_news: Type of upcoming economic news
            recent_returns: Recent returns for volatility regime

        Returns:
            Expected slippage in pips
        """
        # 1. Base spread from pair type
        pair_type = pair_type or self._classify_pair(order.symbol)
        half_spread = self.config.default_spreads[pair_type.value] / 2.0

        # Override with market spread if available
        if market.spread_bps is not None:
            # Convert bps to pips (approximate)
            half_spread = market.spread_bps / 10.0 / 2.0

        # 2. √Participation impact
        participation = max(1e-12, abs(participation_ratio))
        impact = self._adaptive_k * math.sqrt(participation)

        # 3. Session liquidity
        session = session or self._detect_session(market.timestamp)
        session_factor = self.config.session_liquidity.get(
            session.value, 1.0
        )
        # Invert: low liquidity = more slippage
        session_adjustment = 1.0 / max(0.3, session_factor)

        # 4. Volatility regime
        vol_regime = self._detect_volatility_regime(recent_returns)
        vol_mult = self.config.vol_regime_multipliers.get(vol_regime, 1.0)

        # 5. Carry adjustment (high carry = tighter markets)
        carry_factor = 1.0
        if interest_rate_diff is not None:
            # Positive carry (long base, short quote) = more liquidity
            carry_factor = 1.0 - interest_rate_diff * self.config.carry_sensitivity
            carry_factor = max(0.8, min(1.3, carry_factor))

        # 6. DXY correlation (for non-USD pairs)
        dxy_factor = 1.0
        if dxy_correlation is not None and "USD" not in order.symbol:
            # Lower correlation = less liquidity spillover
            dxy_factor = 1.0 + (1.0 - abs(dxy_correlation)) * self.config.dxy_correlation_decay

        # 7. News event impact
        news_factor = 1.0
        if upcoming_news is not None:
            news_factor = self.config.news_event_multipliers.get(
                upcoming_news.lower(), 1.0
            )

        # 8. Pair type multiplier
        pair_mult = self.config.pair_type_multipliers.get(pair_type.value, 1.0)

        # Combine all factors
        total_slippage = (
            half_spread
            * (1.0 + impact * 10000)  # Scale impact
            * session_adjustment
            * vol_mult
            * carry_factor
            * dxy_factor
            * news_factor
            * pair_mult
        )

        # Apply bounds
        return max(
            self.config.min_slippage_pips,
            min(self.config.max_slippage_pips, total_slippage)
        )

    def _classify_pair(self, symbol: str) -> PairType:
        """Classify currency pair."""
        MAJORS = {"EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
                  "AUD_USD", "USD_CAD", "NZD_USD"}
        MINORS = {"EUR_GBP", "EUR_CHF", "GBP_CHF", "EUR_AUD"}
        EXOTICS = {"USD_TRY", "USD_ZAR", "USD_MXN", "USD_PLN"}

        norm = symbol.replace("/", "_").upper()
        if norm in MAJORS:
            return PairType.MAJOR
        if norm in MINORS:
            return PairType.MINOR
        if norm in EXOTICS:
            return PairType.EXOTIC
        return PairType.CROSS

    def _detect_session(self, timestamp: int) -> ForexSession:
        """Detect current forex session from timestamp."""
        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        hour = dt.hour

        # Check overlaps first
        if 12 <= hour < 16:
            return ForexSession.LONDON_NY_OVERLAP
        if 7 <= hour < 9:
            return ForexSession.TOKYO_LONDON_OVERLAP

        # Individual sessions
        if 21 <= hour or hour < 6:
            return ForexSession.SYDNEY
        if 0 <= hour < 9:
            return ForexSession.TOKYO
        if 7 <= hour < 16:
            return ForexSession.LONDON
        if 12 <= hour < 21:
            return ForexSession.NEW_YORK

        return ForexSession.OFF_HOURS
```

**Deliverables**:
- ForexParametricSlippageProvider
- ForexFeeProvider (spread-based)
- Factory function integration
- Profiles: major, exotic, news-sensitive

**Tests**: ~90

---

### Phase 4: Forex Features Pipeline (Week 7)

**Цель**: Forex-специфичные features для training

#### 4.1 forex_features.py (~400 LOC)

```python
"""
Forex-specific features for ML training.

Features:
1. Interest Rate Differential (Carry)
2. Relative Strength vs DXY
3. Session indicator (one-hot)
4. Spread regime
5. COT positioning (Commitments of Traders)
6. News calendar proximity
7. Cross-currency momentum
8. Volatility term structure
"""

@dataclass
class ForexFeatures:
    """Container for forex-specific features."""
    # Interest rates
    base_rate: float
    quote_rate: float
    rate_differential: float

    # DXY correlation
    dxy_value: float
    dxy_return_1d: float
    rs_vs_dxy_20d: float

    # Session indicators
    session: ForexSession
    is_london_open: bool
    is_ny_open: bool
    is_overlap: bool

    # Spread dynamics
    spread_pips: float
    spread_regime: str  # "tight", "normal", "wide"
    spread_zscore: float

    # COT data (weekly)
    cot_net_position: float
    cot_change_1w: float

    # Volatility
    realized_vol_5d: float
    implied_vol: Optional[float]
    vol_term_structure: Optional[float]  # Short/Long vol ratio


def calculate_interest_rate_features(
    base_currency: str,
    quote_currency: str,
    rates_data: pd.DataFrame,
) -> Tuple[float, float, float]:
    """
    Calculate interest rate differential features.

    Data sources:
    - FRED: Federal Reserve rates
    - ECB: Euro rates
    - BOJ: Yen rates
    """
    base_rate = rates_data.get(f"{base_currency}_RATE", 0.0)
    quote_rate = rates_data.get(f"{quote_currency}_RATE", 0.0)
    diff = base_rate - quote_rate
    return base_rate, quote_rate, diff


def calculate_relative_strength_vs_dxy(
    pair_prices: pd.Series,
    dxy_prices: pd.Series,
    window: int = 20,
) -> float:
    """
    Calculate relative strength vs Dollar Index.

    Similar to RS vs SPY for stocks.
    """
    if len(pair_prices) < window or len(dxy_prices) < window:
        return 0.0

    pair_return = (pair_prices.iloc[-1] / pair_prices.iloc[-window]) - 1
    dxy_return = (dxy_prices.iloc[-1] / dxy_prices.iloc[-window]) - 1

    return pair_return - dxy_return
```

#### 4.2 Economic Calendar Integration

```python
class EconomicCalendar:
    """
    Economic events calendar for forex.

    High-impact events:
    - Central bank decisions (FOMC, ECB, BOE, BOJ)
    - Employment reports (NFP, UK Employment)
    - Inflation data (CPI, PPI)
    - GDP releases
    - Trade balance
    """

    HIGH_IMPACT_EVENTS: Dict[str, List[str]] = {
        "USD": ["nfp", "fomc", "cpi", "gdp"],
        "EUR": ["ecb", "german_cpi", "eurozone_gdp"],
        "GBP": ["boe", "uk_cpi", "uk_gdp"],
        "JPY": ["boj", "tankan", "japan_cpi"],
    }

    def get_next_event(
        self,
        currency: str,
        current_ts: int,
    ) -> Optional[Tuple[str, int, str]]:
        """Get next high-impact event for currency."""
        # Returns (event_name, timestamp, impact_level)
        pass

    def hours_to_next_event(
        self,
        currency: str,
        current_ts: int,
    ) -> Optional[float]:
        """Hours until next high-impact event."""
        pass
```

**Deliverables**:
- forex_features.py
- Economic calendar integration
- Feature pipeline extension
- DXY/rates data loader

**Tests**: ~60

---

### Phase 5: L3 Forex Dealer Simulation (Week 8)

**Цель**: High-fidelity dealer quote simulation для 95% реализма

#### 5.1 lob/forex_dealer.py (~500 LOC)

```python
"""
Forex Dealer Quote Simulation (L3)

Unlike exchange LOBs, forex is OTC with dealer quotes.
This module simulates:
1. Multi-dealer quote aggregation
2. Last-look rejection
3. Quote flickering
4. Spread widening on size
5. Latency arbitrage protection
6. Time-of-day liquidity patterns

References:
- Oomen (2017): "Last Look" in FX
- Hasbrouck & Saar (2013): "Low-latency trading"
- King et al. (2012): "Market structure of the FX market"
"""

@dataclass
class DealerQuote:
    """Single dealer quote."""
    dealer_id: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp_ns: int
    is_firm: bool  # Firm vs indicative
    max_size: float
    valid_for_ms: int  # Quote validity window
    last_look_ms: int  # Last-look window


class ForexDealerSimulator:
    """
    Simulates multi-dealer forex market.

    Key behaviors:
    1. Multiple dealers with different spreads/sizes
    2. Quote flickering (rapid updates)
    3. Last-look rejection probability
    4. Size-dependent spread widening
    5. Session-dependent liquidity

    Unlike LOB simulation:
    - No queue position (no FIFO)
    - Dealer discretion (last-look)
    - Indicative vs firm quotes
    """

    def __init__(
        self,
        num_dealers: int = 5,
        config: Optional[ForexDealerConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or ForexDealerConfig()
        self.num_dealers = num_dealers
        self._rng = np.random.default_rng(seed)

        # Initialize dealers with different characteristics
        self._dealers = self._create_dealers()

    def _create_dealers(self) -> List[Dealer]:
        """Create heterogeneous dealers."""
        dealers = []
        for i in range(self.num_dealers):
            dealers.append(Dealer(
                dealer_id=f"dealer_{i}",
                spread_factor=1.0 + self._rng.uniform(-0.2, 0.3),
                max_size=self._rng.uniform(1e6, 10e6),
                last_look_prob=self._rng.uniform(0.02, 0.15),
                latency_ms=self._rng.uniform(10, 100),
            ))
        return dealers

    def get_aggregated_quote(
        self,
        symbol: str,
        session: ForexSession,
        base_mid: float,
    ) -> AggregatedQuote:
        """
        Get best bid/ask from all dealers.

        Returns aggregated (NBBO-like) quote.
        """
        quotes = []
        for dealer in self._dealers:
            quote = dealer.generate_quote(
                symbol=symbol,
                base_mid=base_mid,
                session=session,
                config=self.config,
            )
            quotes.append(quote)

        # Aggregate: best bid, best ask
        best_bid = max(q.bid for q in quotes)
        best_ask = min(q.ask for q in quotes)

        return AggregatedQuote(
            bid=best_bid,
            ask=best_ask,
            dealer_quotes=quotes,
            timestamp_ns=time.time_ns(),
        )

    def attempt_execution(
        self,
        order: Order,
        quote: AggregatedQuote,
    ) -> ExecutionResult:
        """
        Attempt to execute order against dealer quotes.

        Simulates:
        1. Dealer selection (best price)
        2. Latency to dealer
        3. Last-look decision
        4. Fill or reject
        """
        # Select dealer with best price for this side
        if order.is_buy:
            quotes = sorted(quote.dealer_quotes, key=lambda q: q.ask)
        else:
            quotes = sorted(quote.dealer_quotes, key=lambda q: -q.bid)

        for dealer_quote in quotes:
            # Check size
            available = dealer_quote.ask_size if order.is_buy else dealer_quote.bid_size
            if order.qty > available:
                continue

            # Simulate latency
            latency_ns = int(dealer_quote.valid_for_ms * 1e6 * self._rng.uniform(0.5, 1.0))

            # Last-look rejection
            dealer = self._get_dealer(dealer_quote.dealer_id)
            if self._rng.random() < dealer.last_look_prob:
                continue  # Rejected

            # Success
            return ExecutionResult(
                filled=True,
                price=dealer_quote.ask if order.is_buy else dealer_quote.bid,
                qty=order.qty,
                dealer_id=dealer_quote.dealer_id,
                latency_ns=latency_ns,
                last_look_passed=True,
            )

        return ExecutionResult(filled=False, reject_reason="all_dealers_rejected")
```

#### 5.2 Last-Look Simulation

```python
class LastLookSimulator:
    """
    Simulates dealer last-look behavior.

    Last-look allows dealers to reject trades after
    receiving the order if market moved against them.

    Factors affecting rejection:
    1. Price movement since quote
    2. Order size (larger = more scrutiny)
    3. Client classification
    4. Market volatility
    """

    def should_reject(
        self,
        order: Order,
        quote: DealerQuote,
        current_mid: float,
        latency_ms: float,
    ) -> Tuple[bool, str]:
        """
        Determine if dealer rejects via last-look.

        Returns:
            (rejected, reason)
        """
        quote_mid = (quote.bid + quote.ask) / 2.0
        price_move = (current_mid - quote_mid) / quote_mid

        # Adverse selection check
        if order.is_buy and price_move > 0:
            # Client buying, price went up = adverse
            if price_move > self.adverse_threshold:
                return True, "adverse_selection"
        elif not order.is_buy and price_move < 0:
            # Client selling, price went down = adverse
            if abs(price_move) > self.adverse_threshold:
                return True, "adverse_selection"

        # Size-based rejection
        if order.qty > quote.max_size * self.size_threshold:
            if self._rng.random() < self.large_order_reject_prob:
                return True, "size_exceeded"

        return False, ""
```

**Deliverables**:
- ForexDealerSimulator
- LastLookSimulator
- Multi-dealer quote aggregation
- Latency modeling
- Integration with L3ExecutionProvider

**Tests**: ~70

---

### Phase 6: Forex Risk Management (Week 9)

**Цель**: Forex-специфичный risk management (leverage, margin)

#### 6.1 services/forex_risk_guards.py (~350 LOC)

```python
"""
Forex-specific risk guards.

Key differences from stocks:
1. Leverage: 50:1 to 500:1 (vs 4:1 for stocks)
2. Margin: Required margin = Position / Leverage
3. No PDT rules
4. 24/5 monitoring required
5. Swap/rollover costs
"""

class ForexLeverageGuard:
    """
    Monitors forex leverage and margin.

    Margin Call Levels (typical):
    - Warning: 120% margin level
    - Margin Call: 100% margin level
    - Stop Out: 50% margin level

    Calculation:
    - Margin Level = (Equity / Used Margin) × 100%
    - Free Margin = Equity - Used Margin
    - Required Margin = Position Size / Leverage
    """

    def __init__(
        self,
        max_leverage: float = 50.0,
        margin_warning_level: float = 1.20,
        margin_call_level: float = 1.00,
        stop_out_level: float = 0.50,
    ) -> None:
        self.max_leverage = max_leverage
        self.margin_warning_level = margin_warning_level
        self.margin_call_level = margin_call_level
        self.stop_out_level = stop_out_level

    def check_margin_requirement(
        self,
        position_value: float,
        account_equity: float,
        leverage: float,
    ) -> MarginCheckResult:
        """
        Check if position meets margin requirements.
        """
        required_margin = position_value / leverage
        margin_level = account_equity / required_margin if required_margin > 0 else float('inf')

        status = MarginStatus.OK
        if margin_level < self.stop_out_level:
            status = MarginStatus.STOP_OUT
        elif margin_level < self.margin_call_level:
            status = MarginStatus.MARGIN_CALL
        elif margin_level < self.margin_warning_level:
            status = MarginStatus.WARNING

        return MarginCheckResult(
            margin_level=margin_level,
            required_margin=required_margin,
            free_margin=account_equity - required_margin,
            status=status,
            max_additional_position=self._calc_max_additional(
                account_equity, required_margin, leverage
            ),
        )


class SwapCostCalculator:
    """
    Calculate overnight swap/rollover costs.

    Forex positions rolled over at 5pm ET incur:
    - Positive swap: Earning interest (long high-rate currency)
    - Negative swap: Paying interest (long low-rate currency)

    Wednesday swaps are typically 3x (weekend rollover).
    """

    def calculate_swap(
        self,
        symbol: str,
        position_units: float,
        is_long: bool,
        current_price: float,
        days: int = 1,
    ) -> float:
        """
        Calculate swap cost/credit for position.

        Returns:
            Swap amount (positive = credit, negative = cost)
        """
        swap_points = self._get_swap_points(symbol, is_long)
        pip_value = self._get_pip_value(symbol, current_price)

        # Wednesday = 3 days rollover
        effective_days = days

        return position_units * swap_points * pip_value * effective_days / 10.0
```

#### 6.2 Integration with Existing Risk System

```python
# Update risk_guard.py to support forex
def create_risk_guard(
    asset_class: AssetClass,
    config: Dict[str, Any],
) -> BaseRiskGuard:
    """Factory for asset-class-specific risk guards."""
    if asset_class == AssetClass.CRYPTO:
        return CryptoRiskGuard(config)
    elif asset_class == AssetClass.EQUITY:
        return EquityRiskGuard(config)
    elif asset_class == AssetClass.FOREX:
        return ForexRiskGuard(config)  # NEW
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")
```

**Deliverables**:
- ForexLeverageGuard
- SwapCostCalculator
- Integration with existing risk system
- Margin monitoring

**Tests**: ~50

---

### Phase 7: Data Pipeline & Downloader (Week 10)

**Цель**: Скачивание и подготовка данных для Forex

#### 7.1 scripts/download_forex_data.py (~300 LOC)

```python
"""
Download historical forex data from OANDA.

Usage:
    python scripts/download_forex_data.py \
        --pairs EUR_USD GBP_USD USD_JPY \
        --start 2020-01-01 \
        --timeframe H4 \
        --output data/forex/

Data sources:
- OANDA v20 API (primary)
- Dukascopy (tick data, historical)
- FRED (interest rates)
"""

def download_oanda_candles(
    pair: str,
    granularity: str,
    start_date: str,
    end_date: Optional[str] = None,
    output_dir: str = "data/forex/",
) -> str:
    """Download candles from OANDA API."""
    pass

def download_interest_rates(
    currencies: List[str],
    start_date: str,
    output_path: str = "data/forex/rates.csv",
) -> str:
    """Download central bank rates from FRED."""
    pass

def download_economic_calendar(
    start_date: str,
    end_date: str,
    currencies: List[str],
    output_path: str = "data/forex/calendar.csv",
) -> str:
    """Download economic calendar events."""
    pass
```

#### 7.2 Update data_loader_multi_asset.py

```python
def load_forex_data(
    paths: Sequence[str],
    timeframe: str = "4h",
    filter_trading_hours: bool = True,
) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """
    Load forex data with forex-specific preprocessing.

    - Filter weekend data
    - Add session indicators
    - Merge interest rate data
    - Add spread calculations
    """
    pass
```

**Deliverables**:
- download_forex_data.py
- Interest rates downloader
- Economic calendar downloader
- Data loader integration

**Tests**: ~40

---

### Phase 8: Configuration System (Week 10-11)

**Цель**: Гибкая конфигурация для Forex

#### 8.1 configs/forex_defaults.yaml

```yaml
# =============================================================================
# FOREX DEFAULTS CONFIGURATION
# =============================================================================

forex:
  # Core settings
  asset_class: forex
  data_vendor: oanda
  market: spot

  # Trading hours: Sun 5pm - Fri 4pm ET
  session:
    calendar: forex_24x5
    weekend_filter: true
    rollover_time_utc: 21  # 5pm ET = 21:00 UTC (winter)

  # Fees: Spread-based (no commission)
  fees:
    structure: spread_only
    maker_bps: 0.0
    taker_bps: 0.0
    spread_markup_bps: 0.0  # Broker markup
    swap_enabled: true

  # Slippage: Forex-specific model
  slippage:
    profile: forex_major  # or forex_exotic
    k: 0.03
    default_spread_pips: 1.0
    min_slippage_pips: 0.1
    max_slippage_pips: 100.0
    session_adjustment: true
    news_adjustment: true

  # Execution
  execution:
    fill_policy: dealer_quote  # Not LOB-based
    last_look_simulation: true
    quote_validity_ms: 200

  # Leverage
  leverage:
    max_leverage: 50.0
    default_leverage: 30.0
    margin_warning: 1.20
    margin_call: 1.00
    stop_out: 0.50

  # Liquidity
  liquidity:
    min_adv_usd: 0  # Always liquid for majors
    session_scaling: true

  # Risk
  no_trade:
    enabled: true
    enforce_trading_hours: true
    rollover_keepout_minutes: 30  # Around 5pm ET

# Provider mapping
provider_mapping:
  forex:
    fee_provider: ForexFeeProvider
    slippage_provider: ForexParametricSlippageProvider
    slippage_profile: forex_major
    trading_hours_adapter: OandaTradingHoursAdapter
    fill_provider: ForexDealerFillProvider

# Vendor mapping
data_vendor_mapping:
  oanda:
    market_data_adapter: OandaMarketDataAdapter
    exchange_info_adapter: OandaExchangeInfoAdapter
    order_execution_adapter: OandaOrderExecutionAdapter
    default_asset_class: forex
    api_key_env: OANDA_API_KEY
    account_id_env: OANDA_ACCOUNT_ID
```

#### 8.2 configs/config_train_forex.yaml

```yaml
# Training configuration for Forex
mode: train
asset_class: forex
data_vendor: oanda

data:
  timeframe: "4h"
  filter_trading_hours: true
  filter_weekends: true
  paths:
    - "data/forex/*.parquet"

model:
  algo: "ppo"
  optimizer_class: AdaptiveUPGD
  params:
    use_twin_critics: true
    num_atoms: 21
    gamma: 0.99

env:
  session:
    calendar: forex_24x5
  slippage:
    profile: forex_major
  fees:
    structure: spread_only
  leverage:
    max_leverage: 50.0
```

#### 8.3 configs/config_live_oanda.yaml

```yaml
# Live trading configuration for OANDA
mode: live
asset_class: forex
data_vendor: oanda

exchange:
  vendor: oanda
  practice: false  # Live account

execution:
  order_type: market
  slippage_tolerance_pips: 2.0

risk:
  max_position_pct: 0.10
  max_leverage: 30.0
  stop_loss_pips: 50
  take_profit_pips: 100
```

**Deliverables**:
- forex_defaults.yaml
- config_train_forex.yaml
- config_backtest_forex.yaml
- config_live_oanda.yaml
- Integration with asset_class_defaults.yaml

**Tests**: ~30

---

### Phase 9: Training & Backtest Integration (Week 11)

**Цель**: Полная интеграция в training/backtest pipeline

#### 9.1 Update script_live.py

```python
# Add forex to valid asset classes
VALID_ASSET_CLASSES = (ASSET_CLASS_CRYPTO, ASSET_CLASS_EQUITY, ASSET_CLASS_FOREX)

ASSET_CLASS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    ASSET_CLASS_CRYPTO: {...},
    ASSET_CLASS_EQUITY: {...},
    ASSET_CLASS_FOREX: {  # NEW
        "slippage_pips": 1.0,
        "limit_offset_pips": 2.0,
        "tif": "GTC",
        "extended_hours": False,  # N/A for forex
        "default_vendor": "oanda",
        "leverage": 30.0,
    },
}

VENDOR_TO_ASSET_CLASS: Dict[str, str] = {
    "binance": ASSET_CLASS_CRYPTO,
    "alpaca": ASSET_CLASS_EQUITY,
    "polygon": ASSET_CLASS_EQUITY,
    "oanda": ASSET_CLASS_FOREX,  # NEW
}
```

#### 9.2 Update Mediator

```python
# mediator.py - asset class detection
def _create_execution_provider(self, asset_class: str):
    if asset_class == "forex":
        return create_execution_provider(
            AssetClass.FOREX,
            level="L2+",  # ForexParametricSlippageProvider
        )
```

#### 9.3 Training Script Updates

```python
# train_model_multi_patch.py
def create_env_for_asset_class(asset_class: str, config: Dict) -> TradingEnv:
    if asset_class == "forex":
        return ForexTradingEnv(config)  # Or use unified TradingEnv with forex settings
```

**Deliverables**:
- script_live.py updates
- Mediator integration
- Training pipeline support
- Backtest pipeline support

**Tests**: ~40

---

### Phase 10: Testing & Validation (Week 12)

**Цель**: Comprehensive testing и validation

#### 10.1 Unit Tests

```
tests/
├── test_oanda_adapters.py              # 120 tests
│   ├── test_market_data_adapter
│   ├── test_fee_adapter
│   ├── test_trading_hours_adapter
│   ├── test_exchange_info_adapter
│   └── test_order_execution_adapter
│
├── test_forex_execution_providers.py   # 90 tests
│   ├── test_forex_parametric_config
│   ├── test_session_detection
│   ├── test_pair_classification
│   ├── test_slippage_computation
│   └── test_profiles
│
├── test_forex_features.py              # 60 tests
│   ├── test_interest_rate_features
│   ├── test_dxy_correlation
│   ├── test_session_indicators
│   └── test_economic_calendar
│
├── test_forex_dealer_simulation.py     # 70 tests
│   ├── test_dealer_quote_generation
│   ├── test_multi_dealer_aggregation
│   ├── test_last_look_simulation
│   └── test_latency_modeling
│
├── test_forex_risk_guards.py           # 50 tests
│   ├── test_leverage_guard
│   ├── test_margin_calculation
│   └── test_swap_calculator
│
└── test_forex_integration.py           # 40 tests
    ├── test_full_pipeline
    ├── test_backtest_execution
    ├── test_config_loading
    └── test_backward_compatibility
```

#### 10.2 Integration Tests

```python
class TestForexBackwardCompatibility:
    """
    Ensure forex integration doesn't break crypto/equity.
    """

    def test_crypto_pipeline_unchanged(self):
        """Run full crypto backtest, compare metrics."""
        pass

    def test_equity_pipeline_unchanged(self):
        """Run full equity backtest, compare metrics."""
        pass

    def test_mixed_asset_config(self):
        """Test config with multiple asset classes."""
        pass
```

#### 10.3 Validation Metrics

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Spread accuracy | ±10% | Compare vs live OANDA quotes |
| Session liquidity | ±20% | Compare vs historical ADV |
| Fill rate | >95% | Backtest validation |
| Last-look rejects | 5-15% | Industry benchmarks |
| Slippage estimate | ±2 pips | Paper trading validation |

**Deliverables**:
- 400+ unit tests
- Integration test suite
- Backward compatibility tests
- Validation report

---

## 📊 Summary

### Phase Overview

| Phase | Description | Duration | LOC | Tests |
|-------|-------------|----------|-----|-------|
| 0 | Foundation & Research | 1 week | 100 | 20 |
| 1 | Core Enums & Models | 1 week | 200 | 30 |
| 2 | OANDA Adapter | 2 weeks | 1,350 | 120 |
| 3 | ForexParametricSlippage (L2+) | 2 weeks | 600 | 90 |
| 4 | Forex Features | 1 week | 400 | 60 |
| 5 | L3 Dealer Simulation | 1 week | 500 | 70 |
| 6 | Risk Management | 1 week | 350 | 50 |
| 7 | Data Pipeline | 1 week | 400 | 40 |
| 8 | Configuration | 0.5 weeks | 300 | 30 |
| 9 | Training Integration | 1 week | 300 | 40 |
| 10 | Testing & Validation | 1 week | - | 430 (total) |
| **TOTAL** | | **12 weeks** | **4,500** | **430** |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OANDA API changes | Low | Medium | Abstract API layer, version pinning |
| Forex data quality | Medium | High | Multiple data sources, validation |
| Session detection bugs | Medium | Medium | Extensive timezone tests |
| Backward compatibility | Low | High | Comprehensive regression tests |
| L3 simulation accuracy | Medium | Medium | Validate against paper trading |

### Success Criteria

1. **Functional**
   - [ ] Full pipeline from data to live trading works
   - [ ] All 7 major pairs supported
   - [ ] Session-aware execution

2. **Performance**
   - [ ] Slippage estimate accuracy ±15%
   - [ ] Fill rate >95% for liquid pairs
   - [ ] No regressions in crypto/equity

3. **Quality**
   - [ ] 430+ tests passing
   - [ ] Code coverage >90%
   - [ ] Documentation complete

---

## 🔜 Next Steps

1. **Approve plan** - Review and finalize this document
2. **Set up OANDA account** - Practice (demo) account for development
3. **Create branch** - `feature/forex-integration`
4. **Phase 0** - Start research and infrastructure setup

---

**Author**: Claude AI
**Version**: 1.0
**Last Updated**: 2025-11-29
