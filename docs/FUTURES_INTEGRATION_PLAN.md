# 🚀 UNIFIED FUTURES INTEGRATION PLAN

## Comprehensive L3-Level Multi-Asset Futures Trading Integration

**Версия**: 2.0
**Дата**: 2025-11-30
**Статус**: ПЛАН
**Целевой реализм симуляции**: 95%+

---

## 📋 EXECUTIVE SUMMARY

### Цель
Интеграция **всех типов фьючерсов** на уровне L3 с полной симуляцией:

1. **Crypto Futures** (Binance USDT-M Perpetual & Quarterly)
2. **Equity Index Futures** (CME: ES, NQ, YM, RTY)
3. **Commodity Futures** (CME/COMEX: GC, CL, SI)
4. **Currency Futures** (CME: 6E, 6J, 6B)

### Поддерживаемые биржи/брокеры

| Тип | Биржа/Брокер | API | Статус |
|-----|--------------|-----|--------|
| **Crypto** | Binance Futures | REST + WebSocket | 🎯 Primary |
| **Crypto** | Bybit | REST + WebSocket | 📋 Future |
| **Equity Index** | CME via Interactive Brokers | TWS API | 🎯 Primary |
| **Equity Index** | CME via Alpaca | REST | 📋 Alternative |
| **Commodity** | CME via Interactive Brokers | TWS API | 🎯 Primary |
| **Currency** | CME via Interactive Brokers | TWS API | 🎯 Primary |

### Сравнение типов фьючерсов

| Аспект | Crypto Perpetual | Crypto Quarterly | Index Futures (ES) | Commodity (GC) | Currency (6E) |
|--------|------------------|------------------|-------------------|----------------|---------------|
| **Expiration** | Никогда | Mar/Jun/Sep/Dec | Mar/Jun/Sep/Dec | Monthly | Mar/Jun/Sep/Dec |
| **Settlement** | N/A | Cash (USDT) | Cash (USD) | Physical/Cash | Cash (USD) |
| **Funding** | Каждые 8ч | Нет | Нет | Нет | Нет |
| **Basis** | Minimal | Да | Да | Да | Да |
| **Trading Hours** | 24/7 | 24/7 | 23/5* | 23/5* | 23/5* |
| **Max Leverage** | 125x | 125x | 20x | 10x | 50x |
| **Tick Size** | Variable | Variable | 0.25 ($12.50) | 0.10 ($10) | 0.00005 ($6.25) |
| **Contract Size** | 1 unit | 1 unit | $50 × Index | 100 oz | €125,000 |
| **Initial Margin** | 0.8%-50% | 0.8%-50% | ~5-10% | ~5-10% | ~2-3% |

*CME: Sun 6pm - Fri 5pm ET с 15-минутным перерывом 4:15-4:30pm ET

### Ключевые концепции для всех типов

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED FUTURES CONCEPTS                      │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Leverage & Margin          - All futures types               │
│ ✅ Mark Price vs Last Price   - For liquidation calculation     │
│ ✅ Long/Short positions       - Bidirectional trading           │
│ ✅ Contract specifications    - Tick size, multiplier, expiry   │
│ ✅ Settlement mechanics       - Cash vs physical                │
│ ✅ Rollover handling          - Contract expiration             │
├─────────────────────────────────────────────────────────────────┤
│ 🔸 Crypto-specific: Funding rates, Insurance fund, ADL         │
│ 🔸 CME-specific: Settlement times, daily limits, circuit breaks│
│ 🔸 Commodity-specific: Delivery months, storage costs          │
└─────────────────────────────────────────────────────────────────┘
```

### Существующий код для переиспользования

| Компонент | Файл | Применимость | Статус |
|-----------|------|--------------|--------|
| MarketType.CRYPTO_FUTURES | adapters/models.py | Crypto | ✅ Определён |
| crypto_futures defaults | configs/asset_class_defaults.yaml | Crypto | ✅ Базовый |
| Funding rate ingestion | ingest_funding_mark.py | Crypto | ✅ Готов |
| Forex leverage guards | services/forex_risk_guards.py | **Все типы** | ✅ Референс |
| Binance futures URL | adapters/binance/market_data.py | Crypto | ✅ Поддержка |
| L3 LOB simulation | lob/*.py | **Все типы** | ✅ Полностью |
| US market structure | lob/us_market_structure.py | CME futures | ✅ Tick size, circuit breakers |
| Equity parametric TCA | execution_providers.py | Index futures | ✅ Impact models |
| Session router | services/session_router.py | CME futures | ✅ Trading hours |

### Новые MarketType для добавления

```python
# adapters/models.py - расширение
class MarketType(str, Enum):
    # Существующие
    CRYPTO_SPOT = "CRYPTO_SPOT"
    CRYPTO_FUTURES = "CRYPTO_FUTURES"
    CRYPTO_PERP = "CRYPTO_PERP"
    EQUITY = "EQUITY"
    EQUITY_OPTIONS = "EQUITY_OPTIONS"
    FOREX = "FOREX"

    # НОВЫЕ для унифицированных фьючерсов
    INDEX_FUTURES = "INDEX_FUTURES"        # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "COMMODITY_FUTURES" # GC, CL, SI, NG
    CURRENCY_FUTURES = "CURRENCY_FUTURES"   # 6E, 6J, 6B, 6A
    BOND_FUTURES = "BOND_FUTURES"          # ZB, ZN, ZF (Treasury)
```

### Новые ExchangeVendor

```python
# adapters/models.py - расширение
class ExchangeVendor(str, Enum):
    # Существующие
    BINANCE = "binance"
    ALPACA = "alpaca"
    OANDA = "oanda"

    # НОВЫЕ для CME фьючерсов
    INTERACTIVE_BROKERS = "interactive_brokers"  # TWS API
    TRADOVATE = "tradovate"                       # Alternative CME
    NINJATRADER = "ninjatrader"                   # Alternative CME
```

---

## 🏗️ УНИФИЦИРОВАННАЯ АРХИТЕКТУРА

### Принцип: Общее ядро + Vendor-специфичные адаптеры

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: UNIFIED CORE MODELS                         │
│  core_futures.py - Универсальные модели для ВСЕХ типов фьючерсов            │
│  ├── FuturesContract (symbol, expiry, multiplier, tick_size, margin_req)    │
│  ├── FuturesPosition (qty, entry, leverage, margin_mode, unrealized_pnl)    │
│  ├── MarginRequirement (initial, maintenance, variation)                    │
│  ├── SettlementInfo (type, date, price, method)                             │
│  └── ContractRollover (from_contract, to_contract, roll_date, adjustment)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 1: VENDOR ADAPTERS                             │
│                                                                              │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
│  │   BINANCE FUTURES   │ │ INTERACTIVE BROKERS │ │     TRADOVATE       │   │
│  │  (Crypto Perpetual) │ │   (CME/COMEX/ICE)   │ │    (CME Alternative)│   │
│  ├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤   │
│  │ market_data.py      │ │ market_data.py      │ │ market_data.py      │   │
│  │ exchange_info.py    │ │ exchange_info.py    │ │ exchange_info.py    │   │
│  │ order_execution.py  │ │ order_execution.py  │ │ order_execution.py  │   │
│  │ funding_rates.py    │ │ settlement.py       │ │ settlement.py       │   │
│  │ liquidation.py      │ │ margin_req.py       │ │ margin_req.py       │   │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: UNIFIED IMPLEMENTATION                           │
│                                                                              │
│  impl_futures_margin.py      - Margin calculator (ALL types)                │
│  ├── CryptoMarginCalculator     (Tiered brackets, isolated/cross)           │
│  ├── CMEMarginCalculator        (SPAN margin, performance bonds)            │
│  └── UnifiedMarginInterface     (Common API for all)                        │
│                                                                              │
│  impl_futures_settlement.py  - Settlement & Rollover                        │
│  ├── CryptoSettlement           (Funding payments)                          │
│  ├── CMESettlement              (Daily settlement, expiry)                  │
│  └── RolloverManager            (Contract roll handling)                    │
│                                                                              │
│  impl_futures_liquidation.py - Liquidation engine                           │
│  ├── CryptoLiquidation          (Mark price, insurance fund, ADL)           │
│  └── CMELiquidation             (Margin call → forced close)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 3: SERVICES                                    │
│                                                                              │
│  services/futures_risk_guards.py                                            │
│  ├── LeverageGuard              - Max leverage enforcement                  │
│  ├── MarginGuard                - Margin ratio monitoring                   │
│  ├── ConcentrationGuard         - Position limits                           │
│  ├── FundingExposureGuard       - Crypto: funding rate risk                 │
│  └── ExpirationGuard            - CME: contract expiry warnings             │
│                                                                              │
│  services/futures_position_manager.py                                       │
│  ├── PositionTracker            - Real-time position state                  │
│  ├── RolloverScheduler          - Auto-roll near expiry                     │
│  └── PnLCalculator              - Mark-to-market P&L                        │
│                                                                              │
│  services/futures_calendar.py                                               │
│  ├── CMETradingCalendar         - Trading hours, holidays                   │
│  ├── ExpirationCalendar         - Contract expiry dates                     │
│  └── RolloverCalendar           - Standard roll dates                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: EXECUTION PROVIDERS                              │
│                                                                              │
│  execution_providers_futures.py                                             │
│  ├── L2: FuturesParametricSlippageProvider                                  │
│  │   ├── CryptoFuturesSlippage  (Funding impact, liquidation cascade)       │
│  │   ├── IndexFuturesSlippage   (ES/NQ: high liquidity model)               │
│  │   └── CommodityFuturesSlippage (GC/CL: seasonal patterns)                │
│  │                                                                          │
│  └── L3: FuturesL3ExecutionProvider                                         │
│      ├── Uses existing lob/matching_engine.py                               │
│      ├── Liquidation order injection                                        │
│      └── Daily settlement simulation                                        │
│                                                                              │
│  futures_features.py - Type-specific features                               │
│  ├── CryptoFuturesFeatures      (funding_rate, oi, basis, liquidations)     │
│  ├── IndexFuturesFeatures       (roll_yield, term_structure, vix_corr)      │
│  └── CommodityFuturesFeatures   (contango, backwardation, seasonality)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 5: ENTRY POINTS                                │
│                                                                              │
│  script_futures_backtest.py     - Unified backtest (all futures types)      │
│  script_futures_live.py         - Unified live trading                      │
│  train_model_multi_patch.py     - Extended with futures support             │
│                                                                              │
│  configs/                                                                   │
│  ├── config_train_crypto_futures.yaml                                       │
│  ├── config_train_index_futures.yaml                                        │
│  ├── config_train_commodity_futures.yaml                                    │
│  └── config_live_futures.yaml                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vendor-Specific Considerations

#### Crypto Futures (Binance)
```
Уникальные особенности:
├── Perpetual contracts (no expiry)
├── Funding rate payments (8h intervals)
├── Insurance fund + ADL mechanism
├── Mark price = TWAP of index
├── Cross/Isolated margin modes
├── Tiered leverage brackets
└── 24/7 trading
```

#### CME Index Futures (ES, NQ)
```
Уникальные особенности:
├── Quarterly expiration (3rd Friday)
├── Daily settlement at 4pm ET
├── SPAN margin methodology
├── Price limits / circuit breakers
├── Cash settlement
├── Micro contracts available (MES, MNQ)
└── 23/5 trading with maintenance windows
```

#### Commodity Futures (GC, CL)
```
Уникальные особенности:
├── Monthly expiration
├── Physical delivery option (most roll before)
├── Contango/Backwardation dynamics
├── Storage cost implications
├── Seasonal patterns (heating oil, natural gas)
└── Different tick values by product
```

#### Currency Futures (6E, 6J)
```
Уникальные особенности:
├── Quarterly expiration
├── Cash settled in USD
├── Inverse relationship to forex spot
├── High leverage (up to 50x)
└── Correlation with forex markets
```

---

## 📅 ФАЗЫ РЕАЛИЗАЦИИ (ОБНОВЛЁННЫЕ)

### Структура фаз

```
┌─────────────────────────────────────────────────────────────────────┐
│  FOUNDATION (Phases 0-2)                                            │
│  ├── Phase 0: Research & API Analysis (All vendors)                │
│  ├── Phase 1: Unified Core Models                                   │
│  └── Phase 2: Unified Margin & Settlement Interfaces                │
├─────────────────────────────────────────────────────────────────────┤
│  CRYPTO FUTURES TRACK (Phases 3A-6A)                                │
│  ├── Phase 3A: Binance Futures Adapters                            │
│  ├── Phase 4A: Funding Rate & Liquidation (Crypto)                 │
│  ├── Phase 5A: L2/L3 Execution (Crypto)                            │
│  └── Phase 6A: Crypto Futures Features & Training                   │
├─────────────────────────────────────────────────────────────────────┤
│  CME FUTURES TRACK (Phases 3B-6B) - Can run in parallel             │
│  ├── Phase 3B: Interactive Brokers Adapters                        │
│  ├── Phase 4B: SPAN Margin & Daily Settlement                      │
│  ├── Phase 5B: L2/L3 Execution (CME)                               │
│  └── Phase 6B: CME Futures Features & Training                      │
├─────────────────────────────────────────────────────────────────────┤
│  INTEGRATION (Phases 7-10)                                          │
│  ├── Phase 7: Unified Risk Management                              │
│  ├── Phase 8: Multi-Futures Training Pipeline                      │
│  ├── Phase 9: Unified Live Trading                                 │
│  └── Phase 10: Validation & Documentation                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Временная линия

| Фаза | Название | Длительность | Зависимости |
|------|----------|--------------|-------------|
| 0 | Research | 1 week | - |
| 1 | Core Models | 1 week | Phase 0 |
| 2 | Margin/Settlement Interfaces | 1 week | Phase 1 |
| 3A | Binance Adapters | 2 weeks | Phase 2 |
| 3B | IB Adapters | 2 weeks | Phase 2 |
| 4A | Crypto Funding/Liquidation | 1.5 weeks | Phase 3A |
| 4B | CME SPAN/Settlement | 1.5 weeks | Phase 3B |
| 5A | Crypto L2/L3 | 1.5 weeks | Phase 4A |
| 5B | CME L2/L3 | 1.5 weeks | Phase 4B |
| 6A | Crypto Features | 1 week | Phase 5A |
| 6B | CME Features | 1 week | Phase 5B |
| 7 | Risk Management | 1.5 weeks | Phase 6A, 6B |
| 8 | Training Pipeline | 2 weeks | Phase 7 |
| 9 | Live Trading | 1.5 weeks | Phase 8 |
| 10 | Validation | 2 weeks | Phase 9 |

**Общая длительность**: ~14-16 недель (с параллельными tracks)

---

## 📦 PHASE 0: RESEARCH & FOUNDATION

### Цели
- Детальное изучение API всех целевых бирж/брокеров
- Анализ существующего кода для переиспользования
- Определение унифицированной архитектуры
- Сбор референсных данных

### Задачи

#### 0.1 Binance Futures API Analysis (Crypto Track)
```
Endpoints to study:
├── Market Data
│   ├── GET /fapi/v1/klines - Candlesticks
│   ├── GET /fapi/v1/markPriceKlines - Mark price candles
│   ├── GET /fapi/v1/depth - Order book
│   ├── GET /fapi/v1/ticker/24hr - 24h stats
│   ├── GET /fapi/v1/fundingRate - Funding history
│   ├── GET /fapi/v1/premiumIndex - Mark price + funding
│   └── WebSocket streams (aggTrade, markPrice, forceOrder)
├── Account/Trade
│   ├── GET /fapi/v2/account - Account info (margin, positions)
│   ├── GET /fapi/v2/positionRisk - Position risk
│   ├── POST /fapi/v1/order - New order
│   ├── POST /fapi/v1/leverage - Set leverage
│   └── POST /fapi/v1/marginType - Set margin mode
└── Risk
    ├── GET /fapi/v1/adlQuantile - ADL indicator
    └── GET /fapi/v1/forceOrders - Liquidation orders
```

#### 0.2 Interactive Brokers TWS API Analysis (CME Track)
```
TWS API Components to study:
├── Market Data
│   ├── reqMktData() - Real-time quotes
│   ├── reqHistoricalData() - Historical bars
│   ├── reqRealTimeBars() - 5-second bars
│   ├── reqMktDepth() - Level 2 order book
│   └── reqContractDetails() - Contract specifications
├── Orders
│   ├── placeOrder() - Submit orders
│   ├── reqOpenOrders() - Open orders
│   ├── reqPositions() - Current positions
│   ├── reqAccountUpdates() - Account state (margin, PnL)
│   └── Order types: LMT, MKT, STP, TRAIL, BRACKET
├── Contract Definitions
│   ├── FUT - Futures contracts
│   ├── CONTFUT - Continuous futures (auto-roll)
│   ├── expiry, multiplier, exchange
│   └── Symbol mapping: ES, NQ, GC, CL, 6E, ZB
└── Risk & Margin
    ├── reqAccountSummary() - Margin requirements
    ├── Initial Margin, Maintenance Margin
    └── Excess Liquidity, Buying Power

Key Differences from Binance:
├── No funding rates (daily settlement instead)
├── SPAN margin instead of tiered brackets
├── Contract expiration (quarterly)
├── Trading hours: 23/5 with maintenance window
├── Different tick sizes per product
└── Physical vs cash settlement options
```

#### 0.3 CME Contract Specifications
```
Index Futures (via IB):
├── ES (E-mini S&P 500)
│   ├── Exchange: CME
│   ├── Multiplier: $50
│   ├── Tick: 0.25 = $12.50
│   ├── Expiry: Mar/Jun/Sep/Dec
│   ├── Settlement: Cash (3rd Friday)
│   └── Margin: ~5-6% initial
├── NQ (E-mini Nasdaq-100)
│   ├── Multiplier: $20
│   ├── Tick: 0.25 = $5.00
│   └── Similar to ES
├── MES/MNQ (Micro E-mini)
│   ├── Multiplier: $5/$2 (1/10th of E-mini)
│   └── Lower capital requirement

Commodity Futures:
├── GC (Gold)
│   ├── Exchange: COMEX
│   ├── Multiplier: 100 oz
│   ├── Tick: 0.10 = $10.00
│   ├── Expiry: Feb/Apr/Jun/Aug/Oct/Dec
│   └── Settlement: Physical (most roll before)
├── CL (Crude Oil)
│   ├── Exchange: NYMEX
│   ├── Multiplier: 1,000 barrels
│   ├── Tick: 0.01 = $10.00
│   └── Monthly expiration

Currency Futures:
├── 6E (Euro FX)
│   ├── Exchange: CME
│   ├── Multiplier: €125,000
│   ├── Tick: 0.00005 = $6.25
│   ├── Expiry: Mar/Jun/Sep/Dec
│   └── Settlement: Cash
├── 6J (Japanese Yen)
│   ├── Multiplier: ¥12,500,000
│   └── Inverse relationship to USD/JPY
```

#### 0.4 Key Concepts Documentation

**Crypto Futures (Binance) Concepts:**
- **Mark Price**: TWAP of index price + funding basis
- **Index Price**: Weighted average from multiple exchanges
- **Funding Rate**: `(Mark Price - Index Price) / Index Price` + premium (каждые 8ч)
- **Liquidation Price**: `Entry * (1 - IM% + MM%)`
- **ADL (Auto-Deleveraging)**: Forced position close when insurance fund depleted

**CME Futures Concepts:**
- **Settlement Price**: Daily settlement at 4:00pm ET (used for margin)
- **SPAN Margin**: Portfolio-based margin (offsets between correlated products)
- **Initial Margin**: Required to open position (~5-10% notional)
- **Maintenance Margin**: Min to keep position (~80% of initial)
- **Variation Margin**: Daily P&L settlement
- **Circuit Breakers**: 7%/13%/20% limit down for indices
- **Roll Date**: Standard roll ~8 days before expiry

#### 0.5 Existing Code Audit
```bash
# Files to review for reuse
adapters/binance/*.py          # Binance integration (crypto futures base)
adapters/alpaca/*.py           # Equity integration (session/hours reference)
ingest_funding_mark.py         # Funding rate ingestion (crypto)
services/forex_risk_guards.py  # Leverage/margin reference (all futures)
services/session_router.py     # Trading hours handling (CME futures)
lob/*.py                       # L3 LOB simulation (reuse for all)
lob/us_market_structure.py     # Tick sizes, circuit breakers (CME)
execution_providers.py         # L2 TCA (extend for futures)
```

### Deliverables Phase 0
- [ ] Binance Futures API documentation summary
- [ ] IB TWS API documentation summary
- [ ] CME contract specifications database
- [ ] Existing code compatibility report
- [ ] Unified architecture decision record (ADR)
- [ ] Test data collection plan (both crypto + CME)

### Tests
```bash
# No code changes in Phase 0 - documentation only
# Verify existing tests still pass
pytest tests/ -v --tb=short

# Verify IB connectivity (optional, requires IB account)
python -c "from ib_insync import IB; ib = IB(); print('IB available')"
```

---

## 📦 PHASE 1: UNIFIED CORE MODELS

### Цели
- Создать унифицированные futures модели для ВСЕХ типов (crypto, index, commodity, currency)
- Абстрагировать vendor-specific детали в адаптерах
- Обеспечить backward compatibility с существующим кодом

### Ключевой принцип
```
Унифицированные модели НЕ зависят от вендора.
Различия между Binance и CME инкапсулированы в адаптерах.
```

### 1.1 Unified Core Models (`core_futures.py`)

```python
# NEW FILE: core_futures.py
"""
Unified futures trading models for ALL futures types.

Supports:
- Crypto Perpetual (Binance USDT-M, Bybit)
- Crypto Quarterly (Binance delivery)
- Index Futures (CME: ES, NQ, YM, RTY)
- Commodity Futures (COMEX: GC, SI; NYMEX: CL, NG)
- Currency Futures (CME: 6E, 6J, 6B, 6A)
- Bond Futures (CBOT: ZB, ZN, ZF)

Design: Immutable dataclasses, Decimal precision, UTC milliseconds.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List

# ============================================================================
# ENUMS (Unified across all futures types)
# ============================================================================

class FuturesType(str, Enum):
    """Unified futures type classification."""
    # Crypto
    CRYPTO_PERPETUAL = "CRYPTO_PERPETUAL"    # No expiration, funding rate
    CRYPTO_QUARTERLY = "CRYPTO_QUARTERLY"    # Quarterly expiry
    # CME/CBOT/COMEX/NYMEX
    INDEX_FUTURES = "INDEX_FUTURES"          # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "COMMODITY_FUTURES"  # GC, CL, SI, NG
    CURRENCY_FUTURES = "CURRENCY_FUTURES"    # 6E, 6J, 6B
    BOND_FUTURES = "BOND_FUTURES"            # ZB, ZN, ZF

class ContractType(str, Enum):
    """Contract expiration type."""
    PERPETUAL = "PERPETUAL"          # No expiration (crypto only)
    CURRENT_MONTH = "CURRENT_MONTH"  # Monthly contracts (CL, NG)
    CURRENT_QUARTER = "CURRENT_QUARTER"
    NEXT_QUARTER = "NEXT_QUARTER"
    BACK_MONTH = "BACK_MONTH"        # Further out months
    CONTINUOUS = "CONTINUOUS"        # Auto-rolling continuous contract

class SettlementType(str, Enum):
    """Settlement method."""
    CASH = "CASH"                    # Cash settlement (ES, NQ, 6E)
    PHYSICAL = "PHYSICAL"            # Physical delivery (GC, CL)
    FUNDING = "FUNDING"              # Funding rate (crypto perpetual)

class MarginMode(str, Enum):
    """Margin mode for position."""
    CROSS = "CROSS"        # Shared margin (Binance cross, IB portfolio)
    ISOLATED = "ISOLATED"  # Per-position margin (Binance isolated)
    SPAN = "SPAN"          # CME SPAN margin (portfolio-based)

class PositionSide(str, Enum):
    """Position side (for hedge mode)."""
    BOTH = "BOTH"    # One-way mode (net position)
    LONG = "LONG"    # Hedge mode long
    SHORT = "SHORT"  # Hedge mode short

class Exchange(str, Enum):
    """Exchange where contract trades."""
    BINANCE = "BINANCE"
    CME = "CME"           # E-mini indices
    COMEX = "COMEX"       # Gold, Silver
    NYMEX = "NYMEX"       # Oil, Natural Gas
    CBOT = "CBOT"         # Treasuries
    ICE = "ICE"           # Brent, Coffee

# ============================================================================
# CONTRACT SPECIFICATION (Works for ALL futures types)
# ============================================================================

@dataclass(frozen=True)
class FuturesContractSpec:
    """
    Unified futures contract specification.

    Works for Crypto (Binance) AND CME futures.
    """
    # Core identification
    symbol: str                           # BTCUSDT, ES, GC, 6E
    futures_type: FuturesType             # Classification
    contract_type: ContractType           # Perpetual, quarterly, monthly
    exchange: Exchange                    # Where it trades

    # Asset identifiers
    base_asset: str                       # BTC, SPX, Gold, EUR
    quote_asset: str                      # USDT, USD
    margin_asset: str                     # USDT, USD

    # Contract sizing
    contract_size: Decimal = Decimal("1")        # Units per contract
    multiplier: Decimal = Decimal("1")           # Price multiplier (ES=$50, GC=100oz)
    tick_size: Decimal = Decimal("0.01")         # Minimum price increment
    tick_value: Decimal = Decimal("0.01")        # Value per tick (ES=$12.50)

    # Position limits
    min_qty: Decimal = Decimal("0.001")
    max_qty: Decimal = Decimal("1000")
    lot_size: Decimal = Decimal("1")             # Order increment

    # Margin requirements
    max_leverage: int = 125                      # Crypto: 125x, ES: ~20x, GC: ~10x
    initial_margin_pct: Decimal = Decimal("5.0")   # % of notional
    maint_margin_pct: Decimal = Decimal("4.0")     # % of notional

    # Settlement
    settlement_type: SettlementType = SettlementType.CASH
    delivery_date: Optional[str] = None          # For expiring contracts (YYYYMMDD)
    last_trading_day: Optional[str] = None       # Last day to trade

    # Fees (vendor-specific, but defaults provided)
    liquidation_fee_pct: Decimal = Decimal("0.5")  # Crypto: 0.5%, CME: N/A

    # Trading hours
    trading_hours: str = "24/7"                  # "24/7", "23/5", "RTH only"

    @property
    def notional_per_contract(self) -> Decimal:
        """Calculate notional value per contract."""
        # For index futures: price * multiplier
        # For crypto: price * contract_size
        return self.multiplier * self.contract_size

    @property
    def is_perpetual(self) -> bool:
        """Check if perpetual (no expiry)."""
        return self.contract_type == ContractType.PERPETUAL

    @property
    def is_crypto(self) -> bool:
        """Check if crypto futures."""
        return self.futures_type in (
            FuturesType.CRYPTO_PERPETUAL,
            FuturesType.CRYPTO_QUARTERLY
        )

    @property
    def uses_funding(self) -> bool:
        """Check if uses funding rate instead of settlement."""
        return self.settlement_type == SettlementType.FUNDING

@dataclass(frozen=True)
class FuturesPosition:
    """Current futures position state."""
    symbol: str
    side: PositionSide
    entry_price: Decimal
    qty: Decimal  # Positive for long, negative for short
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")  # Isolated margin amount
    maint_margin: Decimal = Decimal("0")
    timestamp_ms: int = 0

@dataclass(frozen=True)
class FundingPayment:
    """Funding rate payment record."""
    symbol: str
    timestamp_ms: int
    funding_rate: Decimal  # e.g., 0.0001 = 0.01%
    mark_price: Decimal
    position_qty: Decimal
    payment_amount: Decimal  # Positive = received, negative = paid
    asset: str = "USDT"

@dataclass(frozen=True)
class LiquidationEvent:
    """Liquidation event record."""
    symbol: str
    timestamp_ms: int
    side: str  # "BUY" or "SELL" (closing side)
    qty: Decimal
    price: Decimal
    liquidation_type: str  # "partial" or "full"
    loss_amount: Decimal
    insurance_fund_contribution: Decimal = Decimal("0")

@dataclass(frozen=True)
class FuturesAccountState:
    """Futures account state snapshot."""
    timestamp_ms: int
    total_wallet_balance: Decimal
    total_margin_balance: Decimal
    total_unrealized_pnl: Decimal
    available_balance: Decimal
    total_initial_margin: Decimal
    total_maint_margin: Decimal
    total_position_initial_margin: Decimal
    total_open_order_initial_margin: Decimal
    max_withdraw_amount: Decimal
    positions: Dict[str, FuturesPosition] = field(default_factory=dict)
```

### 1.2 Futures Market Data Adapter

```python
# NEW FILE: adapters/binance/futures_market_data.py
"""
Binance Futures market data adapter.

Extends BinanceMarketDataAdapter with futures-specific endpoints.
"""

from adapters.binance.market_data import BinanceMarketDataAdapter
from adapters.models import ExchangeVendor, MarketType

class BinanceFuturesMarketDataAdapter(BinanceMarketDataAdapter):
    """
    Binance USDT-M Futures market data adapter.

    Features:
    - Mark price candles (markPriceKlines)
    - Funding rate history
    - Open interest
    - Liquidation stream
    - Premium index
    """

    def __init__(self, config: Optional[Mapping] = None):
        super().__init__(
            vendor=ExchangeVendor.BINANCE,
            config={**(config or {}), "use_futures": True}
        )
        self._market_type = MarketType.CRYPTO_FUTURES

    def get_mark_price_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """Fetch mark price candlesticks."""
        # Uses /fapi/v1/markPriceKlines
        pass

    def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[FundingPayment]:
        """Fetch historical funding rates."""
        pass

    def get_current_funding_rate(self, symbol: str) -> FundingRateInfo:
        """Get current funding rate and next funding time."""
        pass

    def get_open_interest(self, symbol: str) -> OpenInterestInfo:
        """Get current open interest."""
        pass

    def stream_liquidations(
        self,
        symbols: Sequence[str],
    ) -> Iterator[LiquidationEvent]:
        """Stream liquidation orders (forceOrder stream)."""
        pass

    def stream_mark_price(
        self,
        symbols: Sequence[str],
        update_speed: str = "1s",  # "1s" or "3s"
    ) -> Iterator[MarkPriceTick]:
        """Stream real-time mark prices."""
        pass
```

### 1.3 Futures Exchange Info Adapter

```python
# NEW FILE: adapters/binance/futures_exchange_info.py
"""
Binance Futures exchange info adapter.
"""

class BinanceFuturesExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Futures-specific exchange info.

    Provides:
    - Contract specifications
    - Leverage brackets
    - Maintenance margin rates
    - Trading rules
    """

    def get_contract_spec(self, symbol: str) -> FuturesContractSpec:
        """Get contract specification."""
        pass

    def get_leverage_brackets(self, symbol: str) -> List[LeverageBracket]:
        """
        Get leverage brackets with maintenance margin rates.

        Example:
        [
            {"bracket": 1, "notionalCap": 10000, "maintMarginRatio": 0.0040, "maxLeverage": 125},
            {"bracket": 2, "notionalCap": 50000, "maintMarginRatio": 0.0050, "maxLeverage": 100},
            ...
        ]
        """
        pass

    def get_all_contracts(self) -> List[FuturesContractSpec]:
        """Get all tradeable futures contracts."""
        pass
```

### 1.4 Futures Order Execution Adapter

```python
# NEW FILE: adapters/binance/futures_order_execution.py
"""
Binance Futures order execution adapter.
"""

class BinanceFuturesOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Futures order execution with leverage support.

    Features:
    - Set leverage per symbol
    - Set margin mode (cross/isolated)
    - Submit orders with reduceOnly flag
    - Hedge mode support
    """

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol."""
        pass

    def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        """Set margin mode for symbol."""
        pass

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: Decimal,
        price: Optional[Decimal] = None,
        reduce_only: bool = False,
        position_side: PositionSide = PositionSide.BOTH,
        time_in_force: str = "GTC",
    ) -> FuturesOrder:
        """Submit futures order."""
        pass

    def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        """Get current position for symbol."""
        pass

    def get_all_positions(self) -> List[FuturesPosition]:
        """Get all open positions."""
        pass

    def get_account_balance(self) -> FuturesAccountState:
        """Get account state with margin info."""
        pass
```

### 1.5 Registry Updates

```python
# UPDATE: adapters/registry.py
# Add futures adapter registration

from adapters.binance.futures_market_data import BinanceFuturesMarketDataAdapter
from adapters.binance.futures_exchange_info import BinanceFuturesExchangeInfoAdapter
from adapters.binance.futures_order_execution import BinanceFuturesOrderExecutionAdapter

# Register futures adapters
_registry.register(
    vendor=ExchangeVendor.BINANCE,
    adapter_type=AdapterType.MARKET_DATA,
    adapter_class=BinanceFuturesMarketDataAdapter,
    market_type=MarketType.CRYPTO_FUTURES,
    description="Binance USDT-M Futures market data",
)

# Factory function update
def create_market_data_adapter(
    vendor: str,
    config: Optional[Dict] = None,
    market_type: Optional[MarketType] = None,
) -> MarketDataAdapter:
    """Create market data adapter with futures support."""
    v = ExchangeVendor(vendor.lower())

    # Determine market type
    if market_type is None:
        market_type = config.get("market_type") if config else None
        if market_type is None:
            market_type = v.market_type  # Default for vendor

    # Select appropriate adapter
    if v == ExchangeVendor.BINANCE:
        if market_type in (MarketType.CRYPTO_FUTURES, MarketType.CRYPTO_PERP):
            return BinanceFuturesMarketDataAdapter(config)
        return BinanceMarketDataAdapter(config)
    # ... other vendors
```

### Tests for Phase 1

```python
# NEW FILE: tests/test_futures_core_models.py
"""Tests for futures core models."""

import pytest
from decimal import Decimal
from core_futures import (
    ContractType, MarginMode, PositionSide,
    FuturesContractSpec, FuturesPosition, FundingPayment,
)

class TestFuturesContractSpec:
    """Contract specification tests."""

    def test_perpetual_contract_creation(self):
        spec = FuturesContractSpec(
            symbol="BTCUSDT",
            contract_type=ContractType.PERPETUAL,
            base_asset="BTC",
            quote_asset="USDT",
            margin_asset="USDT",
            max_leverage=125,
        )
        assert spec.contract_type == ContractType.PERPETUAL
        assert spec.max_leverage == 125

    def test_quarterly_contract_with_delivery(self):
        spec = FuturesContractSpec(
            symbol="BTCUSDT_240329",
            contract_type=ContractType.CURRENT_QUARTER,
            base_asset="BTC",
            quote_asset="USDT",
            margin_asset="USDT",
            delivery_date="2024-03-29",
        )
        assert spec.delivery_date is not None

class TestFuturesPosition:
    """Position state tests."""

    def test_long_position(self):
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0.1"),
            leverage=20,
            margin_mode=MarginMode.CROSS,
        )
        assert pos.qty > 0

    def test_short_position_negative_qty(self):
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-0.1"),
            leverage=20,
            margin_mode=MarginMode.ISOLATED,
        )
        assert pos.qty < 0

# NEW FILE: tests/test_futures_adapters.py
"""Tests for futures adapters."""

class TestBinanceFuturesMarketDataAdapter:
    """Futures market data adapter tests."""

    @pytest.fixture
    def adapter(self):
        return BinanceFuturesMarketDataAdapter({"use_futures": True})

    def test_adapter_uses_futures_endpoints(self, adapter):
        assert adapter._use_futures is True

    def test_get_mark_price_bars(self, adapter, mocker):
        # Mock API response
        pass

    def test_get_funding_rate_history(self, adapter, mocker):
        pass
```

### Regression Tests
```bash
# Ensure no regressions in existing functionality
pytest tests/test_binance*.py -v
pytest tests/test_execution_providers*.py -v
pytest tests/test_alpaca*.py -v
```

### Deliverables Phase 1
- [ ] `core_futures.py` - Core futures models
- [ ] `adapters/binance/futures_market_data.py`
- [ ] `adapters/binance/futures_exchange_info.py`
- [ ] `adapters/binance/futures_order_execution.py`
- [ ] Updated `adapters/registry.py`
- [ ] `tests/test_futures_core_models.py` (50+ tests)
- [ ] `tests/test_futures_adapters.py` (40+ tests)

---

## 📦 PHASE 2: MARGIN & LIQUIDATION SYSTEM

### Цели
- Реализовать margin calculation engine
- Создать liquidation simulation
- Поддержка cross и isolated margin

### 2.1 Margin Calculator (`impl_futures_margin.py`)

```python
# NEW FILE: impl_futures_margin.py
"""
Futures margin calculation engine.

Implements Binance USDT-M margin rules:
- Initial margin = Notional / Leverage
- Maintenance margin = Notional * MMR (tiered)
- Liquidation price calculation

References:
- https://www.binance.com/en/support/faq/leverage-and-margin-of-usd%E2%93%A2-m-futures-360033162192
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from core_futures import FuturesPosition, MarginMode, LeverageBracket

class MarginCalculator:
    """
    Futures margin calculator with tiered maintenance margin.

    Key formulas:
    - Initial Margin (IM) = Notional Value / Leverage
    - Maintenance Margin (MM) = Notional Value * MMR
    - Position Margin = max(IM, MM)

    Liquidation Price (Long):
        LP = Entry Price * (1 - Initial Margin Rate + Maintenance Margin Rate)

    Liquidation Price (Short):
        LP = Entry Price * (1 + Initial Margin Rate - Maintenance Margin Rate)
    """

    def __init__(self, leverage_brackets: List[LeverageBracket]):
        """
        Initialize with leverage brackets.

        Args:
            leverage_brackets: Tiered margin requirements from exchange
        """
        self._brackets = sorted(leverage_brackets, key=lambda x: x.notional_cap)

    def get_maintenance_margin_rate(self, notional: Decimal) -> Decimal:
        """
        Get maintenance margin rate for notional value.

        Tiered system - higher notional = higher MMR.
        """
        for bracket in self._brackets:
            if notional <= bracket.notional_cap:
                return bracket.maint_margin_rate
        return self._brackets[-1].maint_margin_rate

    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate initial margin requirement."""
        return notional / Decimal(leverage)

    def calculate_maintenance_margin(
        self,
        notional: Decimal,
    ) -> Decimal:
        """Calculate maintenance margin requirement."""
        mmr = self.get_maintenance_margin_rate(notional)
        return notional * mmr

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        qty: Decimal,
        leverage: int,
        wallet_balance: Decimal,
        margin_mode: MarginMode,
        isolated_margin: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate liquidation price.

        For CROSS margin:
            Entire wallet balance is used

        For ISOLATED margin:
            Only isolated margin for position is used
        """
        is_long = qty > 0
        abs_qty = abs(qty)
        notional = entry_price * abs_qty

        mmr = self.get_maintenance_margin_rate(notional)

        if margin_mode == MarginMode.CROSS:
            # Cross margin uses wallet balance
            available_margin = wallet_balance
        else:
            # Isolated uses only position margin
            available_margin = isolated_margin

        # Margin ratio at liquidation
        # MR = (Wallet Balance + Unrealized PnL) / Maintenance Margin
        # At liquidation, MR = 1

        if is_long:
            # Long: liquidation when price drops
            # UPnL = (Mark Price - Entry) * qty
            # At liquidation: available_margin + UPnL = notional * mmr
            # Mark Price = Entry - (available_margin - notional * mmr) / qty
            liq_price = entry_price - (available_margin - notional * mmr) / abs_qty
        else:
            # Short: liquidation when price rises
            # UPnL = (Entry - Mark Price) * |qty|
            liq_price = entry_price + (available_margin - notional * mmr) / abs_qty

        return max(Decimal("0"), liq_price)

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate current margin ratio.

        MR = (Wallet Balance + Unrealized PnL) / Maintenance Margin
        MR < 1 = Liquidation
        MR < 1.5 = Margin warning
        """
        abs_qty = abs(position.qty)
        notional = mark_price * abs_qty

        # Unrealized PnL
        if position.qty > 0:
            upnl = (mark_price - position.entry_price) * abs_qty
        else:
            upnl = (position.entry_price - mark_price) * abs_qty

        mm = self.calculate_maintenance_margin(notional)

        if mm == 0:
            return Decimal("inf")

        margin_ratio = (wallet_balance + upnl) / mm
        return margin_ratio
```

### 2.2 Liquidation Engine (`impl_futures_liquidation.py`)

```python
# NEW FILE: impl_futures_liquidation.py
"""
Futures liquidation simulation engine.

Simulates liquidation cascade with insurance fund mechanics.

References:
- Binance liquidation: https://www.binance.com/en/support/faq/how-is-the-liquidation-price-calculated-360033525271
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from core_futures import FuturesPosition, LiquidationEvent, MarginMode

class LiquidationEngine:
    """
    Simulates futures liquidation mechanics.

    Liquidation process:
    1. Position margin ratio drops below 1 (MM)
    2. Liquidation order placed at bankruptcy price
    3. If filled above bankruptcy → profit to insurance fund
    4. If filled below bankruptcy → ADL or insurance fund covers
    """

    def __init__(
        self,
        insurance_fund_balance: Decimal = Decimal("1000000"),
        liquidation_fee_rate: Decimal = Decimal("0.005"),  # 0.5%
    ):
        self._insurance_fund = insurance_fund_balance
        self._liquidation_fee_rate = liquidation_fee_rate

    def check_liquidation(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        margin_calculator: MarginCalculator,
    ) -> Optional[LiquidationEvent]:
        """
        Check if position should be liquidated.

        Returns:
            LiquidationEvent if liquidation triggered, None otherwise
        """
        margin_ratio = margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if margin_ratio >= Decimal("1"):
            return None  # No liquidation

        # Calculate liquidation parameters
        abs_qty = abs(position.qty)
        notional = mark_price * abs_qty

        # Liquidation fee
        liq_fee = notional * self._liquidation_fee_rate

        # Loss amount
        if position.qty > 0:
            pnl = (mark_price - position.entry_price) * abs_qty
        else:
            pnl = (position.entry_price - mark_price) * abs_qty

        loss = abs(min(Decimal("0"), pnl)) + liq_fee

        # Insurance fund contribution
        insurance_contribution = Decimal("0")
        if pnl < -wallet_balance:
            # Position went bankrupt - insurance fund covers
            insurance_contribution = min(
                self._insurance_fund,
                abs(pnl) - wallet_balance
            )
            self._insurance_fund -= insurance_contribution

        return LiquidationEvent(
            symbol=position.symbol,
            timestamp_ms=0,  # Set by caller
            side="SELL" if position.qty > 0 else "BUY",
            qty=abs_qty,
            price=mark_price,
            liquidation_type="full",
            loss_amount=loss,
            insurance_fund_contribution=insurance_contribution,
        )

    def simulate_partial_liquidation(
        self,
        position: FuturesPosition,
        target_margin_ratio: Decimal = Decimal("2"),
    ) -> Tuple[Decimal, FuturesPosition]:
        """
        Calculate partial liquidation to restore margin ratio.

        Returns:
            (qty_to_liquidate, remaining_position)
        """
        # Partial liquidation reduces position to restore margin
        pass

    def get_adl_ranking(
        self,
        position: FuturesPosition,
        pnl_percentile: Decimal,
        leverage_percentile: Decimal,
    ) -> int:
        """
        Calculate ADL (Auto-Deleveraging) ranking 1-5.

        Higher ranking = higher priority for ADL.
        Based on profit and leverage.
        """
        # ADL ranking formula from Binance
        score = pnl_percentile * leverage_percentile

        if score >= Decimal("0.8"):
            return 5
        elif score >= Decimal("0.6"):
            return 4
        elif score >= Decimal("0.4"):
            return 3
        elif score >= Decimal("0.2"):
            return 2
        return 1
```

### 2.3 Leverage Brackets Data

```python
# NEW FILE: data/futures/leverage_brackets.json
# Or fetch dynamically from Binance API

{
  "BTCUSDT": [
    {"bracket": 1, "notionalCap": 50000, "maintMarginRate": 0.004, "maxLeverage": 125},
    {"bracket": 2, "notionalCap": 250000, "maintMarginRate": 0.005, "maxLeverage": 100},
    {"bracket": 3, "notionalCap": 1000000, "maintMarginRate": 0.01, "maxLeverage": 50},
    {"bracket": 4, "notionalCap": 10000000, "maintMarginRate": 0.025, "maxLeverage": 20},
    {"bracket": 5, "notionalCap": 20000000, "maintMarginRate": 0.05, "maxLeverage": 10},
    {"bracket": 6, "notionalCap": 50000000, "maintMarginRate": 0.10, "maxLeverage": 5},
    {"bracket": 7, "notionalCap": 100000000, "maintMarginRate": 0.125, "maxLeverage": 4},
    {"bracket": 8, "notionalCap": 200000000, "maintMarginRate": 0.15, "maxLeverage": 3},
    {"bracket": 9, "notionalCap": 300000000, "maintMarginRate": 0.25, "maxLeverage": 2},
    {"bracket": 10, "notionalCap": 9223372036854776000, "maintMarginRate": 0.50, "maxLeverage": 1}
  ],
  "ETHUSDT": [
    {"bracket": 1, "notionalCap": 10000, "maintMarginRate": 0.005, "maxLeverage": 100},
    // ...
  ]
}
```

### Tests for Phase 2

```python
# NEW FILE: tests/test_futures_margin.py

class TestMarginCalculator:
    """Margin calculation tests."""

    def test_initial_margin_125x(self):
        """125x leverage = 0.8% initial margin."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)
        im = calc.calculate_initial_margin(
            notional=Decimal("10000"),
            leverage=125,
        )
        assert im == Decimal("80")  # $80 for $10,000 position

    def test_maintenance_margin_tiered(self):
        """Higher notional = higher MMR."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)

        # Small position: 0.4% MMR
        mm_small = calc.calculate_maintenance_margin(Decimal("10000"))
        assert mm_small == Decimal("40")

        # Large position: 5% MMR
        mm_large = calc.calculate_maintenance_margin(Decimal("15000000"))
        # Uses bracket 6: 10% MMR
        assert mm_large == Decimal("1500000")

    def test_liquidation_price_long(self):
        """Long liquidation price below entry."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)
        liq_price = calc.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("0.1"),  # Long
            leverage=20,
            wallet_balance=Decimal("250"),  # 5% margin
            margin_mode=MarginMode.CROSS,
        )
        assert liq_price < Decimal("50000")

    def test_liquidation_price_short(self):
        """Short liquidation price above entry."""
        calc = MarginCalculator(BTCUSDT_BRACKETS)
        liq_price = calc.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("-0.1"),  # Short
            leverage=20,
            wallet_balance=Decimal("250"),
            margin_mode=MarginMode.CROSS,
        )
        assert liq_price > Decimal("50000")

class TestLiquidationEngine:
    """Liquidation simulation tests."""

    def test_no_liquidation_healthy_margin(self):
        """Position with good margin ratio not liquidated."""
        pass

    def test_liquidation_triggers_at_mmr(self):
        """Liquidation when margin ratio < 1."""
        pass

    def test_insurance_fund_covers_bankruptcy(self):
        """Insurance fund used when position goes bankrupt."""
        pass
```

### Deliverables Phase 2
- [ ] `impl_futures_margin.py` - Margin calculator
- [ ] `impl_futures_liquidation.py` - Liquidation engine
- [ ] `data/futures/leverage_brackets.json` - Bracket data
- [ ] `scripts/fetch_leverage_brackets.py` - Auto-fetch script
- [ ] `tests/test_futures_margin.py` (60+ tests)
- [ ] `tests/test_futures_liquidation.py` (40+ tests)

---

## 📦 PHASE 3A: FUNDING RATE MECHANICS (Crypto Track)

### Цели
- Реализовать funding rate tracking для crypto perpetual
- Симуляция funding payments
- Integration с P&L calculation

**Применимость**: Binance USDT-M Perpetual только. CME futures используют daily settlement (Phase 3B).

### 3.1 Funding Rate Tracker (`impl_futures_funding.py`)

```python
# NEW FILE: impl_futures_funding.py
"""
Futures funding rate mechanics.

Perpetual futures use funding to keep price close to spot index.
Funding paid every 8 hours (00:00, 08:00, 16:00 UTC).

Formula:
    Funding Payment = Position Value * Funding Rate

If Funding Rate > 0:
    Longs pay Shorts
If Funding Rate < 0:
    Shorts pay Longs

References:
- Binance funding: https://www.binance.com/en/support/faq/360033525031
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from datetime import datetime, timezone
from core_futures import FundingPayment, FuturesPosition

# Standard funding times (UTC)
FUNDING_TIMES_UTC = [0, 8, 16]  # 00:00, 08:00, 16:00

class FundingRateTracker:
    """
    Tracks and simulates funding rate payments.

    Features:
    - Historical funding rate storage
    - Payment calculation
    - Next funding time prediction
    - Funding cost estimation
    """

    def __init__(self, funding_history: Optional[List[FundingPayment]] = None):
        self._history: List[FundingPayment] = funding_history or []
        self._pending_payments: List[FundingPayment] = []

    def add_funding_rate(
        self,
        symbol: str,
        timestamp_ms: int,
        rate: Decimal,
        mark_price: Decimal,
    ) -> None:
        """Add funding rate to history."""
        pass

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
        timestamp_ms: int,
    ) -> FundingPayment:
        """
        Calculate funding payment for position.

        Payment = Position Value * Funding Rate
        Position Value = Mark Price * |Qty|

        Returns:
            FundingPayment with positive = received, negative = paid
        """
        abs_qty = abs(position.qty)
        position_value = mark_price * abs_qty

        # Payment amount
        payment = position_value * funding_rate

        # Sign depends on position direction
        if position.qty > 0:  # Long
            # Positive funding = longs pay → negative payment for us
            payment = -payment
        else:  # Short
            # Positive funding = shorts receive → positive payment
            payment = payment

        return FundingPayment(
            symbol=position.symbol,
            timestamp_ms=timestamp_ms,
            funding_rate=funding_rate,
            mark_price=mark_price,
            position_qty=position.qty,
            payment_amount=payment,
        )

    def get_next_funding_time(self, current_ts_ms: int) -> int:
        """Get next funding settlement time."""
        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        current_hour = dt.hour

        for funding_hour in FUNDING_TIMES_UTC:
            if current_hour < funding_hour:
                next_dt = dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
                return int(next_dt.timestamp() * 1000)

        # Next funding is tomorrow 00:00 UTC
        next_dt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(next_dt.timestamp() * 1000)

    def estimate_daily_funding_cost(
        self,
        position: FuturesPosition,
        avg_funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Estimate daily funding cost.

        3 fundings per day = 3 * payment
        """
        single_payment = self.calculate_funding_payment(
            position, avg_funding_rate, mark_price, 0
        )
        return single_payment.payment_amount * 3

    def get_average_funding_rate(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> Decimal:
        """Get average funding rate over period."""
        pass
```

### 3.2 Funding Integration with Trading Env

```python
# Updates to trading_patchnew.py or new futures_env.py

class FuturesTradingEnv:
    """
    Trading environment with funding rate simulation.
    """

    def _apply_funding_if_due(
        self,
        position: FuturesPosition,
        current_ts_ms: int,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Apply funding payment if funding time passed.

        Returns:
            Funding payment amount (positive = received)
        """
        if not self._is_funding_time(current_ts_ms):
            return Decimal("0")

        # Get funding rate for this timestamp
        funding_rate = self._get_funding_rate(
            position.symbol, current_ts_ms
        )

        payment = self._funding_tracker.calculate_funding_payment(
            position, funding_rate, mark_price, current_ts_ms
        )

        # Record payment
        self._funding_payments.append(payment)

        return payment.payment_amount
```

### Tests for Phase 3

```python
# NEW FILE: tests/test_futures_funding.py

class TestFundingRateTracker:
    """Funding rate tests."""

    def test_long_pays_positive_funding(self):
        """Long position pays when funding > 0."""
        tracker = FundingRateTracker()
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),  # 1 BTC long
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        payment = tracker.calculate_funding_payment(
            pos,
            funding_rate=Decimal("0.0001"),  # 0.01%
            mark_price=Decimal("50000"),
            timestamp_ms=0,
        )

        # Long pays: 50000 * 0.0001 = $5
        assert payment.payment_amount == Decimal("-5")

    def test_short_receives_positive_funding(self):
        """Short position receives when funding > 0."""
        tracker = FundingRateTracker()
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-1"),  # 1 BTC short
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        payment = tracker.calculate_funding_payment(
            pos,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=0,
        )

        # Short receives: 50000 * 0.0001 = $5
        assert payment.payment_amount == Decimal("5")

    def test_next_funding_time_calculation(self):
        """Correctly predicts next funding settlement."""
        tracker = FundingRateTracker()

        # At 07:30 UTC, next is 08:00 UTC
        ts = int(datetime(2024, 1, 1, 7, 30, tzinfo=timezone.utc).timestamp() * 1000)
        next_ts = tracker.get_next_funding_time(ts)

        expected = int(datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert next_ts == expected
```

### Deliverables Phase 3A
- [ ] `impl_futures_funding.py` - Funding rate tracker
- [ ] `services/futures_funding_tracker.py` - Service wrapper
- [ ] `scripts/download_funding_history.py` - Historical data download
- [ ] `tests/test_futures_funding.py` (50+ tests)

---

## 📦 PHASE 3B: INTERACTIVE BROKERS ADAPTERS (CME Track)

### Цели
- Реализовать IB TWS API адаптеры для CME futures
- Поддержка ES, NQ, GC, CL, 6E и других контрактов
- Daily settlement вместо funding rate
- Rollover handling для expiring contracts

**Применимость**: CME, COMEX, NYMEX, CBOT futures через Interactive Brokers.

### 3B.1 IB Market Data Adapter (`adapters/ib/market_data.py`)

```python
# NEW FILE: adapters/ib/market_data.py
"""
Interactive Brokers market data adapter for CME futures.

Uses ib_insync library for TWS API connectivity.
Supports real-time quotes, historical bars, and Level 2 depth.
"""

from typing import List, Optional, Iterator, Sequence
from decimal import Decimal
from ib_insync import IB, Future, ContFuture, util
from core_models import Bar, Tick
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor

class IBMarketDataAdapter(MarketDataAdapter):
    """
    Interactive Brokers market data adapter.

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live) or Gateway (4002 paper, 4001 live)
        client_id: Unique client ID
        timeout: Connection timeout seconds
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.INTERACTIVE_BROKERS,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._ib: Optional[IB] = None
        self._host = self._config.get("host", "127.0.0.1")
        self._port = self._config.get("port", 7497)  # Paper trading default
        self._client_id = self._config.get("client_id", 1)

    def _do_connect(self) -> None:
        """Connect to TWS/Gateway."""
        self._ib = IB()
        self._ib.connect(self._host, self._port, clientId=self._client_id)

    def _do_disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    def _create_contract(self, symbol: str, use_continuous: bool = True) -> Future:
        """
        Create IB Future contract.

        Args:
            symbol: Contract symbol (ES, NQ, GC, CL, 6E)
            use_continuous: Use continuous contract (auto-roll)
        """
        # Map common symbols to IB contract details
        contract_map = {
            "ES": {"exchange": "CME", "currency": "USD"},
            "NQ": {"exchange": "CME", "currency": "USD"},
            "YM": {"exchange": "CBOT", "currency": "USD"},
            "RTY": {"exchange": "CME", "currency": "USD"},
            "GC": {"exchange": "COMEX", "currency": "USD"},
            "SI": {"exchange": "COMEX", "currency": "USD"},
            "CL": {"exchange": "NYMEX", "currency": "USD"},
            "NG": {"exchange": "NYMEX", "currency": "USD"},
            "6E": {"exchange": "CME", "currency": "USD"},
            "6J": {"exchange": "CME", "currency": "USD"},
            "6B": {"exchange": "CME", "currency": "USD"},
            "ZB": {"exchange": "CBOT", "currency": "USD"},
            "ZN": {"exchange": "CBOT", "currency": "USD"},
        }

        details = contract_map.get(symbol, {"exchange": "CME", "currency": "USD"})

        if use_continuous:
            return ContFuture(symbol, exchange=details["exchange"])
        else:
            # Specific contract (need expiry)
            return Future(symbol, exchange=details["exchange"])

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
        Fetch historical bars from IB.

        Note: IB has pacing limits - max 60 requests per 10 minutes.
        """
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        # Convert timeframe to IB format
        bar_size = self._convert_timeframe(timeframe)
        duration = self._calculate_duration(limit, timeframe)

        bars_raw = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )

        return [self._parse_ib_bar(symbol, b) for b in bars_raw]

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current quote."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(1)  # Wait for data

        if ticker.bid is None:
            return None

        return Tick(
            ts=int(util.dt(ticker.time).timestamp() * 1000),
            symbol=symbol,
            price=Decimal(str(ticker.last)) if ticker.last else None,
            bid=Decimal(str(ticker.bid)),
            ask=Decimal(str(ticker.ask)),
        )

    def get_contract_details(self, symbol: str) -> FuturesContractSpec:
        """Get contract specification from IB."""
        contract = self._create_contract(symbol, use_continuous=False)
        details = self._ib.reqContractDetails(contract)

        if not details:
            raise ValueError(f"Contract not found: {symbol}")

        d = details[0]
        return FuturesContractSpec(
            symbol=symbol,
            futures_type=self._infer_futures_type(symbol),
            contract_type=ContractType.CURRENT_QUARTER,
            exchange=Exchange(d.contract.exchange),
            base_asset=d.contract.symbol,
            quote_asset="USD",
            margin_asset="USD",
            multiplier=Decimal(str(d.contract.multiplier or 1)),
            tick_size=Decimal(str(d.minTick)),
        )

    @staticmethod
    def _convert_timeframe(timeframe: str) -> str:
        """Convert timeframe to IB bar size."""
        mapping = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
            "30m": "30 mins", "1h": "1 hour", "4h": "4 hours",
            "1d": "1 day",
        }
        return mapping.get(timeframe, "1 hour")
```

### 3B.2 IB Order Execution Adapter (`adapters/ib/order_execution.py`)

```python
# NEW FILE: adapters/ib/order_execution.py
"""
Interactive Brokers order execution adapter.

Supports market, limit, stop, and bracket orders for CME futures.
"""

from ib_insync import IB, Future, MarketOrder, LimitOrder, StopOrder
from typing import Optional, List
from decimal import Decimal
from core_futures import FuturesPosition, FuturesOrder

class IBOrderExecutionAdapter:
    """
    IB futures order execution.

    Features:
    - Market, limit, stop orders
    - Bracket orders (entry + take-profit + stop-loss)
    - Position queries
    - Account margin info
    """

    def __init__(self, ib: IB):
        self._ib = ib

    def get_positions(self) -> List[FuturesPosition]:
        """Get all futures positions."""
        positions = []
        for pos in self._ib.positions():
            if isinstance(pos.contract, Future):
                positions.append(FuturesPosition(
                    symbol=pos.contract.symbol,
                    side=PositionSide.LONG if pos.position > 0 else PositionSide.SHORT,
                    entry_price=Decimal(str(pos.avgCost / pos.contract.multiplier)),
                    qty=Decimal(str(abs(pos.position))),
                    leverage=1,  # IB doesn't expose effective leverage
                    margin_mode=MarginMode.SPAN,
                ))
        return positions

    def get_account_margin(self) -> Dict[str, Decimal]:
        """Get account margin info."""
        summary = self._ib.accountSummary()
        margin_info = {}
        for item in summary:
            if item.tag in (
                "InitMarginReq", "MaintMarginReq",
                "AvailableFunds", "ExcessLiquidity", "BuyingPower"
            ):
                margin_info[item.tag] = Decimal(str(item.value))
        return margin_info

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
    ) -> FuturesOrder:
        """Submit market order."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(action, qty)

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade(trade)

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Decimal,
    ) -> FuturesOrder:
        """Submit limit order."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = LimitOrder(action, qty, float(price))

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade(trade)

    def get_margin_requirement(
        self,
        symbol: str,
        qty: int,
    ) -> Dict[str, Decimal]:
        """Get margin requirement for order."""
        contract = self._create_contract(symbol)
        self._ib.qualifyContracts(contract)

        order = MarketOrder("BUY", qty)
        what_if = self._ib.whatIfOrder(contract, order)

        return {
            "initial_margin": Decimal(str(what_if.initMarginChange)),
            "maint_margin": Decimal(str(what_if.maintMarginChange)),
            "impact_on_margin": Decimal(str(what_if.equityWithLoanChange)),
        }
```

### 3B.3 CME Settlement & Rollover (`impl_cme_settlement.py`)

```python
# NEW FILE: impl_cme_settlement.py
"""
CME daily settlement and contract rollover.

CME futures settle daily at 4:00pm ET.
Rollover occurs ~8 days before contract expiry.
"""

from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List

class CMESettlementEngine:
    """
    CME daily settlement simulation.

    Unlike crypto (funding every 8h), CME settles once daily.
    Variation margin is credited/debited to account.
    """

    SETTLEMENT_TIME_ET = 16  # 4:00pm ET

    def __init__(self):
        self._last_settlement_prices: Dict[str, Decimal] = {}

    def calculate_daily_settlement(
        self,
        position: FuturesPosition,
        settlement_price: Decimal,
        contract_spec: FuturesContractSpec,
    ) -> Decimal:
        """
        Calculate daily variation margin.

        Variation = (Settlement - Previous Settlement) * qty * multiplier
        """
        previous = self._last_settlement_prices.get(
            position.symbol, position.entry_price
        )

        price_change = settlement_price - previous
        variation = price_change * abs(position.qty) * contract_spec.multiplier

        # Adjust sign for short positions
        if position.qty < 0:
            variation = -variation

        self._last_settlement_prices[position.symbol] = settlement_price
        return variation

    def is_settlement_time(self, timestamp_ms: int) -> bool:
        """Check if current time is settlement time."""
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        # Convert to ET (UTC-5 or UTC-4 DST)
        et_hour = (dt.hour - 5) % 24
        return et_hour == self.SETTLEMENT_TIME_ET and dt.minute == 0


class ContractRolloverManager:
    """
    Contract rollover management.

    Standard roll: 8 days before expiry (2nd Thursday before 3rd Friday)
    """

    STANDARD_ROLL_DAYS_BEFORE = 8

    def __init__(self, expiration_calendar: Dict[str, List[date]]):
        """
        Args:
            expiration_calendar: Map of symbol to list of expiration dates
        """
        self._calendar = expiration_calendar

    def should_roll(self, symbol: str, current_date: date) -> bool:
        """Check if contract should be rolled."""
        expiry = self._get_current_expiry(symbol)
        if expiry is None:
            return False

        days_to_expiry = (expiry - current_date).days
        return days_to_expiry <= self.STANDARD_ROLL_DAYS_BEFORE

    def get_roll_info(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[Tuple[str, str, Decimal]]:
        """
        Get rollover information.

        Returns:
            (from_contract, to_contract, roll_adjustment)
            Roll adjustment = front price - back price (for continuous data)
        """
        pass

    def _get_current_expiry(self, symbol: str) -> Optional[date]:
        """Get current front month expiry."""
        expirations = self._calendar.get(symbol, [])
        today = date.today()
        for exp in sorted(expirations):
            if exp > today:
                return exp
        return None
```

### 3B.4 CME Trading Calendar (`services/cme_calendar.py`)

```python
# NEW FILE: services/cme_calendar.py
"""
CME trading calendar and hours.

CME Globex: Sunday 6pm ET - Friday 5pm ET
Daily maintenance: 4:15pm - 4:30pm ET
"""

from datetime import datetime, time, timedelta
from typing import Tuple, Optional
import pytz

ET = pytz.timezone("US/Eastern")

class CMETradingCalendar:
    """
    CME trading hours and holiday calendar.
    """

    # Globex hours (ET)
    GLOBEX_OPEN = time(18, 0)    # Sunday 6pm
    GLOBEX_CLOSE = time(17, 0)   # Friday 5pm
    MAINTENANCE_START = time(16, 15)
    MAINTENANCE_END = time(16, 30)

    # US market holidays (2024-2025)
    HOLIDAYS = {
        "2024-01-01", "2024-01-15", "2024-02-19",
        "2024-05-27", "2024-06-19", "2024-07-04",
        "2024-09-02", "2024-11-28", "2024-12-25",
        # Add 2025...
    }

    def is_trading_hours(self, dt: datetime) -> bool:
        """Check if within trading hours."""
        et_dt = dt.astimezone(ET)
        weekday = et_dt.weekday()
        t = et_dt.time()

        # Saturday: closed
        if weekday == 5:
            return False

        # Sunday: open from 6pm
        if weekday == 6:
            return t >= self.GLOBEX_OPEN

        # Friday: close at 5pm
        if weekday == 4 and t >= self.GLOBEX_CLOSE:
            return False

        # Maintenance window
        if self.MAINTENANCE_START <= t < self.MAINTENANCE_END:
            return False

        return True

    def get_next_open(self, dt: datetime) -> datetime:
        """Get next market open time."""
        pass

    def is_holiday(self, d: date) -> bool:
        """Check if date is a market holiday."""
        return d.isoformat() in self.HOLIDAYS
```

### Tests for Phase 3B

```python
# NEW FILE: tests/test_ib_adapters.py

class TestIBMarketDataAdapter:
    """IB market data tests."""

    def test_contract_creation_es(self):
        """Creates correct ES contract."""
        pass

    def test_historical_bars_fetched(self, mock_ib):
        """Fetches historical data from IB."""
        pass

class TestIBOrderExecutionAdapter:
    """IB order execution tests."""

    def test_market_order_submission(self, mock_ib):
        """Submits market order correctly."""
        pass

    def test_margin_requirement_query(self, mock_ib):
        """Queries margin requirements."""
        pass

class TestCMESettlement:
    """CME settlement tests."""

    def test_daily_variation_margin_long(self):
        """Calculates variation for long position."""
        pass

    def test_daily_variation_margin_short(self):
        """Calculates variation for short position."""
        pass

class TestContractRollover:
    """Rollover tests."""

    def test_roll_date_detection(self):
        """Detects roll date correctly."""
        pass

class TestCMETradingCalendar:
    """Trading hours tests."""

    def test_weekday_trading_hours(self):
        """Mon-Thu within hours."""
        pass

    def test_maintenance_window_closed(self):
        """4:15-4:30 ET closed."""
        pass

    def test_friday_close_at_5pm(self):
        """Friday closes at 5pm ET."""
        pass
```

### Deliverables Phase 3B
- [ ] `adapters/ib/__init__.py` - IB adapter package
- [ ] `adapters/ib/market_data.py` - IB market data adapter
- [ ] `adapters/ib/order_execution.py` - IB order execution
- [ ] `adapters/ib/exchange_info.py` - Contract details
- [ ] `impl_cme_settlement.py` - Daily settlement engine
- [ ] `impl_cme_rollover.py` - Contract rollover manager
- [ ] `services/cme_calendar.py` - Trading calendar
- [ ] `tests/test_ib_adapters.py` (80+ tests)
- [ ] `tests/test_cme_settlement.py` (40+ tests)

### Зависимости
```bash
pip install ib_insync  # IB TWS API wrapper
```

---

## 📦 PHASE 4A: L2 EXECUTION PROVIDER (Crypto Track)

### Цели
- Создать L2 execution provider для futures
- Адаптировать slippage model для futures
- Интеграция с существующей инфраструктурой

### 4.1 Futures Slippage Provider

```python
# NEW FILE: execution_providers_futures.py
"""
Futures execution providers (L2).

Extends crypto parametric TCA with futures-specific factors:
1. Funding rate impact on spread
2. Open interest impact on liquidity
3. Liquidation cascade simulation
4. Mark vs last price execution
"""

from decimal import Decimal
from typing import Optional
from execution_providers import (
    SlippageProvider,
    CryptoParametricSlippageProvider,
    CryptoParametricConfig,
    MarketState,
    Order,
    BarData,
)

class FuturesSlippageProvider(CryptoParametricSlippageProvider):
    """
    Futures-specific slippage model.

    Additional factors beyond crypto spot:
    - Funding rate stress (high funding = directional pressure)
    - Open interest concentration
    - Liquidation cascade risk
    """

    def __init__(
        self,
        config: Optional[FuturesSlippageConfig] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self._funding_impact_sensitivity = 5.0  # Funding rate impact multiplier
        self._oi_concentration_threshold = 0.3  # 30% concentration warning

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        recent_liquidations: Optional[float] = None,  # Volume of recent liquidations
        **kwargs,
    ) -> float:
        """
        Compute futures slippage with additional factors.

        Slippage = base_slippage * funding_stress * liquidation_cascade
        """
        # Base slippage from crypto model
        base_bps = super().compute_slippage_bps(
            order, market, participation_ratio,
            funding_rate=funding_rate, **kwargs
        )

        # Funding stress factor
        # High positive funding + buying = extra cost (crowded long)
        # High negative funding + selling = extra cost (crowded short)
        funding_stress = 1.0
        if funding_rate is not None:
            is_same_direction = (
                (funding_rate > 0 and order.side == "BUY") or
                (funding_rate < 0 and order.side == "SELL")
            )
            if is_same_direction:
                funding_stress = 1.0 + abs(funding_rate) * self._funding_impact_sensitivity * 10000

        # Liquidation cascade factor
        cascade_factor = 1.0
        if recent_liquidations is not None and market.adv is not None:
            liquidation_ratio = recent_liquidations / market.adv
            if liquidation_ratio > 0.01:  # >1% of ADV is liquidations
                cascade_factor = 1.0 + liquidation_ratio * 5  # Up to 50% extra slippage

        return base_bps * funding_stress * cascade_factor

class FuturesL2ExecutionProvider:
    """
    L2 execution provider for futures.

    Combines:
    - FuturesSlippageProvider
    - FuturesFeeProvider (with funding)
    - OHLCVFillProvider (uses mark price option)
    """

    def __init__(
        self,
        use_mark_price: bool = True,  # Use mark price for execution
        **kwargs,
    ):
        self._use_mark_price = use_mark_price
        self._slippage = FuturesSlippageProvider(**kwargs)
        self._fees = FuturesFeeProvider(**kwargs)
        self._fill = OHLCVFillProvider(**kwargs)

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        funding_rate: Optional[float] = None,
        mark_bar: Optional[BarData] = None,  # Mark price bar
    ) -> Fill:
        """Execute order with futures mechanics."""
        # Use mark price bar if available and configured
        exec_bar = mark_bar if (self._use_mark_price and mark_bar) else bar

        # Compute fill
        fill = self._fill.compute_fill(order, market, exec_bar)

        if not fill.is_filled:
            return fill

        # Apply slippage
        slippage_bps = self._slippage.compute_slippage_bps(
            order, market, fill.participation_ratio,
            funding_rate=funding_rate,
        )

        # Adjust price
        fill = fill.with_slippage(slippage_bps)

        # Compute fees
        fee = self._fees.compute_fee(fill)

        return fill.with_fee(fee)
```

### 4.2 Futures Fee Provider

```python
# Part of execution_providers_futures.py

class FuturesFeeProvider(FeeProvider):
    """
    Futures fee provider.

    Includes:
    - Maker/taker fees (same as spot)
    - Funding payments (tracked separately)
    - Liquidation fees (if applicable)
    """

    def __init__(
        self,
        maker_bps: float = 2.0,
        taker_bps: float = 4.0,
        liquidation_fee_bps: float = 50.0,  # 0.5%
        **kwargs,
    ):
        self._maker_bps = maker_bps
        self._taker_bps = taker_bps
        self._liquidation_fee_bps = liquidation_fee_bps

    def compute_fee(
        self,
        fill: Fill,
        is_liquidation: bool = False,
    ) -> Decimal:
        """Compute fee for fill."""
        if is_liquidation:
            return fill.notional * Decimal(self._liquidation_fee_bps) / 10000

        bps = self._maker_bps if fill.is_maker else self._taker_bps
        return fill.notional * Decimal(bps) / 10000
```

### Tests for Phase 4

```python
# NEW FILE: tests/test_futures_execution_providers.py

class TestFuturesSlippageProvider:
    """Futures slippage tests."""

    def test_funding_stress_increases_slippage(self):
        """High funding rate in same direction increases slippage."""
        provider = FuturesSlippageProvider()

        # Base slippage without funding
        base = provider.compute_slippage_bps(
            Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
            MarketState(0, Decimal("50000"), Decimal("50001"), adv=1e9),
            participation_ratio=0.001,
        )

        # With high positive funding (crowded long)
        with_funding = provider.compute_slippage_bps(
            Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
            MarketState(0, Decimal("50000"), Decimal("50001"), adv=1e9),
            participation_ratio=0.001,
            funding_rate=0.001,  # 0.1% = very high
        )

        assert with_funding > base

    def test_liquidation_cascade_impact(self):
        """Liquidation cascade increases slippage."""
        pass

class TestFuturesL2ExecutionProvider:
    """L2 futures provider tests."""

    def test_uses_mark_price_when_configured(self):
        """Provider uses mark price bar for execution."""
        pass

    def test_integration_with_funding(self):
        """Full execution with funding rate tracking."""
        pass
```

### Deliverables Phase 4A
- [ ] `execution_providers_futures.py` - L2 crypto futures providers
- [ ] Updated factory functions in `execution_providers.py`
- [ ] `tests/test_futures_execution_providers.py` (80+ tests)

---

## 📦 PHASE 4B: CME SPAN MARGIN & SLIPPAGE (CME Track)

### Цели
- Реализовать SPAN margin calculation для CME
- Создать slippage модель для index/commodity/currency futures
- Учёт circuit breakers и daily limits

**Применимость**: ES, NQ, GC, CL, 6E и все CME/COMEX/NYMEX контракты.

### 4B.1 SPAN Margin Calculator (`impl_span_margin.py`)

```python
# NEW FILE: impl_span_margin.py
"""
CME SPAN Margin Calculator.

SPAN (Standard Portfolio Analysis of Risk) is a risk-based margin system.
Unlike crypto tiered brackets, SPAN uses portfolio-level risk analysis.

Key differences from crypto:
- Portfolio-based (offsets between correlated positions)
- Scenario-based (16 risk scenarios)
- Separate initial and maintenance margins (~80% of initial)

References:
- CME Group SPAN documentation
"""

from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from core_futures import FuturesPosition, FuturesContractSpec

@dataclass
class SPANMarginRequirement:
    """SPAN margin requirements for a position."""
    initial_margin: Decimal
    maintenance_margin: Decimal
    scanning_risk: Decimal          # Max loss across 16 scenarios
    inter_commodity_spread: Decimal  # Credit for correlated positions
    net_option_value: Decimal = Decimal("0")

class SPANMarginCalculator:
    """
    Simplified SPAN margin calculator.

    Full SPAN uses 16 price/volatility scenarios.
    This implementation uses simplified scanning ranges.

    For production, use CME SPAN files or IB margin queries.
    """

    # Default scanning ranges (% of price move)
    SCANNING_RANGES = {
        "ES": 0.08,   # ±8% price move
        "NQ": 0.10,   # ±10% (more volatile)
        "GC": 0.06,   # ±6%
        "CL": 0.12,   # ±12% (high volatility)
        "6E": 0.04,   # ±4% (currency)
        "ZB": 0.03,   # ±3% (bonds)
    }

    # Intra-commodity spread credits (same product, different months)
    SPREAD_CREDITS = {
        "ES": 0.70,  # 70% credit for calendar spreads
        "GC": 0.75,
        "CL": 0.60,  # Lower credit due to contango/backwardation
    }

    # Inter-commodity spread credits (correlated products)
    INTER_COMMODITY_CREDITS = {
        ("ES", "NQ"): 0.40,   # 40% credit for ES/NQ spread
        ("GC", "SI"): 0.30,   # Gold/Silver
        ("CL", "NG"): 0.20,   # Oil/Nat Gas
    }

    def __init__(
        self,
        contract_specs: Dict[str, FuturesContractSpec],
    ):
        self._specs = contract_specs

    def calculate_margin(
        self,
        position: FuturesPosition,
        current_price: Decimal,
    ) -> SPANMarginRequirement:
        """
        Calculate SPAN margin for single position.

        Simplified: margin = notional * scanning_range
        """
        spec = self._specs.get(position.symbol)
        if spec is None:
            raise ValueError(f"Unknown contract: {position.symbol}")

        # Calculate notional
        abs_qty = abs(position.qty)
        notional = current_price * abs_qty * spec.multiplier

        # Scanning risk (max loss under price scenarios)
        scanning_range = self.SCANNING_RANGES.get(position.symbol, 0.08)
        scanning_risk = notional * Decimal(str(scanning_range))

        # Initial margin (typically ~5-10% of notional for index futures)
        # For CME, initial ≈ scanning_risk + additional buffers
        initial_margin = scanning_risk * Decimal("1.25")

        # Maintenance margin (~80% of initial)
        maintenance_margin = initial_margin * Decimal("0.80")

        return SPANMarginRequirement(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            scanning_risk=scanning_risk,
            inter_commodity_spread=Decimal("0"),
        )

    def calculate_portfolio_margin(
        self,
        positions: List[FuturesPosition],
        prices: Dict[str, Decimal],
    ) -> Tuple[Decimal, Decimal, Dict[str, SPANMarginRequirement]]:
        """
        Calculate SPAN margin for portfolio with spread credits.

        Returns:
            (total_initial, total_maintenance, per_position_margins)
        """
        position_margins = {}
        total_scanning = Decimal("0")

        for pos in positions:
            margin = self.calculate_margin(pos, prices[pos.symbol])
            position_margins[pos.symbol] = margin
            total_scanning += margin.scanning_risk

        # Apply inter-commodity spread credits
        spread_credit = self._calculate_spread_credit(positions, position_margins)

        # Net margin after credits
        net_scanning = max(Decimal("0"), total_scanning - spread_credit)

        total_initial = net_scanning * Decimal("1.25")
        total_maintenance = total_initial * Decimal("0.80")

        return total_initial, total_maintenance, position_margins

    def _calculate_spread_credit(
        self,
        positions: List[FuturesPosition],
        margins: Dict[str, SPANMarginRequirement],
    ) -> Decimal:
        """Calculate spread credits for correlated positions."""
        credit = Decimal("0")

        symbols = [p.symbol for p in positions]

        for (sym1, sym2), credit_pct in self.INTER_COMMODITY_CREDITS.items():
            if sym1 in symbols and sym2 in symbols:
                # Credit = smaller margin * credit_pct
                m1 = margins.get(sym1, SPANMarginRequirement(0,0,0,0)).scanning_risk
                m2 = margins.get(sym2, SPANMarginRequirement(0,0,0,0)).scanning_risk
                credit += min(m1, m2) * Decimal(str(credit_pct))

        return credit
```

### 4B.2 CME Slippage Provider (`execution_providers_cme.py`)

```python
# NEW FILE: execution_providers_cme.py
"""
CME futures slippage provider.

CME index futures (ES, NQ) have high liquidity and tight spreads.
Commodity futures (GC, CL) have seasonal and time-of-day patterns.
Currency futures (6E, 6J) correlate with forex spot markets.
"""

from decimal import Decimal
from typing import Optional
from execution_providers import (
    SlippageProvider,
    EquityParametricSlippageProvider,
    EquityParametricConfig,
    MarketState,
    Order,
)

class CMESlippageConfig:
    """CME-specific slippage configuration."""

    # Default spreads in ticks
    DEFAULT_SPREADS = {
        "ES": Decimal("0.25"),   # 1 tick = $12.50
        "NQ": Decimal("0.25"),   # 1 tick = $5.00
        "YM": Decimal("1.0"),    # 1 tick = $5.00
        "GC": Decimal("0.10"),   # 1 tick = $10.00
        "CL": Decimal("0.01"),   # 1 tick = $10.00
        "6E": Decimal("0.00005"), # 1 tick = $6.25
    }

    # Impact coefficients (similar to equity, but product-specific)
    IMPACT_COEFFICIENTS = {
        "ES": 0.03,   # Very liquid, low impact
        "NQ": 0.04,   # Slightly less liquid than ES
        "GC": 0.06,   # Commodity, moderate impact
        "CL": 0.08,   # More volatile
        "6E": 0.05,   # Currency, moderate
    }

class CMESlippageProvider(EquityParametricSlippageProvider):
    """
    CME futures slippage model.

    Based on EquityParametricSlippageProvider with:
    - Product-specific spreads and impact coefficients
    - Trading hours awareness (Globex vs RTH)
    - Daily settlement time effects
    - Circuit breaker considerations
    """

    def __init__(
        self,
        config: Optional[CMESlippageConfig] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self._cme_config = config or CMESlippageConfig()
        self._circuit_breaker_active = False

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        is_rth: bool = True,           # Regular Trading Hours
        minutes_to_settlement: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Compute CME futures slippage.

        Additional factors:
        - RTH vs ETH (extended hours wider spreads)
        - Settlement time (higher volatility near 4pm ET)
        - Circuit breaker (if active, reject or limit trades)
        """
        if self._circuit_breaker_active:
            return 1000.0  # Effectively block trading

        # Get product-specific parameters
        symbol = order.symbol
        base_spread = float(self._cme_config.DEFAULT_SPREADS.get(
            symbol, Decimal("0.01")
        )) / (market.get_mid_price() or 1) * 10000

        impact_k = self._cme_config.IMPACT_COEFFICIENTS.get(symbol, 0.05)

        # RTH/ETH adjustment
        session_mult = 1.0 if is_rth else 1.5  # 50% wider in ETH

        # Settlement time premium
        settlement_mult = 1.0
        if minutes_to_settlement is not None and minutes_to_settlement < 30:
            settlement_mult = 1.0 + (30 - minutes_to_settlement) / 100  # Up to 30%

        # Base calculation
        base_slippage = base_spread / 2 + impact_k * (participation_ratio ** 0.5) * 10000

        return base_slippage * session_mult * settlement_mult

    def set_circuit_breaker(self, active: bool) -> None:
        """Set circuit breaker state (e.g., limit down)."""
        self._circuit_breaker_active = active


class CMECircuitBreaker:
    """
    CME circuit breaker simulation.

    Levels (for S&P 500 index):
    - Level 1: -7% → 15 min halt
    - Level 2: -13% → 15 min halt
    - Level 3: -20% → trading halted for day
    """

    LEVELS = {
        1: Decimal("-0.07"),   # -7%
        2: Decimal("-0.13"),   # -13%
        3: Decimal("-0.20"),   # -20%
    }

    HALT_DURATIONS = {
        1: 15 * 60,  # 15 minutes
        2: 15 * 60,
        3: None,     # Full day halt
    }

    def check_circuit_breaker(
        self,
        current_price: Decimal,
        reference_price: Decimal,  # Previous day close
    ) -> Optional[int]:
        """
        Check if circuit breaker triggered.

        Returns:
            Circuit breaker level (1, 2, 3) or None
        """
        change_pct = (current_price - reference_price) / reference_price

        for level, threshold in sorted(self.LEVELS.items(), reverse=True):
            if change_pct <= threshold:
                return level

        return None
```

### Tests for Phase 4B

```python
# NEW FILE: tests/test_span_margin.py

class TestSPANMarginCalculator:
    """SPAN margin tests."""

    def test_single_position_margin(self):
        """Calculates margin for single ES position."""
        pass

    def test_portfolio_spread_credit(self):
        """ES/NQ spread gets credit."""
        pass

    def test_maintenance_is_80_percent(self):
        """Maintenance margin ~80% of initial."""
        pass

class TestCMESlippageProvider:
    """CME slippage tests."""

    def test_es_tight_spread(self):
        """ES has tight spread ~0.25 ticks."""
        pass

    def test_eth_wider_spread(self):
        """Extended hours has wider spread."""
        pass

    def test_circuit_breaker_blocks_trading(self):
        """Circuit breaker returns high slippage."""
        pass

class TestCMECircuitBreaker:
    """Circuit breaker tests."""

    def test_level_1_at_7_percent(self):
        """Level 1 triggered at -7%."""
        pass

    def test_level_3_halts_trading(self):
        """Level 3 halts for full day."""
        pass
```

### Deliverables Phase 4B
- [ ] `impl_span_margin.py` - SPAN margin calculator
- [ ] `execution_providers_cme.py` - CME slippage provider
- [ ] `impl_circuit_breaker.py` - Circuit breaker simulation
- [ ] `tests/test_span_margin.py` (50+ tests)
- [ ] `tests/test_cme_slippage.py` (40+ tests)

---

## 📦 PHASE 5A: L3 LOB INTEGRATION (Crypto Track)

### Цели
- Интегрировать L3 LOB с futures mechanics
- Адаптировать queue position для futures
- Liquidation order flow simulation

### 5.1 Futures L3 Execution Provider

```python
# NEW FILE: execution_providers_futures_l3.py
"""
L3 Futures execution provider.

Combines:
- Full LOB simulation
- Liquidation order injection
- Mark price execution reference
- Funding-adjusted queue dynamics
"""

from lob import (
    MatchingEngine,
    QueuePositionTracker,
    AlmgrenChrissModel,
    LatencyModel,
)
from execution_providers_l3 import L3ExecutionProvider

class FuturesL3ExecutionProvider(L3ExecutionProvider):
    """
    L3 execution with futures-specific mechanics.

    Extensions:
    - Liquidation order stream injection
    - Market-wide liquidation cascade
    - Insurance fund dynamics
    """

    def __init__(
        self,
        liquidation_stream: Optional[Iterator[LiquidationEvent]] = None,
        insurance_fund_balance: Decimal = Decimal("1000000"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._liquidation_stream = liquidation_stream
        self._insurance_fund = insurance_fund_balance
        self._pending_liquidations: List[LiquidationEvent] = []

    def inject_liquidation_orders(
        self,
        current_ts_ms: int,
    ) -> List[Fill]:
        """
        Inject liquidation orders into order book.

        Liquidation orders are market orders that must be filled.
        They can cause price cascades.
        """
        fills = []

        for event in self._get_liquidations_up_to(current_ts_ms):
            # Liquidation order
            liq_order = Order(
                symbol=event.symbol,
                side=event.side,
                qty=event.qty,
                order_type="MARKET",
                is_liquidation=True,
            )

            # Execute through matching engine
            fill = self._matching_engine.match(liq_order)
            fills.append(fill)

            # Impact on order book
            self._apply_liquidation_impact(fill)

        return fills

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        mark_bar: Optional[BarData] = None,
    ) -> Fill:
        """
        Execute order with liquidation cascade awareness.
        """
        # First, process any pending liquidations
        self.inject_liquidation_orders(bar.timestamp_ms)

        # Then execute our order
        return super().execute(order, market, bar)
```

### 5.2 Liquidation Cascade Simulator

```python
# Part of execution_providers_futures_l3.py

class LiquidationCascadeSimulator:
    """
    Simulates cascade liquidations.

    When price moves against leveraged positions:
    1. Positions hit liquidation price
    2. Liquidation orders placed
    3. Orders fill, push price further
    4. More positions liquidated (cascade)

    Uses historical liquidation data to calibrate.
    """

    def __init__(
        self,
        cascade_decay: float = 0.7,  # Each wave is 70% of previous
        max_waves: int = 5,
    ):
        self._cascade_decay = cascade_decay
        self._max_waves = max_waves

    def simulate_cascade(
        self,
        initial_liquidation: LiquidationEvent,
        order_book: OrderBook,
        open_positions: List[FuturesPosition],
    ) -> List[LiquidationEvent]:
        """
        Simulate liquidation cascade from initial event.
        """
        all_events = [initial_liquidation]

        for wave in range(self._max_waves):
            # Execute current liquidation
            new_price = self._execute_liquidation(
                all_events[-1], order_book
            )

            # Find new liquidations at this price
            new_liquidations = self._find_liquidations(
                new_price, open_positions
            )

            if not new_liquidations:
                break

            # Apply cascade decay
            for event in new_liquidations:
                event = event._replace(
                    qty=event.qty * Decimal(str(self._cascade_decay ** wave))
                )
                all_events.append(event)

        return all_events
```

### Tests for Phase 5

```python
# NEW FILE: tests/test_futures_l3_execution.py

class TestFuturesL3ExecutionProvider:
    """L3 futures tests."""

    def test_liquidation_injection(self):
        """Liquidation orders injected into order book."""
        pass

    def test_cascade_simulation(self):
        """Cascade liquidation simulation."""
        pass

    def test_insurance_fund_depletion(self):
        """Insurance fund properly tracks contributions."""
        pass
```

### Deliverables Phase 5A
- [ ] `execution_providers_futures_l3.py` - L3 crypto futures provider
- [ ] `lob/futures_extensions.py` - LOB extensions for crypto futures
- [ ] `tests/test_futures_l3_execution.py` (60+ tests)

---

## 📦 PHASE 5B: L3 LOB INTEGRATION (CME Track)

### Цели
- Интегрировать L3 LOB с CME-specific mechanics
- Circuit breaker simulation
- Daily settlement simulation
- Globex matching engine emulation

### 5B.1 CME L3 Execution Provider

```python
# NEW FILE: execution_providers_cme_l3.py
"""
L3 CME Futures execution provider.

Combines:
- Globex-style FIFO matching
- Circuit breaker halt simulation
- Daily settlement time effects
- RTH/ETH liquidity differences
"""

from datetime import datetime, time as dt_time
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Tuple

from lob import (
    MatchingEngine,
    QueuePositionTracker,
    AlmgrenChrissModel,
    LatencyModel,
)
from execution_providers_l3 import L3ExecutionProvider


class MarketState(str, Enum):
    """CME market state."""
    PRE_OPEN = "PRE_OPEN"
    OPENING_AUCTION = "OPENING_AUCTION"
    CONTINUOUS = "CONTINUOUS"
    HALTED = "HALTED"
    CLOSING_AUCTION = "CLOSING_AUCTION"
    CLOSED = "CLOSED"


class CMEL3ExecutionProvider(L3ExecutionProvider):
    """
    L3 execution with CME Globex-specific mechanics.

    Features:
    - Circuit breaker levels (7%, 13%, 20%)
    - Daily settlement at 4pm ET
    - Globex hours (Sun 6pm - Fri 5pm ET)
    - RTH (9:30-4:00 ET) vs ETH liquidity
    - Opening/Closing auction simulation
    """

    def __init__(
        self,
        product: str,
        multiplier: Decimal,
        tick_size: Decimal,
        daily_settle_time: dt_time = dt_time(16, 0),  # 4pm ET
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._product = product
        self._multiplier = multiplier
        self._tick_size = tick_size
        self._daily_settle_time = daily_settle_time

        # Circuit breaker state
        self._circuit_breaker = CMECircuitBreakerSimulator(product)
        self._halt_until_ts: Optional[int] = None

        # Last settlement price for daily mark
        self._last_settlement_price: Optional[Decimal] = None
        self._last_settlement_ts: Optional[int] = None

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Fill:
        """
        Execute order with CME mechanics.
        """
        # Check if market is halted
        if self._is_halted(bar.timestamp_ms):
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                qty=Decimal("0"),
                price=Decimal("0"),
                status="REJECTED",
                reject_reason="MARKET_HALTED",
            )

        # Check circuit breaker
        halt_result = self._circuit_breaker.check_and_trigger(
            current_price=bar.close,
            reference_price=self._last_settlement_price,
            timestamp_ms=bar.timestamp_ms,
        )

        if halt_result.is_triggered:
            self._halt_until_ts = halt_result.resume_time_ms
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                qty=Decimal("0"),
                price=Decimal("0"),
                status="REJECTED",
                reject_reason=f"CIRCUIT_BREAKER_L{halt_result.level}",
            )

        # Get session-adjusted slippage
        session = self._get_trading_session(bar.timestamp_ms)
        if session in (TradingSession.ETH, TradingSession.PRE_OPEN):
            # Wider spreads and higher impact in ETH
            adjusted_slippage = self._adjust_slippage_for_eth(market)
        else:
            adjusted_slippage = market

        # Execute through parent L3 provider
        return super().execute(order, adjusted_slippage, bar)

    def _is_halted(self, timestamp_ms: int) -> bool:
        """Check if market is currently halted."""
        if self._halt_until_ts is None:
            return False
        return timestamp_ms < self._halt_until_ts

    def _get_trading_session(self, timestamp_ms: int) -> TradingSession:
        """Determine current trading session."""
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        hour_et = (dt.hour - 5) % 24  # Approximate ET

        if hour_et >= 9.5 and hour_et < 16:
            return TradingSession.RTH
        elif hour_et >= 16 and hour_et < 17:
            return TradingSession.CLOSING
        elif hour_et >= 18 or hour_et < 9.5:
            return TradingSession.ETH
        else:
            return TradingSession.CLOSED

    def _adjust_slippage_for_eth(
        self,
        market: MarketStateData,
    ) -> MarketStateData:
        """
        Adjust market state for ETH conditions.

        ETH typically has:
        - 2-3x wider spreads
        - 50-70% less depth
        - Higher price impact
        """
        return MarketStateData(
            timestamp=market.timestamp,
            bid=market.bid,
            ask=market.ask,
            spread_bps=market.spread_bps * Decimal("2.5"),  # 2.5x wider
            adv=market.adv * Decimal("0.4"),  # 40% of RTH volume
            depth_bid=market.depth_bid * Decimal("0.5") if market.depth_bid else None,
            depth_ask=market.depth_ask * Decimal("0.5") if market.depth_ask else None,
        )


class TradingSession(str, Enum):
    """CME trading session."""
    PRE_OPEN = "PRE_OPEN"
    RTH = "RTH"  # Regular Trading Hours 9:30-16:00 ET
    ETH = "ETH"  # Extended Trading Hours
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
```

### 5B.2 CME Circuit Breaker Simulator

```python
# Part of execution_providers_cme_l3.py

from dataclasses import dataclass
from typing import NamedTuple


class CircuitBreakerResult(NamedTuple):
    """Result of circuit breaker check."""
    is_triggered: bool
    level: int  # 1, 2, or 3
    resume_time_ms: Optional[int]
    halt_duration_minutes: int


class CMECircuitBreakerSimulator:
    """
    Simulates CME S&P 500 circuit breakers.

    Levels (based on prior day's closing price):
    - Level 1: -7%  → 15-minute halt (RTH only)
    - Level 2: -13% → 15-minute halt (RTH only)
    - Level 3: -20% → Halt for remainder of day

    Note: Halts only apply during RTH (9:30-3:25pm ET).
    After 3:25pm, only Level 3 triggers halt.
    """

    # Circuit breaker levels (percentage decline)
    LEVEL_1_PCT = Decimal("-0.07")
    LEVEL_2_PCT = Decimal("-0.13")
    LEVEL_3_PCT = Decimal("-0.20")

    # Halt durations in minutes
    HALT_DURATION_L1_L2 = 15
    HALT_DURATION_L3 = 999  # Rest of day

    # RTH boundaries for halt applicability
    RTH_START_HOUR = 9.5  # 9:30 AM ET
    RTH_CUTOFF_HOUR = 15.417  # 3:25 PM ET (after this only L3 halts)
    RTH_END_HOUR = 16.0  # 4:00 PM ET

    def __init__(self, product: str):
        self._product = product
        self._triggered_levels: set = set()
        self._last_trigger_ts: Dict[int, int] = {}

    def check_and_trigger(
        self,
        current_price: Decimal,
        reference_price: Optional[Decimal],
        timestamp_ms: int,
    ) -> CircuitBreakerResult:
        """
        Check if circuit breaker should trigger.

        Args:
            current_price: Current market price
            reference_price: Prior day settlement price
            timestamp_ms: Current timestamp

        Returns:
            CircuitBreakerResult with halt info
        """
        if reference_price is None or reference_price <= 0:
            return CircuitBreakerResult(False, 0, None, 0)

        # Calculate decline percentage
        decline_pct = (current_price - reference_price) / reference_price

        # Check if within RTH
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        hour_et = (dt.hour - 5) % 24  # Approximate ET

        is_rth = self.RTH_START_HOUR <= hour_et < self.RTH_END_HOUR
        is_pre_cutoff = hour_et < self.RTH_CUTOFF_HOUR

        # Determine triggered level
        triggered_level = 0

        if decline_pct <= self.LEVEL_3_PCT:
            triggered_level = 3
        elif decline_pct <= self.LEVEL_2_PCT and is_rth and is_pre_cutoff:
            triggered_level = 2
        elif decline_pct <= self.LEVEL_1_PCT and is_rth and is_pre_cutoff:
            triggered_level = 1

        if triggered_level == 0:
            return CircuitBreakerResult(False, 0, None, 0)

        # Check if this level already triggered today
        if triggered_level in self._triggered_levels:
            return CircuitBreakerResult(False, 0, None, 0)

        # Trigger circuit breaker
        self._triggered_levels.add(triggered_level)
        self._last_trigger_ts[triggered_level] = timestamp_ms

        # Calculate halt duration
        if triggered_level == 3:
            halt_minutes = self.HALT_DURATION_L3
        else:
            halt_minutes = self.HALT_DURATION_L1_L2

        resume_time_ms = timestamp_ms + (halt_minutes * 60 * 1000)

        return CircuitBreakerResult(
            is_triggered=True,
            level=triggered_level,
            resume_time_ms=resume_time_ms,
            halt_duration_minutes=halt_minutes,
        )

    def reset_daily(self) -> None:
        """Reset circuit breakers for new trading day."""
        self._triggered_levels.clear()
        self._last_trigger_ts.clear()
```

### 5B.3 Daily Settlement Simulator

```python
# Part of execution_providers_cme_l3.py

class DailySettlementSimulator:
    """
    Simulates CME daily settlement process.

    Settlement occurs at 4:00 PM ET each trading day:
    1. Settlement price determined (usually last trade or weighted average)
    2. All open positions marked to settlement price
    3. Variation margin transferred
    4. P&L realized for daily accounting
    """

    def __init__(
        self,
        settlement_time_et: dt_time = dt_time(16, 0),
    ):
        self._settlement_time = settlement_time_et
        self._last_settlement_date: Optional[date] = None
        self._settlement_prices: Dict[str, Decimal] = {}

    def compute_settlement_price(
        self,
        symbol: str,
        bars: List[BarData],
        timestamp_ms: int,
    ) -> Decimal:
        """
        Compute settlement price.

        CME uses various methods depending on product:
        - Volume-weighted average of last N minutes
        - Last trade if liquid
        - Synthetic from related products
        """
        # Simple: use VWAP of last 30 minutes
        cutoff_ms = timestamp_ms - (30 * 60 * 1000)
        relevant_bars = [b for b in bars if b.timestamp_ms >= cutoff_ms]

        if not relevant_bars:
            return bars[-1].close if bars else Decimal("0")

        total_value = sum(
            b.close * b.volume for b in relevant_bars
        )
        total_volume = sum(b.volume for b in relevant_bars)

        if total_volume > 0:
            return total_value / total_volume

        return relevant_bars[-1].close

    def process_settlement(
        self,
        positions: List[CMEFuturesPosition],
        settlement_price: Decimal,
        timestamp_ms: int,
    ) -> List[SettlementRecord]:
        """
        Process daily settlement for all positions.

        Returns:
            List of settlement records with P&L
        """
        records = []

        for pos in positions:
            # Calculate variation margin
            price_change = settlement_price - pos.last_mark_price
            variation = price_change * pos.qty * pos.multiplier

            record = SettlementRecord(
                symbol=pos.symbol,
                position_qty=pos.qty,
                prev_settle_price=pos.last_mark_price,
                new_settle_price=settlement_price,
                variation_margin=variation,
                timestamp_ms=timestamp_ms,
            )

            records.append(record)

            # Update position mark
            pos.last_mark_price = settlement_price
            pos.last_mark_ts = timestamp_ms

        return records


@dataclass
class SettlementRecord:
    """Record of daily settlement."""
    symbol: str
    position_qty: Decimal
    prev_settle_price: Decimal
    new_settle_price: Decimal
    variation_margin: Decimal
    timestamp_ms: int


@dataclass
class CMEFuturesPosition:
    """CME futures position with settlement tracking."""
    symbol: str
    qty: Decimal
    entry_price: Decimal
    multiplier: Decimal
    last_mark_price: Decimal
    last_mark_ts: int
```

### 5B.4 Globex Matching Engine Emulator

```python
# Part of lob/cme_matching.py

class GlobexMatchingEngine(MatchingEngine):
    """
    CME Globex-style matching engine.

    Features:
    - FIFO Price-Time Priority (standard)
    - Market with Protection orders
    - Stop orders with trigger logic
    - Self-Trade Prevention (STP)
    - Minimum Quantity orders
    """

    def __init__(
        self,
        symbol: str,
        tick_size: Decimal,
        max_price_bands: int = 3,  # Price banding levels
    ):
        super().__init__(symbol=symbol, tick_size=tick_size)
        self._max_price_bands = max_price_bands
        self._indicative_opening_price: Optional[Decimal] = None

    def match_with_protection(
        self,
        order: Order,
        protection_points: int = 10,
    ) -> Fill:
        """
        Match Market with Protection order.

        Market with Protection has price limit:
        - Buy: best ask + protection_points * tick_size
        - Sell: best bid - protection_points * tick_size

        Protects against sweeping the book in thin markets.
        """
        if order.side == "BUY":
            limit_price = self._best_ask + (protection_points * self._tick_size)
        else:
            limit_price = self._best_bid - (protection_points * self._tick_size)

        # Convert to limit order with protection price
        protected_order = Order(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type="LIMIT",
            price=limit_price,
        )

        return self.match(protected_order)

    def calculate_opening_price(
        self,
        orders: List[Order],
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate indicative opening price.

        Uses equilibrium price that maximizes matched volume.

        Returns:
            (indicative_price, matched_volume)
        """
        if not orders:
            return Decimal("0"), Decimal("0")

        # Collect buy and sell orders
        buys = [o for o in orders if o.side == "BUY"]
        sells = [o for o in orders if o.side == "SELL"]

        if not buys or not sells:
            return Decimal("0"), Decimal("0")

        # Build cumulative volume curves
        # ... (auction matching logic)

        # Find equilibrium
        best_price = Decimal("0")
        best_volume = Decimal("0")

        # Simplified: midpoint of overlapping range
        max_buy = max(o.price for o in buys if o.price)
        min_sell = min(o.price for o in sells if o.price)

        if max_buy >= min_sell:
            best_price = (max_buy + min_sell) / 2
            best_volume = min(
                sum(o.qty for o in buys if o.price >= best_price),
                sum(o.qty for o in sells if o.price <= best_price),
            )

        self._indicative_opening_price = best_price
        return best_price, best_volume
```

### Tests for Phase 5B

```python
# NEW FILE: tests/test_cme_l3_execution.py
"""Tests for CME L3 execution provider."""

import pytest
from decimal import Decimal
from datetime import datetime, time as dt_time

from execution_providers_cme_l3 import (
    CMEL3ExecutionProvider,
    CMECircuitBreakerSimulator,
    DailySettlementSimulator,
    TradingSession,
    MarketState,
)


class TestCMECircuitBreaker:
    """Circuit breaker simulation tests."""

    def test_level_1_trigger_at_7_percent(self):
        """Level 1 triggers at -7% decline."""
        cb = CMECircuitBreakerSimulator("ES")

        result = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),  # -7%
            timestamp_ms=1700000000000,  # RTH
        )

        assert result.is_triggered
        assert result.level == 1
        assert result.halt_duration_minutes == 15

    def test_level_3_halts_rest_of_day(self):
        """Level 3 halts for remainder of day."""
        cb = CMECircuitBreakerSimulator("ES")

        result = cb.check_and_trigger(
            current_price=Decimal("4000"),
            reference_price=Decimal("5000"),  # -20%
            timestamp_ms=1700000000000,
        )

        assert result.is_triggered
        assert result.level == 3
        assert result.halt_duration_minutes == 999

    def test_no_halt_in_eth_for_level_1(self):
        """Level 1 doesn't halt during ETH."""
        cb = CMECircuitBreakerSimulator("ES")

        # Set timestamp to ETH (e.g., 7am ET)
        eth_ts = 1700000000000  # Adjust to ETH hours

        result = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=eth_ts,
        )

        # Should not trigger during ETH
        # (actual test would verify hour calculation)

    def test_same_level_not_triggered_twice(self):
        """Same circuit breaker level only triggers once per day."""
        cb = CMECircuitBreakerSimulator("ES")

        # First trigger
        result1 = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700000000000,
        )
        assert result1.is_triggered

        # Second trigger attempt at same level
        result2 = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700001000000,
        )
        assert not result2.is_triggered  # Already triggered

    def test_daily_reset(self):
        """Reset allows re-triggering next day."""
        cb = CMECircuitBreakerSimulator("ES")

        # Trigger
        cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700000000000,
        )

        # Reset
        cb.reset_daily()

        # Can trigger again
        result = cb.check_and_trigger(
            current_price=Decimal("4650"),
            reference_price=Decimal("5000"),
            timestamp_ms=1700100000000,
        )
        assert result.is_triggered


class TestDailySettlement:
    """Daily settlement simulation tests."""

    def test_settlement_price_calculation(self):
        """Settlement price computed from VWAP."""
        sim = DailySettlementSimulator()

        bars = [
            BarData(timestamp_ms=1, close=Decimal("5000"), volume=Decimal("100")),
            BarData(timestamp_ms=2, close=Decimal("5010"), volume=Decimal("200")),
            BarData(timestamp_ms=3, close=Decimal("5005"), volume=Decimal("100")),
        ]

        price = sim.compute_settlement_price(
            symbol="ES",
            bars=bars,
            timestamp_ms=1000000,
        )

        # VWAP = (5000*100 + 5010*200 + 5005*100) / 400 = 5006.25
        assert price == Decimal("5006.25")

    def test_variation_margin_calculation(self):
        """Variation margin correctly calculated."""
        sim = DailySettlementSimulator()

        position = CMEFuturesPosition(
            symbol="ES",
            qty=Decimal("2"),
            entry_price=Decimal("5000"),
            multiplier=Decimal("50"),
            last_mark_price=Decimal("4990"),
            last_mark_ts=1000000,
        )

        records = sim.process_settlement(
            positions=[position],
            settlement_price=Decimal("5010"),  # +20 points
            timestamp_ms=2000000,
        )

        # Variation = (5010 - 4990) * 2 * 50 = $2000
        assert len(records) == 1
        assert records[0].variation_margin == Decimal("2000")


class TestCMEL3Provider:
    """CME L3 execution provider tests."""

    @pytest.fixture
    def provider(self):
        return CMEL3ExecutionProvider(
            product="ES",
            multiplier=Decimal("50"),
            tick_size=Decimal("0.25"),
        )

    def test_order_rejected_during_halt(self, provider):
        """Orders rejected when market halted."""
        # Trigger halt
        provider._halt_until_ts = 9999999999999

        order = Order(symbol="ES", side="BUY", qty=Decimal("1"))
        fill = provider.execute(order, market, bar)

        assert fill.status == "REJECTED"
        assert fill.reject_reason == "MARKET_HALTED"

    def test_eth_slippage_adjustment(self, provider):
        """ETH trading has wider spreads."""
        # Test that spread is multiplied for ETH
        pass

    def test_session_detection(self, provider):
        """Trading session correctly detected."""
        # Test RTH vs ETH detection
        pass


class TestGlobexMatching:
    """Globex matching engine tests."""

    def test_market_with_protection(self):
        """Market with protection limits sweep."""
        engine = GlobexMatchingEngine(
            symbol="ES",
            tick_size=Decimal("0.25"),
        )

        # Place some resting orders
        # Then match MWP order
        pass

    def test_opening_auction_price(self):
        """Opening auction calculates equilibrium."""
        engine = GlobexMatchingEngine(
            symbol="ES",
            tick_size=Decimal("0.25"),
        )

        orders = [
            Order(side="BUY", price=Decimal("5000"), qty=Decimal("10")),
            Order(side="BUY", price=Decimal("4995"), qty=Decimal("20")),
            Order(side="SELL", price=Decimal("4998"), qty=Decimal("15")),
            Order(side="SELL", price=Decimal("5002"), qty=Decimal("25")),
        ]

        price, volume = engine.calculate_opening_price(orders)

        # Price should maximize matched volume
        assert price > Decimal("0")
        assert volume > Decimal("0")
```

### Deliverables Phase 5B
- [ ] `execution_providers_cme_l3.py` - L3 CME futures provider
- [ ] `lob/cme_matching.py` - Globex matching engine emulator
- [ ] `tests/test_cme_l3_execution.py` (55+ tests)

---

## 📦 PHASE 6A: RISK MANAGEMENT (Crypto Track)

### Цели
- Создать futures-specific risk guards
- Интеграция с существующим risk_guard.py
- Position sizing с учётом leverage

### 6.1 Futures Risk Guards

```python
# NEW FILE: services/futures_risk_guards.py
"""
Futures-specific risk management guards.

Implements:
1. LeverageGuard - Enforce leverage limits
2. MarginGuard - Monitor margin ratios
3. LiquidationGuard - Pre-liquidation warnings
4. FundingExposureGuard - Funding rate risk management
5. ConcentrationGuard - Position concentration limits
"""

from decimal import Decimal
from typing import Optional, List, Tuple
from core_futures import FuturesPosition, MarginMode
from impl_futures_margin import MarginCalculator
from impl_futures_liquidation import LiquidationEngine

class FuturesLeverageGuard:
    """
    Enforces leverage limits per symbol and account.

    Rules:
    - Max leverage per symbol (from brackets)
    - Max account-wide leverage
    - Gradual leverage reduction at size thresholds
    """

    def __init__(
        self,
        max_account_leverage: int = 20,
        max_symbol_leverage: int = 50,
        concentration_limit: float = 0.5,  # Max 50% in single symbol
    ):
        self._max_account_leverage = max_account_leverage
        self._max_symbol_leverage = max_symbol_leverage
        self._concentration_limit = concentration_limit

    def validate_new_position(
        self,
        proposed_position: FuturesPosition,
        current_positions: List[FuturesPosition],
        account_balance: Decimal,
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate if new position is allowed.

        Returns:
            (is_valid, error_message, suggested_leverage)
        """
        # Check symbol leverage
        if proposed_position.leverage > self._max_symbol_leverage:
            return (
                False,
                f"Leverage {proposed_position.leverage}x exceeds max {self._max_symbol_leverage}x",
                self._max_symbol_leverage,
            )

        # Check account-wide leverage
        total_notional = sum(
            abs(p.entry_price * p.qty) for p in current_positions
        )
        new_notional = abs(proposed_position.entry_price * proposed_position.qty)
        total_after = total_notional + new_notional

        account_leverage = total_after / account_balance if account_balance > 0 else 0

        if account_leverage > self._max_account_leverage:
            return (
                False,
                f"Account leverage {account_leverage:.1f}x exceeds max {self._max_account_leverage}x",
                None,
            )

        # Check concentration
        symbol_notional = new_notional
        for p in current_positions:
            if p.symbol == proposed_position.symbol:
                symbol_notional += abs(p.entry_price * p.qty)

        concentration = symbol_notional / total_after if total_after > 0 else 1

        if concentration > self._concentration_limit:
            return (
                False,
                f"Symbol concentration {concentration:.1%} exceeds limit {self._concentration_limit:.1%}",
                None,
            )

        return (True, None, None)

class FuturesMarginGuard:
    """
    Monitors margin ratios and triggers actions.

    Levels:
    - Warning: margin_ratio < 2.0 (200%)
    - Danger: margin_ratio < 1.5 (150%)
    - Critical: margin_ratio < 1.2 (120%)
    - Liquidation: margin_ratio < 1.0 (100%)
    """

    WARNING_LEVEL = Decimal("2.0")
    DANGER_LEVEL = Decimal("1.5")
    CRITICAL_LEVEL = Decimal("1.2")
    LIQUIDATION_LEVEL = Decimal("1.0")

    def __init__(
        self,
        margin_calculator: MarginCalculator,
        auto_reduce_at_danger: bool = True,
        reduce_by_percent: float = 0.25,  # Reduce 25% at danger level
    ):
        self._calculator = margin_calculator
        self._auto_reduce = auto_reduce_at_danger
        self._reduce_percent = reduce_by_percent

    def check_margin_status(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> MarginStatus:
        """Check margin status for position."""
        ratio = self._calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if ratio < self.LIQUIDATION_LEVEL:
            return MarginStatus.LIQUIDATION
        elif ratio < self.CRITICAL_LEVEL:
            return MarginStatus.CRITICAL
        elif ratio < self.DANGER_LEVEL:
            return MarginStatus.DANGER
        elif ratio < self.WARNING_LEVEL:
            return MarginStatus.WARNING
        return MarginStatus.HEALTHY

class FundingExposureGuard:
    """
    Manages funding rate exposure.

    High funding rates mean holding costs.
    Guard can:
    - Warn on high funding exposure
    - Limit position duration during high funding
    - Force position reduction
    """

    def __init__(
        self,
        max_daily_funding_cost_bps: float = 30,  # 0.3% max daily funding cost
        funding_rate_warning_threshold: float = 0.0005,  # 0.05% per 8h
    ):
        self._max_daily_cost_bps = max_daily_funding_cost_bps
        self._warning_threshold = funding_rate_warning_threshold

    def check_funding_exposure(
        self,
        position: FuturesPosition,
        current_funding_rate: Decimal,
        predicted_funding_rates: List[Decimal],  # Next 3 funding periods
    ) -> FundingExposureStatus:
        """Evaluate funding rate exposure."""
        # Calculate expected daily funding cost
        avg_rate = sum(predicted_funding_rates) / len(predicted_funding_rates)
        daily_cost_bps = abs(float(avg_rate)) * 3 * 10000  # 3 fundings/day

        if daily_cost_bps > self._max_daily_cost_bps:
            return FundingExposureStatus.EXCESSIVE
        elif float(abs(current_funding_rate)) > self._warning_threshold:
            return FundingExposureStatus.WARNING
        return FundingExposureStatus.NORMAL
```

### 6.2 Integration with Risk Guard

```python
# Updates to risk_guard.py

class RiskGuard:
    """Extended RiskGuard with futures support."""

    def __init__(
        self,
        futures_guards: Optional[FuturesRiskGuards] = None,
        **kwargs,
    ):
        # ... existing init
        self._futures_guards = futures_guards

    def validate_trade(
        self,
        trade: Trade,
        market_type: MarketType,
        **kwargs,
    ) -> TradeValidation:
        """Validate trade with asset-class specific guards."""
        # Existing validation
        result = self._base_validate(trade)

        if market_type in (MarketType.CRYPTO_FUTURES, MarketType.CRYPTO_PERP):
            if self._futures_guards:
                futures_result = self._futures_guards.validate(trade, **kwargs)
                result = result.merge(futures_result)

        return result
```

### Tests for Phase 6

```python
# NEW FILE: tests/test_futures_risk_guards.py

class TestFuturesLeverageGuard:
    """Leverage guard tests."""

    def test_blocks_excessive_leverage(self):
        """Rejects leverage above limit."""
        guard = FuturesLeverageGuard(max_symbol_leverage=50)

        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=100,  # Above limit!
            margin_mode=MarginMode.CROSS,
        )

        valid, error, suggested = guard.validate_new_position(
            position, [], Decimal("10000")
        )

        assert not valid
        assert suggested == 50

    def test_concentration_limit(self):
        """Blocks excessive concentration in single symbol."""
        pass

class TestFuturesMarginGuard:
    """Margin guard tests."""

    def test_warning_level_detection(self):
        """Detects margin warning level."""
        pass

    def test_liquidation_level_action(self):
        """Triggers action at liquidation level."""
        pass
```

### Deliverables Phase 6A
- [ ] `services/futures_risk_guards.py` - Crypto futures risk guards
- [ ] Updated `risk_guard.py` with crypto futures integration
- [ ] `tests/test_futures_risk_guards.py` (80+ tests)

---

## 📦 PHASE 6B: RISK MANAGEMENT (CME Track)

### Цели
- SPAN margin integration с risk guards
- Position limits по CME rules
- Circuit breaker integration
- Daily settlement P&L tracking

### 6B.1 CME Risk Guards

```python
# NEW FILE: services/cme_risk_guards.py
"""
CME-specific risk management guards.

Implements:
1. SPANMarginGuard - Monitor SPAN margin requirements
2. PositionLimitGuard - CME position limit enforcement
3. PriceLimitGuard - Circuit breaker awareness
4. SettlementRiskGuard - Daily settlement risk
5. RolloverGuard - Contract expiry management
"""

from datetime import datetime, date, time as dt_time
from decimal import Decimal
from typing import Optional, List, Dict, Tuple
from enum import Enum

from impl_cme_margin import SPANMarginCalculator
from execution_providers_cme_l3 import CMECircuitBreakerSimulator


class MarginLevel(str, Enum):
    """Margin level states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"     # < 150% of maintenance
    DANGER = "DANGER"       # < 125% of maintenance
    MARGIN_CALL = "MARGIN_CALL"  # < 100% of maintenance


class SPANMarginGuard:
    """
    Monitors SPAN margin requirements.

    Levels:
    - Normal: margin_ratio >= 1.5
    - Warning: 1.25 <= margin_ratio < 1.5
    - Danger: 1.0 <= margin_ratio < 1.25
    - Margin Call: margin_ratio < 1.0
    """

    WARNING_RATIO = Decimal("1.50")
    DANGER_RATIO = Decimal("1.25")
    MARGIN_CALL_RATIO = Decimal("1.00")

    def __init__(
        self,
        span_calculator: SPANMarginCalculator,
        auto_reduce_at_danger: bool = True,
        reduce_target_ratio: Decimal = Decimal("2.0"),
    ):
        self._span_calc = span_calculator
        self._auto_reduce = auto_reduce_at_danger
        self._reduce_target = reduce_target_ratio

    def check_margin_status(
        self,
        positions: List[CMEFuturesPosition],
        account_equity: Decimal,
    ) -> Tuple[MarginLevel, Decimal, Optional[str]]:
        """
        Check current margin status.

        Returns:
            (margin_level, margin_ratio, action_message)
        """
        if not positions:
            return MarginLevel.NORMAL, Decimal("999"), None

        # Calculate total SPAN margin requirement
        total_margin = self._span_calc.calculate_portfolio_margin(positions)

        if total_margin <= 0:
            return MarginLevel.NORMAL, Decimal("999"), None

        margin_ratio = account_equity / total_margin

        if margin_ratio >= self.WARNING_RATIO:
            return MarginLevel.NORMAL, margin_ratio, None
        elif margin_ratio >= self.DANGER_RATIO:
            return (
                MarginLevel.WARNING,
                margin_ratio,
                f"Margin ratio {margin_ratio:.2%} below warning level {self.WARNING_RATIO:.0%}",
            )
        elif margin_ratio >= self.MARGIN_CALL_RATIO:
            return (
                MarginLevel.DANGER,
                margin_ratio,
                f"DANGER: Margin ratio {margin_ratio:.2%} - reduce position",
            )
        else:
            return (
                MarginLevel.MARGIN_CALL,
                margin_ratio,
                f"MARGIN CALL: Ratio {margin_ratio:.2%} - immediate action required",
            )

    def calculate_safe_position_size(
        self,
        symbol: str,
        current_price: Decimal,
        account_equity: Decimal,
        target_margin_ratio: Decimal = Decimal("3.0"),
    ) -> Decimal:
        """
        Calculate maximum safe position size.

        Returns max quantity that maintains target margin ratio.
        """
        margin_per_contract = self._span_calc.get_initial_margin(
            symbol, current_price
        )

        if margin_per_contract <= 0:
            return Decimal("0")

        max_contracts = (account_equity / target_margin_ratio) / margin_per_contract

        return max_contracts.quantize(Decimal("1"))


class CMEPositionLimitGuard:
    """
    Enforces CME position limits.

    Position limits vary by:
    - Product (ES: 20k, NQ: 10k, GC: 6k)
    - Account type (speculative vs hedge)
    - Month (spot month limits tighter)
    """

    # Speculative position limits (contracts)
    POSITION_LIMITS = {
        "ES": 20000,
        "NQ": 10000,
        "MES": 50000,
        "MNQ": 50000,
        "GC": 6000,
        "SI": 6000,
        "CL": 10000,
        "6E": 10000,
        "6J": 10000,
        "ZB": 25000,
    }

    # Spot month limits (usually tighter)
    SPOT_MONTH_LIMITS = {
        "ES": 20000,  # Same for financials
        "GC": 3000,   # Physical delivery
        "CL": 3000,   # Physical delivery
    }

    def __init__(
        self,
        account_type: str = "speculative",
    ):
        self._account_type = account_type

    def check_position_limit(
        self,
        symbol: str,
        proposed_qty: Decimal,
        current_qty: Decimal,
        is_spot_month: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[Decimal]]:
        """
        Check if proposed position exceeds limits.

        Returns:
            (is_valid, error_message, max_allowed_qty)
        """
        # Get applicable limit
        product = self._extract_product(symbol)

        if is_spot_month and product in self.SPOT_MONTH_LIMITS:
            limit = self.SPOT_MONTH_LIMITS[product]
        else:
            limit = self.POSITION_LIMITS.get(product, 10000)  # Default

        total_qty = abs(current_qty + proposed_qty)

        if total_qty <= limit:
            return True, None, Decimal(str(limit))

        return (
            False,
            f"Position {total_qty} exceeds limit {limit} for {product}",
            Decimal(str(limit)) - abs(current_qty),
        )

    def _extract_product(self, symbol: str) -> str:
        """Extract product code from symbol (e.g., ESH24 -> ES)."""
        # Remove month/year suffix
        for i, c in enumerate(symbol):
            if c.isdigit() or (i >= 2 and c in "FGHJKMNQUVXZ"):
                return symbol[:i]
        return symbol[:2]


class CircuitBreakerAwareGuard:
    """
    Integrates circuit breaker awareness into risk management.

    Actions:
    - Reduce position sizing near limits
    - Halt new orders after circuit break
    - Widen stops during volatile periods
    """

    def __init__(
        self,
        cb_simulator: CMECircuitBreakerSimulator,
        buffer_pct: Decimal = Decimal("0.02"),  # 2% buffer before L1
    ):
        self._cb = cb_simulator
        self._buffer_pct = buffer_pct

    def check_proximity_to_limit(
        self,
        current_price: Decimal,
        reference_price: Decimal,
    ) -> Tuple[int, Decimal, str]:
        """
        Check how close to circuit breaker levels.

        Returns:
            (nearest_level, distance_pct, recommendation)
        """
        if reference_price <= 0:
            return 0, Decimal("1"), "NO_REFERENCE"

        decline_pct = (current_price - reference_price) / reference_price

        # Check distance to each level
        distance_to_l1 = decline_pct - Decimal("-0.07")
        distance_to_l2 = decline_pct - Decimal("-0.13")
        distance_to_l3 = decline_pct - Decimal("-0.20")

        if distance_to_l1 > self._buffer_pct:
            return 0, distance_to_l1, "NORMAL"
        elif distance_to_l1 > 0:
            return 1, distance_to_l1, "APPROACHING_L1"
        elif distance_to_l2 > 0:
            return 1, distance_to_l2, "BETWEEN_L1_L2"
        elif distance_to_l3 > 0:
            return 2, distance_to_l3, "BETWEEN_L2_L3"
        else:
            return 3, distance_to_l3, "BELOW_L3"

    def get_position_scale_factor(
        self,
        proximity_level: int,
        distance_pct: Decimal,
    ) -> Decimal:
        """
        Get position scaling factor based on circuit breaker proximity.
        """
        if proximity_level == 0:
            return Decimal("1.0")  # Full size
        elif proximity_level == 1:
            # Scale down linearly as approaching
            return max(Decimal("0.5"), min(distance_pct / self._buffer_pct, Decimal("1")))
        elif proximity_level == 2:
            return Decimal("0.25")  # Minimal new positions
        else:
            return Decimal("0")  # No new positions


class RolloverGuard:
    """
    Manages contract expiry and rollover risk.

    Actions:
    - Warn approaching expiry
    - Block new positions in expiring contracts
    - Suggest rollover timing
    """

    def __init__(
        self,
        rollover_days_warning: int = 5,
        rollover_days_block: int = 1,
    ):
        self._warning_days = rollover_days_warning
        self._block_days = rollover_days_block

    def check_expiry_status(
        self,
        symbol: str,
        expiry_date: date,
        today: date,
    ) -> Tuple[str, int, Optional[str]]:
        """
        Check contract expiry status.

        Returns:
            (status, days_to_expiry, message)
        """
        days_to_expiry = (expiry_date - today).days

        if days_to_expiry <= 0:
            return "EXPIRED", days_to_expiry, "Contract has expired"
        elif days_to_expiry <= self._block_days:
            return "BLOCKED", days_to_expiry, f"Contract expires in {days_to_expiry}d - no new positions"
        elif days_to_expiry <= self._warning_days:
            return "ROLLOVER_WARNING", days_to_expiry, f"Consider rolling - {days_to_expiry}d to expiry"
        else:
            return "NORMAL", days_to_expiry, None

    def suggest_rollover_contract(
        self,
        current_symbol: str,
        available_contracts: List[str],
    ) -> Optional[str]:
        """
        Suggest next contract to roll into.

        Usually the next quarterly (ESH24 -> ESM24).
        """
        # Parse current contract month/year
        # Find next available contract
        # Return suggestion
        pass
```

### 6B.2 Daily Settlement Risk

```python
# Part of services/cme_risk_guards.py

class SettlementRiskGuard:
    """
    Manages daily settlement risk.

    Tracks:
    - Unrealized P&L approaching settlement
    - Margin requirements post-settlement
    - Cash flow timing
    """

    def __init__(
        self,
        settlement_time_et: dt_time = dt_time(16, 0),
    ):
        self._settlement_time = settlement_time_et

    def check_settlement_impact(
        self,
        positions: List[CMEFuturesPosition],
        current_prices: Dict[str, Decimal],
        account_equity: Decimal,
    ) -> Dict[str, any]:
        """
        Estimate impact of upcoming settlement.

        Returns:
            {
                'total_variation': Decimal,
                'post_settle_equity': Decimal,
                'largest_loss_position': str,
                'margin_impact': str,
            }
        """
        total_variation = Decimal("0")
        largest_loss = Decimal("0")
        largest_loss_symbol = None

        for pos in positions:
            current = current_prices.get(pos.symbol, pos.last_mark_price)
            variation = (current - pos.last_mark_price) * pos.qty * pos.multiplier
            total_variation += variation

            if variation < largest_loss:
                largest_loss = variation
                largest_loss_symbol = pos.symbol

        post_settle_equity = account_equity + total_variation

        return {
            'total_variation': total_variation,
            'post_settle_equity': post_settle_equity,
            'largest_loss_position': largest_loss_symbol,
            'largest_loss_amount': largest_loss,
            'equity_change_pct': total_variation / account_equity if account_equity > 0 else Decimal("0"),
        }

    def should_reduce_before_settlement(
        self,
        settlement_impact: Dict[str, any],
        risk_threshold_pct: Decimal = Decimal("0.05"),
    ) -> Tuple[bool, Optional[str]]:
        """
        Recommend if positions should be reduced before settlement.

        Args:
            settlement_impact: Result from check_settlement_impact
            risk_threshold_pct: Max acceptable equity change
        """
        equity_change = abs(settlement_impact['equity_change_pct'])

        if equity_change > risk_threshold_pct:
            return (
                True,
                f"Large settlement impact {equity_change:.1%} > {risk_threshold_pct:.1%} threshold",
            )

        return False, None
```

### 6B.3 Tests for CME Risk Guards

```python
# NEW FILE: tests/test_cme_risk_guards.py
"""Tests for CME-specific risk guards."""

import pytest
from decimal import Decimal
from datetime import date, time as dt_time

from services.cme_risk_guards import (
    SPANMarginGuard,
    CMEPositionLimitGuard,
    CircuitBreakerAwareGuard,
    RolloverGuard,
    SettlementRiskGuard,
    MarginLevel,
)


class TestSPANMarginGuard:
    """SPAN margin guard tests."""

    def test_normal_margin_level(self):
        """Above warning level = NORMAL."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("100000"),  # High equity
        )

        assert level == MarginLevel.NORMAL
        assert msg is None

    def test_warning_margin_level(self):
        """Below 150% = WARNING."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("15000"),  # 140% of margin
        )

        assert level == MarginLevel.WARNING
        assert "warning" in msg.lower()

    def test_danger_margin_level(self):
        """Below 125% = DANGER."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("12000"),  # 110% of margin
        )

        assert level == MarginLevel.DANGER
        assert "danger" in msg.lower()

    def test_margin_call_level(self):
        """Below 100% = MARGIN_CALL."""
        guard = SPANMarginGuard(span_calculator=mock_span)

        level, ratio, msg = guard.check_margin_status(
            positions=[mock_position],
            account_equity=Decimal("8000"),  # 80% of margin
        )

        assert level == MarginLevel.MARGIN_CALL
        assert "margin call" in msg.lower()


class TestPositionLimitGuard:
    """Position limit guard tests."""

    def test_within_limit(self):
        """Position within CME limits passes."""
        guard = CMEPositionLimitGuard()

        valid, msg, max_qty = guard.check_position_limit(
            symbol="ESH24",
            proposed_qty=Decimal("100"),
            current_qty=Decimal("50"),
            is_spot_month=False,
        )

        assert valid is True
        assert msg is None

    def test_exceeds_limit(self):
        """Position exceeding limit fails."""
        guard = CMEPositionLimitGuard()

        valid, msg, max_qty = guard.check_position_limit(
            symbol="ESH24",
            proposed_qty=Decimal("25000"),
            current_qty=Decimal("0"),
            is_spot_month=False,
        )

        assert valid is False
        assert "exceeds limit" in msg

    def test_spot_month_tighter_limits(self):
        """Spot month has tighter limits for physical delivery."""
        guard = CMEPositionLimitGuard()

        # Within regular limit but exceeds spot month limit
        valid, msg, max_qty = guard.check_position_limit(
            symbol="GCZ24",
            proposed_qty=Decimal("4000"),
            current_qty=Decimal("0"),
            is_spot_month=True,
        )

        assert valid is False  # Exceeds 3000 spot month limit

    def test_product_extraction(self):
        """Product code extracted correctly from symbol."""
        guard = CMEPositionLimitGuard()

        assert guard._extract_product("ESH24") == "ES"
        assert guard._extract_product("GCZ24") == "GC"
        assert guard._extract_product("6EH24") == "6E"


class TestCircuitBreakerAwareGuard:
    """Circuit breaker awareness tests."""

    def test_normal_distance(self):
        """Far from circuit breaker = NORMAL."""
        guard = CircuitBreakerAwareGuard(cb_simulator=mock_cb)

        level, distance, rec = guard.check_proximity_to_limit(
            current_price=Decimal("5000"),
            reference_price=Decimal("5000"),  # No decline
        )

        assert level == 0
        assert rec == "NORMAL"

    def test_approaching_level_1(self):
        """Near -7% decline = APPROACHING_L1."""
        guard = CircuitBreakerAwareGuard(cb_simulator=mock_cb)

        level, distance, rec = guard.check_proximity_to_limit(
            current_price=Decimal("4700"),
            reference_price=Decimal("5000"),  # -6%
        )

        assert level == 1
        assert rec == "APPROACHING_L1"

    def test_position_scale_factor(self):
        """Position scaling reduces near limits."""
        guard = CircuitBreakerAwareGuard(cb_simulator=mock_cb)

        # Normal
        assert guard.get_position_scale_factor(0, Decimal("0.05")) == Decimal("1.0")

        # Approaching L1
        scale = guard.get_position_scale_factor(1, Decimal("0.01"))
        assert scale < Decimal("1.0")

        # Between L2-L3
        assert guard.get_position_scale_factor(2, Decimal("0.02")) == Decimal("0.25")

        # Below L3
        assert guard.get_position_scale_factor(3, Decimal("0")) == Decimal("0")


class TestRolloverGuard:
    """Contract rollover guard tests."""

    def test_normal_expiry(self):
        """Far from expiry = NORMAL."""
        guard = RolloverGuard(rollover_days_warning=5, rollover_days_block=1)

        status, days, msg = guard.check_expiry_status(
            symbol="ESH24",
            expiry_date=date(2024, 3, 15),
            today=date(2024, 3, 1),  # 14 days to expiry
        )

        assert status == "NORMAL"
        assert days == 14
        assert msg is None

    def test_rollover_warning(self):
        """Within warning period = ROLLOVER_WARNING."""
        guard = RolloverGuard(rollover_days_warning=5, rollover_days_block=1)

        status, days, msg = guard.check_expiry_status(
            symbol="ESH24",
            expiry_date=date(2024, 3, 15),
            today=date(2024, 3, 11),  # 4 days to expiry
        )

        assert status == "ROLLOVER_WARNING"
        assert "rolling" in msg.lower()

    def test_blocked_near_expiry(self):
        """Very close to expiry = BLOCKED."""
        guard = RolloverGuard(rollover_days_warning=5, rollover_days_block=1)

        status, days, msg = guard.check_expiry_status(
            symbol="ESH24",
            expiry_date=date(2024, 3, 15),
            today=date(2024, 3, 14),  # 1 day to expiry
        )

        assert status == "BLOCKED"
        assert "no new positions" in msg.lower()


class TestSettlementRiskGuard:
    """Settlement risk guard tests."""

    def test_settlement_impact_calculation(self):
        """Settlement impact correctly calculated."""
        guard = SettlementRiskGuard()

        positions = [
            CMEFuturesPosition(
                symbol="ES",
                qty=Decimal("2"),
                entry_price=Decimal("5000"),
                multiplier=Decimal("50"),
                last_mark_price=Decimal("5000"),
                last_mark_ts=0,
            ),
        ]

        impact = guard.check_settlement_impact(
            positions=positions,
            current_prices={"ES": Decimal("5020")},  # +20 points
            account_equity=Decimal("50000"),
        )

        # Variation = +20 * 2 * 50 = $2000
        assert impact['total_variation'] == Decimal("2000")
        assert impact['post_settle_equity'] == Decimal("52000")

    def test_large_settlement_triggers_warning(self):
        """Large settlement impact recommends reduction."""
        guard = SettlementRiskGuard()

        impact = {
            'equity_change_pct': Decimal("0.10"),  # 10% change
            'total_variation': Decimal("-5000"),
        }

        should_reduce, msg = guard.should_reduce_before_settlement(
            settlement_impact=impact,
            risk_threshold_pct=Decimal("0.05"),
        )

        assert should_reduce is True
        assert "threshold" in msg
```

### Deliverables Phase 6B
- [ ] `services/cme_risk_guards.py` - CME-specific risk guards
- [ ] `impl_cme_settlement.py` - Daily settlement integration
- [ ] `tests/test_cme_risk_guards.py` (70+ tests)

---

## 📦 PHASE 7A: FEATURES & DATA PIPELINE (Crypto Track)

### Цели
- Создать futures-specific features
- Интеграция funding rate в features
- Open interest и liquidation features

### 7.1 Futures Features

```python
# NEW FILE: futures_features.py
"""
Futures-specific feature engineering.

Features unique to futures:
1. Funding rate features (current, predicted, cumulative)
2. Open interest features (change, concentration)
3. Basis features (spot-futures spread)
4. Liquidation features (recent liquidations, cascade risk)
5. Mark-index spread features
"""

from decimal import Decimal
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def calculate_funding_features(
    funding_rates: pd.Series,
    lookback_periods: int = 8,  # 8 fundings = ~2.67 days
) -> pd.DataFrame:
    """
    Calculate funding rate features.

    Features:
    - funding_rate_current: Current funding rate
    - funding_rate_sma: SMA of funding rates
    - funding_rate_std: Volatility of funding
    - funding_cumulative_24h: Cumulative funding last 24h
    - funding_direction: Sign consistency (-1 to 1)
    """
    features = pd.DataFrame(index=funding_rates.index)

    features['funding_rate_current'] = funding_rates
    features['funding_rate_sma'] = funding_rates.rolling(lookback_periods).mean()
    features['funding_rate_std'] = funding_rates.rolling(lookback_periods).std()
    features['funding_cumulative_24h'] = funding_rates.rolling(3).sum()  # 3 = 24h

    # Direction consistency
    signs = np.sign(funding_rates)
    features['funding_direction'] = signs.rolling(lookback_periods).mean()

    return features

def calculate_open_interest_features(
    open_interest: pd.Series,
    price: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate open interest features.

    Features:
    - oi_change_pct: OI change percentage
    - oi_price_divergence: OI vs price divergence
    - oi_concentration: OI concentration metric
    """
    features = pd.DataFrame(index=open_interest.index)

    features['oi_change_pct'] = open_interest.pct_change(lookback)

    # OI-Price divergence (rising OI + falling price = bearish signal)
    oi_change = open_interest.pct_change(lookback)
    price_change = price.pct_change(lookback)
    features['oi_price_divergence'] = oi_change - price_change

    return features

def calculate_basis_features(
    futures_price: pd.Series,
    spot_price: pd.Series,
    days_to_expiry: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Calculate basis features.

    Basis = Futures Price - Spot Price
    Annualized Basis = (Basis / Spot) * (365 / DTE) for quarterly contracts

    For perpetuals, basis approximates funding expectation.
    """
    features = pd.DataFrame(index=futures_price.index)

    basis = futures_price - spot_price
    features['basis'] = basis
    features['basis_pct'] = basis / spot_price * 100

    if days_to_expiry is not None:
        features['basis_annualized'] = (basis / spot_price) * (365 / days_to_expiry) * 100

    return features

def calculate_liquidation_features(
    liquidation_volume: pd.Series,
    total_volume: pd.Series,
    price: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate liquidation-related features.

    Features:
    - liq_volume_ratio: Liquidation as % of total volume
    - liq_cascade_risk: Indicator of cascade risk
    - liq_price_impact: Price impact from liquidations
    """
    features = pd.DataFrame(index=liquidation_volume.index)

    features['liq_volume_ratio'] = liquidation_volume / total_volume.replace(0, np.nan)

    # Cascade risk: high recent liquidations + volatility
    features['liq_cascade_risk'] = (
        features['liq_volume_ratio'].rolling(lookback).mean() *
        price.pct_change().rolling(lookback).std()
    )

    return features
```

### 7.2 Data Pipeline Updates

```python
# Updates to data_loader_multi_asset.py

def load_futures_data(
    paths: List[str],
    funding_paths: Optional[List[str]] = None,
    mark_price_paths: Optional[List[str]] = None,
    liquidation_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """
    Load futures data with all components.

    Merges:
    - OHLCV bars (last price)
    - Mark price bars
    - Funding rate history
    - Liquidation data
    """
    # Load base OHLCV
    base_frames = load_parquet_files(paths)

    # Merge funding rates
    if funding_paths:
        funding_frames = load_parquet_files(funding_paths)
        base_frames = merge_funding_data(base_frames, funding_frames)

    # Merge mark price
    if mark_price_paths:
        mark_frames = load_parquet_files(mark_price_paths)
        base_frames = merge_mark_price(base_frames, mark_frames)

    # Merge liquidations
    if liquidation_paths:
        liq_frames = load_parquet_files(liquidation_paths)
        base_frames = merge_liquidation_data(base_frames, liq_frames)

    return base_frames, compute_obs_shapes(base_frames)
```

### 7.3 Data Download Script

```python
# NEW FILE: scripts/download_futures_data.py
"""
Download Binance Futures historical data.

Downloads:
- OHLCV candlesticks
- Mark price candlesticks
- Funding rate history
- Liquidation events (aggregated)
"""

import argparse
from binance_public import BinancePublicClient
from ingest_funding_mark import _fetch_all_funding, _fetch_all_mark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--out-dir", default="data/futures")
    args = parser.parse_args()

    client = BinancePublicClient()

    for symbol in args.symbols:
        # Download OHLCV
        download_ohlcv(client, symbol, args)

        # Download mark price
        download_mark_price(client, symbol, args)

        # Download funding
        download_funding(client, symbol, args)

if __name__ == "__main__":
    main()
```

### Tests for Phase 7

```python
# NEW FILE: tests/test_futures_features.py

class TestFundingFeatures:
    """Funding feature tests."""

    def test_funding_cumulative_24h(self):
        """Correctly sums 3 funding periods."""
        pass

    def test_funding_direction_consistency(self):
        """Direction feature in [-1, 1]."""
        pass

class TestOpenInterestFeatures:
    """OI feature tests."""

    def test_oi_price_divergence(self):
        """Divergence shows bullish/bearish signal."""
        pass

class TestBasisFeatures:
    """Basis feature tests."""

    def test_annualized_basis(self):
        """Correctly annualizes basis."""
        pass
```

### Deliverables Phase 7
- [ ] `futures_features.py` - Feature engineering
- [ ] Updated `data_loader_multi_asset.py`
- [ ] `scripts/download_futures_data.py`
- [ ] `tests/test_futures_features.py` (50+ tests)

---

## 📦 PHASE 8: TRAINING INTEGRATION

### Цели
- Интеграция futures в training pipeline
- Position sizing с leverage
- Reward shaping для futures

### 8.1 Futures Trading Environment

```python
# NEW FILE: wrappers/futures_env.py
"""
Futures trading environment wrapper.

Extends TradingEnv with:
- Leverage control action
- Margin tracking
- Funding payment integration
- Liquidation handling
"""

from wrappers.base import TradingEnvWrapper
from core_futures import FuturesPosition, MarginMode

class FuturesTradingEnv(TradingEnvWrapper):
    """
    Futures environment with leverage and margin.

    Action space:
    - Original: position size [-1, 1]
    - Extended: [position_size, leverage_target]

    Observation space:
    - Original features
    - + margin_ratio
    - + funding_rate
    - + liquidation_distance
    """

    def __init__(
        self,
        env,
        initial_leverage: int = 10,
        max_leverage: int = 50,
        margin_mode: MarginMode = MarginMode.CROSS,
        funding_tracker: Optional[FundingRateTracker] = None,
        margin_calculator: Optional[MarginCalculator] = None,
    ):
        super().__init__(env)
        self._leverage = initial_leverage
        self._max_leverage = max_leverage
        self._margin_mode = margin_mode
        self._funding_tracker = funding_tracker
        self._margin_calculator = margin_calculator

        # Position tracking
        self._position: Optional[FuturesPosition] = None
        self._realized_funding: Decimal = Decimal("0")

    def step(self, action):
        """Execute step with futures mechanics."""
        # Parse action (position_size, optional leverage)
        position_target = action[0] if len(action) > 0 else action
        leverage_target = action[1] if len(action) > 1 else self._leverage

        # Execute base environment step
        obs, reward, terminated, truncated, info = self.env.step(position_target)

        # Apply funding if due
        funding_payment = self._apply_funding(info.get("timestamp_ms", 0))

        # Check liquidation
        if self._check_liquidation(info.get("mark_price")):
            terminated = True
            reward = self._liquidation_penalty
            info["liquidated"] = True

        # Augment observation
        obs = self._augment_obs(obs, info)

        # Augment reward with funding
        reward = reward + float(funding_payment)

        info["funding_payment"] = float(funding_payment)
        info["margin_ratio"] = self._get_margin_ratio(info.get("mark_price"))

        return obs, reward, terminated, truncated, info

    def _apply_funding(self, timestamp_ms: int) -> Decimal:
        """Apply funding payment if due."""
        if self._position is None or self._funding_tracker is None:
            return Decimal("0")

        # Check if funding time
        if not self._is_funding_time(timestamp_ms):
            return Decimal("0")

        funding_rate = self._get_current_funding_rate()
        mark_price = self._get_mark_price()

        payment = self._funding_tracker.calculate_funding_payment(
            self._position, funding_rate, mark_price, timestamp_ms
        )

        self._realized_funding += payment.payment_amount
        return payment.payment_amount
```

### 8.2 Training Pipeline Updates

```python
# Updates to train_model_multi_patch.py

def create_futures_env(cfg: Config) -> FuturesTradingEnv:
    """Create futures environment."""
    # Load data
    base_frames, obs_shapes = load_futures_data(
        paths=cfg.data.paths,
        funding_paths=cfg.data.funding_paths,
        mark_price_paths=cfg.data.mark_price_paths,
    )

    # Create base env
    base_env = TradingEnv(
        frames=base_frames,
        config=cfg,
        asset_class="crypto_futures",
    )

    # Wrap with futures mechanics
    env = FuturesTradingEnv(
        base_env,
        initial_leverage=cfg.futures.initial_leverage,
        max_leverage=cfg.futures.max_leverage,
        margin_mode=MarginMode(cfg.futures.margin_mode),
    )

    return env
```

### 8.3 Futures Training Config

```yaml
# NEW FILE: configs/config_train_futures.yaml
mode: train
asset_class: crypto_futures
data_vendor: binance

data:
  paths:
    - "data/futures/*.parquet"
  funding_paths:
    - "data/futures/*_funding.parquet"
  mark_price_paths:
    - "data/futures/*_mark_*.parquet"
  timeframe: "4h"

futures:
  initial_leverage: 10
  max_leverage: 50
  margin_mode: cross
  include_funding_in_reward: true
  liquidation_penalty: -10.0

env:
  session:
    calendar: crypto_24x7

fees:
  structure: maker_taker
  maker_bps: 2.0
  taker_bps: 4.0

slippage:
  profile: crypto_futures
  k: 0.09
```

### Tests for Phase 8

```python
# NEW FILE: tests/test_futures_training.py

class TestFuturesTradingEnv:
    """Futures env tests."""

    def test_funding_applied_at_correct_times(self):
        """Funding only applied at 00:00, 08:00, 16:00 UTC."""
        pass

    def test_liquidation_terminates_episode(self):
        """Episode ends on liquidation."""
        pass

    def test_leverage_scaling_position_size(self):
        """Leverage correctly scales position."""
        pass

class TestFuturesTraining:
    """Training integration tests."""

    def test_training_runs_without_error(self):
        """Basic training loop completes."""
        pass

    def test_model_learns_funding_avoidance(self):
        """Model learns to avoid high funding positions."""
        pass
```

### Deliverables Phase 8
- [ ] `wrappers/futures_env.py` - Futures environment
- [ ] Updated `train_model_multi_patch.py`
- [ ] `configs/config_train_futures.yaml`
- [ ] `tests/test_futures_training.py` (60+ tests)

---

## 📦 PHASE 9: LIVE TRADING

### Цели
- Live trading с futures
- Position synchronization
- Real-time funding tracking

### 9.1 Futures Live Script

```python
# NEW FILE: script_futures_live.py
"""
Futures live trading entry point.

Features:
- Position sync with exchange
- Real-time funding tracking
- Margin monitoring
- Auto-deleveraging detection
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--margin-mode", choices=["cross", "isolated"], default="cross")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Create adapters
    market_data = BinanceFuturesMarketDataAdapter(cfg.binance)
    order_exec = BinanceFuturesOrderExecutionAdapter(cfg.binance)

    # Set leverage and margin mode
    order_exec.set_leverage(cfg.symbol, args.leverage)
    order_exec.set_margin_mode(cfg.symbol, MarginMode(args.margin_mode))

    # Create live runner
    runner = FuturesLiveRunner(
        market_data=market_data,
        order_execution=order_exec,
        model=load_model(cfg.model_path),
        risk_guards=create_futures_risk_guards(cfg),
    )

    # Run
    runner.run()
```

### 9.2 Position Sync

```python
# NEW FILE: services/futures_position_sync.py
"""
Futures position synchronization.

Syncs local state with exchange state.
"""

class FuturesPositionSynchronizer:
    """
    Synchronizes futures positions with exchange.

    Handles:
    - Position quantity mismatches
    - Entry price drift (due to partial fills)
    - Leverage changes
    - ADL events
    """

    def __init__(
        self,
        order_execution: BinanceFuturesOrderExecutionAdapter,
        sync_interval_sec: float = 10.0,
    ):
        self._adapter = order_execution
        self._interval = sync_interval_sec
        self._local_positions: Dict[str, FuturesPosition] = {}

    def sync(self) -> List[PositionDiff]:
        """
        Synchronize with exchange.

        Returns:
            List of position differences found
        """
        exchange_positions = {
            p.symbol: p for p in self._adapter.get_all_positions()
        }

        diffs = []

        for symbol, local_pos in self._local_positions.items():
            exchange_pos = exchange_positions.get(symbol)

            if exchange_pos is None:
                # Position closed externally (liquidation, ADL)
                diffs.append(PositionDiff(
                    symbol=symbol,
                    diff_type="closed_externally",
                    local_qty=local_pos.qty,
                    exchange_qty=Decimal("0"),
                ))
            elif local_pos.qty != exchange_pos.qty:
                # Quantity mismatch
                diffs.append(PositionDiff(
                    symbol=symbol,
                    diff_type="qty_mismatch",
                    local_qty=local_pos.qty,
                    exchange_qty=exchange_pos.qty,
                ))

        return diffs
```

### Tests for Phase 9

```python
# NEW FILE: tests/test_futures_live_trading.py

class TestFuturesLiveRunner:
    """Live trading tests."""

    def test_position_sync_detects_external_close(self):
        """Detects when position closed by exchange (liquidation)."""
        pass

    def test_funding_payment_handling(self):
        """Real-time funding payments tracked."""
        pass
```

### Deliverables Phase 9
- [ ] `script_futures_live.py` - Live trading script
- [ ] `services/futures_position_sync.py` - Position sync
- [ ] `tests/test_futures_live_trading.py` (50+ tests)

---

## 📦 PHASE 10: TESTING & VALIDATION

### Цели
- Comprehensive test suite
- Validation metrics
- Documentation

### 10.1 Validation Metrics

```python
# NEW FILE: tests/test_futures_validation.py
"""
Futures simulation validation.

Target metrics:
- Fill rate: >95%
- Slippage error: <3 bps vs historical
- Funding rate accuracy: >99%
- Liquidation timing: <1 bar delay
- Margin calculation error: <0.1%
"""

class TestFuturesValidation:
    """Validation against historical data."""

    def test_fill_rate(self):
        """Fill rate >95% for limit orders within spread."""
        pass

    def test_slippage_accuracy(self):
        """Simulated slippage within 3 bps of historical."""
        pass

    def test_funding_rate_accuracy(self):
        """Funding payments match historical."""
        pass

    def test_liquidation_timing(self):
        """Liquidation triggered within 1 bar of mark price breach."""
        pass

    def test_margin_calculation_accuracy(self):
        """Margin calculations match exchange."""
        pass
```

### 10.2 Backward Compatibility Tests

```python
# NEW FILE: tests/test_futures_backward_compatibility.py
"""
Ensure crypto spot and equity unchanged.
"""

class TestBackwardCompatibility:
    """No regressions in existing functionality."""

    def test_crypto_spot_unchanged(self):
        """Crypto spot behavior unchanged."""
        pass

    def test_equity_unchanged(self):
        """Equity behavior unchanged."""
        pass

    def test_forex_unchanged(self):
        """Forex behavior unchanged."""
        pass
```

### 10.3 Performance Benchmarks

```python
# NEW FILE: benchmarks/bench_futures_simulation.py
"""
Futures simulation performance benchmarks.

Targets:
- L2 execution: <100μs per fill
- L3 execution: <1ms per fill
- Full day simulation: <10 seconds
"""

def benchmark_l2_execution():
    """L2 execution performance."""
    pass

def benchmark_l3_execution():
    """L3 execution performance."""
    pass
```

### 10.4 Documentation

```
docs/futures/
├── overview.md              # Futures integration overview
├── api_reference.md         # API reference
├── configuration.md         # Configuration guide
├── margin_calculation.md    # Margin mechanics
├── funding_rates.md         # Funding rate handling
├── liquidation.md           # Liquidation simulation
├── deployment.md            # Deployment guide
└── migration_guide.md       # Migration from spot
```

### Deliverables Phase 10
- [ ] `tests/test_futures_validation.py` (100+ tests)
- [ ] `tests/test_futures_backward_compatibility.py` (50+ tests)
- [ ] `benchmarks/bench_futures_simulation.py`
- [ ] Documentation suite in `docs/futures/`
- [ ] `FUTURES_INTEGRATION_REPORT.md` - Final validation report

---

## 📊 SUMMARY

### Total Deliverables by Track

#### Crypto Track (Binance)

| Phase | Description | Files | Tests |
|-------|-------------|-------|-------|
| 0 | Research | 0 | 0 (research) |
| 1 | Core Models (Unified) | 5 | 90+ |
| 2 | Data Download | 4 | 100+ |
| 3A | Binance Adapters | 3 | 50+ |
| 4A | Margin & Liquidation | 3 | 80+ |
| 5A | L3 LOB Integration | 2 | 60+ |
| 6A | Risk Management | 2 | 80+ |
| 7A | Features Pipeline | 3 | 50+ |
| **Subtotal** | | **22** | **510+** |

#### CME Track (Interactive Brokers)

| Phase | Description | Files | Tests |
|-------|-------------|-------|-------|
| 3B | IB TWS Adapters | 5 | 45+ |
| 4B | SPAN Margin | 3 | 55+ |
| 5B | CME L3 Execution | 2 | 55+ |
| 6B | CME Risk Guards | 2 | 70+ |
| 7B | CME Features | 2 | 40+ |
| **Subtotal** | | **14** | **265+** |

#### Shared Phases

| Phase | Description | Files | Tests |
|-------|-------------|-------|-------|
| 8 | Training Pipeline | 3 | 60+ |
| 9 | Live Trading | 3 | 50+ |
| 10 | Validation | 5+ | 150+ |
| **Subtotal** | | **11+** | **260+** |

### Grand Total

| Metric | Count |
|--------|-------|
| **Total Files** | **47+** |
| **Total Tests** | **1,035+** |
| **New Core Models** | 15+ |
| **New Adapters** | 12+ |
| **New Risk Guards** | 10+ |

### Key Metrics

| Metric | Crypto Target | CME Target |
|--------|---------------|------------|
| Simulation realism | 95%+ | 95%+ |
| Fill rate accuracy | >95% | >95% |
| Slippage error | <3 bps | <2 bps |
| Margin calc error | <0.5% | <0.1% (SPAN) |
| Test coverage | >90% | >90% |
| Backward compat | 100% | 100% |

### Futures Type Coverage

| Type | Exchange | Products | Status |
|------|----------|----------|--------|
| Crypto Perpetual | Binance | BTCUSDT, ETHUSDT, ... | Phase 1-7A |
| Crypto Quarterly | Binance | BTCUSDT_YYMMDD | Phase 1-7A |
| Index Futures | CME/CBOT | ES, NQ, MES, MNQ | Phase 3B-7B |
| Commodity Futures | COMEX/NYMEX | GC, SI, CL, NG | Phase 3B-7B |
| Currency Futures | CME | 6E, 6J, 6B, 6A | Phase 3B-7B |
| Bond Futures | CBOT | ZB, ZN, ZT | Phase 3B-7B |

### Risk Mitigation

1. **Backward compatibility**: All changes isolated, existing tests run on every phase
2. **Incremental delivery**: Each phase deployable independently (Crypto first, then CME)
3. **Feature flags**: Futures disabled by default, opt-in per asset class
4. **Parallel development**: Crypto and CME tracks can be developed concurrently
5. **Extensive testing**: 1,035+ tests across all phases
6. **Reuse existing L3**: Same LOB simulation, adapted for each exchange
7. **Documentation**: Complete docs for maintenance

### Development Timeline (Parallel Tracks)

```
Month 1-2: Phase 0-2 (Research + Core + Data) ─────────────────────────┐
                                                                       │
Month 2-3: ┌─ Phase 3A-4A (Crypto Margin)                              │
           └─ Phase 3B-4B (CME SPAN) ──────────────────────────────────┤
                                                                       │
Month 3-4: ┌─ Phase 5A-6A (Crypto L3 + Risk)                           │
           └─ Phase 5B-6B (CME L3 + Risk) ─────────────────────────────┤
                                                                       │
Month 4-5: ┌─ Phase 7A (Crypto Features)                               │
           └─ Phase 7B (CME Features) ─────────────────────────────────┤
                                                                       │
Month 5-6: Phase 8-10 (Training + Live + Validation) ──────────────────┘
```

---

## 🔗 REFERENCES

### Research Papers
1. Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
2. Kyle (1985): "Continuous Auctions and Insider Trading"
3. Cartea, Jaimungal, Penalva (2015): "Algorithmic and HF Trading"
4. Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
5. Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"

### Exchange Documentation

#### Binance (Crypto Futures)
1. [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
2. [Binance Leverage & Margin](https://www.binance.com/en/support/faq/360033162192)
3. [Binance Liquidation](https://www.binance.com/en/support/faq/360033525271)
4. [Binance Funding Rate](https://www.binance.com/en/support/faq/360033525031)

#### CME Group
5. [CME Globex Documentation](https://www.cmegroup.com/confluence/display/EPICSANDBOX/Globex)
6. [CME SPAN Margin](https://www.cmegroup.com/clearing/risk-management/span-overview.html)
7. [CME Circuit Breakers](https://www.cmegroup.com/education/articles-and-reports/understanding-stock-index-futures-circuit-breakers.html)
8. [CME Contract Specifications](https://www.cmegroup.com/trading/equity-index/us-index/e-mini-sandp500_contract_specifications.html)

#### Interactive Brokers
9. [IB TWS API](https://interactivebrokers.github.io/tws-api/)
10. [ib_insync Library](https://ib-insync.readthedocs.io/)
11. [IB Futures Trading](https://www.interactivebrokers.com/en/trading/futures.php)

### Internal Documentation
1. [L3 LOB Simulation](docs/l3_simulator/)
2. [Forex Integration](docs/forex/)
3. [Execution Providers](docs/execution_providers.md)
4. [Asset Class Defaults](configs/asset_class_defaults.yaml)

---

**Дата создания**: 2025-11-30
**Автор**: Claude Code
**Версия плана**: 2.0 (Unified Multi-Asset Futures)
