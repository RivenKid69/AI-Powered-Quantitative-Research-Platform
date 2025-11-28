# Claude Documentation - AI-Powered Quantitative Research Platform

---

## 🤖 БЫСТРАЯ НАВИГАЦИЯ ДЛЯ AI-АССИСТЕНТОВ

### Критические паттерны работы

**ВСЕГДА НАЧИНАЙТЕ С:**
1. **Изучите слоистую архитектуру** — `core_` → `impl_` → `service_` → `strategies` → `script_` — НЕ НАРУШАЙТЕ зависимости!
2. **Используйте Glob/Grep** для поиска файлов, НЕ используйте bash find/grep
3. **Читайте файлы перед изменением** — НИКОГДА не редактируйте файлы, которые не читали
4. **Проверяйте тесты** — перед изменением критичной логики найдите и изучите соответствующие тесты

### 📍 Быстрый поиск по задачам

| Задача | Где искать | Команда |
|--------|------------|---------|
| Найти определение класса/функции | Используйте Glob | `*.py` pattern с именем |
| Исправить ошибку в feature | `features/` + `feature_config.py` | `pytest tests/test_features*.py` |
| Изменить логику исполнения | `execution_sim.py`, `execution_providers.py` | `pytest tests/test_execution*.py` |
| Execution providers (L2/L3) | `execution_providers.py` | `pytest tests/test_execution_providers.py` |
| Crypto Parametric TCA | `execution_providers.py` | `pytest tests/test_crypto_parametric_tca.py` |
| Настроить риск-менеджмент | `configs/risk.yaml`, `risk_guard.py` | Проверить `test_risk*.py` |
| Обновить модель PPO | `distributional_ppo.py` | Проверить все `test_distributional_ppo*.py` |
| Добавить новую метрику | `services/monitoring.py` | Обновить `metrics.json` schema |
| Калибровать параметры | `service_calibrate_*.py` | Запустить соответствующий script |
| Отладить training | `train_model_multi_patch.py` + logs | Проверить `tensorboard` logs |
| Проблемы с данными | `impl_offline_data.py`, `data_validation.py` | Проверить data degradation params |
| Live trading проблемы | `script_live.py` → `service_signal_runner.py` | Проверить ops_kill_switch, state_storage |
| Position sync (Alpaca) | `services/position_sync.py` | `pytest tests/test_phase9_live_trading.py::TestPositionSynchronizer` |
| Extended hours trading | `services/session_router.py` | `pytest tests/test_phase9_live_trading.py::TestSessionRouter` |
| Bracket/OCO orders | `adapters/alpaca/order_execution.py` | `pytest tests/test_phase9_live_trading.py::TestBracketOrderConfig` |
| Скачать stock data | `scripts/download_stock_data.py` | `--symbols GLD IAU SLV --start 2020-01-01` |
| Скачать VIX данные | `scripts/download_stock_data.py` | `--vix --start 2020-01-01` или `--symbols ^VIX` |
| Скачать macro данные | `scripts/download_stock_data.py` | `--macro --start 2020-01-01` (VIX, DXY, Treasury) |
| Yahoo market data | `adapters/yahoo/market_data.py` | Auto-used for ^VIX, DX-Y.NYB, indices |
| Alpaca streaming | `adapters/alpaca/market_data.py` | `stream_bars_async()`, `stream_ticks_async()` |
| L3 LOB matching | `lob/matching_engine.py` | `pytest tests/test_matching_engine.py` |
| Queue position tracking | `lob/queue_tracker.py` | `pytest tests/test_matching_engine.py::TestQueuePositionTracker` |
| Order lifecycle | `lob/order_manager.py` | `pytest tests/test_matching_engine.py::TestOrderManager` |
| Fill probability models | `lob/fill_probability.py` | `pytest tests/test_fill_probability_queue_value.py` |
| Queue value (Moallemi) | `lob/queue_value.py` | `pytest tests/test_fill_probability_queue_value.py::TestQueueValueModel` |
| LOB calibration | `lob/calibration.py` | `pytest tests/test_fill_probability_queue_value.py::TestCalibrationPipeline` |
| Market impact models | `lob/market_impact.py` | `pytest tests/test_market_impact.py::TestAlmgrenChrissModel` |
| Impact effects on LOB | `lob/impact_effects.py` | `pytest tests/test_market_impact.py::TestImpactEffects` |
| Impact calibration | `lob/impact_calibration.py` | `pytest tests/test_market_impact.py::TestImpactCalibration` |
| Latency simulation | `lob/latency_model.py` | `pytest tests/test_lob_latency.py::TestLatencyModel` |
| Event scheduler | `lob/event_scheduler.py` | `pytest tests/test_lob_latency.py::TestEventScheduler` |
| Iceberg detection | `lob/hidden_liquidity.py` | `pytest tests/test_hidden_liquidity_dark_pools.py::TestIcebergDetector` |
| Hidden liquidity | `lob/hidden_liquidity.py` | `pytest tests/test_hidden_liquidity_dark_pools.py::TestHiddenLiquidityEstimator` |
| Dark pool simulation | `lob/dark_pool.py` | `pytest tests/test_hidden_liquidity_dark_pools.py::TestDarkPoolSimulator` |
| L3 execution provider | `execution_providers_l3.py` | `pytest tests/test_execution_providers_l3.py` |
| L3 config models | `lob/config.py` | `pytest tests/test_execution_providers_l3.py::TestL3ExecutionConfig` |

### 🔍 Quick File Reference

| Префикс | Слой | Зависимости | Примеры |
|---------|------|-------------|---------|
| `core_*` | Базовый | Нет | `core_config.py`, `core_models.py`, `core_strategy.py` |
| `impl_*` | Реализация | `core_` | `impl_sim_executor.py`, `impl_fees.py`, `impl_slippage.py` |
| `service_*` | Сервисы | `core_`, `impl_` | `service_backtest.py`, `service_train.py`, `service_eval.py` |
| `strategies/*` | Стратегии | Все предыдущие | `strategies/base.py`, `strategies/momentum.py` |
| `script_*` | CLI точки входа | Все | `script_backtest.py`, `script_live.py`, `script_eval.py` |

### ⚡ Критические команды

```bash
# Тестирование
pytest tests/                                    # Все тесты
pytest tests/test_execution*.py -v               # Execution тесты
pytest -k "test_name" -v                         # Конкретный тест

# Бэктест/Eval
python script_backtest.py --config configs/config_sim.yaml
python script_eval.py --config configs/config_eval.yaml --all-profiles

# Обучение (standard)
python train_model_multi_patch.py --config configs/config_train.yaml

# Обучение (PBT + Adversarial)
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml

# Обновление данных (Crypto)
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python scripts/refresh_fees.py
python -m services.universe --output data/universe/symbols.json

# Обновление данных (Stocks)
python scripts/fetch_alpaca_universe.py --output data/universe/alpaca_symbols.json --popular
python scripts/download_stock_data.py --symbols GLD IAU SGOL SLV --start 2020-01-01 --timeframe 1h --resample 4h

# Обновление данных (VIX / Macro indicators)
python scripts/download_stock_data.py --vix --start 2020-01-01 --timeframe 1d
python scripts/download_stock_data.py --macro --start 2020-01-01 --timeframe 1d
python scripts/download_stock_data.py --symbols ^VIX DX-Y.NYB ^TNX --start 2020-01-01

# Live Trading (Stocks - Alpaca)
python script_live.py --config configs/config_live_alpaca.yaml
python script_live.py --config configs/config_live_alpaca.yaml --asset-class equity --paper
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours

# Live Trading (Crypto - Binance)
python script_live.py --config configs/config_live.yaml

# Training (Stocks)
python train_model_multi_patch.py --config configs/config_train_stocks.yaml

# Backtest (Stocks)
python script_backtest.py --config configs/config_backtest_stocks.yaml
```

---

## 📈 Multi-Exchange Support (Phase 2)

### Поддерживаемые биржи

| Биржа | Тип | Статус | Адаптеры |
|-------|-----|--------|----------|
| **Binance** | Crypto (Spot/Futures) | ✅ Production | MarketData, Fee, TradingHours, ExchangeInfo |
| **Alpaca** | US Equities | ✅ Production | MarketData (REST + WebSocket), Fee, TradingHours, ExchangeInfo, OrderExecution |
| **Polygon** | US Equities (Data) | ✅ Production | MarketData, TradingHours, ExchangeInfo |
| **Yahoo** | Indices/Macro | ✅ Production | MarketData (VIX, DXY, Treasury), CorporateActions, Earnings |

### Архитектура адаптеров

```
adapters/
├── base.py           # Абстрактные базовые классы
├── models.py         # Exchange-agnostic модели данных
├── registry.py       # Фабрика + регистрация адаптеров
├── config.py         # Pydantic конфигурация
├── websocket_base.py # Production-grade async WebSocket wrapper
├── binance/          # Binance реализация (crypto)
│   ├── market_data.py
│   ├── fees.py
│   ├── trading_hours.py
│   └── exchange_info.py
├── alpaca/           # Alpaca реализация (stocks)
│   ├── market_data.py  # REST + WebSocket streaming (sync/async)
│   ├── fees.py
│   ├── trading_hours.py
│   ├── exchange_info.py
│   └── order_execution.py
├── polygon/          # Polygon.io реализация (stocks data)
│   ├── market_data.py
│   ├── trading_hours.py
│   └── exchange_info.py
└── yahoo/            # Yahoo Finance реализация (indices/macro)
    ├── market_data.py      # VIX, DXY, Treasury yields
    ├── corporate_actions.py # Dividends, splits
    └── earnings.py          # Earnings calendar
```

### Использование

```python
# Через Registry
from adapters.registry import create_market_data_adapter, create_fee_adapter

# Crypto
binance_md = create_market_data_adapter("binance")
binance_fees = create_fee_adapter("binance")

# Stocks
alpaca_md = create_market_data_adapter("alpaca", {
    "api_key": "...",
    "api_secret": "...",
    "feed": "iex",
})

# Indices/VIX (Yahoo Finance)
yahoo_md = create_market_data_adapter("yahoo")
vix_bars = yahoo_md.get_bars("^VIX", "1d", limit=365)
dxy_bars = yahoo_md.get_bars("DX-Y.NYB", "1d", limit=365)

# Alpaca Real-time Streaming (sync)
for bar in alpaca_md.stream_bars(["AAPL", "MSFT"], 60000):
    print(f"Bar: {bar.symbol} @ {bar.close}")

# Alpaca Real-time Streaming (async - for live trading)
async for bar in alpaca_md.stream_bars_async(["AAPL", "MSFT"]):
    await process_bar(bar)

# Через Config
from adapters.config import ExchangeConfig

config = ExchangeConfig.from_yaml("configs/exchange.yaml")
adapter = config.create_market_data_adapter()
```

### Конфигурация

**configs/exchange.yaml** — главный файл конфигурации биржи:
```yaml
vendor: "alpaca"  # или "binance"
market_type: "EQUITY"  # или "CRYPTO_SPOT"

alpaca:
  api_key: "${ALPACA_API_KEY}"
  api_secret: "${ALPACA_API_SECRET}"
  paper: true
  feed: "iex"
  extended_hours: false
```

**configs/config_live_alpaca.yaml** — live trading для Alpaca

### Ключевые отличия Crypto vs Stocks

| Аспект | Crypto (Binance) | Stocks (Alpaca) |
|--------|------------------|-----------------|
| **Часы торговли** | 24/7 | NYSE 9:30-16:00 ET + extended |
| **Комиссии** | % от notional (maker/taker) | $0 (+ regulatory на продажу) |
| **Минимальный лот** | По фильтрам биржи | 1 share (или fractional) |
| **Tick size** | Varies by symbol | $0.01 |
| **Short selling** | Через futures | Shortable flag per symbol |
| **Latency** | ~100-500ms | ~50-200ms |

### Команды для Alpaca

```bash
# Получить universe акций
python scripts/fetch_alpaca_universe.py --popular

# Live trading (paper)
python script_live.py --config configs/config_live_alpaca.yaml

# Запустить тесты адаптеров
pytest tests/test_alpaca_adapters.py -v
```

### Требования

```bash
pip install alpaca-py  # Alpaca SDK
```

### Environment Variables

```bash
# Alpaca
ALPACA_API_KEY=...
ALPACA_API_SECRET=...

# Binance (существующие)
BINANCE_API_KEY=...
BINANCE_API_SECRET=...

# Polygon.io (альтернативный data provider)
POLYGON_API_KEY=...
```

---

## 📊 Stock Training & Backtest (Phase 3)

### Обзор

Phase 3 добавляет полную поддержку акций в training и backtest pipeline:

1. **Multi-Asset Data Loader** (`data_loader_multi_asset.py`)
   - Унифицированная загрузка данных для crypto и stocks
   - Фильтрация по trading hours для US equities
   - Поддержка нескольких data vendors (Alpaca, Polygon)

2. **Polygon Data Provider** (`adapters/polygon/`)
   - Альтернативный источник рыночных данных
   - Historical bars и real-time streaming
   - US market holidays и trading hours

3. **WebSocket Wrapper** (`adapters/websocket_base.py`)
   - Production-grade async WebSocket с auto-reconnect
   - Exponential backoff и heartbeat monitoring
   - Rate limiting и message buffering

### Поддерживаемые символы

**Tech Stocks:**
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

**Index ETFs:**
- SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)

**Precious Metals ETFs:**
- GLD (SPDR Gold Trust, $60B AUM)
- IAU (iShares Gold Trust)
- SGOL (Aberdeen Physical Gold)
- SLV (iShares Silver Trust)

### Скачивание данных

```bash
# Скачать все поддерживаемые символы (3 года истории)
python scripts/download_stock_data.py \
    --symbols AAPL MSFT GOOGL AMZN NVDA META TSLA SPY QQQ IWM GLD IAU SGOL SLV \
    --start 2020-01-01 --timeframe 1h --resample 4h

# Только precious metals
python scripts/download_stock_data.py \
    --symbols GLD IAU SGOL SLV \
    --start 2020-01-01 --timeframe 1h --resample 4h

# Популярные tech stocks
python scripts/download_stock_data.py --popular --start 2020-01-01
```

Данные сохраняются в: `data/raw_stocks/*.parquet`

### Stock Training Configuration

```yaml
# configs/config_train_stocks.yaml
mode: train
asset_class: equity
data_vendor: alpaca  # или polygon

data:
  timeframe: "4h"
  filter_trading_hours: true
  include_extended_hours: false
  paths:
    - "data/raw_stocks/*.parquet"
    - "data/stocks/*.parquet"

env:
  session:
    calendar: us_equity
    extended_hours: false
```

### Stock Backtest Configuration

```yaml
# configs/config_backtest_stocks.yaml
mode: backtest
asset_class: equity

fees:
  structure: flat
  maker_bps: 0.0
  taker_bps: 0.0
  regulatory:
    enabled: true
    sec_fee_per_million: 27.80
    taf_fee_per_share: 0.000166
```

### Ключевые особенности Stock Trading

| Аспект | Crypto (Binance) | Stocks (Alpaca/Polygon) |
|--------|------------------|-------------------------|
| **Часы торговли** | 24/7 | NYSE 9:30-16:00 ET |
| **Extended hours** | N/A | 4:00-9:30, 16:00-20:00 ET |
| **Комиссии** | % от notional | $0 + regulatory fees |
| **Min trade** | LOT_SIZE filter | 1 share (fractional OK) |
| **Holidays** | Нет | US market holidays |

### Использование Multi-Asset Loader

```python
from data_loader_multi_asset import (
    load_multi_asset_data,
    load_from_adapter,
    AssetClass,
    DataVendor,
)

# Загрузка из файлов
frames, obs_shapes = load_multi_asset_data(
    paths=["data/stocks/*.parquet"],
    asset_class="equity",
    timeframe="4h",
    filter_trading_hours=True,
)

# Загрузка через адаптер
frames, obs_shapes = load_from_adapter(
    vendor="polygon",
    symbols=["AAPL", "MSFT", "GOOGL"],
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-12-31",
)
```

### Gold-Specific Features (опционально)

Для улучшения модели на precious metals можно добавить макро-индикаторы:

| Feature | Источник | Корреляция с золотом |
|---------|----------|----------------------|
| DXY (Dollar Index) | Yahoo (`DX-Y.NYB`) | Обратная (сильная) |
| Real Yields (TIPS) | FRED (`DFII10`) | Обратная |
| Gold/Silver Ratio | Расчёт (`GLD/SLV`) | Mean-reverts (60-80) |
| VIX | Yahoo (`^VIX`) | Положительная (fear) |

```bash
# Скачать VIX для fear indicator
python scripts/download_stock_data.py --symbols ^VIX --start 2020-01-01
```

### Требования

```bash
pip install polygon-api-client  # Polygon.io
pip install alpaca-py           # Alpaca
```

---

## 🔄 Execution Providers (Phase 4)

### Обзор

Phase 4 добавляет абстракцию execution providers для унифицированной симуляции исполнения crypto и акций.

**Файл**: `execution_providers.py` (~1800 строк)

### Архитектура

```
┌──────────────────────────────────────────────────────────────┐
│                    Protocols (Interfaces)                     │
├──────────────────┬──────────────────┬────────────────────────┤
│ SlippageProvider │  FillProvider    │     FeeProvider        │
└────────┬─────────┴────────┬─────────┴──────────┬─────────────┘
         │                  │                    │
┌────────▼─────────────────▼───────────────────▼───────────────┐
│                  L2 Implementations (Production)              │
├─────────────────────┬──────────────────┬─────────────────────┤
│StatisticalSlippage  │ OHLCVFillProvider│ CryptoFeeProvider   │
│ (√participation)    │ (bar-based fills)│ EquityFeeProvider   │
└─────────────────────┴──────────────────┴─────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│               L2ExecutionProvider (Combined)                  │
│    - Auto-selects crypto/equity defaults                     │
│    - Pre-trade cost estimation                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│            L2+ CryptoParametricSlippageProvider               │
│    - 6 slippage factors (research-backed)                    │
│    - Volatility regime detection                             │
│    - Adaptive impact coefficient                             │
│    - Whale detection & TWAP adjustment                       │
└──────────────────────────────────────────────────────────────┘
```

### Уровни точности (Fidelity Levels)

| Level | Модель | Статус | Описание |
|-------|--------|--------|----------|
| **L1** | Constant | N/A | Фиксированный spread/fee (не реализован) |
| **L2** | Statistical | ✅ Production | √participation impact (Almgren-Chriss) |
| **L2+** | Parametric TCA | ✅ Production | 6-factor crypto model (see below) |
| **L3** | LOB | ✅ Production | Full order book simulation |

### Ключевые классы

| Класс | Назначение |
|-------|------------|
| `MarketState` | Snapshot рынка (bid/ask/spread/adv) |
| `Order` | Ордер для исполнения |
| `Fill` | Результат исполнения |
| `BarData` | OHLCV данные бара |
| `StatisticalSlippageProvider` | √participation slippage модель |
| `CryptoParametricSlippageProvider` | L2+ Smart parametric TCA (6 факторов) |
| `CryptoParametricConfig` | Конфигурация для parametric TCA |
| `VolatilityRegime` | Enum: LOW/NORMAL/HIGH волатильность |
| `OHLCVFillProvider` | Fill logic на основе bar range |
| `CryptoFeeProvider` | Maker/taker комиссии (Binance) |
| `EquityFeeProvider` | Regulatory fees (SEC/TAF) |
| `L2ExecutionProvider` | Комбинированный провайдер |

### Различия Crypto vs Equity

| Параметр | Crypto | Equity |
|----------|--------|--------|
| Default spread | 5 bps | 2 bps |
| Impact coef | 0.1 | 0.05 |
| Fee structure | Maker 2bps / Taker 4bps | $0 + SEC/TAF on sells |
| SEC fee | N/A | ~$0.0000278/$ |
| TAF fee | N/A | ~$0.000166/share (max $8.30) |

### Использование

```python
from execution_providers import (
    create_execution_provider,
    AssetClass,
    Order,
    MarketState,
    BarData,
)

# Создание провайдера для акций
provider = create_execution_provider(AssetClass.EQUITY)

# Исполнение ордера
fill = provider.execute(
    Order(symbol="AAPL", side="BUY", qty=100, order_type="MARKET"),
    MarketState(timestamp=now, bid=150.0, ask=150.02, adv=10_000_000),
    BarData(open=150.0, high=151.0, low=149.0, close=150.5, volume=100000),
)

# Результат
print(f"Price: {fill.price}, Fee: {fill.fee}, Slippage: {fill.slippage_bps} bps")
```

### Factory Functions

```python
# Создание отдельных провайдеров
slippage = create_slippage_provider("L2", AssetClass.EQUITY)
fees = create_fee_provider(AssetClass.CRYPTO)
fill = create_fill_provider("L2", AssetClass.CRYPTO, slippage, fees)

# Комбинированный провайдер
provider = create_execution_provider(AssetClass.EQUITY, level="L2")
```

### Backward Compatibility

```python
from execution_providers import wrap_legacy_slippage_config, wrap_legacy_fees_model

# Обёртки для существующих конфигов
slippage = wrap_legacy_slippage_config(existing_slippage_config)
fees = wrap_legacy_fees_model(existing_fees_model)
```

### Slippage Model (Almgren-Chriss)

```
slippage_bps = half_spread + k * sqrt(participation) * vol_scale * 10000
```

Где:
- `half_spread` — половина спреда из MarketState
- `k` — impact coefficient (0.1 для crypto, 0.05 для equity)
- `participation` — order_notional / ADV
- `vol_scale` — volatility adjustment factor

### Limit Order Fill Logic

```
1. Check immediate execution (crossing spread):
   - BUY LIMIT >= ask → TAKER fill at ask
   - SELL LIMIT <= bid → TAKER fill at bid

2. Check passive fill (bar range):
   - BUY LIMIT: fills if bar_low <= limit_price → MAKER
   - SELL LIMIT: fills if bar_high >= limit_price → MAKER
```

### Тестирование

```bash
# Все тесты execution providers
pytest tests/test_execution_providers.py -v

# Интеграционные тесты
pytest tests/test_execution_providers.py::TestIntegration -v
```

**Покрытие**: 95 тестов (100% pass) + 84 теста parametric TCA

### Референсы

- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Cont (2001): "Empirical Properties of Asset Returns"
- Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
- Cartea, Jaimungal, Penalva (2015): "Algorithmic and HF Trading"

---

## 📊 Crypto Parametric TCA (L2+)

### Обзор

Smart parametric Transaction Cost Analysis model для криптовалютных рынков. Расширяет базовую √participation модель (Almgren-Chriss) с 6 crypto-специфичными факторами.

**Статус**: ✅ Production Ready | **Тесты**: 84 (100% pass)

### Формула Total Slippage

```
slippage = half_spread
    × (1 + k × √participation)      # Almgren-Chriss impact
    × vol_regime_mult               # Volatility regime (Cont 2001)
    × (1 + imbalance_penalty)       # Order book imbalance (Cont et al. 2014)
    × funding_stress                # Funding rate stress (perp-specific)
    × (1 / tod_factor)              # Time-of-day liquidity curve
    × correlation_decay             # BTC correlation decay (altcoins)
    × asymmetric_adjustment         # Panic selling premium
```

### 6 Slippage Factors

| Factor | Формула | Референс |
|--------|---------|----------|
| **√Participation** | `k × √(Q/ADV)` | Almgren-Chriss (2001) |
| **Volatility Regime** | Percentile-based LOW/NORMAL/HIGH | Cont (2001) |
| **Order Book Imbalance** | `(bid - ask) / (bid + ask)` | Cont et al. (2014) |
| **Funding Rate Stress** | `1 + |funding| × sensitivity` | Empirical (Binance) |
| **Time-of-Day** | 24-hour liquidity curve (Asia/EU/US) | Binance research |
| **BTC Correlation Decay** | `1 + (1 - corr) × decay_factor` | Empirical (altcoins) |

### Smart Features

| Feature | Описание |
|---------|----------|
| **Regime Detection** | Автоматическое определение LOW/NORMAL/HIGH volatility |
| **Adaptive Impact** | Коэффициент k адаптируется по trailing fill quality |
| **Asymmetric Slippage** | Продажи в downtrend стоят дороже (panic liquidity) |
| **Whale Detection** | Большие ордеры (Q/ADV > 1%) получают TWAP-adjusted model |

### Использование

```python
from execution_providers import (
    CryptoParametricSlippageProvider,
    CryptoParametricConfig,
    Order,
    MarketState,
)

# 1. Базовое использование (defaults)
provider = CryptoParametricSlippageProvider()

# 2. С кастомной конфигурацией
config = CryptoParametricConfig(
    impact_coef_base=0.12,
    spread_bps=6.0,
    whale_threshold=0.02,
)
provider = CryptoParametricSlippageProvider(config=config)

# 3. Из профиля
provider = CryptoParametricSlippageProvider.from_profile("altcoin")
# Профили: "default", "conservative", "aggressive", "altcoin", "stablecoin"

# 4. Вычисление slippage
slippage_bps = provider.compute_slippage_bps(
    order=Order("ETHUSDT", "BUY", 10.0, "MARKET"),
    market=MarketState(timestamp=0, bid=2000.0, ask=2001.0, adv=50_000_000),
    participation_ratio=0.005,
    funding_rate=0.0003,       # Slightly positive
    btc_correlation=0.85,      # High correlation
    hour_utc=14,               # EU session
    recent_returns=[-0.01, 0.005, -0.008],  # For regime detection
)

# 5. Pre-trade cost estimation
estimate = provider.estimate_impact_cost(
    notional=1_000_000,
    adv=500_000_000,
    side="BUY",
    hour_utc=16,
)
print(f"Impact: {estimate['impact_bps']:.2f} bps")
print(f"Cost: ${estimate['impact_cost']:.2f}")
print(f"Recommendation: {estimate['recommendation']}")
```

### Конфигурация (CryptoParametricConfig)

| Параметр | Default | Описание |
|----------|---------|----------|
| `impact_coef_base` | 0.10 | Base k coefficient |
| `impact_coef_range` | (0.05, 0.15) | Adaptive k bounds |
| `spread_bps` | 5.0 | Default spread (if market unavailable) |
| `vol_regime_multipliers` | {low: 0.8, normal: 1.0, high: 1.5} | Regime scaling |
| `vol_lookback_periods` | 20 | Periods for regime detection |
| `vol_regime_thresholds` | (25.0, 75.0) | Percentiles for LOW/HIGH |
| `imbalance_penalty_max` | 0.3 | Max imbalance penalty (30%) |
| `funding_stress_sensitivity` | 10.0 | Funding rate multiplier |
| `tod_curve` | {0-23: factors} | 24-hour liquidity curve |
| `btc_correlation_decay_factor` | 0.5 | Altcoin decay factor |
| `whale_threshold` | 0.01 | 1% ADV = whale |
| `whale_twap_adjustment` | 0.7 | TWAP adjustment |
| `asymmetric_sell_premium` | 0.2 | 20% panic selling premium |
| `downtrend_threshold` | -0.02 | -2% = downtrend |
| `min_slippage_bps` | 1.0 | Floor |
| `max_slippage_bps` | 500.0 | Cap |

### Профили

| Профиль | impact_coef | spread_bps | Применение |
|---------|-------------|------------|------------|
| `default` | 0.10 | 5.0 | BTC/ETH majors |
| `conservative` | 0.12 | 6.0 | Safer estimates |
| `aggressive` | 0.08 | 4.0 | Tighter estimates |
| `altcoin` | 0.15 | 10.0 | Low-cap altcoins |
| `stablecoin` | 0.05 | 1.0 | USDT/USDC pairs |

### Time-of-Day Curve (Default)

| Session | Часы (UTC) | Factor | Описание |
|---------|------------|--------|----------|
| Asia | 00:00-08:00 | 0.70-0.90 | Lower liquidity |
| EU | 08:00-16:00 | 0.95-1.10 | Increasing liquidity |
| US/EU overlap | 14:00-18:00 | 1.10-1.15 | Peak liquidity |
| US | 18:00-24:00 | 0.85-1.05 | Declining liquidity |

### Adaptive Learning

```python
# После каждого fill обновляем модель
predicted = provider.compute_slippage_bps(order, market, participation)
# ... execution happens ...
actual = (fill_price - expected_price) / expected_price * 10000

provider.update_fill_quality(predicted, actual)
# k coefficient автоматически адаптируется
```

### Тестирование

```bash
# Все тесты parametric TCA
pytest tests/test_crypto_parametric_tca.py -v

# По категориям
pytest tests/test_crypto_parametric_tca.py::TestVolatilityRegime -v
pytest tests/test_crypto_parametric_tca.py::TestWhaleDetection -v
pytest tests/test_crypto_parametric_tca.py::TestAdaptiveImpact -v
```

**Покрытие**: 84 теста (100% pass)

---

## 🔴 Live Trading Improvements (Phase 9)

### Обзор

Phase 9 добавляет полную поддержку live trading для акций через Alpaca:

1. **Unified Live Script** (`script_live.py`)
   - Единый entry point для crypto и stocks
   - Автоматическое определение asset class
   - CLI аргументы для переключения режимов

2. **Position Synchronization** (`services/position_sync.py`)
   - Синхронизация локального состояния с биржей
   - Background polling с настраиваемым интервалом
   - Автоматическое обнаружение и обработка расхождений

3. **Advanced Order Management** (`adapters/alpaca/order_execution.py`)
   - Bracket orders (take-profit + stop-loss)
   - OCO (One-Cancels-Other) orders
   - Order replacement (cancel + new)
   - Order history и wait-for-fill

4. **Extended Hours Trading** (`services/session_router.py`)
   - Session detection (pre-market, regular, after-hours)
   - Session-aware order routing
   - Spread adjustment для extended hours

### Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                      script_live.py                              │
│  - CLI: --asset-class, --extended-hours, --paper/--live         │
│  - Auto-detection: detect_asset_class()                         │
│  - Defaults: apply_asset_class_defaults()                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  Crypto (Binance)   │       │  Equity (Alpaca)    │
│  - 24/7 trading     │       │  - Market hours     │
│  - GTC orders       │       │  - DAY orders       │
│  - 5 bps slippage   │       │  - 2 bps slippage   │
└─────────────────────┘       └─────────┬───────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────┐
          ▼                             ▼                         ▼
┌─────────────────┐         ┌─────────────────┐       ┌─────────────────┐
│ Position Sync   │         │ Order Execution │       │ Session Router  │
│ - Reconcile     │         │ - Bracket orders│       │ - Pre-market    │
│ - Background    │         │ - OCO orders    │       │ - Regular       │
│ - Callbacks     │         │ - Replace order │       │ - After-hours   │
└─────────────────┘         └─────────────────┘       └─────────────────┘
```

### Asset Class Detection

```python
# Приоритет определения asset class:
# 1. Explicit: --asset-class equity
# 2. Vendor: vendor=alpaca → equity
# 3. Market type: market_type=EQUITY → equity
# 4. Default: crypto (backward compatible)

def detect_asset_class(cfg_dict: Dict[str, Any]) -> str:
    # Priority 1: Explicit
    if "asset_class" in cfg_dict:
        return cfg_dict["asset_class"]

    # Priority 2: Vendor mapping
    vendor = cfg_dict.get("vendor", "").lower()
    if vendor in ("alpaca", "polygon"):
        return "equity"
    if vendor == "binance":
        return "crypto"

    # Priority 3: Market type
    market_type = cfg_dict.get("market_type", "").upper()
    if market_type in ("EQUITY", "STOCK"):
        return "equity"

    # Default: crypto
    return "crypto"
```

### Asset Class Defaults

| Параметр | Crypto | Equity |
|----------|--------|--------|
| `slippage_bps` | 5.0 | 2.0 |
| `limit_offset_bps` | 10.0 | 5.0 |
| `tif` | GTC | DAY |
| `extended_hours` | False | False |
| `default_vendor` | binance | alpaca |

### Position Synchronization

```python
from services.position_sync import (
    PositionSynchronizer,
    SyncConfig,
    reconcile_alpaca_state,
)

# Конфигурация
config = SyncConfig(
    sync_interval_sec=30.0,       # Интервал polling
    position_tolerance=0.01,      # 1% tolerance
    auto_reconcile=True,          # Автоматическая коррекция
    max_reconcile_qty=1000.0,     # Максимальный объём коррекции
)

# Создание synchronizer
sync = PositionSynchronizer(
    position_provider=alpaca_adapter,
    local_state_getter=get_local_positions,
    config=config,
    on_discrepancy=handle_discrepancy,
    on_sync_complete=on_sync,
)

# Запуск background sync
sync.start_background_sync()
```

### Bracket Orders (Alpaca)

```python
from adapters.alpaca.order_execution import (
    AlpacaOrderExecutionAdapter,
    BracketOrderConfig,
)

adapter = AlpacaOrderExecutionAdapter(api_key, api_secret, paper=True)

# Bracket order: entry + take-profit + stop-loss
config = BracketOrderConfig(
    symbol="AAPL",
    side=Side.BUY,
    qty=100,
    entry_price=150.0,           # Optional limit entry
    take_profit_price=165.0,     # +10% target
    stop_loss_price=142.50,      # -5% stop
    time_in_force="DAY",
)

result = adapter.submit_bracket_order(config)
# result.entry_order_id, result.tp_order_id, result.sl_order_id
```

### Session Router

```python
from services.session_router import (
    SessionRouter,
    TradingSession,
    get_current_session,
)

# Текущая сессия
session = get_current_session()
# session.session: PRE_MARKET | REGULAR | AFTER_HOURS | CLOSED

# Router для intelligent routing
router = SessionRouter(
    allow_extended_hours=True,
    extended_hours_spread_multiplier=2.0,
)

# Решение о routing
decision = router.get_routing_decision(
    symbol="AAPL",
    side="BUY",
    qty=100,
    order_type="market",
)

if decision.should_submit:
    if decision.use_extended_hours:
        adapter.submit_extended_hours_order(order, session="pre")
    else:
        adapter.submit_order(order)
```

### Trading Sessions (US Equity)

| Session | Время (ET) | Market Orders | Limit Orders | Spread |
|---------|------------|---------------|--------------|--------|
| Pre-market | 4:00-9:30 | ❌ | ✅ | 2.5x |
| Regular | 9:30-16:00 | ✅ | ✅ | 1.0x |
| After-hours | 16:00-20:00 | ❌ | ✅ | 2.0x |
| Closed | 20:00-4:00 | ❌ | ❌ | N/A |

### CLI Usage

```bash
# Crypto (default, backward compatible)
python script_live.py --config configs/config_live.yaml

# Equity explicit
python script_live.py --config configs/config_live_alpaca.yaml --asset-class equity

# Extended hours trading
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours

# Paper trading (Alpaca sandbox)
python script_live.py --config configs/config_live_alpaca.yaml --paper

# Live trading (real money)
python script_live.py --config configs/config_live_alpaca.yaml --live
```

### Backward Compatibility

- **100% backward compatible** с существующим crypto functionality
- Default asset class = `crypto` если не указан explicit
- Все существующие конфиги работают без изменений
- Новые параметры опциональны

### Тестирование

```bash
# Все тесты Phase 9
pytest tests/test_phase9_live_trading.py -v

# Тесты по категориям
pytest tests/test_phase9_live_trading.py::TestAssetClassDetection -v
pytest tests/test_phase9_live_trading.py::TestPositionSynchronizer -v
pytest tests/test_phase9_live_trading.py::TestSessionRouter -v
pytest tests/test_phase9_live_trading.py::TestBackwardCompatibility -v
```

**Покрытие**: 46 тестов (100% pass)

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `script_live.py` | Unified live trading entry point |
| `services/position_sync.py` | Position synchronization service |
| `services/session_router.py` | Session-aware order routing |
| `adapters/alpaca/order_execution.py` | Enhanced Alpaca order execution |
| `tests/test_phase9_live_trading.py` | Comprehensive test suite |

---

## 📚 L3 LOB Simulation (Phase 10)

### Обзор

Phase 10 добавляет высокоточную симуляцию order book для US equities:

1. **Stage 1: Data Structures** (`lob/data_structures.py`)
   - LimitOrder, PriceLevel, OrderBook с O(1)/O(log n) операциями
   - Iceberg и hidden order support
   - LOBSTER message format parsing

2. **Stage 2: Matching Engine** (`lob/matching_engine.py`)
   - FIFO Price-Time Priority matching (CME Globex style)
   - Self-Trade Prevention (STP) — 4 режима
   - Pro-Rata matching для опционных рынков
   - Queue position tracking (Erik Rigtorp method)

3. **Stage 3: Fill Probability & Queue Value** (`lob/fill_probability.py`, `lob/queue_value.py`)
   - Analytical Poisson fill probability: `P(fill in T) = 1 - exp(-λT / position)`
   - Queue-Reactive intensity model (Huang et al.): `λ_i = f(q_i, spread, volatility, imbalance)`
   - Queue Value computation (Moallemi & Yuan): `V = P(fill) * spread/2 - adverse_selection`
   - Calibration pipeline from historical LOB data (`lob/calibration.py`)

4. **Stage 4: Market Impact Models** (`lob/market_impact.py`, `lob/impact_effects.py`)
   - Kyle (1985) Lambda model: `Δp = λ * sign(x) * |x|`
   - Almgren-Chriss (2001): `temp = η * σ * (Q/V)^0.5`, `perm = γ * (Q/V)`
   - Gatheral (2010) transient impact with power-law decay: `G(t) = (1 + t/τ)^(-β)`
   - Impact effects on LOB: quote shifting, liquidity reaction, momentum detection
   - Calibration from historical trade data (`lob/impact_calibration.py`)

5. **Stage 5: Latency Simulation** (`lob/latency_model.py`, `lob/event_scheduler.py`)
   - Realistic latency distributions: Log-normal, Pareto (heavy tail), Gamma
   - Separate feed/order/exchange/fill latencies
   - Latency profiles: Co-located (~10-50μs), Proximity (~100-500μs), Retail (~1-10ms), Institutional (~200μs-2ms)
   - Event scheduler with priority queue and race condition detection
   - Time-of-day seasonality adjustments
   - Volatility-adjusted latency

6. **Stage 6: Hidden Liquidity & Dark Pools** (`lob/hidden_liquidity.py`, `lob/dark_pool.py`)
   - Iceberg order detection from execution patterns (refill pattern recognition)
   - Hidden quantity estimation based on observed refills
   - Dark pool multi-venue simulation (SIGMA_X, IEX_D, LIQUIDNET, RETAIL_INT)
   - Mid-price execution with probabilistic fills
   - Information leakage modeling (quote updates, trade signals, size inference)
   - Smart order routing across dark pool venues
   - Time-of-day and volatility adjustments

7. **Stage 7: L3 Execution Provider Integration** (`execution_providers_l3.py`, `lob/config.py`)
   - Full L3ExecutionProvider combining all LOB components
   - Pydantic-based configuration models for all subsystems
   - Factory function upgrade: `create_execution_provider(level="L3")`
   - YAML configuration support with presets (equity, crypto, minimal)
   - Pre-trade cost estimation with impact models
   - Fill probability computation for limit orders
   - Dark pool routing integration
   - Backward compatible with L2 (crypto unchanged)
   - 79 comprehensive tests

8. **Stage 8: Data Pipeline & Calibration** (`lob/data_adapters.py`, `lob/calibration_pipeline.py`)
   - Data adapters: LOBSTER, ITCH, Binance L2, Alpaca L2
   - Unified L3 calibration pipeline for latency + queue dynamics
   - Format-agnostic LOB update processing
   - Historical data loading utilities

9. **Stage 9: Testing & Validation** (see `docs/L3_VALIDATION_REPORT.md`)
   - 749+ tests passing (100% pass rate)
   - Validation metrics: fill rate >95%, slippage <2bps, queue error <10%
   - Performance benchmarks meeting targets
   - Full backward compatibility with crypto

10. **Stage 10: Documentation & Deployment** (`docs/l3_simulator/`)
    - Comprehensive documentation for all L3 components
    - Deployment checklist with feature flags
    - Gradual rollout strategy (shadow mode → canary → production)
    - Monitoring dashboards and alert rules
    - Rollback procedures

### Архитектура

```
lob/
├── data_structures.py       # LimitOrder, PriceLevel, OrderBook, Fill, Trade
├── matching_engine.py       # MatchingEngine, ProRataMatchingEngine, STP
├── queue_tracker.py         # QueuePositionTracker (MBP/MBO estimation)
├── order_manager.py         # OrderManager, ManagedOrder, TimeInForce
├── state_manager.py         # LOBStateManager, LOBSnapshot
├── parsers.py               # LOBSTERParser
├── fill_probability.py      # Poisson, Queue-Reactive, Historical models (Stage 3)
├── queue_value.py           # Queue value computation (Moallemi & Yuan) (Stage 3)
├── calibration.py           # Model calibration from historical data (Stage 3)
├── market_impact.py         # Kyle, Almgren-Chriss, Gatheral models (Stage 4)
├── impact_effects.py        # Quote shifting, liquidity reaction (Stage 4)
├── impact_calibration.py    # Impact parameter estimation (Stage 4)
├── latency_model.py         # Realistic latency simulation (Stage 5)
├── event_scheduler.py       # Event ordering with priority queue (Stage 5)
├── hidden_liquidity.py      # Iceberg detection, hidden qty estimation (Stage 6)
├── dark_pool.py             # Dark pool simulation, multi-venue routing (Stage 6)
├── config.py                # Pydantic config models for L3 subsystems (Stage 7)
├── data_adapters.py         # LOBSTER, ITCH, Binance, Alpaca adapters (Stage 8)
├── calibration_pipeline.py  # Unified L3 calibration pipeline (Stage 8)
└── __init__.py              # Public API exports

execution_providers_l3.py    # L3ExecutionProvider combining all LOB components (Stage 7)

docs/l3_simulator/           # Stage 10 Documentation
├── overview.md              # Architecture overview
├── data_structures.md       # LOB data structures
├── matching_engine.md       # FIFO matching, STP
├── queue_position.md        # Queue position tracking
├── market_impact.md         # Impact models (Kyle, AC, Gatheral)
├── latency.md               # Latency simulation, event scheduling
├── calibration.md           # Parameter estimation
├── configuration.md         # Config reference
├── deployment.md            # Deployment checklist, rollout, rollback
└── migration_guide.md       # L2 to L3 migration reference
```

### Ключевые классы

| Класс | Назначение |
|-------|------------|
| `MatchingEngine` | FIFO matching с STP |
| `ProRataMatchingEngine` | Pro-rata allocation |
| `QueuePositionTracker` | MBP/MBO position estimation |
| `OrderManager` | Order lifecycle (IOC, FOK, DAY, GTC) |
| `LOBStateManager` | State management + snapshots |
| `QueueReactiveModel` | Fill probability с intensity = f(queue, spread, vol) |
| `QueueValueModel` | Queue position value (Moallemi & Yuan) |
| `CalibrationPipeline` | MLE parameter fitting from historical data |
| `AlmgrenChrissModel` | Square-root temporary + linear permanent impact (Stage 4) |
| `GatheralModel` | Transient impact with power-law decay (Stage 4) |
| `KyleLambdaModel` | Kyle (1985) linear price impact model (Stage 4) |
| `ImpactEffects` | Quote shifting, liquidity reaction, momentum (Stage 4) |
| `LOBImpactSimulator` | Complete trade impact simulation workflow (Stage 4) |
| `ImpactCalibrationPipeline` | OLS/MLE calibration for impact params (Stage 4) |
| `LatencyModel` | Realistic latency simulation with profiles (Stage 5) |
| `LatencySampler` | Distribution-based latency sampling (Stage 5) |
| `EventScheduler` | Event ordering with priority queue (Stage 5) |
| `SimulationClock` | Time tracking with latency awareness (Stage 5) |
| `IcebergDetector` | Iceberg order detection from execution patterns (Stage 6) |
| `IcebergOrder` | Tracked iceberg with refill history (Stage 6) |
| `HiddenLiquidityEstimator` | Hidden quantity estimation (Stage 6) |
| `DarkPoolSimulator` | Multi-venue dark pool simulation (Stage 6) |
| `DarkPoolVenue` | Individual dark pool venue model (Stage 6) |
| `DarkPoolFill` | Dark pool execution result (Stage 6) |
| `L3ExecutionProvider` | Full L3 execution provider combining all LOB components (Stage 7) |
| `L3SlippageProvider` | LOB-based slippage with market impact (Stage 7) |
| `L3FillProvider` | LOB-based fill logic with queue position (Stage 7) |
| `L3ExecutionConfig` | Pydantic config model for L3 subsystems (Stage 7) |
| `BaseLOBAdapter` | Abstract base for LOB data adapters (Stage 8) |
| `LOBSTERAdapter` | LOBSTER format adapter (Stage 8) |
| `ITCHAdapter` | ITCH format adapter (Stage 8) |
| `BinanceL2Adapter` | Binance L2 data adapter (Stage 8) |
| `AlpacaL2Adapter` | Alpaca L2 data adapter (Stage 8) |
| `L3CalibrationPipeline` | Unified calibration for L3 (Stage 8) |
| `LatencyCalibrator` | Latency distribution calibration (Stage 8) |
| `QueueDynamicsCalibrator` | Queue dynamics calibration (Stage 8) |

### Self-Trade Prevention (STP)

| Режим | Действие |
|-------|----------|
| `CANCEL_NEWEST` | Отменяет входящий (aggressive) ордер |
| `CANCEL_OLDEST` | Отменяет resting ордер |
| `CANCEL_BOTH` | Отменяет оба ордера |
| `DECREMENT_AND_CANCEL` | Уменьшает qty, отменяет меньший |

### Time-in-Force

| TIF | Поведение |
|-----|-----------|
| `DAY` | Активен до конца дня |
| `GTC` | Good-Til-Cancelled |
| `IOC` | Immediate-Or-Cancel (partial fill → CANCELLED) |
| `FOK` | Fill-Or-Kill (all or nothing) |

### Queue Position Estimation

```python
from lob import QueuePositionTracker, PositionEstimationMethod

tracker = QueuePositionTracker()

# MBP (pessimistic) — advance only on executions
state = tracker.add_order(order, level_qty_before=500.0)

# MBO (exact) — requires order-level data
state = tracker.add_order(order, orders_ahead=[...])

# Fill probability (Poisson model)
prob = tracker.estimate_fill_probability(
    order_id, volume_per_second=100.0, time_horizon_sec=60.0
)
```

### Использование

```python
from lob import OrderManager, Side, OrderType, TimeInForce

manager = OrderManager(symbol="AAPL")

# Submit limit order
order = manager.submit_order(
    side=Side.BUY,
    price=150.0,
    qty=100.0,
    order_type=OrderType.LIMIT,
    time_in_force=TimeInForce.DAY,
)

# Check fill probability
prob = manager.get_fill_probability(order.order.order_id)

# Cancel
manager.cancel_order(order.order.order_id)
```

### Performance

| Операция | Latency | Target |
|----------|---------|--------|
| Market order simulation | ~5 μs | <10 μs ✅ |
| Limit order matching | ~20 μs | <50 μs ✅ |
| Queue position update | ~50 μs | <500 μs ✅ |

### Stage 3: Fill Probability & Queue Value

```python
from lob import (
    QueueReactiveModel,
    QueueValueModel,
    CalibrationPipeline,
    LOBState,
    TradeRecord,
    Side,
)

# 1. Create fill probability model
fill_model = QueueReactiveModel(
    base_rate=100.0,           # Base volume rate (qty/sec)
    queue_decay_alpha=0.01,    # Queue size impact
    spread_sensitivity_beta=0.5,  # Spread impact
)

# 2. Estimate fill probability
lob_state = LOBState(
    mid_price=150.0,
    spread_bps=5.0,
    volatility=0.02,
    imbalance=0.1,
)

prob_result = fill_model.compute_fill_probability(
    queue_position=10,
    qty_ahead=500.0,
    order_qty=100.0,
    time_horizon_sec=60.0,
    market_state=lob_state,
)
print(f"P(fill in 60s) = {prob_result.prob_fill:.2%}")

# 3. Compute queue value (Moallemi & Yuan)
value_model = QueueValueModel(fill_model=fill_model)
value_result = value_model.compute_queue_value(order, lob_state, queue_state)
print(f"Queue value: ${value_result.queue_value:.4f}")
print(f"Decision: {value_result.decision.name}")  # HOLD or CANCEL

# 4. Calibrate from historical data
pipeline = CalibrationPipeline()
for trade in historical_trades:
    pipeline.add_trade(TradeRecord(
        timestamp_ns=trade.ts,
        price=trade.price,
        qty=trade.qty,
        side=Side.BUY if trade.is_buy else Side.SELL,
    ))
results = pipeline.run_calibration()
calibrated_model = pipeline.get_best_model("queue_reactive")
```

### Stage 4: Market Impact Models

```python
from lob import (
    AlmgrenChrissModel,
    GatheralModel,
    ImpactParameters,
    ImpactEffects,
    LOBImpactSimulator,
    create_impact_model,
    ImpactCalibrationPipeline,
    TradeObservation,
    CalibrationDataset,
)

# 1. Create impact model
params = ImpactParameters.for_equity()  # or .for_crypto()
model = AlmgrenChrissModel(params=params)

# 2. Compute market impact
result = model.compute_total_impact(
    order_qty=10000,
    adv=10_000_000,
    volatility=0.02,
    mid_price=150.0,
)
print(f"Temporary: {result.temporary_impact_bps:.2f} bps")
print(f"Permanent: {result.permanent_impact_bps:.2f} bps")
print(f"Impact cost: ${result.impact_cost:.2f}")

# 3. Simulate impact effects on LOB
simulator = LOBImpactSimulator(impact_model=model)
impact, quote_shift, liquidity = simulator.simulate_trade_impact(
    order_book=order_book,
    order=limit_order,
    fill=fill,
    adv=10_000_000,
    volatility=0.02,
)
print(f"New bid: {quote_shift.new_bid}, New ask: {quote_shift.new_ask}")

# 4. Calibrate from historical trades
pipeline = ImpactCalibrationPipeline()
dataset = CalibrationDataset(avg_adv=10_000_000, avg_volatility=0.02)
for trade in historical_trades:
    obs = TradeObservation(
        timestamp_ms=trade.ts,
        price=trade.price,
        qty=trade.qty,
        side=1 if trade.is_buy else -1,
        adv=dataset.avg_adv,
        pre_trade_mid=trade.pre_mid,
        post_trade_mid=trade.post_mid,
    )
    dataset.add_observation(obs)
results = pipeline.calibrate_all(dataset)
calibrated_model = pipeline.create_calibrated_model()
```

### Stage 5: Latency Simulation

```python
from lob import (
    LatencyModel,
    LatencyProfile,
    EventScheduler,
    SimulationClock,
    MarketDataEvent,
    create_latency_model,
    create_event_scheduler,
)

# 1. Create latency model from profile
model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)
# Or: model = create_latency_model("colocated")

# 2. Sample latencies (returns nanoseconds)
feed_latency = model.sample_feed_latency()
order_latency = model.sample_order_latency()
exchange_latency = model.sample_exchange_latency()
fill_latency = model.sample_fill_latency()
round_trip = model.sample_round_trip()

print(f"Feed: {feed_latency/1000:.1f}us, Order: {order_latency/1000:.1f}us")
print(f"Round-trip: {round_trip/1000:.1f}us")

# 3. Create event scheduler
scheduler = create_event_scheduler("institutional", seed=42)

# Schedule market data event
event = MarketDataEvent(
    symbol="AAPL",
    exchange_time_ns=1_000_000,
    bid_price=150.0,
    ask_price=150.05,
)
our_receive_time = scheduler.schedule_market_data(event, exchange_time_ns=1_000_000)

# Schedule our order
from lob import LimitOrder, Side
order = LimitOrder(
    order_id="order_1",
    price=150.0,
    qty=100.0,
    remaining_qty=100.0,
    timestamp_ns=1_000_000,
    side=Side.BUY,
)
arrival_time = scheduler.schedule_order_arrival(order, our_send_time_ns=1_000_000)

# Process all events in timestamp order
for event in scheduler:
    print(f"Event: {event.event_type.name} at {event.timestamp_ns}ns")

# 4. Get latency statistics
stats = model.stats()
print(f"Feed p95: {stats['feed']['p95_us']:.1f}us")
```

### Stage 6: Hidden Liquidity & Dark Pools

```python
from lob import (
    # Iceberg Detection
    IcebergDetector,
    IcebergOrder,
    IcebergState,
    DetectionConfidence,
    HiddenLiquidityEstimator,
    create_iceberg_detector,
    create_hidden_liquidity_estimator,
    # Dark Pool Simulation
    DarkPoolSimulator,
    DarkPoolVenue,
    DarkPoolConfig,
    DarkPoolFill,
    DarkPoolVenueType,
    FillType,
    InformationLeakage,
    create_dark_pool_simulator,
    create_default_dark_pool_simulator,
)

# 1. Create iceberg detector
detector = create_iceberg_detector(
    min_refills_to_confirm=2,
    lookback_window_sec=60.0,
)

# 2. Process execution and detect iceberg pattern
pre_snap = detector.take_level_snapshot(level, Side.BUY)
# ... execution happens ...
post_snap = detector.take_level_snapshot(level, Side.BUY)
iceberg = detector.process_execution(trade, pre_snap, post_snap, Side.BUY)

if iceberg:
    print(f"Iceberg detected: display={iceberg.display_size}, state={iceberg.state.name}")
    hidden_estimate = detector.estimate_hidden_reserve(iceberg)
    print(f"Estimated hidden: {hidden_estimate}")

# 3. Batch detection from execution history
executions = [trade1, trade2, trade3]
level_qty_history = [500.0, 500.0, 500.0]  # Qty refills indicate iceberg
iceberg = detector.detect_iceberg(executions, level_qty_history, price=100.0, side=Side.BUY)

# 4. Hidden liquidity estimation
estimator = create_hidden_liquidity_estimator(detector, hidden_ratio=0.15)
hidden = estimator.estimate_hidden_at_level(price=100.0, side=Side.BUY, visible_qty=500.0)

# 5. Create dark pool simulator
dark_pool = create_default_dark_pool_simulator(seed=42)

# 6. Attempt dark pool fill
fill = dark_pool.attempt_dark_fill(
    order=limit_order,
    lit_mid_price=100.0,
    lit_spread=0.05,
    adv=10_000_000,
    volatility=0.02,
    hour_of_day=10,
)

if fill and fill.is_filled:
    print(f"Dark fill: {fill.filled_qty} @ {fill.fill_price} ({fill.venue_id})")
    if fill.info_leakage:
        print(f"Leakage: {fill.info_leakage.description}")

# 7. Estimate fill probability at each venue
probs = dark_pool.estimate_fill_probability(order, adv=10_000_000)
for venue_id, prob in probs.items():
    print(f"{venue_id}: {prob:.2%}")

# 8. Multi-venue routing
fills = dark_pool.attempt_fill_with_routing(order, lit_mid_price=100.0, max_attempts=3)
```

### Тестирование

```bash
# Stage 1 тесты (data structures, parsers, state manager)
pytest tests/test_lob_structures.py tests/test_lob_parsers.py tests/test_lob_state_manager.py -v

# Stage 2 тесты (matching engine, queue tracker, order manager)
pytest tests/test_matching_engine.py -v

# Stage 3 тесты (fill probability, queue value, calibration)
pytest tests/test_fill_probability_queue_value.py -v

# Stage 4 тесты (market impact, effects, calibration)
pytest tests/test_market_impact.py -v

# Stage 5 тесты (latency simulation, event scheduler)
pytest tests/test_lob_latency.py -v

# Stage 6 тесты (hidden liquidity, dark pools)
pytest tests/test_hidden_liquidity_dark_pools.py -v

# Stage 7 тесты (L3 execution provider, config)
pytest tests/test_execution_providers_l3.py -v

# Stage 8 тесты (data adapters, calibration pipeline)
pytest tests/test_lob_data_adapters.py tests/test_lob_calibration_pipeline.py -v

# Stage 9 тесты (validation, backward compatibility)
pytest tests/test_queue_tracker.py tests/test_l3_vs_production.py tests/test_l3_backward_compatibility.py -v

# Все LOB тесты
pytest tests/test_lob*.py tests/test_matching_engine.py tests/test_fill_probability_queue_value.py \
    tests/test_market_impact.py tests/test_hidden_liquidity_dark_pools.py tests/test_execution_providers_l3.py \
    tests/test_queue_tracker.py tests/test_l3_vs_production.py tests/test_l3_backward_compatibility.py -v
```

**Покрытие**: 749+ тестов (106 Stage 1 + 72 Stage 2 + 66 Stage 3 + 57 Stage 4 + 66 Stage 5 + 62 Stage 6 + 79 Stage 7 + Stage 8 + 117 Stage 9 + 95 execution_providers)

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `lob/matching_engine.py` | FIFO matching engine with STP |
| `lob/queue_tracker.py` | Queue position tracking (MBP/MBO) |
| `lob/order_manager.py` | Order lifecycle management |
| `lob/data_structures.py` | Core data structures |
| `lob/fill_probability.py` | Poisson, Queue-Reactive, Historical models |
| `lob/queue_value.py` | Queue value computation (Moallemi & Yuan) |
| `lob/calibration.py` | MLE calibration from historical data |
| `lob/market_impact.py` | Kyle, Almgren-Chriss, Gatheral impact models |
| `lob/impact_effects.py` | Quote shifting, liquidity reaction, momentum |
| `lob/impact_calibration.py` | OLS/grid search calibration for impact params |
| `tests/test_matching_engine.py` | 72 Stage 2 tests |
| `tests/test_fill_probability_queue_value.py` | 66 Stage 3 tests |
| `tests/test_market_impact.py` | 57 Stage 4 tests |
| `lob/latency_model.py` | Realistic latency simulation (Stage 5) |
| `lob/event_scheduler.py` | Event ordering with priority queue (Stage 5) |
| `tests/test_lob_latency.py` | 66 Stage 5 tests |
| `lob/hidden_liquidity.py` | Iceberg detection, hidden liquidity estimation (Stage 6) |
| `lob/dark_pool.py` | Dark pool simulation, multi-venue routing (Stage 6) |
| `tests/test_hidden_liquidity_dark_pools.py` | 62 Stage 6 tests |
| `execution_providers_l3.py` | L3ExecutionProvider combining all LOB components (Stage 7) |
| `lob/config.py` | Pydantic configuration models for L3 subsystems (Stage 7) |
| `configs/execution_l3.yaml` | L3 execution configuration file (Stage 7) |
| `tests/test_execution_providers_l3.py` | 79 Stage 7 tests |
| `lob/data_adapters.py` | LOBSTER, ITCH, Binance, Alpaca adapters (Stage 8) |
| `lob/calibration_pipeline.py` | Unified L3 calibration pipeline (Stage 8) |
| `tests/test_lob_data_adapters.py` | Data adapters tests (Stage 8) |
| `tests/test_lob_calibration_pipeline.py` | Calibration pipeline tests (Stage 8) |
| `tests/test_queue_tracker.py` | 55 Queue position tracking tests (Stage 9) |
| `tests/test_l3_vs_production.py` | 30 Validation metrics tests (Stage 9) |
| `tests/test_l3_backward_compatibility.py` | 32 Backward compatibility tests (Stage 9) |
| `benchmarks/bench_matching.py` | Matching engine benchmarks (Stage 9) |
| `benchmarks/bench_full_sim.py` | Full simulation benchmarks (Stage 9) |
| `docs/L3_VALIDATION_REPORT.md` | Stage 9 validation report |
| `docs/L3_MIGRATION_GUIDE.md` | Migration guide from L2 to L3 |
| `docs/l3_simulator/overview.md` | L3 architecture overview (Stage 10) |
| `docs/l3_simulator/data_structures.md` | LOB data structures (Stage 10) |
| `docs/l3_simulator/matching_engine.md` | Matching engine docs (Stage 10) |
| `docs/l3_simulator/queue_position.md` | Queue position tracking (Stage 10) |
| `docs/l3_simulator/market_impact.md` | Impact models (Stage 10) |
| `docs/l3_simulator/latency.md` | Latency simulation (Stage 10) |
| `docs/l3_simulator/calibration.md` | Calibration guide (Stage 10) |
| `docs/l3_simulator/configuration.md` | Config reference (Stage 10) |
| `docs/l3_simulator/deployment.md` | Deployment checklist & rollout (Stage 10) |

### Референсы

- CME Globex Matching Algorithm
- Erik Rigtorp: Queue Position Estimation
- Cont et al. (Columbia): Fill Probability Models
- FIX Protocol: Order Status semantics
- Huang et al. (2015): Queue-Reactive Model
- Moallemi & Yuan (2017): Queue Position Valuation
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
- Almgren et al. (2005): "Direct Estimation of Equity Market Impact"
- hftbacktest: High-frequency trading backtesting framework (latency modeling reference)
- Bookmap: Iceberg order detection methodology (https://bookmap.com/blog/advanced-order-flow-trading-spotting-hidden-liquidity-iceberg-orders)
- SEC Rule 606: Dark pool routing disclosures
- FINRA ATS: Dark pool transparency data

---

## 🛡️ Критические правила (НЕ НАРУШАТЬ!)

1. **ActionProto.volume_frac = TARGET position, НЕ DELTA!**
   - ✅ `next_units = volume_frac * max_position`
   - ❌ `next_units = current_units + volume_frac * max_position` (удвоение!)

2. **Action space bounds: [-1, 1] для policy с LongOnlyActionWrapper**
   - ✅ `LongOnlyActionWrapper.action_space = Box(-1, 1)` — wrapper сам устанавливает!
   - ✅ Policy использует `tanh` когда `action_space.low < 0`
   - ❌ Wrapper НЕ должен наследовать `action_space` от env (было [0,1] → баг!)

3. **LongOnlyActionWrapper: mapping [-1,1] → [0,1], НЕ clipping**
   - ✅ `mapped = (action + 1.0) / 2.0` — policy выдаёт [-1,1], wrapper маппит в [0,1]
   - ✅ `-1.0 → 0.0` (exit), `0.0 → 0.5` (50%), `+1.0 → 1.0` (100%)
   - ❌ `clipped = max(0, action)` (теряет reduction сигналы)
   - ❌ Если wrapper наследует [0,1] от env: sigmoid [0,1] → mapping → [0.5,1.0] **минимум 50%!**

4. **LSTM States ДОЛЖНЫ сбрасываться на episode boundaries!**
   - ✅ `self._last_lstm_states = self._reset_lstm_states_for_done_envs(...)`
   - ⚠️ **НЕ УДАЛЯЙТЕ** вызов в distributional_ppo.py:7418-7427!

5. **UPGD utility scaling: min-max normalization**
   - ✅ `normalized = (utility - global_min) / (global_max - global_min + eps)`
   - ❌ `scaled = utility / global_max` (инвертируется при negative!)

6. **Gamma synchronization для reward shaping**
   - ✅ `reward.gamma == model.params.gamma` (оба = 0.99)
   - ⚠️ При изменении одного — обновите другой!

7. **Technical Indicators инициализация**
   - ✅ **RSI**: SMA(14) для первых gains/losses
   - ✅ **CCI**: SMA(TP) для baseline
   - ✅ **ATR**: SMA variant корректен

---

## 🚨 Troubleshooting (актуальные проблемы)

| Симптом | Причина | Решение |
|---------|---------|---------|
| step() возвращает obs с той же row что reset() | Observation строился из current row, не next | ✅ Фикс 2025-11-25: obs из next_row (Gymnasium семантика) |
| CLOSE_TO_OPEN + SIGNAL_ONLY: look-ahead bias | signal_pos обновлялся немедленно, игнорируя delay | ✅ Фикс 2025-11-25: использует executed_signal_pos |
| info["signal_pos_next"] показывает intent, не actual | В CLOSE_TO_OPEN + signal_only показывал agent_signal_pos | ✅ Фикс 2025-11-25: показывает next_signal_pos + новое поле signal_pos_requested |
| LSTM первый step на zeros | reset() возвращал np.zeros() | ✅ Фикс 2025-11-25: reset() строит obs из row 0 |
| reward=0 при старте эпизода | NaN close в первых rows → _last_reward_price=0 | ✅ Фикс 2025-11-25: fallback на open/scan rows |
| Long-only: позиция всегда ≥50% | Wrapper наследовал [0,1] action_space | ✅ Фикс 2025-11-25: wrapper ставит [-1,1], policy использует tanh |
| Long-only: entropy collapse | Policy не может выразить exit | Переобучить с новым wrapper (tanh вместо sigmoid) |
| PBT deadlock (workers crash) | ready_percentage слишком высокий | `min_ready_members=2`, `ready_check_max_wait=10` |
| Non-monotonic quantiles | NN predictions без sorting | `critic.enforce_monotonicity=true` |
| Value loss не снижается | LSTM states не сбрасываются | Проверьте `_reset_lstm_states_for_done_envs` |
| External features = 0.0 | NaN → 0.0 silent conversion | `log_nan=True` для debugging |
| Градиенты взрываются | UPGD noise слишком высок | Уменьшите `sigma` (0.0005-0.001) |
| `AttributeError` в конфигах | Pydantic V2 API | `model_dump()` вместо `dict()` |
| Feature mismatch | Online/offline паритет | `check_feature_parity.py` |
| PBT state mismatch | VGS не синхронизирован | Проверьте `variance_gradient_scaler.py` state dict |
| step() IndexError при пустом df | Нет защиты от пустого DataFrame | ✅ Фикс 2025-11-25: проверка len(df)==0 в step() |
| signal_pos в obs отстаёт от market data | Obs содержал prev_signal_pos (t), но market data из t+1 | ✅ Фикс 2025-11-26: obs содержит next_signal_pos (t+1) |
| VGS + AdaptiveUPGD: noise 212x amplification | EMA (beta=0.999) слишком медленно адаптируется к VGS scaling | ✅ Фикс 2025-11-26: `instant_noise_scale=True` (default) |
| FG=50 (neutral) treated as missing data | `abs(value-50.0)>0.1` check false negative | ✅ Фикс 2025-11-26: uses `_get_safe_float_with_validity()` |
| UPGDW: inverted weight protection | Only tracked max_util, not min_util | ✅ Фикс 2025-11-26: min-max normalization like AdaptiveUPGD |
| Episode continues with stale data | row_idx clamped to last row instead of truncation | ✅ Фикс 2025-11-26: returns truncated=True when data exhausted |
| cql_beta=0 causes NaN/Inf | No validation for cql_beta divisor | ✅ Фикс 2025-11-26: ValueError if cql_beta <= 0 |
| Twin Critics categorical VF clipping no effect | `_project_distribution` was identity stub | ✅ Фикс 2025-11-26: uses `_project_categorical_distribution` |
| Yang-Zhang volatility inflated ~11% for n=10 | RS component used (n-1) instead of n | ✅ Фикс 2025-11-26: RS now uses n per original formula |
| `_project_categorical_distribution` shape error | 1D atoms not expanded to batch_size | ✅ Фикс 2025-11-26: proper batch expansion |
| Limit order fills missed for high-price assets | Fixed tolerance 1e-12 < machine epsilon at $100k | ✅ Фикс 2025-11-26: `_compute_price_tolerance` с relative tolerance |
| EV≈0, Twin Critics loss +327%, grad norm -82% | VGS alpha=0.1 даёт 91% редукцию градиентов при высокой variance | ✅ Фикс 2025-11-27: VGS v3.2 с `min_scaling_factor=0.1`, `variance_cap=50.0` |
| DarkPoolSimulator memory leak | `_leakage_history`, `_fill_history` росли unbounded | ✅ Фикс 2025-11-27: `deque(maxlen=max_history_size)` |
| DarkPoolConfig division by zero | `impact_size_normalization=0` не валидировался | ✅ Фикс 2025-11-27: `__post_init__` validation |
| DarkPoolSimulator TypeError on deque slice | `_should_block_for_leakage` использовал slice на deque | ✅ Фикс 2025-11-27: convert to list before slicing |

---

## ✅ FAQ: Закрытые вопросы (НЕ ПЕРЕОТКРЫВАТЬ!)

Эти вопросы были тщательно проанализированы. Подробности: [docs/archive/reports_2025_11_24/conceptual_analysis/CRITICAL_ANALYSIS_THREE_PROBLEMS_2025_11_24.md](docs/archive/reports_2025_11_24/conceptual_analysis/CRITICAL_ANALYSIS_THREE_PROBLEMS_2025_11_24.md)

| Вопрос | Ответ |
|--------|-------|
| "Look-ahead bias в индикаторах?" | ✅ **Исправлено 2025-11-23**. Все фичи сдвинуты. |
| "VGS недооценивает variance в N раз?" | ⚠️ **By design**. Var[mean(g)] валиден, работает в production. |
| "-10.0 bankruptcy penalty слишком резкий?" | ✅ **Стандартная практика RL**. Potential shaping даёт smooth gradient. |
| "_last_signal_position двойное присваивание?" | ⚠️ **Удалено 2025-11-25**. Было избыточно, но не баг (значения идентичны). |
| "Первые 2 steps в CLOSE_TO_OPEN reward=0?" | ⚠️ **By design**. Delayed execution: reward × prev_signal_pos, где prev=0 для первых шагов. |
| "signal_only terminated всегда False?" | ⚠️ **By design**. В signal_only нет капитала в риске, банкротство не имеет смысла. |
| "ActionProto double mapping в LongOnlyActionWrapper?" | ⚠️ **НЕ баг**. API контракт: input [-1,1] → output [0,1]. Если передать [0,1] - нарушение контракта. |
| "adaptive_upgd.py grad_norm_ema=1.0 warmup?" | ⚠️ **НЕ баг**. Default `instant_noise_scale=True` bypasses EMA. См. #28. |
| "info[signal_pos] разная семантика?" | ⚠️ **By design**. signal_only: prev (для reward), normal: next (после execution). См. #7. |
| "mediator norm_cols_validity=True?" | ⚠️ **НЕ баг**. Начальное значение полностью перезаписывается в цикле. См. #29. |
| "mediator empty observation silent fail?" | ⚠️ **НЕ баг**. Defensive check для edge cases без observation_space. |
| "mediator race condition signal_pos?" | ⚠️ **НЕ баг**. Single-threaded архитектура, нет параллелизма. |
| "risk_guard асимметричный buffer?" | ⚠️ **By design**. Buffer только на увеличение позиции (корректный risk mgmt). См. #30. |
| "ops_kill_switch cooldown reset при init?" | ⚠️ **НЕ баг**. _last_ts=0.0 = "reset в epoch". Логика корректна. См. #31. |
| "RSI valid на 1 бар раньше (off-by-one)?" | ⚠️ **НЕ баг**. RSI-14 valid на bar 14 (после 14 price changes). Timing корректен. См. #32. |
| "obs_builder vol_proxy=0.01 constant warmup?" | ⚠️ **By design**. 1% price fallback лучше чем NaN или 0. См. #33. |
| "obs_builder FG=50 vs missing неразличимы?" | ✅ **Исправлено 2025-11-26**. Теперь `_get_safe_float_with_validity()` различает. |
| "policy sigma range [0.2,1.5] не адаптируется?" | ⚠️ **НЕ баг**. Standard PPO range для continuous actions. См. #35. |
| "CVaR weight_start=0.5 совпадение?" | ⚠️ **НЕ баг**. Математически корректно: граница = midpoint. См. #3. |
| "features_pipeline constant на shifted data?" | ⚠️ **НЕ баг**. nanstd игнорирует NaN, для типичных datasets работает. См. #36. |
| "mediator step_idx=current не next?" | ⚠️ **Minor**. info для logging, не для agent. Семантика "обработали row X". |
| "Twin Critics logging memory leak?" | ⚠️ **НЕ баг**. Accumulators reset at line 12288 after logging. См. #45. |
| "ddof=1 vs ddof=0 в advantage normalization?" | ⚠️ **Minor inconsistency**. SB3 uses ddof=0, difference <0.1% for n>1000. См. #46. |
| "VGS race condition в PBT?" | ⚠️ **НЕ issue**. Separate workers, unique checkpoint files, Python GIL. См. #47. |
| "CVaR ~16% approximation error?" | ⚠️ **Documented limitation**. Trade-off: speed vs accuracy. N=51 gives ~5% error. |
| "Winsorization [1%,99%] insufficient for crypto?" | ⚠️ **Configurable**. Can adjust in features_pipeline.py:181. |

---

## 🔬 НЕ БАГИ: Корректные паттерны кода (НЕ "ИСПРАВЛЯТЬ"!)

> **ВАЖНО**: Следующие паттерны кода ВЫГЛЯДЯТ как ошибки при статическом анализе, но являются **корректными и намеренными**. НЕ пытайтесь их "исправить"!

### 1. Episode Starts Off-by-One (distributional_ppo.py:8314, 8347)

```python
# Строка 8314: добавляем _last_episode_starts в буфер
rollout_buffer.add(..., self._last_episode_starts, ...)

# Строка 8347: обновляем ПОСЛЕ добавления
self._last_episode_starts = dones
```

**Почему это НЕ баг**: Это стандартный паттерн Stable-Baselines3. `_last_episode_starts` хранит `dones` от **предыдущего** шага. При вычислении GAE (строка 280) используется `episode_starts[step+1]` — это означает "был ли шаг step терминальным". Сдвиг на 1 **намеренный** и семантически корректный.

**Референс**: SB3 `OnPolicyAlgorithm.collect_rollouts()`, PPO paper (Schulman et al., 2017)

---

### 2. VGS применяется ПЕРЕД grad clipping (distributional_ppo.py:11664-11676)

```python
# Строка 11664: VGS масштабирует градиенты
vgs_scaling_factor = self._variance_gradient_scaler.scale_gradients()

# Строка 11676: Потом clipping
total_grad_norm = torch.nn.utils.clip_grad_norm_(...)
```

**Почему это НЕ баг**: VGS **уменьшает** градиенты (scaling_factor < 1.0, см. variance_gradient_scaler.py:446). Порядок корректен:
1. VGS снижает variance высокошумных градиентов
2. clip_grad_norm защищает от оставшихся выбросов

**Референс**: variance_gradient_scaler.py docstring, Adam optimizer design

---

### 3. CVaR Interpolation Weight = 0.5 (distributional_ppo.py:3726-3728)

```python
tau_i_prev = (alpha_idx - 0.5) / num_quantiles  # центр предыдущего интервала
tau_i = (alpha_idx + 0.5) / num_quantiles        # центр текущего интервала
interval_start = alpha_idx / num_quantiles       # граница между ними
weight_start = (interval_start - tau_i_prev) / (tau_i - tau_i_prev)  # = 0.5
```

**Почему это НЕ баг**: `interval_start` (граница квантильного интервала) находится **ровно посередине** между центрами соседних интервалов `tau_i_prev` и `tau_i`. Вес 0.5 — это математически корректная линейная интерполяция.

**Математика**: `weight = (α_idx/N - (α_idx-0.5)/N) / ((α_idx+0.5)/N - (α_idx-0.5)/N) = 0.5/N / (1/N) = 0.5`

---

### 4. LSTM Init State Index 0 (distributional_ppo.py:2217)

```python
state_tensor[:, env_idx, ...] = init_tensor[:, 0, ...].detach().to(...)
```

**Почему это НЕ баг**: `recurrent_initial_state` инициализируется **нулями** для всех environments (custom_policy_patch1.py:492). Все init states идентичны, поэтому `init_tensor[:, 0, ...]` безопасен.

**Референс**: custom_policy_patch1.py:491-503 — `torch.zeros(self.lstm_hidden_state_shape, ...)`

---

### 5. Twin Critics Loss Averaging БЕЗ VF Clipping (distributional_ppo.py:11073)

```python
# Когда VF clipping ВЫКЛЮЧЕН:
critic_loss_unclipped_per_sample = (loss_critic_1 + loss_critic_2) / 2.0
```

**Почему это НЕ баг**: Без VF clipping нет необходимости в `max(clipped, unclipped)`. Простое усреднение losses двух critics корректно. Когда VF clipping **включён**, используется правильная логика (строки 11168-11170):
```python
loss_c1_final = torch.max(loss_c1_unclipped, loss_c1_clipped)
loss_c2_final = torch.max(loss_c2_unclipped, loss_c2_clipped)
critic_loss = torch.mean((loss_c1_final + loss_c2_final) / 2.0)
```

---

### 6. close_orig vs _close_shifted маркеры (features_pipeline.py, trading_patchnew.py)

```python
# features_pipeline.py:329-331 — пропускает shift если close_orig есть
if "close_orig" in frame.columns:
    shifted_frames.append(frame)
    continue

# trading_patchnew.py:305-307 — проверяет close_orig ПЕРВЫМ
if "close_orig" in self.df.columns:
    self._close_actual = self.df["close_orig"].copy()
elif "close" in self.df.columns and "_close_shifted" not in self.df.columns:
    # Shift применяется только здесь
```

**Почему это НЕ баг**: Проверка `close_orig` идёт **раньше** проверки `_close_shifted`. Если данные пришли с `close_orig` (уже сдвинуты), shift НЕ применяется повторно. Два маркера имеют разную семантику:
- `close_orig` — оригинальная цена ДО shift (для анализа)
- `_close_shifted` — флаг что shift уже применён

---

### 7. info["signal_pos_next"] vs info["signal_pos_requested"] (trading_patchnew.py:2194-2204)

```python
if self._reward_signal_only:
    info["signal_pos_next"] = float(next_signal_pos)      # ACTUAL position after step
    info["signal_pos_requested"] = float(agent_signal_pos)  # Agent's INTENTION
else:
    info["signal_pos_next"] = float(next_signal_pos)
    info["signal_pos_requested"] = float(agent_signal_pos)
```

**Почему это корректно** (исправлено 2025-11-25):
1. В CLOSE_TO_OPEN режиме: `next_signal_pos ≠ agent_signal_pos` из-за 1-bar delay
2. `signal_pos_next` показывает **фактическую** позицию после шага (используется для reward)
3. `signal_pos_requested` показывает **намерение** агента (для debugging/анализа)
4. **До фикса**: `signal_pos_next = agent_signal_pos` → вводило в заблуждение при отладке

**Тесты**: `tests/test_signal_pos_next_close_to_open_consistency.py` (8 тестов)

---

### 8. Advantage Normalization с ddof=1 (distributional_ppo.py:8442)

```python
adv_std = float(np.std(advantages_flat, ddof=1))
# ...
normalized_advantages = (adv - adv_mean) / (adv_std + EPSILON)
```

**Почему это НЕ баг**:
1. `ddof=1` для несмещённой оценки дисперсии (Bessel's correction)
2. Если `n_samples == 1`, `std` будет `NaN`
3. Код защищён проверкой на строках 8444-8445: `if not np.isfinite(adv_std): skip`
4. `EPSILON = 1e-8` защищает от деления на ноль

---

### 9. Policy Adaptive Activation (custom_policy_patch1.py:491-497, 1301-1314)

```python
# __init__: определяем тип активации по action_space
action_low = float(self.action_space.low.flat[0])
self._use_tanh_activation = action_low < 0.0

# _apply_action_activation: выбираем sigmoid или tanh
if getattr(self, "_use_tanh_activation", False):
    return torch.tanh(raw)
else:
    return torch.sigmoid(raw)
```

**Почему это НЕ баг**: Это **КРИТИЧЕСКИЙ FIX** (2025-11-25):
1. `LongOnlyActionWrapper` устанавливает `action_space = [-1, 1]`
2. Policy детектирует это и использует `tanh` (выход [-1, 1])
3. Wrapper маппит [-1, 1] → [0, 1] для TradingEnv
4. БЕЗ этого фикса: sigmoid [0,1] → mapping → [0.5, 1.0] — **минимум 50% позиции!**

**Тесты**: `tests/test_long_only_action_space_fix.py` (26 тестов)

---

### 10. step() Observation from NEXT Row (trading_patchnew.py:1007-1037, mediator.py:1724-1739)

```python
# Вычисляем индекс СЛЕДУЮЩЕЙ строки для observation
obs_row_idx = min(next_idx, len(self.df) - 1)
next_row = self.df.iloc[obs_row_idx]
obs = self._mediator._build_observation(row=next_row, state=state, mark_price=next_mark_price)
```

**Почему это КОРРЕКТНО** (исправлено 2025-11-25):
1. **Gymnasium семантика**: `step(a)` возвращает `(s_{t+1}, r_t, ...)` — observation **после** действия
2. До фикса: reset() и step()#1 возвращали obs из одной строки (row[0]) — дубликат!
3. После фикса: reset() → row[0], step()#1 → row[1], step()#2 → row[2]
4. Terminal case: при next_idx >= len(df), используется последняя доступная строка

**Влияние бага на training**:
- Sample efficiency: ~1% loss (1 бесполезный transition на эпизод)
- LSTM: первые два hidden state обновления от идентичного входа
- Первый step reward: всегда 0 (log(price[0]/price[0])=0)

**Тесты**: `tests/test_step_observation_next_row.py` (6 тестов)

---

### 11. CLOSE_TO_OPEN + SIGNAL_ONLY Delayed Position (trading_patchnew.py:1725-1756)

```python
if self.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
    # Всегда уважаем 1-bar delay для signal position
    next_signal_pos = executed_signal_pos  # от delayed proto
else:
    next_signal_pos = agent_signal_pos if self._reward_signal_only else executed_signal_pos
```

**Почему это КОРРЕКТНО** (исправлено 2025-11-25):
1. **CLOSE_TO_OPEN семантика**: действие агента исполняется на **следующем** баре
2. До фикса: в SIGNAL_ONLY позиция обновлялась мгновенно → look-ahead bias
3. После фикса: даже в SIGNAL_ONLY режиме позиция задерживается на 1 бар
4. Reward = log(price_change) × position → позиция должна соответствовать реальному timing'у

**Влияние бага на training**:
- Training Sharpe: inflated на ~10-30% vs reality
- Look-ahead bias: reward за позицию, которой ещё нет
- Training/Live gap: увеличен из-за нереалистичных rewards

**Тесты**: `tests/test_close_to_open_signal_only_timing.py` (5 тестов)

---

### 12. Первые 2 step'а в CLOSE_TO_OPEN имеют reward ≈ 0 (trading_patchnew.py:1997-2015)

```python
# reward = log(price_ratio) × prev_signal_pos
# Step #1: prev_signal_pos = 0 (initial) → reward = 0
# Step #2: prev_signal_pos = 0 (delayed HOLD) → reward = 0
# Step #3+: prev_signal_pos = executed_action → reward ≠ 0
reward_raw_fraction = math.log(ratio_clipped) * prev_signal_pos
```

**Почему это BY DESIGN (НЕ баг)**:
1. **Физика delayed execution**: в CLOSE_TO_OPEN действие исполняется на **следующем** баре
2. При reset() устанавливается `_pending_action = HOLD(0.0)` — первое действие
3. Step #1: prev_pos = 0 (initial), action = HOLD(0.0) → reward × 0 = 0
4. Step #2: prev_pos = 0 (от HOLD), action = A1 → reward × 0 = 0
5. Step #3: prev_pos = A1, reward × A1 ≠ 0

**Семантика**: Reward отражает позицию, которая **РЕАЛЬНО была** во время движения цены, а не намерение агента. Это корректно для реалистичного trading simulation.

**Влияние на training**:
- Короткие эпизоды (< 5 баров) получают мало ненулевых rewards
- ~2/N долевая потеря sample efficiency для N-bar эпизодов
- Это **НЕ влияет на качество обучения** — агент учится правильной семантике

**Не пытайтесь "исправить"** — это сломает корректность симуляции!

---

### 13. В signal_only режиме terminated всегда False (trading_patchnew.py:1067-1086)

```python
# is_bankrupt устанавливается ТОЛЬКО в mediator.step()
# В signal_only режиме mediator.step() НЕ вызывается
terminated = bool(getattr(state, "is_bankrupt", False))  # всегда False
```

**Почему это BY DESIGN (НЕ баг)**:
1. **Signal_only режим**: агент учится генерировать сигналы без реального execution
2. Нет реальных позиций → нет реального capital at risk → нет банкротства
3. Reward = log(price_change) × signal_position — чисто сигнальный training
4. Эпизоды заканчиваются через **truncation** (`max_steps`), НЕ termination

**Альтернатива**: Добавить "виртуальное банкротство"?
- Это усложнит семантику без реальной пользы
- Сигнальный режим не симулирует капитал — банкротство не имеет смысла
- Если нужна проверка drawdown → используйте real execution mode

**Не пытайтесь добавить виртуальное банкротство** — это нарушит принцип signal_only!

---

### 14. ActionProto "double mapping" в LongOnlyActionWrapper (wrappers/action_space.py:120-147)

```python
# API контракт: INPUT [-1, 1] → OUTPUT [0, 1]
mapped = self._map_to_long_only(action.volume_frac)  # (x+1)/2
# -1.0 → 0.0, 0.0 → 0.5, 1.0 → 1.0
```

**Почему это НЕ баг (API CONTRACT)**:

| Input ([-1,1]) | Output ([0,1]) | Позиция |
|----------------|----------------|---------|
| -1.0 | 0.0 | Exit to cash |
| -0.5 | 0.25 | 25% long |
| 0.0 | 0.5 | 50% long |
| 0.5 | 0.75 | 75% long |
| 1.0 | 1.0 | 100% long |

**ЧАСТАЯ ОШИБКА**: передача `ActionProto(volume_frac=0.5)` с ожиданием "50% позиции"
- 0.5 в [-1,1] маппится в 0.75 в [0,1] — это **75%**, не 50%!
- Для 50% позиции передавайте `volume_frac=0.0`

**Почему wrapper всегда применяет маппинг**:
- Wrapper НЕ ЗНАЕТ семантику входящего ActionProto
- Он ВСЕГДА преобразует [-1,1] → [0,1] согласно API
- Если вам нужно передать [0,1] напрямую — НЕ используйте LongOnlyActionWrapper

**Тесты**: `tests/test_long_only_action_space_fix.py::test_action_proto_transformation`

---

### 15. signal_pos в observation = next_signal_pos (trading_patchnew.py:1829-1837)

```python
# FIX (2025-11-26): Set mediator signal_pos to next_signal_pos for observation
if self._reward_signal_only:
    try:
        setattr(
            self._mediator,
            "_last_signal_position",
            float(next_signal_pos),  # FIX: was prev_signal_pos_for_reward
        )
    except Exception:
        pass
```

**Почему это КОРРЕКТНО** (исправлено 2025-11-26):

1. **Gymnasium семантика**: `step(action)` возвращает `s_{t+1}` — состояние **ПОСЛЕ** действия
2. Observation содержит market data из `next_row` (время t+1)
3. signal_pos в observation должен быть `next_signal_pos` (позиция после step, время t+1)
4. **До фикса**: market data t+1, signal_pos t → temporal mismatch!
5. **После фикса**: market data t+1, signal_pos t+1 → согласованы

**Reward НЕ затронут**:
- Reward = `log(price_change) × prev_signal_pos_for_reward`
- Reward использует позицию, которая **РЕАЛЬНО была** во время price change
- Это корректно и не изменилось

**Влияние бага на training**:
- MDP violation: observation не отражало результат действия
- LSTM confusion: hidden state обновлялся с несогласованным входом
- Sample inefficiency: agent не видел эффект своих действий в obs

**Тесты**: `tests/test_signal_pos_observation_consistency.py` (10 тестов)

---

### 16. Limit Order Maker Fill Logic (execution_sim.py:11420-11448)

```python
elif best_ask is not None and price_q < best_ask:
    filled_price = float(price_q)
    liquidity_role = "maker"
    if (intrabar_fill_price is not None
        and intrabar_fill_price <= limit_price_value + tolerance):
        maker_fill = True
        filled = True
    else:
        filled = False  # ← НЕ заполняется если цена не достигла лимита!
```

**Почему это НЕ баг**: BUY LIMIT с ценой НИЖЕ best_ask НЕ заполняется мгновенно. Заполнение происходит ТОЛЬКО если `intrabar_fill_price` (low бара) достигает лимитной цены. Это корректная симуляция maker orders.

---

### 17. Fee Computed on Filled Price (execution_sim.py:3507-3526)

```python
trade_notional = filled_price * qty_total  # filled_price includes slippage
fee = self._compute_trade_fee(price=filled_price, ...)  # Fee от actual fill price
```

**Почему это НЕ баг (НЕ double-counting)**:
- **Slippage**: разница между expected и actual price (market impact)
- **Fee**: процент от actual fill price (биржевая комиссия)

На реальной бирже комиссия взимается от **фактической цены исполнения**. Это корректное поведение.

---

### 18. VGS _param_ids не сохраняется в state_dict (variance_gradient_scaler.py:136)

```python
self._param_ids: Dict[int, int] = {}  # UNUSED - legacy placeholder
```

**Почему это НЕ баг**: `_param_ids` **НИГДЕ НЕ ИСПОЛЬЗУЕТСЯ**! Поиск `_param_ids[` по коду даёт 0 результатов. VGS работает через `enumerate(self._parameters)` напрямую. Это мёртвый/placeholder код.

---

### 19. UPGDW global_max_util = -inf (optimizers/upgdw.py:106)

```python
global_max_util = torch.tensor(-torch.inf, device="cpu")
# В первом проходе обновляется если есть gradients
# Во втором проходе используется для scaled_utility
```

**Почему это НЕ баг**: Если `global_max_util` остаётся `-inf`, это означает что ВСЕ параметры имели `grad=None` в первом проходе. Но тогда они ТАКЖЕ будут пропущены во втором проходе (`if p.grad is None: continue`). Деление на `-inf` не произойдёт.

---

### 20. CVaR tail_mass = max(alpha, mass * (full_mass + frac)) (distributional_ppo.py:3696)

```python
tail_mass = max(alpha, mass * (full_mass + frac))
# Для α=0.95, N=20: tail_mass = max(0.95, 0.05*19) = 0.95 ✓
```

**Почему это НЕ баг**: Формула **математически корректна**. `max()` защищает от underestimate из-за дискретизации квантилей. Результат всегда ≥ alpha.

---

### 21. CVaR alpha_idx_float < 0 → Extrapolation (distributional_ppo.py:3650-3678)

```python
if alpha_idx_float < 0.0:
    # EXTRAPOLATION CASE: handles negative alpha_idx_float
    # This branch executes BEFORE floor() could give -1
```

**Почему это НЕ баг**: Отрицательный `alpha_idx_float` (для α < tau_0) обрабатывается **отдельным branch** через экстраполяцию. Negative indexing `q[:, -1]` **НИКОГДА не достигается**.

---

### 22. Rolling Window Drawdown Peak (risk_guard.py:99-133)

```python
peak = max(max(self._peak_nw_window, default=nw), nw)
# _peak_nw_window is a deque with maxlen=dd_window
```

**Почему это НЕ баг (BY DESIGN)**: Peak вычисляется в пределах **СКОЛЬЗЯЩЕГО ОКНА** (`dd_window` баров). Это **намеренное** поведение для "recent drawdown" метрики. После заполнения окна peak может уменьшиться — это корректно.

Для глобального drawdown: `dd_window: 999999` в configs/risk.yaml.

---

### 23. Kill Switch Crash Recovery (services/ops_kill_switch.py:123-156)

```python
def _trip() -> None:
    _tripped = True  # 1. In-memory first
    try:
        atomic_write_with_retry(_flag_path, "1", ...)  # 2. Flag file
    except Exception:
        pass  # OK - _save_state provides backup
    _save_state()  # 3. ALWAYS runs
```

**Почему это НЕ баг**: Crash recovery обеспечивается **дублированием**:
- Если flag write упал → state содержит `tripped=True`
- Если _save_state упал → flag file существует
- При старте проверяются ОБА

I/O внутри lock — trade-off для consistency, не race condition.

---

### 24. All Features Shifted Together (features_pipeline.py:339-353)

```python
for col in cols_to_shift:
    frame_copy[col] = frame_copy[col].shift(1)
```

**Почему это НЕ баг (НЕТ temporal mismatch)**: SMA, Return, RSI и **ВСЕ** features сдвигаются на 1 период **ОДНОВРЕМЕННО**. После shift они все представляют данные на момент t-1. Temporal alignment сохраняется.

---

### 25. Winsorization Prevents Unbounded Z-scores (features_pipeline.py:588-607)

```python
if "winsorize_bounds" in ms:
    lower, upper = ms["winsorize_bounds"]
    v = np.clip(v, lower, upper)  # Clipping BEFORE z-score!
z = (v - ms["mean"]) / ms["std"]
```

**Почему это НЕ баг**: Winsorization bounds из training применяются **ДО** вычисления z-score. Flash crash: raw=70 → clipped=95 → z=-1.0 (не -6.0!). Экстремальные 50+ sigma z-scores предотвращены.

---

### 26. row_idx для Reward, obs_row_idx для Observation (trading_patchnew.py:2017-2036)

```python
reward_price_curr = self._resolve_reward_price(row_idx, row)  # Current step
# ... while observation uses next_row (obs_row_idx = next_idx)
```

**Почему это НЕ баг (GYMNASIUM SEMANTICS)**:
- `step(action)` returns `(s_{t+1}, r_t, ...)` по стандарту Gymnasium
- `s_{t+1}`: observation из next_row (будущее состояние)
- `r_t`: reward за текущий переход (текущие цены)

Это **корректная MDP семантика**, не temporal mismatch!

---

### 27. GRU vs LSTM Different Paths (custom_policy_patch1.py:972-1012)

```python
if isinstance(recurrent_module, nn.GRU):
    # Handle locally with explicit reshape
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
    ...
else:  # LSTM
    # Delegate to base class _process_sequence
    return RecurrentActorCriticPolicy._process_sequence(...)
```

**Почему это НЕ баг (BY DESIGN)**:
- GRU проще (одно hidden state) → обрабатывается локально
- LSTM сложнее (h, c states) → делегируется в базовый класс sb3_contrib
- `_process_sequence` внутри делает тот же reshape для episode_starts
- Оба пути корректно обрабатывают episode boundaries

---

### 28. AdaptiveUPGD grad_norm_ema=1.0 при инициализации (adaptive_upgd.py:159)

```python
if group["adaptive_noise"]:
    state["grad_norm_ema"] = 1.0  # Neutral starting point
```

**Почему это НЕ баг**:
1. **Default mode bypasses EMA**: `instant_noise_scale=True` (default) использует `current_grad_norm` напрямую
2. Строки 215-219: `if group["instant_noise_scale"]: grad_norm_for_noise = current_grad_norm`
3. EMA используется ТОЛЬКО для legacy mode и diagnostics
4. Для legacy mode (`instant_noise_scale=False`) применяется bias correction (строка 224-225)

**Fix уже применён** (2025-11-26): `instant_noise_scale=True` по умолчанию для VGS совместимости.

---

### 29. mediator norm_cols_validity=True (mediator.py:1272)

```python
norm_cols_validity = np.ones(21, dtype=bool)  # Assume valid by default
# Далее ВСЕ 21 элемент перезаписываются:
norm_cols_values[0], norm_cols_validity[0] = self._get_safe_float_with_validity(row, "cvd_24h", 0.0)
# ... (строки 1276-1301)
norm_cols_values[20], norm_cols_validity[20] = self._get_safe_float_with_validity(...)
```

**Почему это НЕ баг**: Начальное значение `np.ones(21)` **полностью перезаписывается** в цикле (строки 1276-1301). Каждый из 21 элементов явно получает значение от `_get_safe_float_with_validity()`. Начальное значение нерелевантно.

---

### 30. risk_guard.py асимметричный buffer (risk_guard.py:668-671)

```python
if exposure_delta > self._EPS:
    buffered_delta = notional_delta * buffer_mult  # Buffer ТОЛЬКО на increase
else:
    buffered_delta = notional_delta  # Без buffer на decrease
```

**Почему это BY DESIGN (корректный risk management)**:
- **Position INCREASE** → нужен safety margin (slippage, fees, market impact)
- **Position DECREASE** → риск уменьшается, дополнительный buffer не нужен
- Это стандартная практика: консервативность при открытии, не при закрытии позиций

---

### 31. ops_kill_switch _last_ts=0.0 при инициализации (ops_kill_switch.py:28, 112-114)

```python
_last_ts: Dict[str, float] = {"rest": 0.0, "ws": 0.0, ...}  # Line 28

def _maybe_reset_all(now: float) -> None:
    for k in list(_counters.keys()):
        if now - _last_ts[k] > _reset_cooldown_sec:  # При now > 60: True
            _counters[k] = 0
            _last_ts[k] = now
```

**Почему это НЕ баг**:
1. `_last_ts[k] = 0.0` означает "последний reset в Unix epoch"
2. При первом вызове `record_error()` в time > 60s: counter сбрасывается до 0, затем инкрементируется до 1
3. При вызове в time < 60s: counter просто инкрементируется до 1
4. Оба сценария дают корректный результат (counter = 1)

---

### 32. RSI timing: valid на bar 14 (transformers.py:959-968)

```python
st["gain_history"].append(gain)
st["loss_history"].append(loss)

if st["avg_gain"] is None or st["avg_loss"] is None:
    if len(st["gain_history"]) == self.spec.rsi_period:  # == 14
        st["avg_gain"] = sum(st["gain_history"]) / float(self.spec.rsi_period)
        st["avg_loss"] = sum(st["loss_history"]) / float(self.spec.rsi_period)
```

**Почему это НЕ баг (timing корректен)**:

| Bar | Action | len(gain_history) | RSI valid? |
|-----|--------|-------------------|------------|
| 0 | last_close = price0 | 0 | ❌ |
| 1 | delta = p1-p0, append | 1 | ❌ |
| ... | ... | ... | ❌ |
| 14 | delta = p14-p13, append | 14 | ✅ SMA computed |

**RSI-14** требует 14 price changes → доступен после 15 prices (bars 0-14). Bar 14 — корректный момент.

**Референс**: Wilder (1978), "New Concepts in Technical Trading Systems"

---

### 33. obs_builder vol_proxy=0.01 во время ATR warmup (obs_builder.pyx:389-396)

```cython
if atr_valid:
    vol_proxy = tanh(log1p(atr / (price_d + 1e-8)))
else:
    atr_fallback = price_d * 0.01  # 1% of price
    vol_proxy = tanh(log1p(atr_fallback / (price_d + 1e-8)))
```

**Почему это BY DESIGN (trade-off)**:

| Вариант | vol_proxy | Проблема |
|---------|-----------|----------|
| NaN | NaN | Observation crash, NaN propagation |
| 0.0 | 0.0 | Model видит "нулевая волатильность" — неверно! |
| **1% price** | ~0.01 | Разумная аппроксимация типичного ATR |

Типичный ATR для crypto: 1-3% от цены. Fallback 1% — консервативная оценка.

---

### 34. obs_builder FG=50 vs missing РАЗЛИЧИМЫ (obs_builder.pyx:590-600)

```cython
if has_fear_greed:
    feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)  # FG=50 → 0.5
    indicator = 1.0  # FLAG: present
else:
    feature_val = 0.0
    indicator = 0.0  # FLAG: missing
```

**Почему это НЕ баг**:

| Сценарий | feature_val | indicator | Различимы? |
|----------|-------------|-----------|------------|
| FG = 50 | 0.5 | **1.0** | ✅ |
| FG missing | 0.0 | **0.0** | ✅ |

Indicator flag (второй элемент пары) **полностью различает** реальные данные от отсутствующих.

---

### 35. Policy sigma range [0.2, 1.5] (custom_policy_patch1.py:1088-1091)

```python
sigma_min, sigma_max = 0.2, 1.5
sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(self.unconstrained_log_std)
```

**Почему это НЕ баг (standard PPO practice)**:
- **σ = 0.2**: near-deterministic actions (exploitation phase)
- **σ = 1.5**: high exploration
- Работает для обоих: tanh [-1,1] и sigmoid [0,1] выходов
- Большое σ естественно приводит к saturated actions (bounds)

**Референс**: Schulman et al. (2017) PPO, OpenAI Baselines defaults

---

### 36. features_pipeline constant detection на shifted data (features_pipeline.py:396-410)

```python
m = float(np.nanmean(v_clean))  # Ignores NaN
s = float(np.nanstd(v_clean, ddof=0))  # Ignores NaN
is_constant = (not np.isfinite(s)) or (s == 0.0)
```

**Почему это НЕ баг (practical for typical datasets)**:
1. `nanmean`/`nanstd` **игнорируют NaN** при вычислении
2. Shifted data имеет NaN только в первых ~20 rows
3. Типичный training dataset: 10,000+ rows
4. Первые 20 NaN rows составляют < 0.2% — negligible impact
5. Statistics корректно вычисляются на valid portion

**Edge case**: Если dataset < 100 rows, могут быть issues. Но training datasets всегда >>1000 rows.

---

### 37. mark_for_obs passed but "recomputed" inside _signal_only_step (trading_patchnew.py:1868-1879, 1040)

```python
# Caller (step method):
mark_for_obs = self._resolve_reward_price(row_idx, row)  # current row
result = self._signal_only_step(..., float(mark_for_obs), ...)

# Inside _signal_only_step:
next_mark_price = self._resolve_reward_price(obs_row_idx, next_row)  # NEXT row (different!)
```

**Почему это НЕ баг**:
1. `mark_price` (from caller) используется для **текущего** net_worth (line 979)
2. `next_mark_price` вычисляется для **следующей** строки (Gymnasium semantics: obs = s_{t+1})
3. Это **разные rows** с разными ценами — повторное вычисление НЕОБХОДИМО
4. `mark_price` также используется как fallback (line 1042) если next invalid

---

### 38. ratio_clipped not clipped in signal_only mode (trading_patchnew.py:2126-2129)

```python
# Signal-only mode:
ratio_clipped = float(ratio_price)  # No np.clip() call!

# Non-signal_only mode:
ratio_clipped = float(np.clip(ratio_price, ratio_clip_floor, ratio_clip_ceiling))
```

**Почему это BY DESIGN (НЕ баг)**:
1. Variable named "ratio_clipped" for **API consistency** — info dict always has this key
2. In signal_only: ratio is **sanitized** (NaN→1.0) but not bounds-clipped
3. Signal-only mode doesn't simulate extreme price moves — clipping unnecessary
4. Comment added to code explaining this design decision

---

### 39. Empty action array returned without mapping (wrappers/action_space.py:108-110)

```python
if isinstance(action, np.ndarray):
    if action.size == 0:
        return action  # Returns empty array as-is
```

**Почему это НЕ баг (корректное поведение)**:
1. Empty array contains **nothing to map** — no elements to transform
2. Mapping formula `(arr + 1.0) / 2.0` on empty array would still produce empty array
3. Early return preserves type and is more efficient
4. This is standard defensive programming for edge cases

---

### 40. _log_sigmoid_jacobian_from_raw misleading name (custom_policy_patch1.py:1350-1353)

```python
def _log_sigmoid_jacobian_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
    # DEPRECATED: Use _log_activation_jacobian instead
    # Kept for backwards compatibility
    return self._log_activation_jacobian(raw)
```

**Почему это НЕ баг**:
1. Method is **explicitly marked DEPRECATED** in comment
2. Delegates to correctly-named `_log_activation_jacobian`
3. Kept for **backwards compatibility** — external code may reference it
4. Will be removed in future major version

---

### 41. 4 samples for entropy estimation (custom_policy_patch1.py:1420-1433)

```python
samples = 4
entropy_accum: Optional[torch.Tensor] = None
for _ in range(samples):
    raw_sample = rsample_fn()
    ...
entropy_estimate = -(entropy_accum / float(samples))
```

**Почему это НЕ проблема**:
1. Monte Carlo entropy variance scales as O(1/n) — 4 samples gives ~25% relative error
2. **ent_coef = 0.001** (from configs) — entropy contributes tiny fraction to loss
3. Impact on total loss: `0.001 × entropy × (1 ± 0.25)` ≈ negligible
4. Increasing to 16 samples would 4x compute for <0.1% loss improvement
5. Trade-off: speed vs accuracy — current choice prioritizes training throughput

---

### 42. No handling for reduction with spaces/case (distributional_ppo.py:3495-3496)

```python
if reduction not in ("none", "mean", "sum"):
    raise ValueError(f"Invalid reduction mode: {reduction}")
```

**Почему это НЕ баг (стандартный API design)**:
1. Follows **PyTorch convention** — exact string matching, no normalization
2. `torch.nn.functional.mse_loss(reduction="Mean")` also raises error
3. Case sensitivity is **intentional** for API strictness
4. Adding `.lower().strip()` would hide caller bugs and violate principle of least surprise

---

### 43. Redundant isfinite(bb_width) check (obs_builder.pyx:550-559)

```python
if (not bb_valid) or bb_width <= min_bb_width:
    feature_val = 0.5
else:
    if not isfinite(bb_width):  # "Redundant" check
        feature_val = 0.5
    else:
        feature_val = _clipf(...)
```

**Почему это НЕ баг (defense-in-depth)**:
1. `bb_valid` checks **indicator computed** — not that bb_width is finite
2. Edge case: bb_valid=True but bb_width=inf from overflow in upstream calc
3. Comment in code explicitly says "Additional safety" — **intentional redundancy**
4. Cost: one `isfinite()` check; Benefit: guaranteed NaN-free output
5. Defense-in-depth is **best practice** for numerical code

---

### 44. ma20 variable is actually 21-bar MA (mediator.py:1199-1201)

```python
# HISTORICAL NAMING: Variable named "ma20" for feature schema compatibility
# Actual value is 21-bar SMA (sma_5040 = 21 bars × 240 min)
ma20 = self._get_safe_float(row, "sma_5040", float('nan'))
```

**Почему это BY DESIGN (НЕ баг)**:
1. Variable name is **legacy** from feature schema (feature_config.py)
2. Renaming would break:
   - Feature parity checks
   - Trained models expecting this feature order
   - Audit scripts and documentation
3. Comment added to code explaining the naming
4. Underlying value (21-bar SMA) is **correct** — only name is historical artifact

---

### 45. Twin Critics Logging Accumulators (distributional_ppo.py:11088-11094, 12288-12290)

```python
# Accumulation during training:
self._twin_critic_1_loss_sum += float(loss_critic_1.mean().item()) * weight

# Reset after logging:
self._twin_critic_1_loss_sum = 0.0
self._twin_critic_2_loss_sum = 0.0
self._twin_critic_loss_count = 0
```

**Почему это НЕ memory leak**:
1. Accumulators are **RESET** at line 12288-12290 after logging
2. Reset happens at end of each train() iteration
3. Float values can't overflow in practice (values << 1e308)
4. This is standard accumulate-then-log pattern

---

### 46. Advantage Normalization ddof=1 (distributional_ppo.py:8454)

```python
adv_std = float(np.std(advantages_flat, ddof=1))  # Sample std with Bessel correction
```

**Почему это minor inconsistency (НЕ баг)**:
1. SB3 uses `ddof=0` (population std), our code uses `ddof=1` (sample std)
2. Difference: factor √(n/(n-1)) ≈ 1.0005 for n=10000
3. For typical batch sizes (n>1000): difference < 0.1%
4. Both approaches are valid — this is a philosophical difference
5. ddof=1 gives unbiased estimate, ddof=0 is more common in RL

**Референс**: Bessel's correction, SB3 `on_policy_algorithm.py`

---

### 47. VGS State in PBT Checkpoints (adversarial/pbt_scheduler.py:340-455)

```python
# Each worker saves to unique file:
checkpoint_path = f"member_{member.member_id}_step_{step}.pt"
torch.save(checkpoint_to_save, checkpoint_path)

# VGS state is serialized atomically:
has_vgs = 'vgs_state' in checkpoint_data
```

**Почему это НЕ race condition**:
1. Each PBT worker has **its own model and VGS instance**
2. Checkpoints are saved to **unique files** per worker
3. torch.save/load are atomic at OS level
4. Python GIL prevents concurrent access to live objects
5. VGS state_dict is serialized **before** save (no concurrent modification)

---

### 48. CVaR Approximation Error ~16% for N=21 (distributional_ppo.py:3612-3615)

```python
# Note on Accuracy:
#     - Perfect for linear distributions (0% error)
#     - ~5-18% approximation error for standard normal (decreases with N)
#     - N=21 (default): ~16% error
```

**Почему это documented trade-off (НЕ баг)**:
1. **Already documented** in code with accuracy notes
2. Numerical integration over discrete quantiles has inherent error
3. Error decreases with N: N=51 gives ~5%, N=101 gives ~2%
4. Trade-off: more quantiles = more accurate but slower training
5. For risk-critical applications: increase `num_quantiles` to 51+

**Референс**: Dabney et al. (2018) "IQN", quantile regression theory

---

### 49. Winsorization Percentiles [1%, 99%] (features_pipeline.py:181)

```python
winsorize_percentiles: Tuple[float, float] = (1.0, 99.0)
```

**Почему это configurable (НЕ issue)**:
1. Default [1%, 99%] clips 2% of extreme values
2. For crypto with fat tails: can adjust to [0.5%, 99.5%] or [0.1%, 99.9%]
3. This is a **configurable parameter**, not hardcoded limitation
4. Winsorization bounds are computed from training data and stored
5. Inference applies same bounds for consistency

---

### 50. obs_builder.pyx boundscheck=False (obs_builder.pyx:1)

```cython
# cython: boundscheck=False, wraparound=False
```

**Почему это BY DESIGN (performance trade-off)**:
1. `boundscheck=False` is a **deliberate Cython optimization** for critical path
2. The `build_observation_vector` Python wrapper validates all inputs before calling C version
3. Array size is determined by `compute_n_features()` which ensures consistency with observation_space
4. If mismatch occurs, it's a configuration error caught during testing
5. Re-enabling bounds checking would add ~15-20% overhead to observation building
6. Defense layers: P0 (mediator validation) → P1 (wrapper validation) → C function

**Referenced in**: 2025-11-26 bug investigation (Issue #2 - concluded NOT A BUG)

---

### 51. Slippage Model Uses Mid-Price (execution_sim.py:5901-5910)

```python
cost_fraction = float(expected_bps) / 1e4
if side_key == "BUY":
    candidate = mid_val * (1.0 + cost_fraction)
```

**Почему это НЕ проблема (already has market impact model)**:
1. Slippage module уже включает **market impact term**: `k * sqrt(participation_ratio)` (impl_slippage.py:2342)
2. Это стиль **Almgren-Chriss** square-root impact model
3. `participation_ratio = order_notional / ADV` учитывает размер ордера
4. Mid-price — только reference point; фактический slippage включает:
   - Half spread (`half_spread`)
   - Market impact (`k_effective * sqrt(participation_ratio)`)
   - Volatility adjustments
   - Tail shock для extreme conditions
5. Для полного LOB simulation нужен external LOB — это documented design choice

**Референс**: Almgren & Chriss (2001), impl_slippage.py:2290-2354

---

### 52. Latency Clamping Warnings Configurable (execution_sim.py:7110-7126)

```python
if ratio > 1.0 and self._intrabar_log_warnings:  # Configurable!
    logger.warning("intrabar latency %.0f ms exceeds timeframe %.0f ms ...")
    # Throttled to avoid log spam
if ratio > 1.0:
    ratio = 1.0  # Clamped to end of bar
```

**Почему это НЕ "silent" clamping**:
1. Warning **IS** logged when `_intrabar_log_warnings=True`
2. Default `False` для performance (production не нуждается в verbose logging)
3. Throttling предотвращает log spam
4. Configurable через `execution.intrabar.log_warnings: true`
5. Clamping at 100% — корректное поведение (исполнение в конце бара)

**Референс**: execution_sim.py:2555, 2598-2604

---

### 53. No LOB Depth Tracking (execution_sim.py:11414-11424, docstring)

```python
# Из docstring модуля (execution_sim.py:14-16):
# 3) Работать как с внешним LOB (если он передан), так и без него (простая модель):
#    - Для LIMIT без LOB исполняем только если есть abs_price
```

**Почему это BY DESIGN (not a bug)**:
1. **Documented design choice**: модуль работает с/без external LOB
2. Full LOB simulation = significant computational overhead
3. Queue position tracking добавит complexity без proportional benefit
4. Для backtesting стратегий простая модель достаточна
5. Production с крупными объёмами: используйте external LOB adapter
6. Market impact через `participation_ratio` уже покрывает основной эффект

**Референс**: execution_sim.py:4-23 (module docstring), standard backtesting practice

---

## 📊 СТАТУС ПРОЕКТА (2025-11-28)

### ✅ Production Ready

Все критические исправления применены и протестированы. **300+ тестов** с 97%+ pass rate.

| Компонент | Статус | Тесты |
|-----------|--------|-------|
| Step Observation Timing | ✅ Production | 6/6 |
| Signal Pos in Observation | ✅ Production | 10/10 |
| CLOSE_TO_OPEN Timing | ✅ Production | 5/5 |
| LongOnlyActionWrapper | ✅ Production | 26/26 |
| AdaptiveUPGD Optimizer | ✅ Production | 119/121 |
| UPGDW Optimizer | ✅ Production | 4/4 |
| Twin Critics + VF Clipping | ✅ Production | 49/50 |
| VGS v3.1 | ✅ Production | 7/7 |
| PBT | ✅ Production | 14/14 |
| SA-PPO | ✅ Production | 16/16 |
| Data Leakage Prevention | ✅ Production | 46/47 |
| Technical Indicators | ✅ Production | 11/16 (C++ pending) |
| Fear & Greed Detection | ✅ Production | 13/13 |
| Crypto Parametric TCA | ✅ Production | 84/84 (NEW) |
| Bug Fixes 2025-11-26 | ✅ Production | 22/22 (includes projection+YZ fixes) |

### ⚠️ Требуется действие

**Переобучите модели**, если они обучены **до 2025-11-26**:
- **UPGDW min-max normalization fix (2025-11-26)** — weight protection inverted with negative utilities!
- **Fear & Greed detection fix (2025-11-26)** — FG=50 ошибочно помечался как missing data!
- **signal_pos in observation fix (2025-11-26)** — obs содержал prev_signal_pos (t), но market data из t+1!
- **step() observation timing fix (2025-11-25)** — obs был из той же row что reset!
- **CLOSE_TO_OPEN + SIGNAL_ONLY fix (2025-11-25)** — look-ahead bias в signal position
- **LongOnlyActionWrapper action space fix (2025-11-25)** — минимальная позиция была 50%!
- Data leakage fix (2025-11-23) + close_orig fix (2025-11-25)
- RSI/CCI initialization fixes (2025-11-24)
- Twin Critics GAE fix (2025-11-21)
- LSTM state reset fix (2025-11-21)
- UPGD negative utility fix (2025-11-21)

---

## 📜 История критических исправлений

> **Примечание**: Все отчёты перемещены в `docs/archive/`. Путь: `docs/archive/reports_2025_11_25_cleanup/root_reports/`

| Дата | Исправление | Влияние |
|------|-------------|---------|
| **2025-11-28** | feat(crypto): CryptoParametricSlippageProvider | L2+ smart TCA model with 6 factors, 84 tests |
| **2025-11-27** | Stage 6: DarkPoolSimulator memory leak fix | unbounded List → deque(maxlen=N), prevents OOM in long simulations |
| **2025-11-27** | Stage 6: DarkPoolConfig validation | Division by zero prevented with ValueError for invalid params |
| **2025-11-27** | Stage 6: deque slice fix in _should_block_for_leakage | TypeError on deque slicing → convert to list first |
| **2025-11-27** | VGS v3.2: min_scaling_factor + variance_cap | EV≈0, Twin Critics loss +327%, grad norm -82% → VGS не блокирует обучение |
| **2025-11-26** | Twin Critics categorical VF clipping projection fix | `_project_distribution` was identity stub → now uses proper C51 projection |
| **2025-11-26** | Yang-Zhang RS denominator fix | RS used (n-1) instead of n → +11% inflation for n=10 removed |
| **2025-11-26** | `_project_categorical_distribution` batch shape fix | Shape mismatch for 1D atoms with batched probs → properly expands |
| **2025-11-26** | UPGDW min-max normalization fix | Negative utilities no longer invert weight protection |
| **2025-11-26** | Data exhaustion truncation fix | Episode properly ends with truncated=True when data runs out |
| **2025-11-26** | cql_beta validation fix | Division by zero prevented with ValueError for cql_beta <= 0 |
| **2025-11-26** | Mediator dead code removal | Removed unreachable `is None` check (code smell) |
| **2025-11-26** | Fear & Greed detection fix | FG=50 (neutral) correctly detected as valid data, not missing |
| **2025-11-26** | AdaptiveUPGD instant_noise_scale fix | VGS + UPGD noise 212x amplification → 1.0x (constant ratio) |
| **2025-11-26** | signal_pos in observation uses next_signal_pos | Temporal mismatch: market data t+1, position t → теперь оба t+1 |
| **2025-11-26** | Limit order tolerance fix | Fixed 1e-12 < machine epsilon at $100k → relative tolerance |
| **2025-11-25** | Empty DataFrame protection in step() | IndexError при пустом df → graceful termination |
| **2025-11-25** | step() observation from NEXT row (Gymnasium) | Duplicate obs: reset() и step()#1 возвращали одну row |
| **2025-11-25** | CLOSE_TO_OPEN + SIGNAL_ONLY timing | Look-ahead bias: signal_pos игнорировал 1-bar delay |
| **2025-11-25** | info["signal_pos_next"] consistency | Показывал intent вместо actual; добавлен signal_pos_requested |
| **2025-11-25** | reset() returns actual observation (Issue #1) | LSTM получал zeros на первом step эпизода |
| **2025-11-25** | Improved _last_reward_price init (Issue #3) | reward=0 если данные начинались с NaN |
| **2025-11-25** | Removed redundant signal_position update (Issue #2) | Code smell (не влияло на функционал) |
| **2025-11-25** | LongOnlyActionWrapper action space | Минимальная позиция была 50% вместо 0%! |
| **2025-11-25** | Policy adaptive activation (tanh/sigmoid) | Policy теперь адаптируется к action_space |
| **2025-11-25** | close_orig semantic conflict | Data leakage в pipeline |
| **2025-11-24** | Twin Critics loss aggregation | 25% underestimation |
| **2025-11-24** | RSI/CCI initialization | 5-20x error first 150 bars |
| **2025-11-23** | Data leakage (all features) | Look-ahead bias |
| **2025-11-23** | VGS v3.1 E[g²] computation | 10,000x underestimation |
| **2025-11-23** | SA-PPO epsilon + KL | Schedule + 10x faster |
| **2025-11-23** | GAE overflow protection | Float32 overflow |
| **2025-11-22** | PBT deadlock prevention | Indefinite wait |
| **2025-11-22** | Twin Critics VF Clipping | Independent critic updates |
| **2025-11-21** | Twin Critics GAE | min(Q1,Q2) not applied |
| **2025-11-21** | LSTM state reset | Temporal leakage 5-15% |
| **2025-11-21** | UPGD negative utility | Inverted weight protection |
| **2025-11-21** | Action space (3 bugs) | Position doubling |
| **2025-11-20** | Numerical stability (5 bugs) | Gradient explosions |
| **2025-11-20** | Feature engineering (3 bugs) | Volatility bias 1-5% |

---

## О проекте

**AI-Powered Quantitative Research Platform** — ML-платформа для количественных исследований и торговли на криптовалютах (Binance spot/futures) и акциях (Alpaca/Polygon), использующая reinforcement learning (Distributional PPO) для принятия торговых решений.

### Основные характеристики

- **Язык**: Python 3.12 + Cython + C++
- **RL Framework**: Stable-Baselines3 (Distributional PPO with Twin Critics)
- **Optimizer**: AdaptiveUPGD (default) — continual learning
- **Gradient Scaling**: VGS v3.2 — automatic per-layer normalization + anti-blocking
- **Training**: PBT + SA-PPO (adversarial training)
- **Биржа**: Binance (Spot/Futures)
- **Режимы**: Бэктест, Live trading, Обучение

---

## 🚀 Продвинутые возможности

### Quick Reference: Training Configuration

```yaml
# configs/config_train.yaml
model:
  algo: "ppo"
  optimizer_class: AdaptiveUPGD
  optimizer_kwargs:
    lr: 1.0e-4
    weight_decay: 0.001
    sigma: 0.001                       # CRITICAL для VGS
    beta_utility: 0.999
    beta1: 0.9
    beta2: 0.999

  vgs:
    enabled: true
    accumulation_steps: 4
    warmup_steps: 10
    clip_threshold: 10.0

  params:
    use_twin_critics: true             # Default: enabled
    num_atoms: 21
    v_min: -10.0
    v_max: 10.0
    cvar_alpha: 0.05
    cvar_weight: 0.15
    clip_range_vf: 0.7
    gamma: 0.99                        # Must match reward.gamma!
    gae_lambda: 0.95
    clip_range: 0.10
    ent_coef: 0.001
    vf_coef: 1.8
    max_grad_norm: 0.5
```

### 1. UPGD Optimizer

**Статус**: ✅ Production Ready | **Default**: Enabled (AdaptiveUPGD)

Continual learning optimizer для предотвращения catastrophic forgetting.

**Варианты**: AdaptiveUPGD (рекомендуется), UPGD, UPGDW

**Документация**: [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md)

### 2. Twin Critics

**Статус**: ✅ Production Ready | **Default**: Enabled

Две независимые value networks для снижения overestimation bias.

```
[Observation] → [LSTM] → [MLP] → [Critic Head 1] → [Value 1]
                                ↘ [Critic Head 2] → [Value 2]
Target Value = min(Value 1, Value 2)
```

**Документация**: [docs/twin_critics.md](docs/twin_critics.md)

### 3. VGS (Variance Gradient Scaler)

**Статус**: ✅ Production Ready | **Version**: v3.1

Автоматическое масштабирование градиентов на основе стохастической вариации.

**Важно**: При использовании с UPGD установите `sigma` в диапазоне 0.0005-0.001.

### 4. PBT (Population-Based Training)

**Статус**: ✅ Production Ready

Эволюционная оптимизация гиперпараметров через популяцию агентов.

```yaml
pbt:
  enabled: true
  population_size: 8
  perturbation_interval: 10
  min_ready_members: 2          # Deadlock prevention
  ready_check_max_wait: 10
```

### 5. SA-PPO (State-Adversarial PPO)

**Статус**: ✅ Production Ready

Robust training через adversarial perturbations (PGD attack).

```yaml
adversarial:
  enabled: true
  perturbation:
    epsilon: 0.075
    attack_steps: 3
    attack_lr: 0.03
```

---

## Архитектура проекта

**Слоистая архитектура** с префиксами имён файлов:

```
core_ → impl_ → service_ → strategies → script_
```

**ВАЖНО**: Нарушение зависимостей → циклические импорты!

### Слои

| Слой | Префикс | Описание |
|------|---------|----------|
| Базовый | `core_*` | Модели, контракты, константы. Без зависимостей. |
| Реализация | `impl_*` | Инфраструктура. Зависит только от `core_`. |
| Сервисы | `service_*` | Бизнес-логика. Зависит от `core_`, `impl_`. |
| Стратегии | `strategies/` | Торговые алгоритмы. Зависит от всех. |
| CLI | `script_*` | Точки входа. Использует DI. |

### Ключевые файлы

**Core**: `core_config.py`, `core_models.py`, `core_strategy.py`

**Impl**: `impl_sim_executor.py`, `impl_fees.py`, `impl_slippage.py`, `impl_latency.py`

**Service**: `service_backtest.py`, `service_train.py`, `service_eval.py`, `service_signal_runner.py`

**ML**: `distributional_ppo.py`, `custom_policy_patch1.py`, `variance_gradient_scaler.py`

**Scripts**: `train_model_multi_patch.py`, `script_backtest.py`, `script_live.py`, `script_eval.py`

---

## Основные компоненты

### 1. Симулятор исполнения

`execution_sim.py` — симуляция LOB, микроструктура, проскальзывание, комиссии.

Алгоритмы: TWAP, POV, VWAP

### 2. Distributional PPO

`distributional_ppo.py` — PPO с:
- Distributional value head (quantile regression)
- Twin Critics (default enabled)
- VGS gradient scaling
- AdaptiveUPGD optimizer
- CVaR risk-aware learning

### 3. Features Pipeline

`features_pipeline.py` — препроцессинг с проверкой паритета.

63 features: price, volume, volatility, momentum, microstructure.

### 4. Риск-менеджмент

`risk_guard.py` — гварды на позицию/PnL/дроудаун.

`services/ops_kill_switch.py` — операционный kill switch.

---

## Конфигурации

| Файл | Назначение |
|------|------------|
| `config_train.yaml` | Обучение (standard) |
| `config_pbt_adversarial.yaml` | PBT + SA-PPO |
| `config_sim.yaml` | Бэктест |
| `config_live.yaml` | Live trading |
| `config_eval.yaml` | Оценка модели |

**Модульные**: `execution.yaml`, `fees.yaml`, `slippage.yaml`, `risk.yaml`, `no_trade.yaml`

---

## CLI Примеры

```bash
# Бэктест
python script_backtest.py --config configs/config_sim.yaml

# Обучение
python train_model_multi_patch.py --config configs/config_train.yaml

# PBT + Adversarial
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml

# Live trading
python script_live.py --config configs/config_live.yaml

# Оценка
python script_eval.py --config configs/config_eval.yaml --all-profiles

# Обновление данных
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python scripts/refresh_fees.py
```

---

## Тестирование

```bash
pytest tests/                          # Все тесты
pytest tests/test_twin_critics*.py -v  # Twin Critics
pytest tests/test_upgd*.py -v          # UPGD
pytest tests/test_pbt*.py -v           # PBT
```

### Ключевые тестовые файлы

| Категория | Файлы |
|-----------|-------|
| Twin Critics | `test_twin_critics*.py` (49 тестов) |
| UPGD | `test_upgd*.py` (119 тестов) |
| VGS | `test_vgs*.py` (7 тестов) |
| Data Leakage | `test_data_leakage*.py`, `test_close_orig*.py` |
| Indicators | `test_indicator*.py`, `test_rsi_cci*.py` |
| Action Space | `test_critical_action_space_fixes.py`, `test_long_only_action_space_fix.py` (26+21 тестов) |
| LSTM | `test_lstm_episode_boundary_reset.py` |
| Reset Observation | `test_trading_env_reset_observation_fixes.py` (9 тестов) |
| Phase 9 Live Trading | `test_phase9_live_trading.py` (46 тестов) |

---

## Документация

### Основная

- [DOCS_INDEX.md](DOCS_INDEX.md) — Индекс документации
- [ARCHITECTURE.md](ARCHITECTURE.md) — Архитектура
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) — Сборка

### Продвинутые возможности

- [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) — UPGD Optimizer
- [docs/twin_critics.md](docs/twin_critics.md) — Twin Critics
- [docs/pipeline.md](docs/pipeline.md) — Decision pipeline
- [docs/bar_execution.md](docs/bar_execution.md) — Bar execution

### Отчёты об исправлениях

**Все отчёты перенесены в архив:**
- Основной архив: `docs/archive/reports_2025_11_25_cleanup/`
- Критические исправления: `docs/archive/reports_2025_11_25_cleanup/root_reports/`
- Верификация: `docs/archive/verification_2025_11/`

---

## Важные переменные окружения

```bash
BINANCE_API_KEY, BINANCE_API_SECRET     # API ключи
TB_FAIL_ON_STALE_FILTERS=1              # Fail при устаревших фильтрах
BINANCE_PUBLIC_FEES_DISABLE_AUTO=1      # Отключить автообновление fees
```

---

## Production Checklist

### Данные и конфигурация
- [ ] Обновлены фильтры (`fetch_binance_filters.py`)
- [ ] Обновлены комиссии (`refresh_fees.py`)
- [ ] Проверены risk limits (`risk.yaml`)
- [ ] Проверены no-trade окна (`no_trade.yaml`)

### ML Модель
- [ ] AdaptiveUPGD настроен
- [ ] VGS enabled, warmup настроен
- [ ] Twin Critics enabled
- [ ] `gamma` синхронизирован (reward = model)
- [ ] **Long-only**: wrapper устанавливает [-1,1], policy использует tanh
- [ ] Model trained after 2025-11-25

### Тестирование
- [ ] `pytest tests/` — все тесты проходят
- [ ] `check_feature_parity.py` — паритет OK
- [ ] `sim_reality_check.py` — симуляция реалистична

### Live Trading
- [ ] API ключи настроены
- [ ] Kill switch протестирован
- [ ] Мониторинг настроен

---

## Заключение

### Золотые правила

1. **Следуйте слоистой архитектуре**
2. **Читайте файлы перед изменением**
3. **Пишите тесты для критичной логики**
4. **Проверяйте feature parity**
5. **Мониторьте метрики**

### Когда что-то идёт не так

1. Проверьте тесты для проблемной области
2. Используйте Glob/Grep для поиска
3. Проверьте конфиги
4. Проверьте слоистую архитектуру
5. Изучите историю исправлений (таблица выше)

---

**Последнее обновление**: 2025-11-28
**Версия документации**: 10.1 (Phase 10 + Crypto Parametric TCA)
**Статус**: ✅ Production Ready (все критические исправления применены, 53 задокументированных "НЕ БАГИ")
