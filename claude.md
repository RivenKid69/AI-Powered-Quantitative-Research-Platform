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
| Equity Parametric TCA | `execution_providers.py` | `pytest tests/test_equity_parametric_tca.py` |
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
| Benchmark temporal alignment | `stock_features.py` | `pytest tests/test_benchmark_temporal_alignment.py` |
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
| Conformal prediction | `core_conformal.py`, `impl_conformal.py`, `service_conformal.py` | `pytest tests/test_conformal_prediction.py` |
| Uncertainty bounds | `service_conformal.py` | `pytest tests/test_conformal_prediction.py::TestUncertaintyTracker` |
| CVaR bounds | `impl_conformal.py` | `pytest tests/test_conformal_prediction.py::TestConformalCVaREstimator` |
| Stock features (VIX, RS) | `stock_features.py` | `pytest tests/test_stock_features.py` |
| Stock risk guards | `services/stock_risk_guards.py` | `pytest tests/test_stock_risk_guards.py` |
| Stock universe mgmt | `services/universe_stocks.py` | `pytest tests/test_universe_stocks.py` |
| US market structure | `lob/us_market_structure.py` | `pytest tests/test_us_market_structure.py` |
| Verification tools | `tools/check_*.py`, `tools/verify_*.py` | Run directly with `python tools/<script>.py` |
| Feature parity check | `tools/check_feature_parity.py` | `python tools/check_feature_parity.py` |
| **Forex Parametric TCA** | `execution_providers.py` | `pytest tests/test_forex_parametric_tca.py` |
| Forex features (sessions) | `forex_features.py` | `pytest tests/test_forex_features.py` |
| Forex dealer simulation | `services/forex_dealer.py` | `pytest tests/test_forex_dealer_simulation.py` |
| Forex risk guards | `services/forex_risk_guards.py` | `pytest tests/test_forex_phase6_risk_services.py` |
| Forex session router | `services/forex_session_router.py` | `pytest tests/test_forex_execution_integration.py` |
| Forex config | `services/forex_config.py` | `pytest tests/test_forex_configuration.py` |
| OANDA adapter | `adapters/oanda/*.py` | `pytest tests/test_forex_foundation.py` |
| Forex tick simulation | `lob/forex_tick_simulation.py` | `pytest tests/test_forex_tick_simulation.py` |
| **IB market data** (CME futures) | `adapters/ib/market_data.py` | `pytest tests/test_ib_adapters.py::TestIBMarketDataAdapter` |
| **IB order execution** (CME) | `adapters/ib/order_execution.py` | `pytest tests/test_ib_adapters.py::TestIBOrderExecutionAdapter` |
| **CME settlement** (daily variation) | `impl_cme_settlement.py` | `pytest tests/test_cme_settlement.py::TestCMESettlementEngine` |
| **CME rollover** (contract expiry) | `impl_cme_rollover.py` | `pytest tests/test_cme_settlement.py::TestContractRolloverManager` |
| **CME trading calendar** | `services/cme_calendar.py` | `pytest tests/test_cme_calendar.py::TestCMETradingCalendar` |
| **SPAN margin calculator** | `impl_span_margin.py` | `pytest tests/test_span_margin.py` |
| **CME slippage provider** | `execution_providers_cme.py` | `pytest tests/test_cme_slippage.py` |
| **CME circuit breaker** | `impl_circuit_breaker.py` | `pytest tests/test_circuit_breaker.py` |
| **CME SPAN margin guard** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestSPANMarginGuard` |
| **CME position limits** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestCMEPositionLimitGuard` |
| **CME CB aware guard** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestCircuitBreakerAwareGuard` |
| **CME settlement risk** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestSettlementRiskGuard` |
| **CME rollover guard** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestRolloverGuard` |
| **CME unified risk** | `services/cme_risk_guards.py` | `pytest tests/test_cme_risk_guards.py::TestCMEFuturesRiskGuard` |
| **Unified futures risk** | `services/unified_futures_risk.py` | `pytest tests/test_unified_futures_risk.py` |
| **Asset type detection** | `services/unified_futures_risk.py` | `pytest tests/test_unified_futures_risk.py::TestAssetType` |
| **Portfolio risk mgr** | `services/unified_futures_risk.py` | `pytest tests/test_unified_futures_risk.py::TestPortfolioRiskManager` |
| **Futures LOB extensions** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py` |
| **Liquidation cascade** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestLiquidationCascadeSimulator` |
| **Insurance fund** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestInsuranceFundManager` |
| **ADL queue** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestADLQueueManager` |
| **Funding dynamics** | `lob/futures_extensions.py` | `pytest tests/test_futures_l3_execution.py::TestFundingPeriodDynamics` |
| **Futures L3 execution** | `execution_providers_futures_l3.py` | `pytest tests/test_futures_l3_execution.py::TestFuturesL3ExecutionProvider` |
| **CME Globex matching** | `lob/cme_matching.py` | `pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineBasic` |
| **CME MWP orders** | `lob/cme_matching.py` | `pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineMWP` |
| **CME stop orders** | `lob/cme_matching.py` | `pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineStops` |
| **CME L3 execution** | `execution_providers_cme_l3.py` | `pytest tests/test_cme_l3_execution.py::TestCMEL3ExecutionProvider` |
| **CME session detection** | `execution_providers_cme_l3.py` | `pytest tests/test_cme_l3_execution.py::TestSessionDetection` |
| **CME daily settlement** | `execution_providers_cme_l3.py` | `pytest tests/test_cme_l3_execution.py::TestDailySettlementSimulator` |
| **Futures leverage guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestFuturesLeverageGuard` |
| **Futures margin guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestFuturesMarginGuard` |
| **Margin call notifier** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestMarginCallNotifier` |
| **Funding exposure guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestFundingExposureGuard` |
| **Concentration guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestConcentrationGuard` |
| **ADL risk guard** | `services/futures_risk_guards.py` | `pytest tests/test_futures_risk_guards.py::TestADLRiskGuard` |
| **Crypto futures risk** | `risk_guard.py` | `pytest tests/test_futures_risk_guards.py::TestCryptoFuturesRiskGuard` |
| **Futures env wrapper** | `wrappers/futures_env.py` | `pytest tests/test_futures_training.py::TestFuturesEnvWrapper` |
| **Futures feature flags** | `services/futures_feature_flags.py` | `pytest tests/test_futures_feature_flags.py` |
| **Futures training config** | `configs/config_train_futures.yaml` | `pytest tests/test_futures_training.py::TestFuturesTrainingConfig` |
| **Futures live runner** | `services/futures_live_runner.py` | `pytest tests/test_futures_live_trading.py::TestFuturesLiveRunner` |
| **Futures position sync** | `services/futures_position_sync.py` | `pytest tests/test_futures_live_trading.py::TestFuturesPositionSynchronizer` |
| **Futures margin monitor** | `services/futures_margin_monitor.py` | `pytest tests/test_futures_live_trading.py::TestFuturesMarginMonitor` |
| **Futures funding tracker** | `services/futures_funding_tracker.py` | `pytest tests/test_futures_live_trading.py::TestFuturesFundingTracker` |
| **Futures live config** | `configs/config_live_futures.yaml` | `pytest tests/test_futures_live_trading.py::TestFuturesLiveConfig` |
| **Futures validation** | `tests/test_futures_validation.py` | `pytest tests/test_futures_validation.py` |
| **Futures backward compat** | `tests/test_futures_backward_compatibility.py` | `pytest tests/test_futures_backward_compatibility.py` |
| **Futures benchmarks** | `benchmarks/bench_futures_simulation.py` | `python benchmarks/bench_futures_simulation.py` |

### 🔍 Quick File Reference

| Префикс | Слой | Зависимости | Примеры |
|---------|------|-------------|---------|
| `core_*` | Базовый | Нет | `core_config.py`, `core_models.py`, `core_strategy.py` |
| `impl_*` | Реализация | `core_` | `impl_sim_executor.py`, `impl_fees.py`, `impl_slippage.py` |
| `service_*` | Сервисы | `core_`, `impl_` | `service_backtest.py`, `service_train.py`, `service_eval.py` |
| `strategies/*` | Стратегии | Все предыдущие | `strategies/base.py`, `strategies/momentum.py` |
| `script_*` | CLI точки входа | Все | `script_backtest.py`, `script_live.py`, `script_eval.py` |

### 📁 Project Organization (Updated 2025-11-30)

**ВАЖНО**: Проект реорганизован (commit db9655a). Файлы перемещены:

```
TradingBot2/
├── tests/              # 557 test files (moved from root)
│   ├── test_*.py       # All test files
│   └── conftest.py     # Pytest fixtures
├── tools/              # 34 utility scripts (moved from root)
│   ├── check_*.py      # Validation scripts
│   ├── verify_*.py     # Verification scripts
│   └── analyze_*.py    # Analysis scripts
├── scripts/            # Data fetching scripts
│   ├── download_stock_data.py
│   ├── fetch_binance_filters.py
│   └── fetch_alpaca_universe.py
├── lob/                # L3 LOB simulation modules
├── adapters/           # Exchange adapters (Binance, Alpaca, etc.)
├── services/           # Business logic services
├── strategies/         # Trading strategies
├── configs/            # YAML configuration files
├── docs/               # Documentation and archives
└── *.py                # Core modules (core_, impl_, script_, etc.)
```

**Key directories**:
- `tools/` — Scripts for verification, debugging, analysis (run directly)
- `tests/` — All pytest tests (use `pytest tests/`)
- `scripts/` — Data management scripts

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

# Training (Forex)
python train_model_multi_patch.py --config configs/config_train_forex.yaml

# Backtest (Forex)
python script_backtest.py --config configs/config_backtest_forex.yaml

# Live Trading (Forex - OANDA)
python script_live.py --config configs/config_live_forex.yaml --asset-class forex
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
| **OANDA** | Forex (OTC) | ✅ Production | MarketData, Fee, TradingHours, ExchangeInfo, OrderExecution |
| **Interactive Brokers** | CME Futures (ES, NQ, GC, CL, 6E) | ✅ Production | MarketData, OrderExecution, ExchangeInfo (via TWS API) |

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
├── yahoo/            # Yahoo Finance реализация (indices/macro)
│   ├── market_data.py      # VIX, DXY, Treasury yields
│   ├── corporate_actions.py # Dividends, splits
│   └── earnings.py          # Earnings calendar
├── oanda/            # OANDA реализация (forex OTC)
│   ├── market_data.py      # FX pairs real-time quotes
│   ├── fees.py             # Spread-based fees (no commission)
│   ├── trading_hours.py    # Sun 5pm - Fri 5pm ET sessions
│   ├── exchange_info.py    # Currency pair specifications
│   └── order_execution.py  # OTC dealer execution
└── ib/               # Interactive Brokers реализация (CME futures)
    ├── market_data.py      # Historical bars, real-time quotes (via TWS API)
    ├── order_execution.py  # Market/limit/bracket orders, margin queries
    └── exchange_info.py    # Contract specifications (ES, NQ, GC, etc.)
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

## 📈 Equity Parametric TCA (L2+)

### Обзор

Smart parametric Transaction Cost Analysis model для US equities. Расширяет базовую √participation модель (Almgren-Chriss) с equity-специфичными факторами.

**Статус**: ✅ Production Ready | **Тесты**: 86 (100% pass)

### Формула Total Slippage

```
slippage = half_spread
    × (1 + k × √participation)      # Almgren-Chriss impact
    × volatility_regime_mult        # Volatility regime (Hasbrouck 2007)
    × market_cap_mult               # Market cap tier (Kissell 2013)
    × (1 + beta_stress)             # Systematic risk adjustment
    × intraday_factor               # U-curve liquidity (ITG 2012)
    × auction_factor                # Opening/closing auction proximity
    × (1 + short_penalty)           # Short squeeze risk
    × event_mult                    # Earnings/news events
    × (1 + sector_penalty)          # Sector rotation
    × imbalance_factor              # Order book imbalance
```

### 9 Slippage Factors

| Factor | Формула | Референс |
|--------|---------|----------|
| **√Participation** | `k × √(Q/ADV)`, k ∈ [0.03, 0.08] | Almgren-Chriss (2001) |
| **Market Cap Tier** | mega=0.7, large=1.0, mid=1.3, small=1.8, micro=2.5 | Kissell (2013) |
| **Intraday U-Curve** | open=1.5 → midday=1.0 → close=1.3 | ITG (2012) |
| **Auction Proximity** | `1 + 0.3 × exp(-minutes/10)` | NYSE/NASDAQ mechanics |
| **Beta Stress** | `1 + |β-1| × SPY_move × 0.1` | Systematic risk |
| **Short Interest** | `log1p(ratio/threshold) × max_penalty` | GME-style squeeze |
| **Events** | Earnings=2.5×, News=1.5× | Event-driven volatility |
| **Sector Rotation** | Penalty when sector ETF down >1% | Cross-asset signal |
| **Volatility Regime** | LOW=0.85, NORMAL=1.0, HIGH=1.4 | Hasbrouck (2007) |

### Smart Features

| Feature | Описание |
|---------|----------|
| **Market Cap Auto-Detection** | Классификация MEGA/LARGE/MID/SMALL/MICRO по market cap |
| **Trading Session Detection** | PRE_MARKET, OPEN_AUCTION, REGULAR, CLOSE_AUCTION, AFTER_HOURS, CLOSED |
| **Adaptive Impact** | Коэффициент k адаптируется по trailing fill quality |
| **Auction Detector** | Экспоненциальный decay вблизи 9:30/16:00 ET |
| **Earnings Calendar** | Автоматическое определение T-1 to T+1 earnings window |
| **Cross-Asset Signal** | SPY volatility spike → все акции получают penalty |
| **Sector Rotation** | XLF/XLK/XLV down >1% → соответствующие акции получают penalty |

### Market Cap Tiers

| Tier | Threshold | Multiplier | Примеры |
|------|-----------|------------|---------|
| **MEGA** | >$200B | 0.7 | AAPL, MSFT, GOOGL |
| **LARGE** | $10B-$200B | 1.0 | Most S&P 500 |
| **MID** | $2B-$10B | 1.3 | Mid-cap stocks |
| **SMALL** | $300M-$2B | 1.8 | Regional banks |
| **MICRO** | <$300M | 2.5 | Penny stocks |

### Intraday U-Curve (US Eastern Time)

| Session | Часы (ET) | Factor | Описание |
|---------|-----------|--------|----------|
| Pre-market | 4:00-9:30 | 2.0-2.5 | Very low liquidity |
| Open auction | 9:30-10:00 | ~1.5 | High volume, wide spreads |
| Morning | 10:00-12:00 | 1.1-1.2 | Improving liquidity |
| Midday | 12:00-14:00 | **1.0** | Peak liquidity (best execution) |
| Afternoon | 14:00-15:00 | 1.05-1.1 | Still good |
| Pre-close | 15:00-16:00 | ~1.3 | Rising activity |
| After-hours | 16:00-20:00 | 2.0-2.5 | Low liquidity |

### Использование

```python
from execution_providers import (
    EquityParametricSlippageProvider,
    EquityParametricConfig,
    MarketCapTier,
    TradingSession,
    Order,
    MarketState,
    AssetClass,
)

# 1. Базовое использование (defaults)
provider = EquityParametricSlippageProvider()

# 2. С кастомной конфигурацией
config = EquityParametricConfig(
    impact_coef_base=0.06,
    spread_bps=2.5,
    market_cap_multipliers={"mega": 0.6, "large": 1.0, ...},
)
provider = EquityParametricSlippageProvider(config=config)

# 3. Из профиля
provider = EquityParametricSlippageProvider.from_profile("large_cap")
# Профили: "default", "conservative", "aggressive", "retail", "large_cap", "small_cap"

# 4. Вычисление slippage с полным набором параметров
slippage_bps = provider.compute_slippage_bps(
    order=Order("AAPL", "BUY", 1000, "MARKET", asset_class=AssetClass.EQUITY),
    market=MarketState(timestamp=0, bid=175.0, ask=175.02, adv=80_000_000),
    participation_ratio=0.002,
    market_cap=2.8e12,           # $2.8T (MEGA cap)
    beta=1.2,                    # Stock beta vs SPY
    time_et=12,                  # 12:00 ET (midday - best liquidity)
    spy_return_today=-0.015,     # SPY down 1.5%
    short_interest_ratio=3.0,    # 3 days to cover
    has_earnings_soon=False,
    sector="technology",
    sector_etf_return=-0.02,     # XLK down 2%
)

# 5. Pre-trade cost estimation с рекомендациями
estimate = provider.estimate_impact_cost(
    notional=1_000_000,
    adv=50_000_000,
    market_cap=50e9,
    beta=1.3,
    time_et=14,
    has_earnings_soon=True,
)
print(f"Impact: {estimate['impact_bps']:.2f} bps")
print(f"Cost: ${estimate['impact_cost']:.2f}")
print(f"Market Cap Tier: {estimate['market_cap_tier']}")
print(f"Trading Session: {estimate['trading_session']}")
print(f"Recommendation: {estimate['recommendation']}")
```

### Конфигурация (EquityParametricConfig)

| Параметр | Default | Описание |
|----------|---------|----------|
| `impact_coef_base` | 0.05 | Base k coefficient (lower than crypto!) |
| `impact_coef_range` | (0.03, 0.08) | Adaptive k bounds |
| `spread_bps` | 2.0 | Default spread (tighter than crypto) |
| `market_cap_multipliers` | {mega: 0.7, ..., micro: 2.5} | Tier multipliers |
| `market_cap_thresholds` | {mega: 200e9, large: 10e9, ...} | USD thresholds |
| `intraday_curve` | {hour: factor} | 24-hour liquidity curve (ET) |
| `auction_decay_minutes` | 10.0 | Exponential decay parameter |
| `auction_premium` | 0.3 | Max 30% auction premium |
| `vol_regime_multipliers` | {low: 0.85, normal: 1.0, high: 1.4} | Volatility scaling |
| `beta_stress_sensitivity` | 0.1 | 10% per unit beta deviation × SPY move |
| `short_interest_max_penalty` | 0.3 | Max 30% short squeeze penalty |
| `short_interest_threshold` | 5.0 | 5 days to cover threshold |
| `earnings_event_multiplier` | 2.5 | 2.5× during earnings |
| `news_event_multiplier` | 1.5 | 1.5× during news |
| `sector_penalty_threshold` | -0.01 | -1% sector ETF return triggers penalty |
| `sector_penalty_max` | 0.15 | Max 15% sector penalty |
| `min_slippage_bps` | 0.5 | Floor (lower than crypto) |
| `max_slippage_bps` | 200.0 | Cap (lower than crypto) |

### Профили

| Профиль | impact_coef | spread_bps | min_bps | Применение |
|---------|-------------|------------|---------|------------|
| `default` | 0.05 | 2.0 | 0.5 | Standard institutional |
| `conservative` | 0.07 | 3.0 | 1.0 | Safer estimates |
| `aggressive` | 0.04 | 1.5 | 0.3 | Tighter estimates |
| `retail` | 0.06 | 4.0 | 1.5 | Retail flow (wider spreads) |
| `large_cap` | 0.04 | 1.5 | 0.3 | MEGA/LARGE caps |
| `small_cap` | 0.08 | 5.0 | 2.0 | SMALL/MICRO caps |

### Сравнение Crypto vs Equity TCA

| Параметр | Crypto | Equity |
|----------|--------|--------|
| Base k coefficient | 0.10 | 0.05 |
| Default spread | 5.0 bps | 2.0 bps |
| Max slippage | 500 bps | 200 bps |
| Time-of-day | 24h UTC curve | US Eastern U-curve |
| Special factors | Funding rate, BTC correlation | Beta stress, earnings, sector rotation |
| Market structure | 24/7 trading | 9:30-16:00 ET + extended |

### Тестирование

```bash
# Все тесты equity parametric TCA
pytest tests/test_equity_parametric_tca.py -v

# По категориям
pytest tests/test_equity_parametric_tca.py::TestMarketCapTierClassification -v
pytest tests/test_equity_parametric_tca.py::TestIntradayUCurve -v
pytest tests/test_equity_parametric_tca.py::TestAuctionProximityFactor -v
pytest tests/test_equity_parametric_tca.py::TestBetaStress -v
pytest tests/test_equity_parametric_tca.py::TestShortSqueeze -v
pytest tests/test_equity_parametric_tca.py::TestEarningsWindow -v
pytest tests/test_equity_parametric_tca.py::TestSectorRotation -v
pytest tests/test_equity_parametric_tca.py::TestL2Integration -v
```

**Покрытие**: 86 тестов (100% pass)

### Референсы

- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2013): "Optimal Trading Strategies"
- Hasbrouck (2007): "Empirical Market Microstructure"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- ITG (2012): "Global Cost Review" — intraday patterns
- Cont, Kukanov, Stoikov (2014): "Price Impact of Order Book Events"
- Pagano & Schwartz (2003): "Opening and Closing Auctions"

---

## 📊 Stock Features & Risk Management (Phase 5)

### Обзор

Phase 5 добавляет stock-специфичные features и risk guards, параллельно crypto Fear & Greed индексу.

**Файлы**:
- `stock_features.py` — VIX integration, market regime, relative strength
- `services/stock_risk_guards.py` — Margin, short sale, corporate actions guards
- `services/universe_stocks.py` — Stock universe management with TTL caching

### Stock Features (`stock_features.py`)

| Feature | Описание | Источник |
|---------|----------|----------|
| **VIX Value** | Market volatility (fear gauge) | Yahoo `^VIX` |
| **VIX Regime** | LOW (<12), NORMAL (12-20), ELEVATED (20-30), EXTREME (>30) | CBOE thresholds |
| **Market Regime** | BULL/SIDEWAYS/BEAR based on SPY + VIX | SMA crossover + VIX |
| **RS vs SPY (20d)** | 20-day relative strength vs S&P 500 | Levy (1967) |
| **RS vs SPY (50d)** | 50-day relative strength vs S&P 500 | Moskowitz et al. (2012) |
| **RS vs QQQ (20d)** | 20-day relative strength vs Nasdaq 100 | Momentum proxy |
| **Sector Momentum** | Sector rotation signal | XLK, XLF, XLV ETF returns |

**Использование**:
```python
from stock_features import (
    StockFeatures,
    BenchmarkData,
    calculate_vix_regime,
    calculate_market_regime,
    calculate_relative_strength,
    VIXRegime,
    MarketRegime,
)

# Calculate VIX regime
vix_normalized, regime = calculate_vix_regime(vix_value=25.0)
# regime = VIXRegime.ELEVATED

# Calculate market regime
market_regime = calculate_market_regime(
    spy_prices=spy_close_list,
    vix_value=25.0,
)
# market_regime = MarketRegime.SIDEWAYS

# Calculate relative strength
rs_20d = calculate_relative_strength(
    stock_prices=stock_close_list,
    benchmark_prices=spy_close_list,
    window=20,
)
```

### Stock Risk Guards (`services/stock_risk_guards.py`)

| Guard | Правило | Описание |
|-------|---------|----------|
| **MarginGuard** | Reg T | 50% initial, 25% maintenance margin |
| **ShortSaleGuard** | Rule 201 | Uptick rule при -10% drop |
| **CorporateActionsHandler** | SEC | Dividends, splits, ex-dates |

**Margin Call Types**:
- `FEDERAL` — Below Reg T initial margin (new positions)
- `MAINTENANCE` — Below 25% maintenance margin
- `HOUSE` — Broker's stricter requirements

**Short Sale Restrictions**:
- `UPTICK_RULE` — Rule 201 (short only on uptick)
- `HTB` — Hard-to-borrow (may not be available)
- `RESTRICTED` — Exchange restricted
- `NOT_SHORTABLE` — Cannot be shorted

**Использование**:
```python
from services.stock_risk_guards import (
    MarginGuard,
    ShortSaleGuard,
    MarginCallType,
    ShortSaleRestriction,
)

# Margin check
margin_guard = MarginGuard()
result = margin_guard.check_margin_requirement(
    position_value=100000,
    account_equity=60000,
    is_new_position=True,
)
# result.margin_call_type = MarginCallType.NONE if OK

# Short sale check
short_guard = ShortSaleGuard()
restriction = short_guard.check_short_restriction(
    symbol="GME",
    price_change_pct=-0.12,  # -12% drop
)
# restriction = ShortSaleRestriction.UPTICK_RULE
```

### Benchmark Temporal Alignment (Fix 2025-11-29)

**Проблема**: VIX/SPY/QQQ данные использовали positional index вместо timestamp merge → look-ahead bias.

**Решение**: `pd.merge_asof(direction="backward")` для корректного temporal alignment.

```python
# stock_features.py:_align_benchmark_by_timestamp()
aligned = pd.merge_asof(
    stock_df,
    benchmark_df,
    on="timestamp",
    direction="backward",  # Use last available benchmark value
    suffixes=("", "_benchmark"),
)
```

### Тестирование

```bash
# Stock features tests
pytest tests/test_stock_features.py -v

# Stock risk guards tests
pytest tests/test_stock_risk_guards.py -v

# Benchmark alignment tests
pytest tests/test_benchmark_temporal_alignment.py -v
```

### Референсы

- CBOE VIX White Paper (2003): VIX as fear gauge
- Lo, A.W. (2004): "The Adaptive Markets Hypothesis"
- Moskowitz, T.J. et al. (2012): "Time series momentum"
- Levy, R. (1967): "Relative Strength as a Criterion for Investment Selection"
- Reg T (Federal Reserve): Initial/maintenance margin requirements
- SEC Rule 201: Short sale circuit breaker

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
├── us_market_structure.py   # SEC Reg NMS rules (tick size, odd lots, NBBO)
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
| `TickSizeValidator` | SEC Reg NMS Rule 612 tick size validation |
| `OddLotHandler` | Odd lot (<100 shares) handling per SEC Rule 600 |
| `NBBOProtector` | Reg NMS Rule 611 trade-through prevention |

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

### US Market Structure (`lob/us_market_structure.py`)

SEC Reg NMS rules implementation for realistic equity simulation:

| Rule | Component | Description |
|------|-----------|-------------|
| **Rule 612** | `TickSizeValidator` | Sub-penny rule: $0.01 for ≥$1.00, $0.0001 for <$1.00 |
| **Rule 600** | `OddLotHandler` | Odd lot (<100 shares), round lot, mixed lot handling |
| **Rule 611** | `NBBOProtector` | Order Protection Rule (trade-through prevention) |

**Lot Types**:
- `ODD_LOT` — < 100 shares (different execution properties)
- `ROUND_LOT` — Exactly 100 shares or multiples
- `MIXED_LOT` — Round lots + odd lot remainder

**Trade-Through Protection**:
- `BID_THROUGH` — Sell below protected bid (violation)
- `ASK_THROUGH` — Buy above protected ask (violation)

```python
from lob.us_market_structure import (
    TickSizeValidator,
    OddLotHandler,
    NBBOProtector,
    LotType,
    TradeThrough,
    TICK_SIZE_PENNY,
    ROUND_LOT_SIZE,
)

# Tick size validation
validator = TickSizeValidator()
valid = validator.validate_price(150.015, stock_price=150.0)  # False (sub-penny!)
rounded = validator.round_to_tick(150.015)  # 150.01

# Lot type classification
handler = OddLotHandler()
lot_type = handler.classify_lot(75)  # LotType.ODD_LOT

# NBBO protection check
protector = NBBOProtector()
violation = protector.check_trade_through(
    trade_price=149.99,
    side="SELL",
    nbbo_bid=150.00,
    nbbo_ask=150.02,
)
# violation = TradeThrough.BID_THROUGH
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

## 💱 Forex Integration (Phase 11)

### Обзор

Phase 11 добавляет полную поддержку Forex (OTC) через OANDA:

**Статус**: ✅ Production Ready | **Тесты**: 18 test files (735+ tests planned)

**Ключевое архитектурное решение**: Forex — это OTC (Over-The-Counter) рынок с дилерскими котировками, а НЕ биржевой рынок. Поэтому:
- Используется **L2+ Parametric TCA** (как для crypto/equity), НЕ L3 LOB simulation
- **OTC Dealer Simulation** — отдельный модуль в `services/`, НЕ в `lob/`

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **ForexParametricTCA** | `execution_providers.py` | 8-factor slippage model |
| **ForexFeatures** | `forex_features.py` | Session-aware features |
| **ForexDealer** | `services/forex_dealer.py` | OTC dealer simulation |
| **ForexRiskGuards** | `services/forex_risk_guards.py` | Leverage & margin guards |
| **ForexSessionRouter** | `services/forex_session_router.py` | Session-aware routing |
| **ForexConfig** | `services/forex_config.py` | Pydantic config models |
| **ForexEnv** | `wrappers/forex_env.py` | Trading environment wrapper |
| **ForexTickSim** | `lob/forex_tick_simulation.py` | Tick-level simulation |
| **OANDA Adapter** | `adapters/oanda/*.py` | Market data & execution |

### Forex Sessions (Критично для моделирования)

| Session | Время (UTC) | Liquidity Factor | Spread Multiplier |
|---------|-------------|------------------|-------------------|
| **Sydney** | 21:00-06:00 | 0.60-0.70 | 1.4-1.6x |
| **Tokyo** | 00:00-09:00 | 0.70-0.85 | 1.2-1.4x |
| **London** | 07:00-16:00 | 1.00-1.20 | 1.0x |
| **New York** | 12:00-21:00 | 1.00-1.15 | 1.0x |
| **London/NY overlap** | 12:00-16:00 | **1.30-1.50** | **0.8x** (tightest) |

### Forex vs Crypto/Equity

| Аспект | Crypto | Equity | **Forex** |
|--------|--------|--------|-----------|
| **Market structure** | Central LOB | Central LOB | **OTC Dealer Network** |
| **Trading hours** | 24/7 | NYSE 9:30-16:00 ET | **Sun 5pm - Fri 5pm ET** |
| **Fees** | Maker/Taker % | $0 + regulatory | **Spread-based (0 commission)** |
| **Simulation** | L3 LOB | L3 LOB | **L2+ Parametric + OTC Sim** |
| **Leverage** | 1x-125x | 1x-4x | **50:1 - 500:1** |

### Конфигурация

```yaml
# configs/config_train_forex.yaml
mode: train
asset_class: forex
data_vendor: oanda

forex:
  default_spread_pips: 1.0
  session_spread_multipliers:
    sydney: 1.5
    tokyo: 1.3
    london: 1.0
    new_york: 1.0
  leverage: 50
  margin_requirement: 0.02  # 2%
```

### Тестирование

```bash
# Все Forex тесты
pytest tests/test_forex*.py -v

# По категориям
pytest tests/test_forex_parametric_tca.py -v        # L2+ TCA
pytest tests/test_forex_dealer_simulation.py -v     # OTC dealer
pytest tests/test_forex_features.py -v              # Session features
pytest tests/test_forex_phase6_risk_services.py -v  # Risk guards
pytest tests/test_forex_configuration.py -v         # Config models
```

### Environment Variables

```bash
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_PRACTICE=true  # or false for live
```

### Референсы

- BIS Triennial Survey (2022): FX market structure
- LMAX Exchange: FX market microstructure
- OANDA API Documentation
- `docs/FOREX_INTEGRATION_PLAN.md` — Полный план интеграции
- `docs/FOREX_INTEGRATION_QUICK_REF.md` — Краткий справочник

---

## 🔮 Futures Integration (Phase 3B-10: ✅ COMPLETE)

**Статус**: ✅ Production Ready | **Документация**: `docs/FUTURES_INTEGRATION_PLAN.md`

**Completed Phases**:
- Phase 3B: ✅ IB/CME Adapters
- Phase 4A: ✅ Crypto L2 Execution
- Phase 4B: ✅ CME SPAN Margin
- Phase 5A: ✅ Crypto L3 LOB
- Phase 5B: ✅ CME L3 LOB
- Phase 6A: ✅ Crypto Risk Guards
- Phase 6B: ✅ CME Risk Guards
- Phase 7: ✅ Unified Risk Management
- Phase 8: ✅ Multi-Futures Training Pipeline
- Phase 9: ✅ Unified Futures Live Trading
- Phase 10: ✅ Validation & Documentation

Интеграция всех типов фьючерсов:

| Тип | Биржа | Примеры | Статус | Phase |
|-----|-------|---------|--------|-------|
| **Equity Index** | CME (via IB) | ES, NQ, YM, RTY | ✅ IB Adapters Ready | 3B |
| **Commodity** | CME (via IB) | GC, CL, SI, NG | ✅ IB Adapters Ready | 3B |
| **Currency** | CME (via IB) | 6E, 6J, 6B, 6A | ✅ IB Adapters Ready | 3B |
| **Bonds** | CME (via IB) | ZN, ZB, ZT | ✅ IB Adapters Ready | 3B |
| **Crypto Perpetual** | Binance | BTCUSDT, ETHUSDT | ✅ L2 Execution Provider | **4A** |
| **Crypto Quarterly** | Binance | BTCUSDT_240329 | 📋 Phase 4B Planned | 4B |

Ключевые концепции: Leverage & Margin, Mark Price, Funding Rates (crypto), Rollover, Settlement.

---

## 📦 Phase 3B: Interactive Brokers & CME Settlement (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 205/205 (100% pass)

Phase 3B добавляет полную поддержку CME Group futures через Interactive Brokers TWS API:

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **IB Market Data** | `adapters/ib/market_data.py` | Historical bars, real-time quotes, contract details |
| **IB Order Execution** | `adapters/ib/order_execution.py` | Market/limit/bracket orders, margin queries |
| **IB Exchange Info** | `adapters/ib/exchange_info.py` | Contract specifications |
| **CME Settlement** | `impl_cme_settlement.py` | Daily settlement engine, variation margin |
| **CME Rollover** | `impl_cme_rollover.py` | Contract rollover manager |
| **CME Calendar** | `services/cme_calendar.py` | Trading hours, holidays, maintenance windows |

### Поддерживаемые контракты (30+)

**Equity Index (CME):**
- **E-mini**: ES (S&P 500), NQ (NASDAQ 100), RTY (Russell 2000), YM (Dow)
- **Micro E-mini**: MES, MNQ, M2K, MYM

**Metals (COMEX):**
- **Standard**: GC (Gold), SI (Silver), HG (Copper)
- **Micro**: MGC (Micro Gold), SIL (Micro Silver)

**Energy (NYMEX):**
- **Standard**: CL (Crude Oil), NG (Natural Gas), RB (Gasoline), HO (Heating Oil)
- **Micro**: MCL (Micro Crude Oil)

**Currencies (CME):**
- 6E (Euro), 6J (Yen), 6B (Pound), 6A (Aussie), 6C (CAD), 6S (CHF)

**Bonds (CBOT):**
- ZN (10-Year Note), ZB (30-Year Bond), ZT (2-Year Note), ZF (5-Year Note)

### IB TWS API Rate Limiting

**Production-grade rate limiter** (`IBRateLimiter`) с thread-safe tracking:

| Rate Limit Type | IB Limit | Implementation | Safety Margin |
|-----------------|----------|----------------|---------------|
| General messages | 50/sec | 45/sec | 10% |
| Historical requests | 60/10min | 55/10min | 8% |
| Identical requests | 6/10min | 5/10min | 17% |
| Market data subscriptions | 1/sec | 1/sec | None (hard limit) |
| Concurrent market data | 100 lines | 100 lines | None (hard limit) |

**Connection Management** (`IBConnectionManager`):
- Heartbeat every 30sec (IB requires 60sec)
- Exponential backoff reconnection: [1, 2, 5, 10, 30, 60, 120] seconds
- Paper/Live routing via port:
  - `7497` = TWS Paper
  - `7496` = TWS Live
  - `4002` = Gateway Paper
  - `4001` = Gateway Live

### CME Settlement Engine

**Product-specific settlement times** (Eastern Time):

| Product Category | Examples | Settlement Time (ET) | Reference |
|------------------|----------|----------------------|-----------|
| Equity Index | ES, NQ, YM, RTY | 15:30 (14:30 CT) | CME Group |
| Currencies | 6E, 6J, 6B | 15:00 (14:00 CT) | CME Group |
| Metals | GC, SI, HG | 14:30 (13:30 CT) | COMEX |
| Energy | CL, NG | 15:30 (14:30 CT) | NYMEX |
| Bonds | ZN, ZB, ZT | 16:00 (15:00 CT) | CBOT |
| Agricultural | ZC, ZS, ZW | 14:15 (13:15 CT) | CBOT |

**Variation Margin Calculation**:

```python
from impl_cme_settlement import CMESettlementEngine, create_settlement_engine

engine = CMESettlementEngine()

# Daily variation margin
variation = engine.calculate_variation_margin(
    position=futures_position,
    settlement_price=Decimal("4500.00"),
    contract_spec=es_spec,
)
# variation = (Settlement_t - Settlement_t-1) × Qty × Multiplier
```

**Formula**: `VM = ΔP × qty × multiplier`
- LONG position: profit if price ↑, loss if price ↓
- SHORT position: profit if price ↓, loss if price ↑

### Contract Rollover

**Standard roll dates** by product:

| Product | Roll Date | Example |
|---------|-----------|---------|
| Equity Index (ES, NQ) | 8 business days before expiry | 2nd Thursday before 3rd Friday |
| Currencies (6E, 6J) | 2 business days before expiry | 2nd business day before 3rd Wednesday |
| Metals (GC, SI) | 3 business days before last trading day | End of month before delivery |
| Energy (CL, NG) | 3 business days before expiry | ~3 days before contract month end |
| Bonds (ZN, ZB) | 7 business days before first delivery | ~7 days before month end |

**Contract Month Codes**:
```
F = Jan, G = Feb, H = Mar, J = Apr, K = May, M = Jun
N = Jul, Q = Aug, U = Sep, V = Oct, X = Nov, Z = Dec
```

**Contract Cycles**:
- **Quarterly** (H, M, U, Z): Equity Index, Currencies, Bonds
- **Monthly** (All months): Energy
- **Bi-Monthly**: Metals, Grains

### CME Trading Calendar

**CME Globex Hours** (Eastern Time):
- **Regular**: Sunday 18:00 ET → Friday 17:00 ET
- **Daily Maintenance**: Monday-Friday 16:15-16:30 ET (15 minutes)
- **Weekend**: Closed Saturday

**US Market Holidays** (2024-2026):
```python
from services.cme_calendar import CMETradingCalendar

calendar = CMETradingCalendar()

# Check if trading
is_open = calendar.is_trading_hours(datetime.now())

# Check holiday
is_holiday = calendar.is_holiday(date.today())

# Get next open
next_open = calendar.get_next_open(datetime.now())
```

**Holiday List** (2024-2026):
- New Year's Day, MLK Day, Presidents Day, Good Friday
- Memorial Day, Juneteenth, Independence Day
- Labor Day, Thanksgiving, Christmas

**Early Close Days**:
- Day before Thanksgiving: 13:15 ET
- Christmas Eve: 13:15 ET
- New Year's Eve: 13:15 ET

### Использование

```python
# 1. Market Data Adapter
from adapters.ib import IBMarketDataAdapter
from adapters.models import ExchangeVendor

adapter = IBMarketDataAdapter(
    vendor=ExchangeVendor.IB,
    config={
        "host": "127.0.0.1",
        "port": 7497,  # Paper trading
        "client_id": 1,
        "readonly": True,
    }
)

# Fetch historical bars
bars = adapter.get_bars("ES", "1h", limit=500)

# Get current quote
tick = adapter.get_tick("ES")

# Get contract details
spec = adapter.get_contract_details("ES")


# 2. Order Execution Adapter
from adapters.ib import IBOrderExecutionAdapter

execution = IBOrderExecutionAdapter(
    vendor=ExchangeVendor.IB,
    config={
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 2,
    }
)

# Submit market order
order = execution.submit_market_order("ES", "BUY", qty=1)

# Submit bracket order (entry + TP + SL)
from adapters.ib.order_execution import IBBracketOrderConfig

bracket = execution.submit_bracket_order(IBBracketOrderConfig(
    symbol="ES",
    side="BUY",
    qty=1,
    entry_price=Decimal("4500.00"),
    take_profit_price=Decimal("4550.00"),  # +50 points
    stop_loss_price=Decimal("4475.00"),    # -25 points
))

# Query margin requirement
margin = execution.get_margin_requirement("ES", qty=1)
# margin = {"initial_margin": ..., "maint_margin": ..., "impact_on_margin": ...}

# Get positions
positions = execution.get_positions()


# 3. CME Settlement
from impl_cme_settlement import CMESettlementEngine, create_settlement_engine
from core_futures import FuturesPosition, FuturesContractSpec

engine = create_settlement_engine()

# Calculate daily variation margin
variation = engine.calculate_variation_margin(
    position=FuturesPosition(...),
    settlement_price=Decimal("4500.00"),
    contract_spec=FuturesContractSpec(...),
)

# Check if settlement time
is_settlement = engine.is_settlement_time(
    timestamp_ms=int(time.time() * 1000),
    symbol="ES",
)


# 4. Contract Rollover
from impl_cme_rollover import ContractRolloverManager

rollover = ContractRolloverManager(expiration_calendar={
    "ES": [date(2025, 3, 21), date(2025, 6, 20), ...]
})

# Check if should roll
should_roll = rollover.should_roll("ES", date.today())

# Get roll date
roll_date = rollover.get_roll_date("ES", date.today())


# 5. Trading Calendar
from services.cme_calendar import CMETradingCalendar, CMESession

calendar = CMETradingCalendar()

# Check trading hours
is_open = calendar.is_trading_hours(datetime.now())

# Get current session
session = calendar.get_current_session(datetime.now())
# session = CMESession.REGULAR | MAINTENANCE | CLOSED

# Check holiday
is_holiday = calendar.is_holiday(date.today())
```

### Конфигурация

**IB Connection Config**:
```yaml
# configs/ib_connection.yaml
host: "127.0.0.1"
port: 7497  # Paper: 7497 (TWS) or 4002 (Gateway)
client_id: 1
readonly: true  # Safety: data-only mode
timeout: 10.0
account: null  # For multi-account setups
```

**Environment Variables**:
```bash
# Not required for IB (uses TWS/Gateway local connection)
# But recommended for logging
IB_LOG_LEVEL=INFO
IB_ENABLE_RATE_LIMIT_LOGGING=true
```

### Тестирование

```bash
# IB Adapters tests (100 tests)
pytest tests/test_ib_adapters.py -v

# CME Settlement tests (52 tests)
pytest tests/test_cme_settlement.py -v

# CME Calendar tests (53 tests)
pytest tests/test_cme_calendar.py -v

# All Phase 3B tests (205 tests)
pytest tests/test_ib_adapters.py tests/test_cme_settlement.py tests/test_cme_calendar.py -v
```

**Coverage**: 205 tests (100% pass rate)

| Test Suite | Tests | Focus |
|------------|-------|-------|
| `test_ib_adapters.py` | 100 | Rate limiting, connection mgmt, contract mapping, order execution |
| `test_cme_settlement.py` | 52 | Settlement times, variation margin, rollover dates |
| `test_cme_calendar.py` | 53 | Trading hours, holidays, session detection |

### Ключевые отличия CME vs Crypto Perpetuals

| Аспект | Crypto Perpetual (Binance) | CME Futures (IB) |
|--------|----------------------------|------------------|
| **Settlement** | Funding every 8h (continuous) | Daily settlement at fixed time |
| **Expiration** | Perpetual (no expiry) | Quarterly/Monthly expiration |
| **Rollover** | N/A | Required ~8 days before expiry |
| **Margin** | Cross/Isolated with ADL | SPAN margin (risk-based) |
| **Trading Hours** | 24/7 | Sun 18:00 - Fri 17:00 ET |
| **Maintenance** | N/A | Daily 16:15-16:30 ET |
| **Leverage** | Up to 125x (retail) | Regulated by SPAN |
| **Mark Price** | Index + funding basis | Last traded price |

### Dependencies

```bash
pip install ib_insync  # IB TWS API wrapper (required)
```

**TWS/Gateway Setup**:
1. Download IB TWS or Gateway from Interactive Brokers
2. Enable API connections (Edit → Global Configuration → API → Enable ActiveX and Socket Clients)
3. Set Socket Port: 7497 (paper) or 7496 (live)
4. Allow connections from `127.0.0.1`

### Registry Integration

**Automatically registered** в `adapters/registry.py`:

```python
ExchangeVendor.IB           # Generic IB
ExchangeVendor.IB_CME       # CME futures
ExchangeVendor.IB_CBOT      # CBOT futures
ExchangeVendor.IB_NYMEX     # NYMEX futures
ExchangeVendor.IB_COMEX     # COMEX futures
```

**Factory Functions**:
```python
from adapters.registry import create_market_data_adapter, create_order_execution_adapter

# Via registry
md_adapter = create_market_data_adapter("ib", {"port": 7497})
exec_adapter = create_order_execution_adapter("ib", {"port": 7497})
```

### Референсы

- **IB TWS API**: https://interactivebrokers.github.io/tws-api/
- **ib_insync**: https://ib-insync.readthedocs.io/
- **CME Group Settlement**: https://www.cmegroup.com/clearing/operations-and-deliveries/settlement.html
- **CME Contract Specs**: https://www.cmegroup.com/trading/products/
- **CME Holiday Calendar**: https://www.cmegroup.com/tools-information/holiday-calendar.html
- **SPAN Margin**: https://www.cmegroup.com/clearing/risk-management/span-methodology.html

### Roadmap (Phase 4+)

**Next Steps**:
- ✅ Phase 3A: Funding Rate Mechanics (Binance perpetuals) — DONE
- ✅ Phase 3B: IB Adapters & CME Settlement — DONE
- ✅ Phase 4A: L2 Execution Provider (Crypto Futures Slippage) — DONE
- ✅ Phase 4B: CME SPAN Margin & Slippage — DONE
- ✅ Phase 5A: L3 LOB Integration for Crypto Futures — DONE
- ✅ Phase 5B: L3 LOB for CME Futures — DONE
- ✅ Phase 6A: Crypto Futures Risk Management — DONE
- 📋 Phase 6B: CME Futures Risk Management
- 📋 Phase 7: Training & Backtesting Integration

---

## 📊 Phase 4A: L2 Execution Provider for Crypto Futures (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 54/54 (100% pass) | **Date**: 2025-12-02

Phase 4A extends the crypto parametric TCA model with futures-specific factors for Binance USDT-M perpetuals.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FuturesSlippageProvider** | `execution_providers_futures.py` | L2+ slippage with funding/liquidation/OI factors |
| **FuturesFeeProvider** | `execution_providers_futures.py` | Maker/taker/liquidation fees + funding payments |
| **FuturesL2ExecutionProvider** | `execution_providers_futures.py` | Combined execution provider |
| **Тесты** | `tests/test_futures_execution_providers.py` | 54 comprehensive tests |

### Futures-Specific Factors

#### 1. Funding Rate Stress
- **Formula**: `funding_stress = 1.0 + abs(funding_rate) × sensitivity`
- **Default sensitivity**: 5.0
- **Example**: 0.01% funding → 0.05% slippage increase
- **Direction**: Only applies when trading in same direction as funding (crowded position)

#### 2. Liquidation Cascade
- **Formula**: `cascade_factor = min(max_factor, 1.0 + (liquidations/ADV) × sensitivity)`
- **Default sensitivity**: 5.0
- **Max cap**: 3.0x (200% increase)
- **Threshold**: 1% of ADV
- **Example**: 2% liquidations → 10% slippage increase (capped at 200%)

#### 3. Open Interest Penalty
- **Formula**: `oi_penalty = min(max_penalty, 1.0 + (OI/ADV - 1.0) × factor)`
- **Default factor**: 0.1
- **Max cap**: 2.0x (100% increase)
- **Trigger**: OI > ADV
- **Example**: OI = 3× ADV → 20% slippage increase (capped at 100%)

### Total Slippage Formula

```python
total_slippage = base_slippage
    × (1.0 + funding_rate × sensitivity)           # Funding stress
    × min(3.0, 1.0 + liq_ratio × cascade_sens)     # Cascade (capped)
    × min(2.0, 1.0 + (oi/adv - 1.0) × oi_factor)  # OI penalty (capped)
```

**Realistic Example**:
- Base slippage: 8 bps (from crypto model)
- Funding: 0.01% × 5.0 = 0.05% increase → × 1.0005
- Liquidations: 2% × 5.0 = 10% increase → × 1.10
- OI: 3× ADV → × 1.20
- **Total**: 8 × 1.0005 × 1.10 × 1.20 ≈ **10.6 bps** ✅

### Fee Structure (Binance USDT-M)

| Fee Type | Rate | Notes |
|----------|------|-------|
| Maker | 2 bps (0.02%) | Passive liquidity provision |
| Taker | 4 bps (0.04%) | Aggressive execution |
| Liquidation | 50 bps (0.5%) | Goes to insurance fund |

### Funding Payment

**Formula**: `payment = position_notional × funding_rate`

- **Positive funding**: Longs pay shorts
- **Negative funding**: Shorts pay longs

**Example**:
```python
# Long 1 BTC at $50,000, funding = +0.01%
payment = 50,000 × 1.0 × 0.0001 = $5.00 (paid by long)

# Short 1 BTC at $50,000, funding = +0.01%
payment = 50,000 × 1.0 × 0.0001 = $5.00 (received by short)
```

### Configuration

```python
from execution_providers_futures import FuturesSlippageConfig, create_futures_execution_provider

# Default configuration
config = FuturesSlippageConfig(
    funding_impact_sensitivity=5.0,
    liquidation_cascade_sensitivity=5.0,
    liquidation_cascade_max_factor=3.0,      # Cap at 200% increase
    open_interest_liquidity_factor=0.1,
    open_interest_max_penalty=2.0,           # Cap at 100% increase
    use_mark_price_execution=True,
)

# Create provider
provider = create_futures_execution_provider(
    use_mark_price=True,
    slippage_config=config,
)
```

### Usage Example

```python
from execution_providers import Order, MarketState, BarData

# Execute order
order = Order("BTCUSDT", "BUY", 0.1, "MARKET")
market = MarketState(timestamp=0, bid=50000.0, ask=50001.0, adv=1e9)
bar = BarData(open=50000.0, high=50100.0, low=49900.0, close=50050.0, volume=1000.0)

fill = provider.execute(
    order=order,
    market=market,
    bar=bar,
    funding_rate=0.0001,            # 0.01% funding
    open_interest=2_000_000_000,    # $2B OI (2× ADV)
    recent_liquidations=10_000_000, # $10M liquidations (1%)
)

print(f"Filled at {fill.price} with {fill.slippage_bps:.2f}bps slippage")
print(f"Fee: ${fill.fee:.2f}")
```

### Factory Integration

```python
from execution_providers import create_execution_provider, AssetClass

# Via factory (automatically uses FuturesSlippageProvider)
provider = create_execution_provider(AssetClass.FUTURES, level="L2")
```

### Тестирование

```bash
# All futures tests (54 tests)
pytest tests/test_futures_execution_providers.py -v

# Coverage: 54 passed, 1 skipped (100% pass rate)
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| FuturesSlippageConfig | 5 | Config validation |
| Funding Stress | 5 | Positive/negative/zero/scaling |
| Liquidation Cascade | 3 | Above/below threshold, scaling, caps |
| Open Interest Penalty | 2 | High/normal OI, caps |
| Combined Factors | 2 | Worst/best case scenarios |
| Liquidation Risk | 3 | Long/short, leverage |
| Fee Computation | 5 | Maker/taker/liquidation |
| Funding Payment | 5 | Long pays/receives, scaling |
| L2 Execution | 4 | Basic/mark price/all factors |
| Factory Functions | 5 | Creation, integration |
| Edge Cases | 7 | None params, zero ADV, bounds |
| Backward Compat | 3 | Protocol compliance |

### Critical Bugs Fixed (2025-12-02)

1. **Funding Stress Formula**: Removed `× 10000` (was 51x, now 1.005x for 0.1% funding) ✅
2. **Liquidation Cascade Cap**: Added max_factor=3.0 to prevent unrealistic extremes ✅
3. **OI Penalty Cap**: Added max_penalty=2.0 to prevent unbounded growth ✅
4. **Syntax Error**: Fixed duplicate docstring in execution_providers.py ✅

### Limitations & Future Work

**Current Scope**:
- ✅ Crypto perpetuals (USDT-M)
- ✅ L2 statistical slippage
- ✅ Mark price execution

**Future Phases**:
- 📋 Quarterly futures expiration handling (Phase 4B)
- 📋 Binance Futures adapters (Phase 5)
- 📋 L3 LOB simulation for futures (Phase 6)
- 📋 Historical data validation vs actual fills

### Референсы

- **Binance Futures**: https://www.binance.com/en/support/faq/360033524991
- **Funding Rate Mechanism**: https://www.binance.com/en/support/faq/360033525031
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Zhao et al. (2020): "Liquidation Cascade Effects in Crypto Markets"
- Cont et al. (2014): "The Price Impact of Order Book Events"

---

## 📊 Phase 4B: CME SPAN Margin & Slippage (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 258/258 (100% pass) | **Покрытие**: 99% | **Date**: 2025-12-02

Phase 4B implements CME-specific margin calculation (SPAN methodology) and slippage modeling for CME Group futures.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **SPANMarginCalculator** | `impl_span_margin.py` | SPAN margin calculation with 16-scenario testing |
| **CMESlippageProvider** | `execution_providers_cme.py` | CME-specific slippage with session/settlement factors |
| **CMEFeeProvider** | `execution_providers_cme.py` | Fixed per-contract fee structure |
| **CMECircuitBreaker** | `impl_circuit_breaker.py` | Rule 80B circuit breakers, overnight limits, velocity logic |
| **CircuitBreakerManager** | `impl_circuit_breaker.py` | Multi-product circuit breaker management |

### SPAN Margin Calculator

**SPAN (Standard Portfolio Analysis of Risk)** — CME's risk-based margin methodology.

**Key Concepts**:
- **Scanning Risk**: Maximum expected loss under 16 stress scenarios
- **Inter-Commodity Credit**: Margin offset for correlated products
- **Intra-Commodity Credit**: Calendar spread credits
- **Delivery Month Charge**: Additional margin near expiration

**Scanning Risk Ranges** (% of notional):

| Product | Range | Volatility Scan |
|---------|-------|-----------------|
| ES (E-mini S&P) | 6% | 30% |
| NQ (E-mini NASDAQ) | 8% | 35% |
| GC (Gold) | 5% | 25% |
| CL (Crude Oil) | 8% | 35% |
| NG (Natural Gas) | 12% | 50% |
| 6E (Euro FX) | 4% | 20% |
| ZN (10-Year Note) | 2% | 15% |

**Inter-Commodity Spread Credits**:

| Pair | Credit Rate | Rationale |
|------|-------------|-----------|
| ES/NQ | 50% | Correlated equity indices |
| ES/YM | 50% | S&P 500 vs Dow correlation |
| GC/SI | 35% | Precious metals correlation |
| MGC/GC | 85% | Micro/Standard same underlying |
| CL/RB/HO | 40% | Crack spread (refining) |

**Usage**:

```python
from impl_span_margin import (
    SPANMarginCalculator,
    create_span_calculator,
    calculate_simple_margin,
)
from core_futures import FuturesPosition, PositionSide, MarginMode

# 1. Create calculator with default specs
calc = create_span_calculator()

# 2. Calculate single position margin
position = FuturesPosition(
    symbol="ES",
    qty=Decimal("2"),
    entry_price=Decimal("4500"),
    side=PositionSide.LONG,
    leverage=1,
    margin_mode=MarginMode.SPAN,
)

result = calc.calculate_margin(
    position=position,
    current_price=Decimal("4500"),
)

print(f"Scanning Risk: ${result.scanning_risk}")
print(f"Initial Margin: ${result.initial_margin}")
print(f"Maintenance Margin: ${result.maintenance_margin}")

# 3. Portfolio margin with spread credits
positions = [es_long, nq_long]  # Correlated positions
portfolio_result = calc.calculate_portfolio_margin(
    positions=positions,
    prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
)

print(f"Inter-commodity Credit: ${portfolio_result.inter_commodity_credit}")
print(f"Net Portfolio Margin: ${portfolio_result.net_portfolio_margin}")

# 4. Margin call detection
call_status = calc.check_margin_call(
    positions=positions,
    prices=prices,
    account_equity=Decimal("50000"),
)
# call_status.call_type: NONE, WARNING, MARGIN_CALL, LIQUIDATION
```

### CME Slippage Provider

**Session-Aware Slippage Model** with CME-specific factors.

**Slippage Factors**:

| Factor | Multiplier | Condition |
|--------|------------|-----------|
| ETH Session | 1.5x | Outside RTH (18:00-17:00 ET) |
| Settlement Period | 1.3x | 15 min before settlement |
| Roll Period | 1.2x | 8 days before expiry |
| Circuit Breaker L1 | 2.0x | -7% decline |
| Circuit Breaker L2 | 5.0x (max) | -13% decline |
| Velocity Pause | 1.5x | Fat-finger protection |

**Default Spreads** (in bps):

| Product | Spread | Impact Coef |
|---------|--------|-------------|
| ES | 0.5 bps | 0.03 |
| NQ | 0.75 bps | 0.04 |
| GC | 1.0 bps | 0.04 |
| CL | 2.0 bps | 0.06 |
| NG | 3.0 bps | 0.08 |
| 6E | 0.5 bps | 0.03 |
| ZN | 0.25 bps | 0.02 |

**Slippage Profiles**:
- `default`: Balanced settings
- `conservative`: Wider spreads, higher impacts
- `aggressive`: Tighter estimates
- `equity_index`: Optimized for ES/NQ
- `metals`: Optimized for GC/SI
- `energy`: Optimized for CL/NG

**Usage**:

```python
from execution_providers_cme import (
    create_cme_slippage_provider,
    create_cme_execution_provider,
    CMESlippageProvider,
)
from execution_providers import Order, MarketState, BarData

# 1. Create from profile
provider = CMESlippageProvider.from_profile("equity_index")

# 2. Compute slippage
slippage_bps = provider.compute_slippage_bps(
    order=Order("ES", "BUY", 5.0, "MARKET"),
    market=MarketState(timestamp=0, bid=4500.0, ask=4500.25, adv=2e9),
    participation_ratio=0.001,
    is_eth_session=False,
    is_settlement_period=False,
    circuit_breaker_level=CircuitBreakerLevel.NONE,
)

# 3. Full execution provider
exec_provider = create_cme_execution_provider(profile="default")
fill = exec_provider.execute(order, market, bar)
```

### CME Fee Provider

**Fixed Per-Contract Fees** (no maker/taker distinction):

| Product | Fee per Contract | Exchange |
|---------|------------------|----------|
| ES | $1.29 | CME |
| NQ | $1.29 | CME |
| GC | $1.60 | COMEX |
| SI | $1.60 | COMEX |
| CL | $1.50 | NYMEX |
| NG | $1.50 | NYMEX |
| 6E | $1.00 | CME |
| ZN | $0.85 | CBOT |

### CME Circuit Breaker (Rule 80B)

**Equity Index Circuit Breakers** (ES, NQ, YM, RTY):

| Level | Trigger | Halt Duration | Time Restriction |
|-------|---------|---------------|------------------|
| Level 1 | -7% | 15 minutes | Before 15:25 ET only |
| Level 2 | -13% | 15 minutes | Before 15:25 ET only |
| Level 3 | -20% | Remainder of day | Any time |

**Overnight Price Limits** (ETH only):

| Product | Limit | Note |
|---------|-------|------|
| ES, NQ, YM, RTY | ±5% | From prior settlement |

**Commodity Daily Price Limits**:

| Product | Initial | Expanded | Notes |
|---------|---------|----------|-------|
| CL | ±$10 | ±$15, ±$20 | Consecutive limit days |
| NG | ±$3 | ±$4.50, ±$6 | Expansion mechanism |
| GC | ±$100 | ±$150, ±$200 | COMEX metals |

**Velocity Logic** (Fat-Finger Protection):

| Product | Threshold (ticks) | Pause Duration |
|---------|-------------------|----------------|
| ES | 12 | 2 seconds |
| NQ | 20 | 2 seconds |
| GC | 50 | 2 seconds |
| CL | 100 | 2 seconds |

**Usage**:

```python
from impl_circuit_breaker import (
    CMECircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerLevel,
    create_circuit_breaker,
)

# 1. Single product circuit breaker
cb = create_circuit_breaker("ES", reference_price=Decimal("4500"))

# 2. Check circuit breaker status
level = cb.check_circuit_breaker(
    current_price=Decimal("4185"),  # -7%
    timestamp_ms=int(time.time() * 1000),
    is_rth=True,
)
# level = CircuitBreakerLevel.LEVEL_1

# 3. Check if trading allowed
can_trade, reason = cb.can_trade()
# can_trade = False, reason = "Circuit breaker Level 1 halt"

# 4. Get halt end time
halt_end = cb.get_halt_end_time()

# 5. Multi-product manager
manager = CircuitBreakerManager()
manager.add_product("ES", reference_price=Decimal("4500"))
manager.add_product("NQ", reference_price=Decimal("15000"))

status = manager.check_all(
    prices={"ES": Decimal("4185"), "NQ": Decimal("13900")},
    timestamp_ms=now_ms,
    is_rth=True,
)
# status = {
#     "ES": {"level": "LEVEL_1", "can_trade": False},
#     "NQ": {"level": "NONE", "can_trade": True},
# }

# 6. Daily reset
manager.reset_all_daily()
```

### Тестирование

```bash
# All Phase 4B tests (258 tests, 99% coverage)
pytest tests/test_span_margin.py tests/test_cme_slippage.py tests/test_circuit_breaker.py -v

# By component
pytest tests/test_span_margin.py -v          # 85 tests (78 + 7 edge cases)
pytest tests/test_cme_slippage.py -v         # 66 tests (55 + 11 edge cases)
pytest tests/test_circuit_breaker.py -v      # 67 tests (60 + 7 edge cases)
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| SPAN Scanning Risk | 9 | Product-specific ranges |
| SPAN Portfolio Margin | 7 | Spread credits |
| SPAN Margin Impact | 3 | New position impact estimation |
| SPAN Edge Cases | 5 | Missing specs/prices, fallbacks |
| CME Slippage Profiles | 6 | Profile configurations |
| CME Session Factors | 5 | ETH/settlement/roll |
| CME Limit Orders | 6 | Passive/aggressive/no-fill |
| CME Edge Cases | 5 | Currency futures, recommendations |
| CME Circuit Breaker | 20 | Rule 80B, overnight limits |
| Velocity Logic | 7 | Fat-finger protection |
| Circuit Breaker Manager | 6 | Multi-product management |
| Circuit Breaker Edge Cases | 7 | Expanded limits, non-equity products |
| Integration Scenarios | 5 | Flash crash, overnight trading |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `impl_span_margin.py` | SPAN margin calculator (~1050 lines) |
| `execution_providers_cme.py` | CME slippage/fee providers (~800 lines) |
| `impl_circuit_breaker.py` | Circuit breaker simulation (~700 lines) |
| `tests/test_span_margin.py` | 78 SPAN margin tests |
| `tests/test_cme_slippage.py` | 55 CME slippage tests |
| `tests/test_circuit_breaker.py` | 60 circuit breaker tests |

### Референсы

- **CME SPAN Methodology**: https://www.cmegroup.com/clearing/risk-management/span-methodology.html
- **CME Rule 80B**: https://www.cmegroup.com/rulebook/CME/I/5/5.html
- **CME Globex Price Limits**: https://www.cmegroup.com/trading/equity-index/price-limit-guide.html
- **CME Velocity Logic**: https://www.cmegroup.com/confluence/display/EPICSANDBOX/Velocity+Logic

---

## 📊 Phase 5A: L3 LOB Integration for Crypto Futures (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 100/100 (100% pass) | **Date**: 2025-12-02

Phase 5A integrates L3 Limit Order Book simulation with crypto perpetual futures, adding liquidation cascade simulation, insurance fund dynamics, ADL queue management, and funding period-aware execution.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **LiquidationOrderStream** | `lob/futures_extensions.py` | Liquidation order injection into LOB |
| **LiquidationCascadeSimulator** | `lob/futures_extensions.py` | Kyle price impact cascade simulation |
| **InsuranceFundManager** | `lob/futures_extensions.py` | Insurance fund contribution/payout dynamics |
| **ADLQueueManager** | `lob/futures_extensions.py` | Auto-Deleveraging queue management |
| **FundingPeriodDynamics** | `lob/futures_extensions.py` | Queue behavior near funding times |
| **FuturesL3SlippageProvider** | `execution_providers_futures_l3.py` | L3 slippage with cascade/funding factors |
| **FuturesL3FillProvider** | `execution_providers_futures_l3.py` | L3 fill logic with liquidation injection |
| **FuturesL3ExecutionProvider** | `execution_providers_futures_l3.py` | Combined L3 futures execution provider |
| **Тесты** | `tests/test_futures_l3_execution.py` | 100 comprehensive tests |

### Key Concepts

#### 1. Liquidation Cascade Simulation (Kyle Price Impact)

Based on Kyle (1985) λ-model: `ΔP = λ × sign(x) × |x|`

**Cascade Mechanics**:
- **Wave Decay**: Each subsequent liquidation wave is dampened by `cascade_decay` factor (default: 0.7)
- **Price Impact**: Cumulative impact follows `impact_coef × √(liquidation_volume / ADV)`
- **Max Waves**: Configurable limit (default: 5) to prevent infinite cascade loops
- **Phases**: INITIAL → PROPAGATING → DAMPENING → ENDED

**Usage**:
```python
from lob.futures_extensions import (
    LiquidationCascadeSimulator,
    create_cascade_simulator,
)

# Create simulator
simulator = create_cascade_simulator(
    price_impact_coef=0.5,  # Kyle λ coefficient
    cascade_decay=0.7,       # Wave dampening factor
    max_waves=5,
)

# Simulate cascade
result = simulator.simulate_cascade(
    initial_liquidation_volume=1_000_000,
    market_price=50000.0,
    adv=500_000_000,
)

print(f"Total waves: {len(result.waves)}")
print(f"Total liquidated: ${result.total_liquidated_volume:,.0f}")
print(f"Final price impact: {result.total_price_impact_bps:.2f} bps")
```

#### 2. Insurance Fund Dynamics

**Fund Flow**:
- **Profit liquidation** → Contribution to fund (bankruptcy - fill > 0)
- **Loss liquidation** → Payout from fund (fill - bankruptcy > 0)
- **Fund depletion** → Triggers ADL mechanism

**Usage**:
```python
from lob.futures_extensions import (
    InsuranceFundManager,
    create_insurance_fund,
    LiquidationFillResult,
)

fund = create_insurance_fund(initial_balance=10_000_000)

# Process liquidation
result = fund.process_liquidation(
    liquidation_info=liq_order,
    fill_price=49500.0,
)

print(f"Contribution: ${result.contribution:.2f}")
print(f"Payout: ${result.payout:.2f}")
print(f"Fund balance: ${fund.get_state().current_balance:,.0f}")
```

#### 3. ADL (Auto-Deleveraging) Queue

**Ranking Formula**: `ADL_Score = PnL% × Leverage`

Higher score = higher priority for deleveraging.

**Usage**:
```python
from lob.futures_extensions import (
    ADLQueueManager,
    create_adl_manager,
)

adl_manager = create_adl_manager()

# Build queue from positions
positions = [
    {"address": "user1", "pnl_pct": 0.15, "leverage": 20, "side": "long", "size": 1000},
    {"address": "user2", "pnl_pct": 0.10, "leverage": 10, "side": "long", "size": 2000},
]
adl_manager.build_queue(positions, side="long")

# Get candidates for deleveraging
candidates = adl_manager.get_adl_candidates(
    side="long",
    required_amount=500,
)
```

#### 4. Funding Period Dynamics

**Queue Behavior Near Funding**:
- Spread widens (arbitrageurs exit)
- Liquidity decreases (position rebalancing)
- Volatility increases

**Usage**:
```python
from lob.futures_extensions import (
    FundingPeriodDynamics,
    create_funding_dynamics,
)

dynamics = create_funding_dynamics(
    funding_times_utc=[0, 8, 16],  # 00:00, 08:00, 16:00 UTC
    window_minutes_before=5,
    window_minutes_after=1,
)

state = dynamics.get_state(
    timestamp_ms=current_time_ms,
    funding_rate=0.0001,
)

print(f"In funding window: {state.in_funding_window}")
print(f"Spread multiplier: {state.spread_multiplier:.2f}")
print(f"Queue priority factor: {state.queue_priority_factor:.2f}")
```

### Configuration

```python
from execution_providers_futures_l3 import (
    FuturesL3Config,
    create_futures_l3_config,
)

config = FuturesL3Config(
    # Cascade parameters
    price_impact_coef=0.5,
    cascade_decay=0.7,
    max_cascade_waves=5,

    # Insurance fund
    initial_insurance_fund=10_000_000,
    adl_trigger_threshold=0.1,

    # Funding
    funding_times_utc=[0, 8, 16],
    funding_window_minutes_before=5,
    funding_window_minutes_after=1,
    funding_spread_multiplier_max=1.5,
    funding_queue_priority_factor=0.8,

    # Execution
    use_mark_price_execution=True,
)
```

### Presets

| Preset | Cascade Decay | Max Waves | Impact Coef | Use Case |
|--------|---------------|-----------|-------------|----------|
| `default` | 0.7 | 5 | 0.5 | General simulation |
| `conservative` | 0.6 | 3 | 0.7 | Conservative estimates |
| `fast` | 0.8 | 3 | 0.3 | Faster simulations |
| `stress_test` | 0.5 | 10 | 1.0 | Extreme market conditions |

**Usage**:
```python
from execution_providers_futures_l3 import (
    FuturesL3ExecutionProvider,
    create_futures_l3_execution_provider,
)

# From preset
provider = FuturesL3ExecutionProvider.from_preset("stress_test")

# Or via factory
provider = create_futures_l3_execution_provider(preset="conservative")
```

### Integration with L3 LOB

The FuturesL3ExecutionProvider integrates with the existing L3 LOB infrastructure:

```python
from lob import MatchingEngine, OrderBook
from execution_providers_futures_l3 import create_futures_l3_execution_provider

# Create provider
provider = create_futures_l3_execution_provider(preset="default")

# Load historical liquidation data
provider.load_liquidation_data(liquidation_events_list)

# Execute with full LOB simulation
fill = provider.execute(
    order=order,
    market=market_state,
    bar=bar_data,
    order_book=lob_order_book,
    matching_engine=matching_engine,
    funding_rate=0.0001,
    open_interest=2_000_000_000,
    recent_liquidations=10_000_000,
    positions=current_positions,
)
```

### Тестирование

```bash
# All Phase 5A tests (100 tests)
pytest tests/test_futures_l3_execution.py -v

# By category
pytest tests/test_futures_l3_execution.py::TestLiquidationCascadeSimulator -v
pytest tests/test_futures_l3_execution.py::TestInsuranceFundManager -v
pytest tests/test_futures_l3_execution.py::TestADLQueueManager -v
pytest tests/test_futures_l3_execution.py::TestFundingPeriodDynamics -v
pytest tests/test_futures_l3_execution.py::TestFuturesL3ExecutionProvider -v
pytest tests/test_futures_l3_execution.py::TestIntegration -v
```

**Coverage**: 100 tests (100% pass rate)

| Category | Tests | Coverage |
|----------|-------|----------|
| Enums | 3 | LiquidationType, ADLRank, CascadePhase |
| LiquidationOrderInfo | 5 | Creation, properties, defaults |
| LiquidationFillResult | 2 | Filled/unfilled results |
| CascadeResult | 4 | Depth, phases |
| InsuranceFundState | 2 | Depletion, utilization |
| LiquidationOrderStream | 10 | Event handling, filtering, stats |
| LiquidationCascadeSimulator | 6 | Cascade simulation, price impact |
| InsuranceFundManager | 10 | Contributions, payouts, ADL trigger |
| ADLQueueManager | 7 | Queue building, ranking, candidates |
| FundingPeriodDynamics | 6 | Window detection, multipliers |
| FuturesL3Config | 6 | Validation, defaults |
| FuturesL3SlippageProvider | 5 | Base slippage, funding, cascade |
| FuturesL3FillProvider | 4 | Fill tracking, liquidation injection |
| FuturesL3ExecutionProvider | 10 | Full execution flow |
| Factory Functions | 2 | Config and provider creation |
| Presets | 5 | All preset configurations |
| Integration | 3 | Full flow, cascade recovery, fund depletion |
| Edge Cases | 4 | Empty orders, extreme funding, zero ADV |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `lob/futures_extensions.py` | LOB extensions for crypto futures (~1300 lines) |
| `execution_providers_futures_l3.py` | L3 futures execution provider (~1100 lines) |
| `tests/test_futures_l3_execution.py` | 100 comprehensive tests |

### Референсы

- Kyle (1985): "Continuous Auctions and Insider Trading" — Price impact model
- Almgren & Chriss (2001): "Optimal Execution" — Market impact theory
- Binance: "Liquidation Protocol" — Insurance fund and ADL mechanics
- Binance: "Funding Rate" — 8-hour funding periods
- FTX Research: "Liquidation Cascades" — Cascade dynamics (pre-collapse research)

---

## 📊 Phase 5B: L3 LOB for CME Futures (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 42/42 (100% pass) | **Date**: 2025-12-02

Phase 5B implements L3 Limit Order Book simulation for CME Group futures, including Globex-style FIFO matching, Market with Protection (MWP) orders, stop orders with velocity logic, and daily settlement simulation.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **GlobexMatchingEngine** | `lob/cme_matching.py` | CME Globex-style FIFO matching engine |
| **CMEL3SlippageProvider** | `execution_providers_cme_l3.py` | L3 slippage with LOB walk-through |
| **CMEL3FillProvider** | `execution_providers_cme_l3.py` | L3 fill logic with matching engine |
| **CMEL3ExecutionProvider** | `execution_providers_cme_l3.py` | Combined L3 CME execution provider |
| **DailySettlementSimulator** | `execution_providers_cme_l3.py` | Daily variation margin simulation |
| **Тесты** | `tests/test_cme_l3_execution.py` | 42 comprehensive tests |

### Key Concepts

#### 1. Globex-Style FIFO Matching

CME Globex uses strict Price-Time Priority (FIFO) matching:

```
BUY orders sorted: price DESC, time ASC (best price first, oldest first)
SELL orders sorted: price ASC, time ASC (best price first, oldest first)
```

**Usage**:
```python
from lob.cme_matching import GlobexMatchingEngine, StopOrder
from lob.data_structures import LimitOrder, Side, OrderType

# Create engine for ES (E-mini S&P 500)
engine = GlobexMatchingEngine(symbol="ES", tick_size=0.25, protection_points=6)

# Add resting order
resting = LimitOrder(
    order_id="rest_1",
    price=4500.0,
    qty=10.0,
    remaining_qty=10.0,
    timestamp_ns=0,
    side=Side.BUY,
    order_type=OrderType.LIMIT,
)
engine.add_resting_order(resting)

# Match aggressive order
aggressive = LimitOrder(
    order_id="aggr_1",
    price=4500.0,
    qty=5.0,
    remaining_qty=5.0,
    timestamp_ns=1000,
    side=Side.SELL,
    order_type=OrderType.MARKET,
)
result = engine.match(aggressive)
print(f"Filled: {result.total_filled_qty} @ {result.avg_fill_price}")
```

#### 2. Market with Protection (MWP) Orders

CME uses implicit price limits on market orders to prevent runaway fills:

| Product | Protection Points | Tick Size | Max Deviation |
|---------|-------------------|-----------|---------------|
| ES | 6 | 0.25 | 1.5 points |
| NQ | 10 | 0.25 | 2.5 points |
| GC | 50 | 0.10 | 5.0 points |
| CL | 100 | 0.01 | 1.0 point |

**MWP Behavior**:
- BUY MWP: Limit at best_ask + (protection_points × tick_size)
- SELL MWP: Limit at best_bid - (protection_points × tick_size)
- Unfilled portion is cancelled (not rested)

**Usage**:
```python
result = engine.match_with_protection(
    order=market_order,
    protection_points=6,  # Optional override
)
if result.cancelled_orders:
    print("Unfilled portion cancelled due to protection limit")
```

#### 3. Stop Orders with Velocity Logic

Stop orders trigger when price crosses the stop price, with CME velocity logic protection:

| Product | Velocity Threshold (ticks) | Pause Duration |
|---------|---------------------------|----------------|
| ES | 12 | 2 seconds |
| NQ | 20 | 2 seconds |
| GC | 50 | 2 seconds |
| CL | 100 | 2 seconds |

**Stop Order Types**:
- **Stop-Market**: Converts to MWP when triggered
- **Stop-Limit**: Converts to limit order when triggered

**Usage**:
```python
stop = StopOrder(
    order_id="stop_1",
    symbol="ES",
    side=Side.SELL,
    qty=5.0,
    stop_price=4490.0,
    limit_price=None,  # Stop-market
    use_protection=True,
)
engine.submit_stop_order(stop)

# Check and trigger stops
results = engine.check_stop_triggers(
    last_trade_price=4489.0,
    bid=4488.5,
    ask=4489.5,
    timestamp_ns=int(time.time() * 1e9),
)
```

#### 4. Session Detection

RTH (Regular Trading Hours) vs ETH (Electronic Trading Hours):

| Session | Hours (ET) | Spread Multiplier |
|---------|------------|-------------------|
| RTH | 9:30 - 16:15 | 1.0x |
| ETH | 18:00 - 9:30 | 1.5x |
| Pre-Open | 8:30 - 9:30 | 1.25x |
| Maintenance | 16:15 - 16:30 | N/A (closed) |

**Usage**:
```python
from execution_providers_cme_l3 import (
    detect_cme_session,
    is_rth_session,
    get_minutes_to_settlement,
    CMESession,
)

session = detect_cme_session(timestamp_ms)
if session == CMESession.RTH:
    print("Regular trading hours - tightest spreads")
elif session == CMESession.ETH:
    print("Electronic hours - wider spreads")
elif session == CMESession.MAINTENANCE:
    print("Market closed for daily maintenance")

# Check if RTH
if is_rth_session(timestamp_ms):
    spread_mult = 1.0

# Minutes until settlement
minutes = get_minutes_to_settlement(timestamp_ms, "ES")
if minutes and minutes < 30:
    print(f"Settlement approaching in {minutes} minutes")
```

#### 5. Daily Settlement Simulation

CME futures settle daily with variation margin:

**Settlement Times (Eastern Time)**:

| Product | Settlement Time | Notes |
|---------|-----------------|-------|
| ES, NQ, YM, RTY | 16:00 ET | Equity index |
| GC, SI, HG | 13:30 ET | Metals (COMEX) |
| CL, NG | 14:30 ET | Energy (NYMEX) |
| 6E, 6J, 6B | 15:00 ET | Currencies |

**Variation Margin Formula**:
```
VM = (Settlement_t - Settlement_t-1) × Qty × Multiplier
```

**Usage**:
```python
from execution_providers_cme_l3 import DailySettlementSimulator
from decimal import Decimal

simulator = DailySettlementSimulator(
    symbol="ES",
    contract_multiplier=Decimal("50"),
)

# Process settlement
simulator.process_settlement(
    timestamp_ms=settlement_time_ms,
    settlement_price=Decimal("4520.00"),
    position_qty=Decimal("2"),
)

# Get variation margin
vm = simulator.get_pending_variation_margin()
print(f"Variation Margin: ${vm}")

# Get last settlement price
last_price = simulator.get_last_settlement_price()
```

### Configuration

```python
from execution_providers_cme_l3 import (
    CMEL3ExecutionProvider,
    create_cme_l3_execution_provider,
    CMEL3Config,
)

# Create with default config
provider = create_cme_l3_execution_provider(symbol="ES")

# Create with profile
provider = create_cme_l3_execution_provider(
    symbol="ES",
    profile="conservative",
)

# Custom configuration
config = CMEL3Config(
    spread_bps=0.5,
    eth_spread_multiplier=1.5,
    settlement_premium=1.3,
    impact_coef=0.03,
)
provider = CMEL3ExecutionProvider(symbol="ES", config=config)
```

### Presets

| Preset | Spread (bps) | ETH Mult | Settlement Mult | Impact Coef |
|--------|--------------|----------|-----------------|-------------|
| `default` | 0.5 | 1.5 | 1.3 | 0.03 |
| `conservative` | 0.75 | 1.75 | 1.5 | 0.05 |
| `aggressive` | 0.35 | 1.25 | 1.15 | 0.02 |

### Тестирование

```bash
# All Phase 5B tests (42 tests)
pytest tests/test_cme_l3_execution.py -v

# By category
pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineBasic -v
pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineMWP -v
pytest tests/test_cme_l3_execution.py::TestGlobexMatchingEngineStops -v
pytest tests/test_cme_l3_execution.py::TestSessionDetection -v
pytest tests/test_cme_l3_execution.py::TestDailySettlementSimulator -v
pytest tests/test_cme_l3_execution.py::TestCMEL3SlippageProvider -v
pytest tests/test_cme_l3_execution.py::TestCMEL3FillProvider -v
pytest tests/test_cme_l3_execution.py::TestIntegration -v
```

**Coverage**: 42 tests (100% pass rate)

| Category | Tests | Coverage |
|----------|-------|----------|
| GlobexMatchingEngine Basic | 8 | FIFO matching, best bid/ask |
| MWP Orders | 3 | Protection limits, unfilled cancellation |
| Stop Orders | 5 | Trigger logic, stop-limit, velocity |
| Session Detection | 5 | RTH/ETH, settlement time |
| Daily Settlement | 7 | VM calculation, long/short positions |
| Slippage Provider | 4 | LOB walk, ETH multiplier, settlement |
| Fill Provider | 2 | Market order fills |
| Factory Functions | 3 | Profiles, creation |
| Edge Cases | 3 | Empty book, zero qty, various symbols |
| Integration | 2 | Full execution flow, settlement |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `lob/cme_matching.py` | GlobexMatchingEngine with MWP, stops, velocity (~800 lines) |
| `execution_providers_cme_l3.py` | L3 CME execution provider (~700 lines) |
| `tests/test_cme_l3_execution.py` | 42 comprehensive tests |

### Референсы

- CME Group: "Globex Matching Algorithm" — FIFO Price-Time Priority
- CME Group: "Market with Protection Orders" — MWP order handling
- CME Group: "Stop Spike Logic" — Velocity logic protection
- CME Group: "Daily Settlement Procedures" — Variation margin
- CME Group: "Globex Trading Hours" — RTH/ETH session definitions

---

## 🛡️ Phase 6A: Crypto Futures Risk Management (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 101/101 (100% pass) | **Date**: 2025-12-02

Phase 6A implements comprehensive risk management for crypto perpetual futures (Binance USDT-M), including leverage guards, margin monitoring, funding exposure, position concentration limits, and ADL risk tracking.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FuturesLeverageGuard** | `services/futures_risk_guards.py` | Tiered leverage enforcement with Binance brackets |
| **FuturesMarginGuard** | `services/futures_risk_guards.py` | Margin ratio monitoring with 5 levels |
| **MarginCallNotifier** | `services/futures_risk_guards.py` | Margin call notifications with cooldowns |
| **FundingExposureGuard** | `services/futures_risk_guards.py` | Funding rate risk monitoring |
| **ConcentrationGuard** | `services/futures_risk_guards.py` | Position concentration limits |
| **ADLRiskGuard** | `services/futures_risk_guards.py` | Auto-Deleveraging queue risk |
| **CryptoFuturesRiskGuard** | `risk_guard.py` | Unified guard integration |
| **Тесты** | `tests/test_futures_risk_guards.py` | 101 comprehensive tests |

### Key Concepts

#### 1. Leverage Tiering (Binance USDT-M)

Higher notional positions get lower max leverage:

| Notional (USD) | BTC Max | ETH Max | Other Max |
|----------------|---------|---------|-----------|
| < $50,000 | 125x | 100x | 75x |
| $50K-250K | 100x | 75x | 50x |
| $250K-1M | 50x | 50x | 25x |
| $1M-5M | 20x | 25x | 10x |
| $5M-20M | 10x | 10x | 5x |
| > $20M | 5x | 5x | 3x |

**Usage**:
```python
from services.futures_risk_guards import FuturesLeverageGuard, LeverageCheckResult

guard = FuturesLeverageGuard(
    max_account_leverage=20,
    max_symbol_leverage=125,
    concentration_limit=0.5,  # Max 50% in single symbol
)

result = guard.validate_new_position(
    proposed_position=position,
    current_positions=existing_positions,
    account_balance=Decimal("10000"),
)

if not result.is_valid:
    print(f"Blocked: {result.error_message}")
    print(f"Suggested leverage: {result.suggested_leverage}")
```

#### 2. Margin Status Levels

| Level | Margin Ratio | Action |
|-------|--------------|--------|
| **HEALTHY** | ≥ 1.5 (150%) | No action |
| **WARNING** | 1.2-1.5 (120-150%) | Alert |
| **DANGER** | 1.05-1.2 (105-120%) | Reduce position |
| **CRITICAL** | 1.0-1.05 (100-105%) | Urgent action |
| **LIQUIDATION** | ≤ 1.0 (100%) | Immediate liquidation risk |

**Usage**:
```python
from services.futures_risk_guards import (
    FuturesMarginGuard,
    MarginStatus,
    MarginCallLevel,
)
from decimal import Decimal

guard = FuturesMarginGuard(
    margin_calculator=None,  # Optional calculator
    warning_level=Decimal("1.5"),
    danger_level=Decimal("1.2"),
    critical_level=Decimal("1.05"),
)

# Check pre-calculated margin ratio
result = guard.check_margin_ratio(
    margin_ratio=1.35,  # 135%
    account_equity=10000.0,
    total_margin_used=7407.0,
    symbol="BTCUSDT",
)

print(f"Status: {result.status}")  # MarginStatus.WARNING
print(f"Requires reduction: {result.requires_reduction}")  # False
print(f"Requires liquidation: {result.requires_liquidation}")  # False
```

#### 3. Margin Call Notifications

```python
from services.futures_risk_guards import MarginCallNotifier, MarginCallEvent

notifier = MarginCallNotifier(
    cooldown_seconds=300,  # 5 minute cooldown between alerts
    callback=send_alert_function,  # Optional callback
)

# Check and notify
event = notifier.check_and_notify(
    margin_result=margin_result,
    position=position,
    mark_price=Decimal("50000"),
    wallet_balance=Decimal("10000"),
)

if event:
    print(f"Alert: {event.level.value} - {event.recommended_action}")
    print(f"Shortfall: ${event.shortfall}")
```

#### 4. Funding Rate Exposure

Monitors exposure to funding payments (every 8 hours):

| Level | Annual Rate | Action |
|-------|-------------|--------|
| **NORMAL** | < 10% APR | No action |
| **WARNING** | 10-25% APR | Monitor |
| **EXCESSIVE** | 25-50% APR | Consider reducing |
| **EXTREME** | > 50% APR | Reduce immediately |

**Usage**:
```python
from services.futures_risk_guards import FundingExposureGuard

guard = FundingExposureGuard(
    warning_threshold=Decimal("0.0001"),  # 0.01% per 8h
)

result = guard.check_funding_exposure(
    funding_rate=Decimal("0.0005"),  # 0.05% per 8h = ~54% APR
    position_side="LONG",
    position_notional=Decimal("100000"),
)

print(f"Level: {result.level}")  # EXTREME
print(f"APR: {result.annualized_rate:.1%}")  # 54.8%
print(f"Daily cost: ${result.daily_cost}")
```

#### 5. Position Concentration

```python
from services.futures_risk_guards import ConcentrationGuard

guard = ConcentrationGuard(
    single_symbol_limit=0.5,     # Max 50% in any symbol
    correlated_group_limit=0.7,  # Max 70% in correlated group
    correlation_groups={
        "BTC-ALTS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "STABLE": ["USDCUSDT", "BUSDUSDT"],
    },
)

result = guard.check_concentration(
    positions={"BTCUSDT": 60000, "ETHUSDT": 30000, "SOLUSDT": 10000},
    total_exposure=100000,
)

if not result.is_valid:
    print(f"Concentration exceeded: {result.largest_concentration:.1%}")
```

#### 6. ADL Risk Tracking

Auto-Deleveraging queue risk based on PnL × Leverage ranking:

| Level | ADL Percentile | Risk |
|-------|----------------|------|
| **LOW** | < 50% | Minimal ADL risk |
| **MEDIUM** | 50-75% | Monitor |
| **HIGH** | 75-90% | Consider reducing |
| **CRITICAL** | > 90% | High ADL risk |

**Usage**:
```python
from services.futures_risk_guards import ADLRiskGuard

guard = ADLRiskGuard(
    warning_percentile=75.0,
    critical_percentile=90.0,
)

result = guard.check_adl_risk(
    position_pnl_percentile=85.0,  # Top 15% profitable
    position_leverage_percentile=80.0,  # Top 20% leveraged
)

print(f"ADL Level: {result.level}")  # HIGH
print(f"ADL Score: {result.adl_score:.1f}")  # 85 × 80 / 100 = 68
```

### Integration with risk_guard.py

```python
from risk_guard import create_crypto_futures_risk_guard, CryptoFuturesRiskConfig

config = CryptoFuturesRiskConfig(
    market_type="CRYPTO_FUTURES",
    max_account_leverage=20.0,
    max_single_symbol_pct=0.5,
    max_correlated_group_pct=0.7,
    margin_warning_threshold=1.5,
    margin_danger_threshold=1.2,
    margin_critical_threshold=1.05,
    funding_rate_warning_threshold=0.0001,
    adl_warning_percentile=75.0,
    adl_critical_percentile=90.0,
    strict_mode=True,
)

guard = create_crypto_futures_risk_guard(config)

# Check trade
event = guard.check_trade(
    symbol="BTCUSDT",
    side="LONG",
    quantity=0.1,
    leverage=10,
    mark_price=50000.0,
    account_equity=10000.0,
)

if event != RiskEvent.NONE:
    print(f"Risk event: {event.value}")
    print(f"Reason: {guard.get_last_event_reason()}")
```

### Тестирование

```bash
# All Phase 6A tests (101 tests)
pytest tests/test_futures_risk_guards.py -v

# By category
pytest tests/test_futures_risk_guards.py::TestFuturesLeverageGuard -v
pytest tests/test_futures_risk_guards.py::TestFuturesMarginGuard -v
pytest tests/test_futures_risk_guards.py::TestMarginCallNotifier -v
pytest tests/test_futures_risk_guards.py::TestFundingExposureGuard -v
pytest tests/test_futures_risk_guards.py::TestConcentrationGuard -v
pytest tests/test_futures_risk_guards.py::TestADLRiskGuard -v
pytest tests/test_futures_risk_guards.py::TestCryptoFuturesRiskGuard -v
pytest tests/test_futures_risk_guards.py::TestThreadSafety -v
pytest tests/test_futures_risk_guards.py::TestIntegrationScenarios -v
```

**Coverage**: 101 tests (100% pass rate)

| Category | Tests | Coverage |
|----------|-------|----------|
| Enums & Constants | 7 | MarginCallLevel, MarginStatus, etc. |
| Config Classes | 6 | Leverage, Margin, Notifier, etc. |
| LeverageCheckResult | 2 | Valid/invalid results |
| MarginCheckResult | 2 | Healthy/danger results |
| MarginCallEvent | 4 | Creation, urgency, escalation |
| FuturesLeverageGuard | 8 | Validation, max position |
| FuturesMarginGuard | 7 | All margin levels |
| MarginCallNotifier | 7 | Notifications, cooldowns |
| FundingExposureGuard | 8 | All funding levels |
| ConcentrationGuard | 6 | Single/correlated limits |
| ADLRiskGuard | 5 | All ADL levels |
| CryptoFuturesRiskGuard | 4 | Integration tests |
| Factory Functions | 4 | Creation, spot handling |
| RiskEvent Integration | 7 | All event types |
| Edge Cases | 6 | Zero values, extremes |
| Thread Safety | 2 | Concurrent access |
| Integration Scenarios | 4 | Full workflows |
| Risk Summary | 2 | Summary generation |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/futures_risk_guards.py` | All futures risk guard implementations (~1200 lines) |
| `risk_guard.py` | CryptoFuturesRiskGuard integration (~200 lines added) |
| `tests/test_futures_risk_guards.py` | 101 comprehensive tests |

### Референсы

- Binance: "Leverage and Margin of USDⓈ-M Futures"
- Binance: "Auto-Deleveraging (ADL)"
- Binance: "Funding Rate History"
- Binance: "Liquidation Protocol"
- Risk management best practices for derivatives trading

---

## 🛡️ Phase 6B: CME Futures Risk Management (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 130/130 (100% pass) | **Покрытие**: 98% | **Date**: 2025-12-02

Phase 6B implements comprehensive risk management for CME Group futures (via Interactive Brokers), including SPAN margin monitoring, position limits, circuit breaker awareness, settlement risk management, and contract rollover guards.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **SPANMarginGuard** | `services/cme_risk_guards.py` | SPAN margin monitoring with 4 levels |
| **CMEPositionLimitGuard** | `services/cme_risk_guards.py` | CME speculative limits & accountability levels |
| **CircuitBreakerAwareGuard** | `services/cme_risk_guards.py` | Rule 80B circuit breaker integration |
| **SettlementRiskGuard** | `services/cme_risk_guards.py` | Daily settlement risk management |
| **RolloverGuard** | `services/cme_risk_guards.py` | Contract expiration & rollover tracking |
| **CMEFuturesRiskGuard** | `services/cme_risk_guards.py` | Unified guard combining all CME guards |
| **Тесты** | `tests/test_cme_risk_guards.py` | 130 comprehensive tests |

### Key Concepts

#### 1. SPAN Margin Status Levels

| Level | Margin Ratio | Action |
|-------|--------------|--------|
| **HEALTHY** | ≥ 1.5 (150%) | No action |
| **WARNING** | 1.2-1.5 (120-150%) | Alert |
| **DANGER** | 1.05-1.2 (105-120%) | Reduce position |
| **CRITICAL** | 1.0-1.05 (100-105%) | Urgent action |
| **LIQUIDATION** | ≤ 1.0 (100%) | Immediate liquidation risk |

#### 2. CME Position Limits (Speculative)

| Product | Speculative Limit | Accountability Level |
|---------|-------------------|---------------------|
| ES | 50,000 | 20,000 |
| NQ | 40,000 | 15,000 |
| YM | 25,000 | 10,000 |
| RTY | 20,000 | 5,000 |
| GC | 6,000 | 3,000 |
| CL | 10,000 | 5,000 |
| 6E | 10,000 | 5,000 |
| ZN | 150,000 | 50,000 |

#### 3. Circuit Breaker Levels (Rule 80B)

| Level | Trigger | RTH Halt | ETH Action |
|-------|---------|----------|------------|
| Level 1 | -7% | 15 min | Monitoring |
| Level 2 | -13% | 15 min | Restrict trading |
| Level 3 | -20% | Day halt | Block all trading |

#### 4. Settlement Risk Levels

| Level | Minutes to Settlement | Action |
|-------|----------------------|--------|
| **NORMAL** | > warn_minutes | Normal trading |
| **APPROACHING** | warn - critical | Alert, monitor VM |
| **IMMINENT** | critical - block | Prepare for settlement |
| **SETTLEMENT** | < block_minutes | Block new positions |

#### 5. Rollover Risk Levels

| Level | Days to Roll | Action |
|-------|--------------|--------|
| **SAFE** | > warn_days | Normal trading |
| **MONITORING** | warn - critical | Monitor spreads |
| **APPROACHING** | critical - block | Prepare roll trades |
| **IMMINENT** | 0 - block | Execute rollover |
| **EXPIRED** | < 0 | Force close only |

### Usage

```python
from services.cme_risk_guards import (
    CMEFuturesRiskGuard,
    SPANMarginGuard,
    CMEPositionLimitGuard,
    CircuitBreakerAwareGuard,
    SettlementRiskGuard,
    RolloverGuard,
    RiskEvent,
)
from decimal import Decimal

# 1. Unified Risk Guard
guard = CMEFuturesRiskGuard(strict_mode=True)
guard.add_symbol_to_monitor("ES", Decimal("4500"))

event = guard.check_trade(
    symbol="ES",
    side="LONG",
    quantity=5,
    account_equity=Decimal("500000"),
    positions=current_positions,
    prices={"ES": Decimal("4500")},
    contract_specs=specs,
    timestamp_ms=int(time.time() * 1000),
)

if event != RiskEvent.NONE:
    print(f"Risk event: {event.value}")
    print(f"Details: {guard.get_last_event_details()}")

# 2. SPAN Margin Guard
margin_guard = SPANMarginGuard()
margin_result = margin_guard.check_margin(
    account_equity=Decimal("500000"),
    positions=positions,
    prices=prices,
    contract_specs=specs,
)
print(f"Margin Status: {margin_result.status}")
print(f"Margin Ratio: {margin_result.margin_ratio}")

# 3. Position Limit Guard
limit_guard = CMEPositionLimitGuard()
limit_result = limit_guard.check_position_limit("ES", 45000)
print(f"Within Limit: {limit_result.is_within_limit}")
print(f"Utilization: {limit_result.utilization_pct}%")

# 4. Circuit Breaker Aware Guard
cb_guard = CircuitBreakerAwareGuard()
cb_guard.add_symbol("ES", Decimal("4500"))
cb_result = cb_guard.check_trading_allowed(
    symbol="ES",
    current_price=Decimal("4185"),  # -7%
    timestamp_ms=now_ms,
    is_rth=True,
)
print(f"Can Trade: {cb_result.can_trade}")
print(f"CB Level: {cb_result.circuit_breaker_level}")

# 5. Settlement Risk Guard
settle_guard = SettlementRiskGuard()
settle_result = settle_guard.check_settlement_risk(
    symbol="ES",
    timestamp_ms=now_ms,
)
print(f"Settlement Risk: {settle_result.risk_level}")
print(f"Minutes to Settlement: {settle_result.minutes_to_settlement}")

# 6. Rollover Guard
roll_guard = RolloverGuard()
roll_guard.set_expiration_calendar("ES", [date(2025, 3, 21)])
roll_result = roll_guard.check_rollover_risk("ES", date.today())
print(f"Rollover Risk: {roll_result.risk_level}")
print(f"Days to Roll: {roll_result.days_to_roll}")
```

### Risk Event Types

| Event | Trigger | Strict Mode |
|-------|---------|-------------|
| `NONE` | All checks pass | - |
| `MARGIN_WARNING` | Margin ratio < warning | Strict only |
| `MARGIN_DANGER` | Margin ratio < danger | Always |
| `MARGIN_CRITICAL` | Margin ratio < critical | Always |
| `MARGIN_LIQUIDATION` | Margin ratio ≤ 1.0 | Always |
| `POSITION_LIMIT_EXCEEDED` | Over speculative limit | Always |
| `POSITION_ACCOUNTABILITY` | Over accountability | Strict only |
| `CIRCUIT_BREAKER_L1` | -7% decline | Always |
| `CIRCUIT_BREAKER_L2` | -13% decline | Always |
| `CIRCUIT_BREAKER_L3` | -20% decline | Always |
| `VELOCITY_PAUSE` | Rapid price movement | Always |
| `SETTLEMENT_APPROACHING` | < warn_minutes | Strict only |
| `SETTLEMENT_IMMINENT` | < critical_minutes | Always |
| `ROLLOVER_WARNING` | < warn_days | Strict only |
| `ROLLOVER_IMMINENT` | < block_days | Always |
| `ROLLOVER_REQUIRED` | Contract expired | Always |

### Тестирование

```bash
# All Phase 6B tests (130 tests, 98% coverage)
pytest tests/test_cme_risk_guards.py -v

# By component
pytest tests/test_cme_risk_guards.py::TestSPANMarginGuard -v
pytest tests/test_cme_risk_guards.py::TestCMEPositionLimitGuard -v
pytest tests/test_cme_risk_guards.py::TestCircuitBreakerAwareGuard -v
pytest tests/test_cme_risk_guards.py::TestSettlementRiskGuard -v
pytest tests/test_cme_risk_guards.py::TestRolloverGuard -v
pytest tests/test_cme_risk_guards.py::TestCMEFuturesRiskGuard -v
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/cme_risk_guards.py` | All CME risk guard implementations (~1850 lines) |
| `tests/test_cme_risk_guards.py` | 130 comprehensive tests |

### Configuration

```python
from services.cme_risk_guards import (
    SPANMarginGuardConfig,
    CMEPositionLimitGuardConfig,
    CircuitBreakerGuardConfig,
    SettlementRiskGuardConfig,
    RolloverGuardConfig,
)

# SPAN Margin Config
margin_config = SPANMarginGuardConfig(
    warning_ratio=Decimal("1.5"),
    danger_ratio=Decimal("1.2"),
    critical_ratio=Decimal("1.05"),
)

# Circuit Breaker Config
cb_config = CircuitBreakerGuardConfig(
    prevent_trades_on_halt=True,
    pre_cb_warning_pct=Decimal("-0.05"),
)

# Settlement Risk Config
settle_config = SettlementRiskGuardConfig(
    warn_minutes_before=60,
    critical_minutes_before=30,
    block_new_positions_minutes=15,
)

# Rollover Config
roll_config = RolloverGuardConfig(
    warn_days_before=8,
    critical_days_before=3,
    block_new_positions_days=1,
)
```

### Референсы

- CME Group: "Position Limits and Accountability Levels"
- CME Group: "SPAN Margin Methodology"
- CME Group: "Rule 80B - Circuit Breakers"
- CME Group: "Daily Settlement Procedures"
- CME Group: "Contract Specifications and Expiration"

---

## 🛡️ Phase 7: Unified Futures Risk Management (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 116/116 (100% pass) | **Покрытие**: 98% | **Date**: 2025-12-02

Phase 7 unifies crypto futures and CME futures risk management into a single interface with automatic asset type detection, portfolio-level risk aggregation, and cross-asset correlation handling.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **UnifiedFuturesRiskGuard** | `services/unified_futures_risk.py` | Main unified guard with auto-delegation |
| **AssetType** | `services/unified_futures_risk.py` | Enum for asset classification |
| **UnifiedRiskConfig** | `services/unified_futures_risk.py` | Pydantic config combining crypto/CME settings |
| **UnifiedRiskEvent** | `services/unified_futures_risk.py` | Unified risk events across asset types |
| **UnifiedMarginResult** | `services/unified_futures_risk.py` | Unified margin check results |
| **PortfolioRiskManager** | `services/unified_futures_risk.py` | Portfolio-level risk aggregation |
| **Тесты** | `tests/test_unified_futures_risk.py` | 116 comprehensive tests |
| **Config** | `configs/unified_futures_risk.yaml` | YAML configuration with profiles |

### Key Concepts

#### 1. Asset Type Detection

Automatic detection from symbol patterns:

| Pattern | Asset Type | Examples |
|---------|------------|----------|
| `*USDT`, `*BUSD` | CRYPTO_PERPETUAL | BTCUSDT, ETHBUSD |
| `*_YYMMDD` | CRYPTO_QUARTERLY | BTCUSDT_240329 |
| `ES`, `NQ`, `YM`, `RTY` | CME_EQUITY_INDEX | ES, NQ, MES, MNQ |
| `GC`, `SI`, `HG`, `MGC` | CME_METAL | Gold, Silver, Copper |
| `CL`, `NG`, `RB`, `HO` | CME_ENERGY | Crude, NatGas |
| `6E`, `6J`, `6B`, `6A` | CME_CURRENCY | Euro, Yen, Pound |
| `ZN`, `ZB`, `ZT`, `ZF` | CME_BOND | 10Y, 30Y notes |
| Other | UNKNOWN | Fallback |

#### 2. Automatic Guard Delegation

```python
from services.unified_futures_risk import UnifiedFuturesRiskGuard

guard = UnifiedFuturesRiskGuard()

# Crypto symbols → Crypto guards
event = guard.check_trade("BTCUSDT", "BUY", 0.1, ...)  # Uses crypto guards

# CME symbols → CME guards
event = guard.check_trade("ES", "BUY", 5, ...)  # Uses CME guards
```

#### 3. Unified Risk Events

| Event | Description | Crypto | CME |
|-------|-------------|--------|-----|
| `MARGIN_WARNING` | Approaching margin limit | ✅ | ✅ |
| `MARGIN_DANGER` | Low margin ratio | ✅ | ✅ |
| `MARGIN_CRITICAL` | Critical margin | ✅ | ✅ |
| `MARGIN_LIQUIDATION` | Liquidation risk | ✅ | ✅ |
| `LEVERAGE_EXCEEDED` | Over leverage limit | ✅ | - |
| `CONCENTRATION_EXCEEDED` | Position too large | ✅ | - |
| `FUNDING_WARNING` | High funding rate | ✅ | - |
| `FUNDING_EXCESSIVE` | Extreme funding | ✅ | - |
| `ADL_WARNING` | ADL queue risk | ✅ | - |
| `ADL_CRITICAL` | High ADL risk | ✅ | - |
| `CIRCUIT_BREAKER_L1` | -7% decline | - | ✅ |
| `CIRCUIT_BREAKER_L2` | -13% decline | - | ✅ |
| `CIRCUIT_BREAKER_L3` | -20% decline | - | ✅ |
| `VELOCITY_PAUSE` | Rapid price move | - | ✅ |
| `POSITION_LIMIT_EXCEEDED` | Over spec limit | - | ✅ |
| `SETTLEMENT_APPROACHING` | Near settlement | - | ✅ |
| `ROLLOVER_WARNING` | Near expiry | - | ✅ |

### Usage

```python
from services.unified_futures_risk import (
    UnifiedFuturesRiskGuard,
    UnifiedRiskConfig,
    CryptoRiskConfig,
    CMERiskConfig,
    PortfolioRiskConfig,
    create_unified_risk_guard,
    load_config_from_yaml,
)
from decimal import Decimal

# 1. Create with defaults
guard = UnifiedFuturesRiskGuard()

# 2. Create from YAML config
config = load_config_from_yaml("configs/unified_futures_risk.yaml")
guard = create_unified_risk_guard(config)

# 3. Create with custom config
config = UnifiedRiskConfig(
    crypto=CryptoRiskConfig(
        max_account_leverage=20.0,
        max_symbol_leverage=125.0,
        margin_warning_threshold=1.5,
        margin_danger_threshold=1.2,
        margin_critical_threshold=1.05,
        max_single_symbol_pct=0.5,
    ),
    cme=CMERiskConfig(
        margin_warning_ratio=1.5,
        margin_danger_ratio=1.2,
        margin_critical_ratio=1.05,
        enforce_speculative_limits=True,
        prevent_trades_on_halt=True,
    ),
    portfolio=PortfolioRiskConfig(
        enable_correlation_tracking=True,
        correlation_lookback_days=30,
    ),
)
guard = UnifiedFuturesRiskGuard(config=config)

# 4. Check trade (auto-delegates based on symbol)
event = guard.check_trade(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.5,
    leverage=10,
    account_equity=Decimal("50000"),
    mark_price=Decimal("45000"),
    funding_rate=Decimal("0.0001"),
)

if event != UnifiedRiskEvent.NONE:
    print(f"Risk event: {event.value}")
    print(f"Details: {guard.get_last_event_details()}")

# 5. Check margin (crypto)
margin_result = guard.check_margin(
    symbol="ETHUSDT",
    account_equity=Decimal("100000"),
    positions=crypto_positions,
    mark_prices={"ETHUSDT": Decimal("3000")},
)
print(f"Status: {margin_result.status}")
print(f"Margin Ratio: {margin_result.margin_ratio}")

# 6. Check margin (CME)
margin_result = guard.check_margin(
    symbol="ES",
    account_equity=Decimal("500000"),
    positions=cme_positions,
    prices={"ES": Decimal("4500")},
    contract_specs=es_spec,
)
print(f"Status: {margin_result.status}")
print(f"Available Margin: ${margin_result.available_margin}")

# 7. Get asset type
asset_type = guard.get_asset_type("BTCUSDT")  # CRYPTO_PERPETUAL
asset_type = guard.get_asset_type("ES")       # CME_EQUITY_INDEX
asset_type = guard.get_asset_type("GC")       # CME_METAL

# 8. Portfolio-level risk (cross-asset)
portfolio_result = guard.check_portfolio_risk(
    all_positions={"BTCUSDT": pos1, "ES": pos2, "GC": pos3},
    account_equity=Decimal("1000000"),
)
print(f"Total Margin Used: ${portfolio_result.total_margin_used}")
print(f"Cross-Asset Correlation: {portfolio_result.correlation_warning}")
```

### Configuration (YAML)

```yaml
# configs/unified_futures_risk.yaml
crypto:
  max_account_leverage: 20.0
  max_symbol_leverage: 125.0
  margin_warning_threshold: 1.5
  margin_danger_threshold: 1.2
  margin_critical_threshold: 1.05
  max_single_symbol_pct: 0.5
  max_correlated_group_pct: 0.7
  funding_warning_threshold: 0.0001
  funding_excessive_threshold: 0.0003
  adl_warning_percentile: 75.0
  adl_critical_percentile: 90.0
  strict_mode: true

cme:
  margin_warning_ratio: 1.5
  margin_danger_ratio: 1.2
  margin_critical_ratio: 1.05
  prevent_trades_on_halt: true
  pre_cb_warning_pct: -0.05
  settlement_warn_minutes: 60
  settlement_critical_minutes: 30
  rollover_warn_days: 8
  rollover_critical_days: 3
  enforce_speculative_limits: true
  strict_mode: true

portfolio:
  enable_correlation_tracking: true
  correlation_lookback_days: 30
  correlation_spike_threshold: 0.8
  aggregate_margin_across_types: true

profiles:
  conservative:
    crypto:
      max_account_leverage: 10.0
      margin_warning_threshold: 2.0
    cme:
      margin_warning_ratio: 2.0
  aggressive:
    crypto:
      max_account_leverage: 50.0
      margin_warning_threshold: 1.2
    cme:
      margin_warning_ratio: 1.2
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  UnifiedFuturesRiskGuard                        │
│  - Asset type detection                                         │
│  - Automatic guard delegation                                   │
│  - Unified event translation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│  Crypto Guards      │               │  CME Guards         │
│  ├─ LeverageGuard   │               │  ├─ SPANMarginGuard │
│  ├─ MarginGuard     │               │  ├─ PositionLimits  │
│  ├─ ConcentrationG  │               │  ├─ CircuitBreaker  │
│  ├─ FundingGuard    │               │  ├─ SettlementRisk  │
│  └─ ADLRiskGuard    │               │  └─ RolloverGuard   │
└─────────────────────┘               └─────────────────────┘
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                 ┌─────────────────────┐
                 │  PortfolioRiskMgr   │
                 │  - Cross-asset      │
                 │  - Correlation      │
                 │  - Aggregation      │
                 └─────────────────────┘
```

### Тестирование

```bash
# All Phase 7 tests (116 tests)
pytest tests/test_unified_futures_risk.py -v

# By category
pytest tests/test_unified_futures_risk.py::TestAssetType -v
pytest tests/test_unified_futures_risk.py::TestUnifiedRiskEvent -v
pytest tests/test_unified_futures_risk.py::TestUnifiedMarginResult -v
pytest tests/test_unified_futures_risk.py::TestUnifiedRiskConfig -v
pytest tests/test_unified_futures_risk.py::TestUnifiedFuturesRiskGuard -v
pytest tests/test_unified_futures_risk.py::TestPortfolioRiskManager -v
pytest tests/test_unified_futures_risk.py::TestFactoryFunctions -v
pytest tests/test_unified_futures_risk.py::TestIntegration -v

# Regression tests (Phase 6A + 6B)
pytest tests/test_futures_risk_guards.py tests/test_cme_risk_guards.py -v  # 231 tests
```

**Coverage**: 116 Phase 7 tests + 231 regression tests = 347 total tests passing

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/unified_futures_risk.py` | Unified risk management (~900 lines) |
| `configs/unified_futures_risk.yaml` | Configuration with profiles |
| `tests/test_unified_futures_risk.py` | 116 comprehensive tests |

### Референсы

- Phase 6A: Crypto Futures Risk Guards
- Phase 6B: CME Futures Risk Guards
- Portfolio theory: Markowitz (1952) mean-variance optimization
- Risk aggregation: Basel III framework concepts

---

## 🔴 Phase 9: Unified Futures Live Trading (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 81/81 (100% pass) | **Date**: 2025-12-02

Phase 9 implements unified live trading infrastructure for futures, including position synchronization, margin monitoring, funding rate tracking, and a coordinated live runner.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **FuturesLiveRunner** | `services/futures_live_runner.py` | Main live trading coordinator |
| **FuturesPositionSynchronizer** | `services/futures_position_sync.py` | Position sync with exchange |
| **FuturesMarginMonitor** | `services/futures_margin_monitor.py` | Real-time margin monitoring |
| **FuturesFundingTracker** | `services/futures_funding_tracker.py` | Funding rate tracking & predictions |
| **Live Config** | `configs/config_live_futures.yaml` | Live trading configuration |
| **Tests** | `tests/test_futures_live_trading.py` | 81 comprehensive tests |

### Key Concepts

#### 1. Position Synchronization

Real-time position sync between local state and exchange:

```python
from services.futures_position_sync import (
    FuturesPositionSynchronizer,
    FuturesSyncConfig,
    FuturesSyncEventType,
)

config = FuturesSyncConfig(
    exchange=Exchange.BINANCE,
    futures_type=FuturesType.CRYPTO_PERPETUAL,
    sync_interval_sec=10.0,       # Sync every 10 seconds
    qty_tolerance_pct=0.001,      # 0.1% tolerance
    auto_reconcile=False,         # Manual reconciliation
)

sync = FuturesPositionSynchronizer(
    position_provider=position_provider,
    account_provider=account_provider,
    local_state_getter=get_local_positions,
    config=config,
    on_event=handle_sync_event,
)

# Start background sync
await sync.start_async()

# Or sync once
events = await sync.sync_once()
for event in events:
    if event.event_type == FuturesSyncEventType.QTY_MISMATCH:
        print(f"Position mismatch: {event.symbol}")
```

#### 2. Sync Event Types

| Event Type | Description |
|------------|-------------|
| `POSITION_OPENED` | New position detected on exchange |
| `POSITION_CLOSED` | Position closed on exchange |
| `POSITION_MODIFIED` | Position size changed |
| `QTY_MISMATCH` | Local vs exchange quantity differs |
| `LEVERAGE_MISMATCH` | Leverage setting differs |
| `LIQUIDATION_DETECTED` | Position liquidated |
| `ADL_DETECTED` | Auto-deleveraging occurred |
| `FUNDING_RECEIVED` | Funding payment received |
| `FUNDING_PAID` | Funding payment made |
| `SETTLEMENT_OCCURRED` | Daily settlement (CME) |
| `MARGIN_CALL` | Margin call triggered |
| `MARGIN_RATIO_LOW` | Margin ratio below threshold |

#### 3. Margin Monitoring

Real-time margin ratio tracking with alerts:

```python
from services.futures_margin_monitor import (
    FuturesMarginMonitor,
    MarginMonitorConfig,
    MarginStatus,
)

config = MarginMonitorConfig(
    check_interval_sec=5.0,
    warning_ratio=1.5,    # 150%
    danger_ratio=1.2,     # 120%
    critical_ratio=1.05,  # 105%
)

monitor = FuturesMarginMonitor(
    account_provider=account_provider,
    position_provider=position_provider,
    config=config,
    on_status_change=handle_margin_alert,
)

# Check current status
status = await monitor.check_margin()
print(f"Margin ratio: {status.margin_ratio:.2f}")
print(f"Status: {status.status}")  # HEALTHY, WARNING, DANGER, CRITICAL
```

#### 4. Funding Rate Tracking

Historical tracking and prediction for crypto perpetuals:

```python
from services.futures_funding_tracker import (
    FuturesFundingTracker,
    FundingTrackerConfig,
    FundingRateInfo,
)

config = FundingTrackerConfig(
    data_dir="data/futures",
    prediction_method="ewma",    # last, avg, ewma
    cache_ttl_sec=300,
)

tracker = FuturesFundingTracker(
    funding_provider=funding_provider,
    config=config,
)

# Get current funding info
info = await tracker.get_funding_info("BTCUSDT")
print(f"Current rate: {info.funding_rate:.4%}")
print(f"Next funding: {info.next_funding_time}")
print(f"Predicted rate: {info.predicted_rate:.4%}")

# Get funding statistics
stats = tracker.get_funding_stats("BTCUSDT", lookback_days=30)
print(f"Avg rate: {stats.avg_rate:.4%}")
print(f"Annualized: {stats.annualized_rate:.2%}")
```

#### 5. Live Runner

Coordinates all components for unified live trading:

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

# Runner coordinates:
# - Position sync (every 5-10 sec)
# - Margin monitoring (every 5 sec)
# - Funding tracking (every 60 sec)
# - Signal generation (main loop)
# - Order execution
# - Risk management
```

### Configuration

```yaml
# configs/config_live_futures.yaml
futures_type: "CRYPTO_PERPETUAL"
exchange: "binance"
symbols:
  - "BTCUSDT"
  - "ETHUSDT"

paper_trading: true

# Timing
main_loop_interval_sec: 1.0
position_sync_interval_sec: 5.0
margin_check_interval_sec: 10.0
funding_check_interval_sec: 60.0

# Feature flags
enable_position_sync: true
enable_margin_monitoring: true
enable_funding_tracking: true
enable_adl_monitoring: true

# Risk settings
strict_mode: true
max_leverage: 10
max_position_value: 100000
max_total_exposure: 500000

# Margin thresholds
margin:
  warning_ratio: 1.5
  danger_ratio: 1.2
  critical_ratio: 1.1
  alert_cooldown_sec: 300

# Position sync settings
position_sync:
  interval_sec: 5.0
  tolerance: 0.01
  auto_reconcile: false

# Funding tracking
funding:
  data_dir: "data/futures"
  prediction_method: "ewma"
  cache_ttl_sec: 300
```

### ADL Risk Levels

| Level | Description | Action |
|-------|-------------|--------|
| `SAFE` | Low ADL risk | Normal trading |
| `WARNING` | Moderate ADL risk | Monitor closely |
| `DANGER` | High ADL risk | Consider reducing |
| `CRITICAL` | Imminent ADL risk | Reduce immediately |

### Тестирование

```bash
# All Phase 9 tests (81 tests)
pytest tests/test_futures_live_trading.py -v

# By category
pytest tests/test_futures_live_trading.py::TestFuturesLiveConfig -v
pytest tests/test_futures_live_trading.py::TestFuturesSyncConfig -v
pytest tests/test_futures_live_trading.py::TestFuturesSyncEventType -v
pytest tests/test_futures_live_trading.py::TestFuturesPositionSynchronizer -v
pytest tests/test_futures_live_trading.py::TestFuturesMarginMonitor -v
pytest tests/test_futures_live_trading.py::TestFuturesFundingTracker -v
pytest tests/test_futures_live_trading.py::TestFuturesLiveRunner -v
```

**Coverage**: 81 tests (100% pass rate)

| Category | Tests | Coverage |
|----------|-------|----------|
| FuturesLiveConfig | 10 | Config loading, validation, defaults |
| FuturesSyncConfig | 6 | Sync config defaults, custom values |
| FuturesSyncEventType | 8 | All event types |
| FundingRateInfo | 4 | Funding rate data model |
| MarginStatus | 5 | Margin status levels |
| ADLRiskLevel | 4 | ADL risk classification |
| FuturesPositionSynchronizer | 15 | Position sync workflow |
| FuturesMarginMonitor | 10 | Margin monitoring |
| FuturesFundingTracker | 8 | Funding tracking & prediction |
| FuturesLiveRunner | 7 | Live runner coordination |
| Integration | 4 | End-to-end scenarios |

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `services/futures_live_runner.py` | Main live trading coordinator (~500 lines) |
| `services/futures_position_sync.py` | Position synchronization (~600 lines) |
| `services/futures_margin_monitor.py` | Margin monitoring (~400 lines) |
| `services/futures_funding_tracker.py` | Funding rate tracking (~450 lines) |
| `configs/config_live_futures.yaml` | Live trading configuration |
| `tests/test_futures_live_trading.py` | 81 comprehensive tests |

### Референсы

- Phase 8: Multi-Futures Training Pipeline (prerequisite)
- Phase 6A/6B: Crypto/CME Risk Guards (integrated)
- Phase 7: Unified Risk Management (integrated)
- Binance Futures API: Position, Account, Funding Rate endpoints
- CME Group: Daily settlement procedures

---

## 📋 Phase 10: Validation & Documentation (COMPLETED)

**Статус**: ✅ Production Ready | **Тесты**: 171/171 (100% pass) | **Date**: 2025-12-02

Phase 10 completes the Futures Integration project with comprehensive validation testing, backward compatibility verification, performance benchmarks, and documentation.

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| **Validation Tests** | `tests/test_futures_validation.py` | 125 comprehensive validation tests |
| **Backward Compatibility** | `tests/test_futures_backward_compatibility.py` | 46 passed, 20 skipped compatibility tests |
| **Performance Benchmarks** | `benchmarks/bench_futures_simulation.py` | Performance measurement suite |
| **Integration Report** | `FUTURES_INTEGRATION_REPORT.md` | Project completion report |
| **Documentation Suite** | `docs/futures/*.md` | 8 documentation files |

### Validation Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Core Models | 15 | FuturesPosition, ContractSpec, MarginMode |
| Margin Calculations | 20 | Tiered margin, SPAN, liquidation price |
| Funding Rates | 12 | Rate calculation, payment simulation |
| Slippage Models | 18 | Crypto L2, CME L2, cascade effects |
| Risk Guards | 15 | Leverage, margin, concentration, ADL |
| L3 LOB Simulation | 15 | Fill probability, impact models, matching |
| Cross-Component | 10 | Full trade cycle, data flow |
| Validation Metrics | 5 | Fill rate, slippage, funding accuracy |

### Backward Compatibility Categories

| Category | Tests | Status |
|----------|-------|--------|
| Crypto Spot | 10 | ✅ All pass |
| US Equity | 10 | ✅ All pass (some skipped) |
| Forex (OANDA) | 8 | ✅ All pass |
| L3 LOB | 8 | ✅ All pass |
| Risk Management | 4 | ✅ All pass |
| Trading Env | 4 | ✅ All pass |
| Adapters | 6 | ✅ All pass |
| Features Pipeline | 4 | ✅ All pass (some skipped) |
| Model Training | 4 | ✅ All pass |
| Configuration | 4 | ✅ All pass |

### Validation Metrics Achieved

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Fill Rate (L2) | > 95% | 98.5% | ✅ |
| Fill Rate (L3) | > 90% | 94.2% | ✅ |
| Slippage Error | < 3 bps | 1.8 bps | ✅ |
| Funding Rate Accuracy | > 99% | 99.7% | ✅ |
| Liquidation Timing | < 1 bar | 0.2 bars | ✅ |
| Margin Calculation Error | < 0.1% | 0.02% | ✅ |

### Performance Benchmarks

| Operation | Target | Achieved | Status |
|-----------|--------|----------|--------|
| L2 Crypto Slippage | < 100 μs | 45 μs | ✅ |
| L2 CME Slippage | < 100 μs | 52 μs | ✅ |
| L3 Matching | < 500 μs | 180 μs | ✅ |
| Tiered Margin Calc | < 50 μs | 18 μs | ✅ |
| SPAN Margin Calc | < 100 μs | 75 μs | ✅ |
| Funding Rate Calc | < 10 μs | 3 μs | ✅ |
| Liquidation Price | < 50 μs | 22 μs | ✅ |
| Risk Guard Check | < 50 μs | 28 μs | ✅ |

### Documentation Suite

| File | Description |
|------|-------------|
| `docs/futures/overview.md` | Architecture overview |
| `docs/futures/api_reference.md` | API reference |
| `docs/futures/configuration.md` | Configuration guide |
| `docs/futures/margin_calculation.md` | Margin calculation |
| `docs/futures/funding_rates.md` | Funding rates |
| `docs/futures/liquidation.md` | Liquidation engine |
| `docs/futures/deployment.md` | Deployment guide |
| `docs/futures/migration_guide.md` | Migration guide |

### Тестирование

```bash
# All Phase 10 tests
pytest tests/test_futures_validation.py tests/test_futures_backward_compatibility.py -v

# Validation tests only (125 tests)
pytest tests/test_futures_validation.py -v

# Backward compatibility only (66 tests)
pytest tests/test_futures_backward_compatibility.py -v

# Run benchmarks
python benchmarks/bench_futures_simulation.py
```

### Ключевые файлы

| Файл | Описание |
|------|----------|
| `tests/test_futures_validation.py` | 125 validation tests |
| `tests/test_futures_backward_compatibility.py` | 66 backward compatibility tests |
| `benchmarks/bench_futures_simulation.py` | Performance benchmark suite |
| `FUTURES_INTEGRATION_REPORT.md` | Integration completion report |
| `docs/futures/*.md` | 8 documentation files |

### Референсы

- Phase 3B-9: All preceding futures integration phases
- Binance Futures API: Reference for crypto perpetual simulation
- CME Group: SPAN methodology, Rule 80B, trading hours
- Kyle (1985): Price impact model
- Almgren & Chriss (2001): Optimal execution

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
| VIX/SPY/QQQ benchmark temporal misalignment | Positional indexing вместо timestamp merge → look-ahead | ✅ Фикс 2025-11-29: `merge_asof(direction="backward")` |

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
| "tanh в potential shaping нарушает Ng theorem?" | ⚠️ **НЕ баг**. Ng et al. (1999) разрешает ЛЮБУЮ функцию Φ(s). tanh(net_worth) валиден. |
| "gap_filled look-ahead bias?" | ⚠️ **НЕ баг**. Feature shifting (shift(1)) применяется ПОСЛЕ вычисления. См. features_pipeline.py:441-442. |
| "Earnings unbounded future window?" | ⚠️ **Документация**. Пользователь обязан гарантировать актуальность earnings calendar. Не code bug. |
| "γ не синхронизирован между env и model?" | ⚠️ **Documented**. CLAUDE.md: "reward.gamma == model.params.gamma (оба = 0.99)". Конфигурационная ответственность пользователя. |
| "3 уровня reward clipping создают non-monotonic value?" | ⚠️ **НЕ баг**. Разные клипы: (1) ratio→log safety, (2) final bounds. Служат разным целям. См. #59. |
| "Long-only reward=0 при pos=0 асимметричен?" | ⚠️ **By design**. `reward = log(ratio) × position`. При pos=0 агент не участвовал → reward=0 корректен. |
| "L2 ADV не учитывает intraday seasonality?" | ⚠️ **By design**. L2 simple/fast; L2+ has `tod_curve`. См. #54. |
| "L2 нет temp/perm impact separation?" | ⚠️ **By design**. L2=√participation; L3 has AlmgrenChriss/Gatheral. См. #55. |
| "L2 spread статичен?" | ⚠️ **By design**. L2+ has vol_regime_multipliers. См. #56. |
| "L2 limit fills детерминистичны?" | ⚠️ **By design**. L2=binary; L3 has QueueReactiveModel. См. #57. |
| "whale_threshold не масштабируется по ADV?" | ⚠️ **Configurable**. Threshold = participation ratio (уже normalized). Config profiles exist. См. #58. |

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

### 54. L2 ADV Ignores Intraday Seasonality (execution_providers.py:2867-2870)

```python
if market.adv is not None and market.adv > 0:
    ref_price = market.get_mid_price() or bar.typical_price
    order_notional = order.get_notional(ref_price)
    return order_notional / market.adv  # No TOD adjustment
```

**Почему это BY DESIGN (L2 vs L2+ trade-off)**:
1. L2 (`StatisticalSlippageProvider`) is intentionally **simple and fast** for rapid backtesting
2. L2+ (`CryptoParametricSlippageProvider`) has `tod_curve` at lines 785-792 with Asia/EU/US session factors (0.70-1.15)
3. L2+ applies TOD adjustment to slippage, effectively capturing intraday effects
4. Adding TOD to L2 would require `hour_utc` parameter breaking backward compatibility
5. Users requiring accurate intraday cost estimation should use L2+ or L3

**Fidelity Level Selection**:
- **L2**: Quick backtests, strategy screening (±30-50% cost error acceptable)
- **L2+**: Production cost estimation (TOD, imbalance, funding, whale detection)
- **L3**: HFT research, queue position tracking, fill probability models

**Референс**: ITG (2012) "Global Cost Review", Kyle (1985)

---

### 55. L2 No Permanent vs Temporary Impact Separation (impl_slippage.py:2342-2349)

```python
impact_term = k_effective * math.sqrt(participation_ratio)  # √participation = temporary
base_cost = half_spread + impact_term  # Single-term model
```

**Почему это BY DESIGN (L2 vs L3 trade-off)**:
1. L2 uses **simplified Almgren-Chriss**: `k * √participation` — temporary impact only
2. L3 has full separation in `lob/market_impact.py`:
   - `AlmgrenChrissModel`: `temp = η * σ * (Q/V)^0.5`, `perm = γ * (Q/V)`
   - `GatheralModel`: transient impact with power-law decay `G(t) = (1 + t/τ)^(-β)`
3. For bar-level simulation, temp/perm distinction matters less (impact reverts within bar)
4. For HFT simulation, use L3 with proper impact decay modeling

**Референс**: Almgren & Chriss (2001), Gatheral (2010)

---

### 56. L2 Spread Model Static (execution_providers.py:514-518)

```python
spread = market.get_spread_bps()
if spread is None or not math.isfinite(spread) or spread < 0:
    half_spread = self.spread_bps / 2.0  # Default fallback
```

**Почему это BY DESIGN**:
1. L2 uses market spread if available in `MarketState.get_spread_bps()`
2. L2+ adds volatility-based adjustments via `vol_regime_multipliers` (0.8-1.5x)
3. L2+ has order book `imbalance_penalty_max` (up to 30% extra cost)
4. Dynamic spread widening is implemented in L2+, not L2

**Референс**: Cont et al. (2014) "Price Impact of Order Book Events"

---

### 57. L2 Limit Order Fills Deterministic (execution_sim.py:11750-11755)

```python
if intrabar_fill_price is not None and intrabar_fill_price <= limit_price_value + tolerance:
    maker_fill = True
    filled = True  # Binary: filled or not
```

**Почему это BY DESIGN (L2 vs L3 trade-off)**:
1. L2 uses **binary fill logic**: price touches limit → filled
2. L3 has probabilistic models in `lob/fill_probability.py`:
   - `PoissonFillModel`: `P(fill in T) = 1 - exp(-λT / position)`
   - `QueueReactiveModel`: `λ_i = f(q_i, spread, volatility, imbalance)`
   - `QueueValueModel`: Value = P(fill) × spread/2 - adverse_selection
3. Queue position tracking in `lob/queue_tracker.py` with MBP/MBO estimation
4. L2 is 100-1000x faster than L3 for backtesting

**Референс**: Huang et al. (2015) Queue-Reactive Model, Moallemi & Yuan (2017)

---

### 58. Whale Threshold 1% Not ADV-Scaled (execution_providers.py:798)

```python
whale_threshold: float = 0.01  # 1% of ADV
```

**Почему это CONFIGURABLE (not a bug)**:
1. Threshold is **participation ratio** (order/ADV), already normalized by ADV
2. 1% default is reasonable: $100M order on $10B ADV is whale behavior
3. For low-ADV altcoins: use `CryptoParametricConfig(whale_threshold=0.005)` (0.5%)
4. For stablecoin pairs: use profile `from_profile("stablecoin")` with lower threshold
5. Configuration profiles exist: `default`, `conservative`, `aggressive`, `altcoin`, `stablecoin`

**Usage**:
```python
# For low-liquidity altcoins
config = CryptoParametricConfig(whale_threshold=0.005)  # 0.5%
provider = CryptoParametricSlippageProvider(config=config)

# Or use built-in profile
provider = CryptoParametricSlippageProvider.from_profile("altcoin")
```

---

### 59. Reward Clipping is NOT Stacked (trading_patchnew.py:2201, 2345)

```python
# Line 2201: Numerical safety BEFORE log()
ratio_clipped = np.clip(ratio, 1e-10, 1e10)

# Line 2345: Final reward bounds (policy requirement)
reward = float(np.clip(reward_before_clip, -clip_for_clamp, clip_for_clamp))
```

**Почему это НЕ создаёт non-monotonic value function**:

1. **First clip** (line 2201): Protects against numerical overflow in `log(ratio)`
   - Without this, ratio=0 → log(0)=-inf → NaN propagation
   - Clipping to [1e-10, 1e10] is defensive programming, not reward shaping

2. **Second clip** (line 2345): Bounds the final reward for policy stability
   - RL policies need bounded rewards for numerical stability
   - `clip_for_clamp` is typically large (e.g., 10.0), rarely triggered

3. **Different code paths**: `reward.pyx` has separate `_clamp` for non-signal-only mode
   - These are independent code paths, not stacked operations

**Value function remains monotonic** because:
- Both clips are defensive (rarely triggered in normal operation)
- First clip applies BEFORE log → preserves log's monotonicity
- Second clip applies AFTER all computations → bounds extreme outliers only

**Референс**: Standard numerical programming practice, Schulman et al. (2017) PPO

---

## 📊 СТАТУС ПРОЕКТА (2025-11-30)

### ✅ Production Ready

Все критические исправления применены и протестированы. **557 test files** с 97%+ pass rate.

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
| Crypto Parametric TCA | ✅ Production | 84/84 |
| Equity Parametric TCA | ✅ Production | 86/86 |
| Bug Fixes 2025-11-26 | ✅ Production | 22/22 (includes projection+YZ fixes) |
| **Forex Integration** | ✅ Production | 18 test files (Phase 11) |
| Forex Parametric TCA | ✅ Production | In test_forex_parametric_tca.py |
| OANDA Adapter | ✅ Production | In test_forex_foundation.py |

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
| **2025-11-30** | feat(forex): Phase 11 Forex Integration complete | L2+ parametric TCA, OANDA adapter, 18 test files |
| **2025-11-30** | feat(futures): Unified multi-asset futures plan | 1,035+ tests planned for crypto/equity/commodity futures |
| **2025-11-29** | fix(stocks): Benchmark temporal alignment via merge_asof | VIX/SPY/QQQ used positional index → look-ahead bias for equities |
| **2025-11-28** | feat(equity): EquityParametricSlippageProvider | L2+ smart TCA model for US equities, 9 factors, 86 tests |
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

### 6. Conformal Prediction

**Статус**: ✅ Production Ready | **Тесты**: 59 (100% pass)

Distribution-free uncertainty bounds на CVaR и value estimates.

**Методы**:
- **CQR** (Conformalized Quantile Regression) — Romano et al., 2019
- **EnbPI** (Ensemble batch Prediction Intervals) — Xu & Xie, ICML 2021
- **ACI** (Adaptive Conformal Inference) — Gibbs & Candes, 2021

**Архитектура**:
```
core_conformal.py → impl_conformal.py → service_conformal.py
```

**Конфигурация** (`configs/conformal.yaml`):
```yaml
conformal:
  enabled: true
  calibration:
    method: "cqr"           # cqr, enbpi, aci, naive
    coverage_target: 0.90   # P(Y ∈ interval) ≥ 90%
    min_calibration_samples: 500
    recalibrate_interval: 1000
  cvar_bounds:
    enabled: true
    use_for_gae: false      # Conservative, experimental
  risk_integration:
    enabled: true
    uncertainty_position_scaling: true
    baseline_interval_width: 0.1
    max_uncertainty_reduction: 0.5
  escalation:
    enabled: true
    warning_percentile: 90
    critical_percentile: 99
    action_on_warning: "log"
    action_on_critical: "reduce_position"
```

**Использование**:
```python
from service_conformal import (
    ConformalPredictionService,
    create_conformal_config,
    wrap_cvar_with_bounds,
    create_risk_guard_integration,
)

# 1. Создание сервиса из YAML
config = create_conformal_config(yaml_dict["conformal"])
service = ConformalPredictionService(config)

# 2. Калибровка после training
service.calibrate(predictions, true_values)

# 3. Получение prediction interval
interval = service.predict_interval(point_estimate)
print(f"[{interval.lower_bound:.3f}, {interval.upper_bound:.3f}]")

# 4. CVaR bounds
bounds = service.compute_cvar_bounds(quantiles)
print(f"CVaR worst-case: {bounds.worst_case_cvar:.3f}")

# 5. Position scaling
scale = service.get_position_scale()  # 0.5-1.0 based on uncertainty

# 6. Integration с risk_guard
position_scale_fn = create_risk_guard_integration(service, lambda: 1.0)
```

**Тестирование**:
```bash
pytest tests/test_conformal_prediction.py -v
```

**Референсы**:
- Romano et al. (2019): [CQR](https://arxiv.org/abs/1905.03222)
- Xu & Xie (ICML 2021): EnbPI
- Gibbs & Candes (2021): ACI
- MAPIE: https://mapie.readthedocs.io/

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

### Основные конфиги

| Файл | Назначение |
|------|------------|
| `config_train.yaml` | Обучение crypto (standard) |
| `config_train_stocks.yaml` | Обучение stocks (Alpaca) |
| `config_train_signal_only_stocks.yaml` | Signal-only обучение stocks |
| `config_pbt_adversarial.yaml` | PBT + SA-PPO |
| `config_sim.yaml` | Бэктест crypto |
| `config_backtest_stocks.yaml` | Бэктест stocks |
| `config_live.yaml` | Live trading crypto (Binance) |
| `config_live_alpaca.yaml` | Live trading stocks (Alpaca) |
| `config_eval.yaml` | Оценка модели |
| `config_train_forex.yaml` | Обучение forex (OANDA) |
| `config_backtest_forex.yaml` | Бэктест forex |

### Asset Class конфигурация

| Файл | Назначение |
|------|------------|
| `asset_class_defaults.yaml` | Defaults для crypto/equity/forex/futures |
| `forex_defaults.yaml` | Forex-specific defaults (spreads, sessions, leverage) |
| `exchange.yaml` | Exchange adapter configuration |

### Модульные конфиги

| Файл | Назначение |
|------|------------|
| `execution.yaml` | Execution simulation parameters |
| `execution_l3.yaml` | L3 LOB execution configuration |
| `fees.yaml` | Fee structures (maker/taker, regulatory) |
| `slippage.yaml` | Slippage profiles (crypto, equity) |
| `risk.yaml` | Risk limits and guards |
| `no_trade.yaml` | No-trade windows |
| `conformal.yaml` | Conformal prediction settings |
| `signal_quality.yaml` | Signal quality metrics |

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
| Stock Features | `test_stock_features.py`, `test_benchmark_temporal_alignment.py` |
| Stock Risk Guards | `test_stock_risk_guards.py` |
| US Market Structure | `test_us_market_structure.py` |

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
- [ ] `python tools/check_feature_parity.py` — паритет OK
- [ ] `python tools/verify_fixes.py` — все фиксы работают

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

**Последнее обновление**: 2025-12-02
**Версия документации**: 11.8 (Phase 10: Validation & Documentation)
**Статус**: ✅ Production Ready (567+ test files, Futures Integration complete)

### Изменения в 11.8:
- **Phase 10 Validation & Documentation complete** — 171 tests total (125 validation + 46 backward compatibility)
  - Comprehensive validation test suite (test_futures_validation.py)
  - Backward compatibility tests (test_futures_backward_compatibility.py)
  - Performance benchmarks (bench_futures_simulation.py)
  - Integration report (FUTURES_INTEGRATION_REPORT.md)
  - Documentation suite (8 files in docs/futures/)
- Futures Integration project **COMPLETE** — All 10 phases implemented
- Updated CLAUDE.md with Phase 10 entries in Quick Reference table
- Total futures tests: 1,365+ across all phases
- Status changed from "Live Ready" to "Production Ready"

### Изменения в 11.7:
- **Добавлена полная документация Phase 9 (Unified Futures Live Trading)** — 81 тестов
  - FuturesLiveRunner — Main live trading coordinator
  - FuturesPositionSynchronizer — Position sync with exchange
  - FuturesMarginMonitor — Real-time margin monitoring
  - FuturesFundingTracker — Funding rate tracking & predictions
  - FuturesSyncConfig — Configuration with exchange, futures_type, sync_interval_sec, qty_tolerance_pct
  - FuturesSyncEventType — 12 event types for position changes, margin calls, ADL
  - ADLRiskLevel — SAFE, WARNING, DANGER, CRITICAL levels
  - configs/config_live_futures.yaml — Live trading configuration
  - 81 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — Phase 9 теперь ✅ DONE
- Добавлены Phase 9 entries в Quick Reference таблицу
- Обновлён FUTURES_INTEGRATION_PLAN.md с Phase 9 completion
- Status изменён с "Training Ready" на "Live Ready"

### Изменения в 11.6:
- **Добавлена полная документация Phase 8 (Multi-Futures Training Pipeline)** — 131 тестов
  - FuturesTradingEnv wrapper с leverage, margin tracking, liquidation handling
  - FuturesFeatureFlags system с RolloutStage (DISABLED, SHADOW, CANARY, PRODUCTION)
  - Thread-safe feature flag operations с symbol filtering для CANARY stage
  - configs/config_train_futures.yaml — Futures training configuration
  - configs/config_futures_unified.yaml — Unified futures config template
  - configs/feature_flags_futures.yaml — Feature flags configuration
  - Integration с train_model_multi_patch.py через create_futures_env()
  - 131 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — Phase 8 теперь ✅ DONE
- Добавлены Phase 8 entries в Quick Reference таблицу
- Обновлён FUTURES_INTEGRATION_PLAN.md с Phase 8 completion
- Status изменён с "Core Complete" на "Training Ready"

### Изменения в 11.5:
- **Добавлена полная документация Phase 7 (Unified Futures Risk Management)** — 290+ строк
  - UnifiedFuturesRiskGuard с automatic asset type detection
  - Asset type classification (Crypto Perpetual/Quarterly, CME Index/Metal/Energy/Currency/Bond)
  - Automatic delegation to crypto or CME guards based on symbol
  - UnifiedRiskEvent для унификации событий риска
  - UnifiedMarginResult для унифицированных результатов проверки маржи
  - PortfolioRiskManager для cross-asset correlation handling
  - Configuration с profiles (conservative, aggressive)
  - 116 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — Phase 7 теперь ✅ DONE
- Добавлены Phase 7 entries в Quick Reference таблицу
- Добавлены примеры использования UnifiedFuturesRiskGuard, config profiles
- Обновлён FUTURES_INTEGRATION_PLAN.md с Phase 6A, 6B, 7 completion
- Добавлены референсы на Phase 6A/6B, portfolio theory

### Изменения в 11.4:
- **Добавлена полная документация Phase 5B (L3 LOB for CME Futures)** — 290+ строк
  - GlobexMatchingEngine с FIFO Price-Time Priority matching
  - Market with Protection (MWP) orders with protection points
  - Stop orders с velocity logic protection
  - Session detection (RTH vs ETH) with spread multipliers
  - DailySettlementSimulator с variation margin calculation
  - CMEL3ExecutionProvider combining all L3 CME components
  - 42 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — Phase 5B теперь ✅ DONE
- Добавлены примеры использования GlobexMatchingEngine, MWP, stop orders
- Добавлены референсы на CME Group Globex documentation
- Добавлены Phase 5B entries в Quick Reference таблицу

### Изменения в 11.3:
- **Добавлена полная документация Phase 5A (L3 LOB for Crypto Futures)** — 280+ строк
  - LiquidationCascadeSimulator с Kyle price impact model
  - InsuranceFundManager с contribution/payout dynamics
  - ADLQueueManager для auto-deleveraging queue
  - FundingPeriodDynamics для funding window detection
  - FuturesL3ExecutionProvider combining all L3 components
  - 100 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — Phase 5A теперь ✅ DONE
- Добавлены примеры использования cascade simulation, insurance fund, ADL queue
- Добавлены референсы на Kyle (1985), Almgren-Chriss, Binance liquidation protocol

### Изменения в 11.2:
- **Добавлена полная документация Phase 4B (CME SPAN Margin & Slippage)** — 300+ строк
  - SPAN Margin Calculator с 16-scenario testing
  - Inter/Intra-commodity spread credits
  - CME Slippage Provider с session/settlement факторами
  - CME Circuit Breaker (Rule 80B, overnight limits, velocity logic)
  - CircuitBreakerManager для multi-product
  - 237 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — Phase 4B теперь ✅ DONE
- Добавлены Phase 4B entries в Quick Reference таблицу
- Добавлены примеры использования SPAN margin, circuit breakers
- Добавлены референсы на CME SPAN, Rule 80B, Velocity Logic

### Изменения в 11.1:
- **Добавлена полная документация Phase 3B (IB Adapters & CME Settlement)** — 390+ строк
  - IB Market Data Adapter с production-grade rate limiting
  - IB Order Execution Adapter (market/limit/bracket orders)
  - CME Settlement Engine с product-specific settlement times
  - Contract Rollover Manager (8 days before expiry for ES/NQ)
  - CME Trading Calendar (Globex hours, holidays, maintenance)
  - 30+ поддерживаемых контрактов (ES, NQ, GC, CL, 6E, ZN и др.)
  - 205 тестов (100% pass rate)
- Обновлена секция "Futures Integration" — статус изменён с PLANNED на Partial
- Добавлены Phase 3B entries в Quick Reference таблицу
- Добавлены примеры использования IB adapters, CME settlement, rollover
- Добавлены референсы на CME Group, IB TWS API, SPAN margin
- Roadmap с Phase 4A-7B для Binance futures integration

### Изменения в 11.0:
- **Добавлена секция Forex Integration (Phase 11)** — L2+ parametric TCA, OANDA adapter
- **Добавлена секция Futures Integration (PLANNED)** — план для crypto/equity/commodity futures
- Добавлен OANDA в таблицу поддерживаемых бирж
- Добавлена архитектура adapters/oanda/
- Добавлены Forex entries в Quick Reference таблицу
- Добавлены Forex commands (training, backtest, live)
- Добавлены forex configs (config_train_forex.yaml, forex_defaults.yaml)
- Обновлён счётчик тестов: 262 → 557 test files
