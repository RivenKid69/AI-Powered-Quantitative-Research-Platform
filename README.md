# AI-Powered Quantitative Research Platform

**ML-платформа для количественных исследований и торговли** на криптовалютах и акциях с использованием **Reinforcement Learning (Distributional PPO)** для принятия торговых решений.

---

## 🎯 Статус Проекта (2025-12-01)

**Версия**: 4.1 (Stable)
**Статус**: ✅ **Production Ready**
**Последнее обновление**: 2025-12-01

### Поддерживаемые рынки

| Рынок | Адаптер | Статус |
|-------|---------|--------|
| **Crypto** (Binance Spot/Futures) | `adapters/binance/` | ✅ Production |
| **US Equities** (Alpaca) | `adapters/alpaca/` | ✅ Production |
| **US Equities Data** (Polygon) | `adapters/polygon/` | ✅ Production |

### ✅ Последние Критические Исправления

- 🔴 **UPGD Learning Rate Multiplier** (2025-11-30) - **VERIFIED** ✅
  - `AdaptiveUPGD.step` confirmed to use correct `-1.0 * lr` multiplier.
  - **VERIFIED**: Code review confirmed fix at line 292 of `optimizers/adaptive_upgd.py`.

- 🔴 **Twin Critics Implementation** (2025-11-30) - **VERIFIED** ✅
  - `min(Q1, Q2)` logic confirmed in `distributional_ppo.py` and `custom_policy_patch1.py`.
  - Independent VF clipping for each critic verified.
  - **VERIFIED**: Comprehensive code audit confirmed implementation.

- 🔴 **VGS v3.2** (2025-11-30) - **VERIFIED** ✅
  - `min_scaling_factor` and `variance_cap` confirmed in `variance_gradient_scaler.py`.
  - Stochastic variance calculation `E[g^2]` confirmed fixed.

- 🔴 **PBT Optimizer State** (2025-11-30) - **VERIFIED** ✅
  - Optimizer state preservation during PBT exploit confirmed "HANDLED / BY DESIGN".

- 🔴 **Twin Critics Categorical VF Clipping** (2025-11-26) - **FIXED** ✅
  - `_project_distribution` was identity stub → proper C51 projection
  - Yang-Zhang RS denominator: used (n-1) instead of n → +11% inflation removed

- 🔴 **UPGDW Min-Max Normalization** (2025-11-26) - **FIXED** ✅
  - Negative utilities no longer invert weight protection

- 🔴 **Signal Position in Observation** (2025-11-26) - **FIXED** ✅
  - Temporal mismatch: market data t+1, position t → now both t+1

- 🔴 **LongOnlyActionWrapper** (2025-11-25) - **FIXED** ✅
  - Minimum position was 50% instead of 0% → policy now uses tanh

- 🔴 **Step Observation Timing** (2025-11-25) - **FIXED** ✅
  - Obs from same row as reset() → now Gymnasium-compliant

- 🔴 **Data Leakage** (2025-11-23) - **FIXED** ⚠️ **REQUIRES RETRAINING**
  - ALL technical indicators were NOT shifted → look-ahead bias removed

**⚠️ Переобучите модели**, если они обучены до 2025-11-26.

**Test Coverage**: **215+ тестов** (97%+ pass rate)

---

## 🚀 Основные Возможности

### RL Framework
- **Distributional PPO** с Twin Critics (default enabled)
- **AdaptiveUPGD Optimizer** - continual learning без catastrophic forgetting
- **VGS (Variance Gradient Scaler)** - автоматическое per-layer gradient scaling
- **PBT (Population-Based Training)** - эволюционная оптимизация гиперпараметров
- **SA-PPO (State-Adversarial PPO)** - robust training против distribution shift

### Execution Simulator
- Full LOB (Limit Order Book) simulation с микроструктурой
- Модели проскальзывания (linear, sqrt, calibrated)
- Учет комиссий (maker/taker, BNB discount)
- TTL (time-to-live) для лимитных заявок, TIF (GTC/IOC/FOK)
- Алгоритмические исполнители: TWAP, POV, VWAP

### Data & Features
- 63 features (price, volume, volatility, momentum, microstructure)
- Seasonality framework (168 часов недели)
- Data degradation simulation (stale data, drops, delays)
- Multi-symbol support с portfolio management

### Risk Management
- Position limits, leverage limits, stop-loss
- Daily loss limits, drawdown protection
- Operational kill switch
- No-trade windows (funding windows, custom intervals)

---

## 📚 Документация

### Главная Документация
- **[CLAUDE.md](CLAUDE.md)** - **Полная документация проекта** (Russian) ⭐ **Начните здесь!**
- **[docs/AI_GUIDE.md](docs/AI_GUIDE.md)** - **Guide for AI Agents** (English) 🤖 **For AI Assistants**
- **[DOCS_INDEX.md](DOCS_INDEX.md)** - Навигация по всей документации
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Архитектура системы
- **[QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)** - Быстрый старт

### Критические Исправления
- **[DATA_LEAKAGE_MIGRATION_GUIDE.md](DATA_LEAKAGE_MIGRATION_GUIDE.md)** - ⭐ **MIGRATION GUIDE** (2025-11-23) - **READ FIRST**
- **[DATA_LEAKAGE_FIX_REPORT_2025_11_23.md](DATA_LEAKAGE_FIX_REPORT_2025_11_23.md)** - ⭐ Data leakage fix (2025-11-23) **CRITICAL**
- **[CRITICAL_ANALYSIS_REPORT_2025_11_24.md](CRITICAL_ANALYSIS_REPORT_2025_11_24.md)** - ⭐ Twin Critics loss fix (2025-11-24)
- **[SA_PPO_BUG_FIXES_REPORT_2025_11_23.md](SA_PPO_BUG_FIXES_REPORT_2025_11_23.md)** - SA-PPO fixes (2025-11-23)
- **[VGS_E_G_SQUARED_BUG_REPORT.md](VGS_E_G_SQUARED_BUG_REPORT.md)** - VGS v3.1 fix (2025-11-23)
- **[GAE_OVERFLOW_PROTECTION_FIX_REPORT.md](GAE_OVERFLOW_PROTECTION_FIX_REPORT.md)** - GAE overflow (2025-11-23)
- **[TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md](TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md)** - Twin Critics VF Clipping (2025-11-22)
- **[CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md)** - Action space fixes (2025-11-21)
- **[NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md)** - LSTM + NaN fixes (2025-11-21)
- **[REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md)** - Обязательный checklist

### Продвинутые Возможности
- **[docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md)** - UPGD optimizer integration
- **[docs/twin_critics.md](docs/twin_critics.md)** - Twin critics architecture
- **[docs/seasonality.md](docs/seasonality.md)** - Seasonality framework

---

## ⚡ Быстрый Старт

### Установка

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements_extra.txt

# Build Cython modules
python setup.py build_ext --inplace
```

### Основные Команды

```bash
# Бэктест
python script_backtest.py --config configs/config_sim.yaml

# Обучение (standard)
python train_model_multi_patch.py --config configs/config_train.yaml

# Обучение (PBT + Adversarial)
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml

# Live trading (Crypto)
python script_live.py --config configs/config_live.yaml

# Live trading (Stocks - Alpaca)
python script_live.py --config configs/config_live_alpaca.yaml --paper
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours

# Evaluation
python script_eval.py --config configs/config_eval.yaml --all-profiles

# Тестирование
pytest tests/                          # Все тесты
pytest tests/test_critical*.py -v     # Критические тесты
```

### Stock Data Commands

```bash
# Скачать данные по акциям
python scripts/download_stock_data.py \
    --symbols AAPL MSFT GOOGL AMZN NVDA META TSLA SPY QQQ IWM GLD IAU SGOL SLV \
    --start 2020-01-01 --timeframe 1h --resample 4h

# Получить universe акций
python scripts/fetch_alpaca_universe.py --popular
```

### Обновление Данных

```bash
# Обновить universe символов
python -m services.universe --output data/universe/symbols.json

# Обновить биржевые фильтры
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json

# Обновить комиссии
python scripts/refresh_fees.py
```

---

## 🏗️ Архитектура

Проект использует **строгую слоистую архитектуру** с dependency injection:

```
core_ → impl_ → service_ → strategies → script_
```

- **core_*** - Базовые сущности, contracts, models
- **impl_*** - Конкретные реализации (execution, fees, slippage, etc)
- **service_*** - Бизнес-логика (backtest, train, eval, live trading)
- **strategies/** - Торговые стратегии
- **script_*** - CLI точки входа

**ВАЖНО**: Не нарушайте зависимости между слоями!

---

## 🧪 Тестирование

```bash
# Все тесты
pytest tests/

# Критические тесты
pytest tests/test_twin_critics_vf_clipping_correctness.py -v  # Twin Critics VF Clipping (2025-11-22)
pytest tests/test_lstm_episode_boundary_reset.py -v           # LSTM state reset (2025-11-21)
pytest tests/test_critical_action_space_fixes.py -v           # Action space fixes (2025-11-21)
pytest tests/test_nan_handling_external_features.py -v        # NaN handling (2025-11-21)

# По категориям
pytest tests/test_execution*.py -v     # Execution simulator
pytest tests/test_distributional_ppo*.py -v  # PPO implementation
pytest tests/test_upgd*.py -v          # UPGD optimizer
pytest tests/test_pbt*.py -v           # PBT scheduler
```

---

## 📖 Дополнительная Документация

Скрипты `script_*.py` и `train_model_multi_patch.py` выступают CLI-точками
входа в сервисы. Все они используют dependency injection и не содержат
бизнес-логики, ограничиваясь описанием аргументов и вызовом соответствующих
сервисов.

### Компоненты
- [docs/moving_average.md](docs/moving_average.md) - Moving average implementation
- [docs/pipeline.md](docs/pipeline.md) - Decision pipeline architecture
- [docs/bar_execution.md](docs/bar_execution.md) - Bar execution mode
  - JSON schemas: [envelope](docs/spot_signal_envelope.schema.json), [target_weight](docs/spot_signal_target_weight.schema.json), [delta_weight](docs/spot_signal_delta_weight.schema.json)
- [docs/large_orders.md](docs/large_orders.md) - Large order execution (TWAP, POV, VWAP)
- [docs/no_trade.md](docs/no_trade.md) - No-trade windows
- [docs/data_degradation.md](docs/data_degradation.md) - Data degradation simulation

## File Ownership and Permissions

Информация о ролях, владении файлами и необходимых правах доступа приведена в [docs/permissions.md](docs/permissions.md).

## Установка зависимостей для скриптов

Скрипты построения и валидации данных полагаются на библиотеки
`pandas` и `numpy`. Их можно установить вместе с прочими дополнительными
зависимостями:

```bash
pip install -r requirements_extra.txt
# или через extras из pyproject.toml
pip install ".[extra]"
```

## Примеры запуска

Сравнить результаты нескольких запусков можно с помощью
`script_compare_runs.py`. Передайте ему пути к каталогам или файлам
`metrics.json`. По умолчанию таблица будет сохранена в
`compare_runs.csv`, а флаг `--stdout` выведет её в консоль.

```bash
python script_backtest.py --config configs/config_sim.yaml
python train_model_multi_patch.py --config configs/config_train.yaml \
  --regime-config configs/market_regimes.json \
  --liquidity-seasonality data/latency/liquidity_latency_seasonality.json
python script_compare_runs.py run1 run2 run3            # сохранит compare_runs.csv
python script_compare_runs.py run1 metrics.json --stdout  # вывод в stdout
python script_fetch_exchange_specs.py --market futures --symbols BTCUSDT,ETHUSDT --out data/exchange_specs.json
python scripts/validate_seasonality.py --historical path/to/trades.csv --multipliers data/latency/liquidity_latency_seasonality.json
```

### Полный цикл обновления и инференса

Для запуска полного пайплайна без отдельного YAML используйте
`scripts/run_full_cycle.py`. Скрипт сначала выгружает и агрегирует свечи через
`ingest_orchestrator`, а затем запускает обновление данных и инференс сигналов из
`update_and_infer.py`.

Пример одноразового запуска с базовым интервалом `1m` и агрегацией в `5m` и
`15m`:

```bash
python scripts/run_full_cycle.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1m,5m,15m \
  --start 2024-01-01 --end 2024-12-31 \
  --prepare-args "--config configs/config_train.yaml" \
  --infer-args "--config configs/config_live.yaml"
```

Основные аргументы CLI:

- `--symbols` -- список символов (через запятую или несколько флагов).
- `--interval` -- первый интервал выгружается напрямую, остальные агрегируются
  из базового.
- `--start` / `--end` -- границы окна дат для скачивания свечей.
- `--loop` и `--sleep-min` -- включают бесконечный цикл с паузой между
  итерациями.
- `--prepare-args`, `--infer-args` -- дополнительные параметры, передаваемые
  соответственно в `prepare_advanced_data.py`/`prepare_and_run.py` и
  `infer_signals.py`.
- Дополнительные флаги (`--klines-dir`, `--futures-dir`, `--prices-out` и т.д.)
  переопределяют пути и задержки, повторяя настройки `ingest_orchestrator`.

`update_and_infer.py` по-прежнему читает переменные окружения (`SYMS`,
`LOOP`, `SLEEP_MIN`, `EVENTS_DAYS`, `SKIP_EVENTS`, `EXTRA_ARGS_PREPARE`,
`EXTRA_ARGS_INFER`) при прямом запуске:

```bash
SYMS=BTCUSDT,ETHUSDT LOOP=1 SLEEP_MIN=30 python update_and_infer.py
```

Runners load the symbol universe from ``data/universe/symbols.json`` by default.
Override it with the ``--symbols`` CLI flag or an explicit ``data.symbols``
entry in the YAML configuration.

### Intrabar price configuration

Simulation and training configs now ship with explicit intrabar execution
settings.  The block ``execution`` in
[`configs/config_sim.yaml`](configs/config_sim.yaml) defines:

- ``intrabar_price_model`` -- choose the price sampling mode.  ``bridge`` keeps
  the legacy Brownian bridge sampling, while ``reference`` uses an external M1
  reference feed for deterministic fills.
- ``timeframe_ms`` -- bar length in milliseconds.  Set it to ``3600000`` for H1
  profiles to align latency fractions with hourly candles.
- ``reference_prices_path`` -- optional path to the precomputed M1 reference
  dataset required by the ``reference`` mode.  Leave it ``null`` when using
  ``bridge``/``linear`` sampling.

The nested ``execution.bridge`` block mirrors the same fields for scenarios
where the simulator works as a bridge adapter.  Override the values there when
the adapter consumes a different intrabar data source than the main simulator.

The list is managed by ``services/universe.py`` which caches Binance spot
symbols trading against USDT.  The cache is refreshed on first use if it is
missing or older than 24 hours and can be updated manually:

```bash
python -m services.universe --output data/universe/symbols.json --liquidity-threshold 1e6
```

To confirm the refresh succeeded, inspect the modification time and a sample
of the cached symbols:

```bash
python - <<'PY'
import json, os, time
path = "data/universe/symbols.json"
print("age_s", round(time.time() - os.path.getmtime(path), 1))
with open(path, "r", encoding="utf-8") as fh:
    symbols = json.load(fh)
print("first", symbols[:5])
print("count", len(symbols))
PY
```

Runners resolve the same list through ``core_config.get_symbols``.  Load a
configuration to verify the symbols wired into the service:

```bash
python - <<'PY'
from core_config import load_config
cfg = load_config("configs/config_live.yaml")
print("runner_symbols", cfg.data.symbols[:5])
PY
```

Schedule the command daily via cron or rely on the automatic refresh at
startup.  Use ``--liquidity-threshold 0`` to bypass the volume filter or
point ``--output`` to maintain a custom symbols file.  See
[docs/universe.md](docs/universe.md) for details.

### Bar-mode quickstart

Bar execution swaps the per-order runtime for deterministic bar-level
rebalances driven by signed signal envelopes.  The recommended starting point
is [`configs/runtime_trade.yaml`](configs/runtime_trade.yaml); include the
snippet below in your live or simulation runtime overrides to enable the mode:

```yaml
portfolio:
  equity_usd: 1_000_000.0
costs:
  taker_fee_bps: 7.5
  half_spread_bps: 1.5
  impact:
    sqrt_coeff: 15.0
    linear_coeff: 2.5
execution:
  mode: bar
  bar_price: close
  min_rebalance_step: 0.05
```

CLI runners expose the same knobs for quick experimentation.  The following
command flips a backtest into bar mode with custom economics and portfolio
size, keeping everything else from the YAML unchanged:

```bash
python script_backtest.py --config configs/config_sim.yaml \
  --execution-mode bar --execution-bar-price close \
  --portfolio-equity-usd 1_000_000 \
  --costs-taker-fee-bps 7.5 --costs-half-spread-bps 1.5 \
  --costs-impact-sqrt 15 --costs-impact-linear 2.5
```

Signals delivered to the bar executor must follow the
[spot signal envelope](docs/bar_execution.md#signal-envelope) contract.  Each
payload carries pre-computed economics (``edge_bps``, ``cost_bps``, ``net_bps``,
``turnover_usd``) alongside an ``act_now`` flag.  The executor rechecks the net
edge after subtracting the optional ``execution.safety_margin_bps`` buffer; only
signals with positive ``net_bps`` and non-zero turnover keep ``act_now=True`` and
immediately enter the schedule.

### Обновление биржевых фильтров и спецификаций

JSON-файлы `binance_filters.json` и `exchange_specs.json` теперь содержат блок
`metadata` с подробной диагностикой. Для фильтров фиксируются момент выгрузки
(`built_at`), источник (`source`) и количество символов (`symbols_count`).
Спецификации добавляют отметку времени (`generated_at`), имя датасета
(`source_dataset`) и версию (`version`). Эти поля помогают быстро проверить,
какие данные были загружены и из какого окружения.

Сформировать свежие фильтры и спецификации можно так:

```bash
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python script_fetch_exchange_specs.py --market futures --symbols BTCUSDT,ETHUSDT --out data/exchange_specs.json
```

Флаг `--universe` принимает необязательный путь: `--universe` без аргумента
подтягивает `data/universe/symbols.json`, а `--universe custom.json` позволяет
использовать собственный список. Дополнительно можно перечислить тикеры через
позиционные аргументы или вообще отказаться от фильтра по вселенной. Ключ
`--dry-run` печатает план запросов и статистику `RestBudgetSession` без
обращений к API. Скрипт перезаписывает целевой JSON атомарно, чтобы
долгосрочные сервисы могли читать его без гонок.

`script_fetch_exchange_specs.py` поддерживает расширенный набор флагов:
`--volume-threshold` и `--days` управляют минимальным объёмом и окном расчёта,
`--volume-out` сохраняет метрики ликвидности, `--shuffle` перемешивает порядок
запросов, а `--checkpoint-path` вместе с `--resume/--no-resume` позволяет
возобновлять прерванные сессии. Опция `--rest-budget-config` подключает YAML с
лимитами REST-квоты, а `--dry-run` формирует сводку запросов без сетевых
вызовов.

Параметр `auto_refresh_days` в YAML-конфигурациях (`quantizer.auto_refresh_days`)
задаёт допустимый возраст фильтров. После превышения порога квантайзер
помечает файл как устаревший, записывает возраст, размер и SHA-256 в логи и
экспортирует показатель `filters_age_days`. При активном
`quantizer.refresh_on_start` квантайзер автоматически вызовет
`scripts/fetch_binance_filters.py`, если файл отсутствует или устарел. Без
`refresh_on_start` сервис ограничится предупреждением, оставляя обновление на
ручной запуск или расписание.

Реакция на предупреждения зависит от окружения. При `auto_refresh_days > 0`
квантайзер только логирует устаревшие фильтры; рекомендуется сразу запустить
CLI, чтобы избавиться от предупреждений. Переменная
`TB_FAIL_ON_STALE_FILTERS=1` ужесточает политику -- симулятор или сервис
завершится с ошибкой, если фильтры старше порога. После обновления убедитесь,
что логи сообщают актуальный `age_days`, `size_bytes` и `sha256`; это сигнал,
что файл подхвачен и предупреждения исчезли.

Простой пример ежедневного обновления через `cron`:

```cron
15 4 * * * /bin/bash -lc 'cd /opt/ai-quant-platform && python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json'
```

Скрипт `scripts/validate_seasonality.py` воспроизводит почасовое поведение
ликвидности, спреда и задержек и сравнивает его с историческим датасетом.
Проверка завершится ошибкой, если максимальное относительное отклонение
превысит допуск `--threshold` (по умолчанию 10%).
Подробные шаги и критерии приёмки описаны в [docs/seasonality_QA.md](docs/seasonality_QA.md).

### Обновление таблицы комиссий

`scripts/refresh_fees.py` автоматизирует обновление `data/fees/fees_by_symbol.json`.
Скрипт использует модуль :mod:`binance_fee_refresh`, чтобы прочитать активные
спот-символы из `exchangeInfo`, подтянуть ставки maker/taker через публичные
endpoint'ы Binance и пересобрать JSON c метаданными. Укажите
`BINANCE_API_KEY`/`BINANCE_API_SECRET` (или аргументы `--api-key` / `--api-secret`),
чтобы вместо публичного снимка использовать приватный endpoint `tradeFee`. Если
Binance публикует CSV-файл, его можно передать через `--csv`. Режим `--dry-run`
печатает пример диффа без перезаписи файла, а встроенная проверка предупреждает,
если предыдущая версия моложе 30 дней.

`FeesImpl` автоматически инициирует тот же рефреш при старте, если локальная
таблица отсутствует, повреждена или устарела (по умолчанию старше 30 дней).
Авто-режим работает на публичных endpoint'ах и, при успехе, заполняет
базовые ставки, BNB-множители и VIP-уровень даже в signal-only окружении. Чтобы
отключить сетевой вызов, задайте `BINANCE_PUBLIC_FEES_DISABLE_AUTO=1` или
явно пропишите `maker_bps`/`taker_bps`/`use_bnb_discount` в конфигурации.
Дополнительно поддерживаются переменные `BINANCE_FEE_SNAPSHOT_CSV`,
`BINANCE_PUBLIC_FEE_URL`, `BINANCE_FEE_TIMEOUT` и `BINANCE_BNB_DISCOUNT_RATE`
для тонкой настройки источника и таймаутов.


Блок `fees.rounding` в YAML-конфигурациях управляет пост-обработкой комиссий.
При `enabled: true` движок стремится округлять рассчитанные комиссии вверх до
ближайшего шага (`mode: "up"`) или по выбранной стратегии (`nearest`, `down`).
Шаг можно задать явно через `rounding.step`, но безопаснее оставить `null` --
в этом случае используется `commission_step`, извлечённый из биржевых фильтров
(`quantizer` анализирует `commissionStep`, `commissionPrecision` и
`quotePrecision`), либо из `symbol_fee_table.*.quantizer.commission_step`.
Дополнительные поля `minimum_fee`/`maximum_fee` задают жёсткие границы после
округления, а `per_symbol` позволяет переопределять правила по отдельным
тикерам.

Параметр `fees.settlement.enabled` управляет альтернативной валютой оплаты
комиссий. Например, `mode: "bnb"` и `currency: "BNB"` заставят симулятор
моделировать списание комиссий в BNB, включая округление по шагу и учёт
скидочной логики Binance (`prefer_discount_asset: true`). При `enabled: false`
комиссии остаются в котируемой валюте, как в предыдущих версиях.
=======
### Проверка округления комиссий

Скрипт `scripts/verify_fees.py` загружает `FeesImpl`/`Quantizer` и сравнивает
комиссии симулятора с эталонным значением, округлённым до шага комиссии.
По умолчанию проверяются пары `BTCUSDT`, `ETHUSDT` и `BNBUSDT`, но список можно
задать через `--symbols` (передайте `ALL`, чтобы обойти весь набор фильтров).
Количество случайных сделок на символ задаётся опцией `--samples`, а флаг
`--settlement-mode` позволяет принудительно включить расчёт комиссий в BNB
(`--bnb-price` задаёт курс конвертации). Все обнаруженные расхождения больше
одного шага комиссии выводятся в лог.


### Offline REST budget configuration

Файл `configs/offline.yaml` содержит общие параметры для офлайн-скриптов,
использующих `services.rest_budget.RestBudgetSession`. Ключ `rest_budget`
сгруппирован по блокам:

- `limits.global` задаёт базовый токен-бакет (`qps`, `burst`) и паузы
  (`jitter_ms`, `cooldown_sec`).
- `limits.endpoints` позволяет переопределять квоты и вспомогательные параметры
  для конкретных REST-маршрутов Binance (например, `exchangeInfo.min_refresh_days`).
- `cache` управляет путём к каталогу, режимом (`read`, `read_write`, `off`) и TTL
  кэшированных ответов.
- `checkpoint.enabled`/`checkpoint.path` переключают сохранение прогресса и путь
  до файла контрольной точки.
- `concurrency.workers` и `concurrency.batch_size` ограничивают число рабочих
  потоков и размер очереди задач внутри `RestBudgetSession`.
- `shuffle.enabled` включает перемешивание очереди символов, чтобы равномернее
  распределять нагрузку между запусками.
- Флаг `dynamic_from_headers` разрешает автоматически подстраивать вес запросов
  согласно заголовкам Binance, если они присутствуют.

Дополнительно файл описывает подготовленные датасеты в секции `datasets`.
Каждый из сплитов `train`/`val`/`test` содержит тег версии, временные границы
источников данных и вложенные артефакты (`seasonality`, `adv`, `fees`) с
описанием диапазона входных данных, путей выгрузки и контрольных хэшей.

Параметры симуляции можно временно переопределить через CLI:

```bash
python train_model_multi_patch.py --config configs/config_train.yaml --slippage.bps 5 --latency.mean_ms 50
```

Дополнительно доступны опции `--regime-config` и `--liquidity-seasonality`,
позволяющие указать пути к откалиброванным JSON-файлам с параметрами
рыночных режимов и сезонностью ликвидности и задержек соответственно. По
умолчанию используются файлы из каталога `data/latency/`, где
`liquidity_latency_seasonality.json` содержит массивы `liquidity` и
`latency` для 168 часов недели. Файл может быть как плоским, так и
вложенным по символам, например `{ "BTCUSDT": {"liquidity": [...], "latency": [...] }}`.
Шаблон с единичными множителями по-прежнему доступен в
`configs/liquidity_latency_seasonality.sample.json`.

### PopArt отключён

Ранее `DistributionalPPO` поддерживал оффлайн PopArt-регулятор, однако его
использование приводило к существенным накладным расходам и сложной
эксплуатации. В актуальной версии проекта PopArt полностью отключён: любые
параметры в секции `value_scale_controller` конфигураций игнорируются, а
привязанные holdout-лоадеры больше не создаются. Метрики `config/popart/*`,
`shadow_popart/*`, `popart/*` и `gate/*` недоступны, а обновление статистик
нормализации выполняется стандартными средствами PPO без внешнего контроллера.

### Параметры исполнения

Блоки `execution_params` и `execution_config` в YAML управляют поведением
симулятора при выставлении заявок.

- `execution_params.limit_offset_bps` -- задаёт смещение лимитной цены от
  mid-цены, если используется профиль `LIMIT_MID_BPS`. Значение указывается
  в базисных пунктах; положительное смещение смещает цену «хуже» рынка,
  отрицательное -- агрессивнее в сторону фила.
- `execution_params.ttl_steps` -- время жизни лимитного ордера в шагах
  симуляции. Когда счётчик достигает нуля, заявка автоматически отменяется.
  Нуль отключает автосписание. (Тесты реализованы в `test_limit_order_ttl.py`)
- `execution_params.tif` -- режим исполнения (`GTC`, `IOC`, `FOK`). Для `IOC`
  незаполненный остаток снимается немедленно, для `FOK` ордер либо
  исполняется полностью, либо отменяется.
- `execution_config.notional_threshold` -- порог крупного ордера. Заявки с
  нотоционалом выше порога будут исполняться алгоритмически.
- `execution_config.large_order_algo` -- алгоритм для крупных ордеров.
  Поддерживается `TWAP`, который дробит заявку на равные части.
- `execution_config.pov.participation` -- целевая доля участия в объёме
  рынка при исполнении крупного ордера через POV-алгоритм.
- `execution_config.pov.child_interval_s` -- минимальный интервал между
  дочерними заявками POV в секундах.
- `execution_config.pov.min_child_notional` -- минимальный нотоционал
  дочернего ордера; помогает избегать микроскопических сделок.

#### Definition of Done

- TTL-логика фиксирует снятие просроченных лимитных ордеров и отсутствуют
  fills с нарушением квантования (`ActionProto.ttl_steps` и журнал
  симуляции согласованы).
- Параметры `limit_offset_bps` и `tif` отражаются в отчётах исполнения.
  (Тесты реализованы в `test_execution_profiles.py`)
- Для крупного ордера превышение `notional_threshold` переводит исполнение
  в режим `large_order_algo`, а статистика POV-алгоритма фиксирует долю
  участия и интервал дочерних ордеров в `risk.jsonl`.

## Rate limiter configuration

`SignalRateLimiter` ограничивает частоту исходящих торговых сигналов, помогая
не превышать лимиты биржевого API. При достижении порога используется
экспоненциальный бэкофф.

Настройки задаются в YAML-конфигурациях:

- `max_signals_per_sec` -- максимальное число сигналов в секунду (0 отключает лимит);
- `backoff_base_s` -- базовая задержка экспоненциального бэкоффа в секундах;
- `max_backoff_s` -- максимальная задержка бэкоффа в секундах.

Эти параметры передаются в `Mediator` и затем в `SignalRateLimiter` для контроля
частоты торговых сигналов. Пример конфигурации:

```yaml
max_signals_per_sec: 5.0
backoff_base_s: 2.0
max_backoff_s: 60.0
```

После завершения эпизода агрегированная статистика записывается в `risk.jsonl`
в виде JSON-строки:

```json
{"etype": "SIGNAL_RATE_STATS", "total": 120, "delayed_ratio": 0.05, "rejected_ratio": 0.01}
```

При использовании публичных клиентов аналогичная сводка выводится в логи, например:

```
BinancePublicClient rate limiting: delayed=5.00% (6/120), rejected=1.00% (1/120)
```

## Operational kill switch reset

The operational kill switch persists its counters in `state/ops_state.json` and
sets a flag file at `state/ops_kill_switch.flag` when tripped. To recover
manually, remove the flag and reset the counters:

```bash
python scripts/reset_kill_switch.py
```

The script deletes the flag file and calls `ops_kill_switch.manual_reset()`.

## Large order execution

Orders whose notional exceeds `notional_threshold` are split by a deterministic algorithm.
Select the strategy with `large_order_algo` (`TWAP` or `POV`). POV accepts extra fields under `pov`:
`participation`, `child_interval_s`, and `min_child_notional`.

```yaml
notional_threshold: 10000.0
large_order_algo: POV
pov:
  participation: 0.2
  child_interval_s: 1
  min_child_notional: 1000.0
```

This configuration slices a 50k parent into 2k notional children every second, matching 20% of observed volume.
See [docs/large_orders.md](docs/large_orders.md) for additional examples and trajectories.
Parameters are deterministic but should be calibrated on historical data to align with market impact.


Сезонные множители позволяют масштабировать базовые значения ликвидности и
задержек для каждого часа недели (от понедельника 00:00 до воскресенья
23:00 UTC). Формат файла и процесс пересчёта коэффициентов из исторических
данных описаны в [docs/seasonality.md](docs/seasonality.md).
Краткие шаги для локального воспроизведения результатов приведены в
[docs/seasonality_quickstart.md](docs/seasonality_quickstart.md).

Те же значения можно задать в YAML-конфиге:

```yaml
slippage:
  bps: 5
latency:
  mean_ms: 50
```

### Деградация данных

`DataDegradationConfig` позволяет смоделировать пропуски и задержки в
потоке маркет-данных. Полный список полей описан в
[docs/data_degradation.md](docs/data_degradation.md).

Пример настройки в YAML-конфиге:

```yaml
data_degradation:
  stale_prob: 0.1      # вероятность повторить предыдущий бар
  drop_prob: 0.05      # вероятность пропустить бар
  dropout_prob: 0.2    # вероятность добавить задержку
  max_delay_ms: 50     # верхняя граница задержки
  seed: 42
```

Во время работы сервисы выводят сводку вида
`OfflineCSVBarSource degradation: ...`, `BinanceWS degradation: ...` или
`LatencyQueue degradation: ...` -- по этим сообщениям можно контролировать
доли пропусков и задержек.

Обработка окон **no-trade** описывается в конфигурации; подробности
см. [docs/no_trade.md](docs/no_trade.md).

Параллельные окружения и контроль случайности описаны в
[docs/parallel.md](docs/parallel.md).

### no-trade-mask утилита

Для предварительной фильтрации датасетов можно воспользоваться
консольным скриптом `no-trade-mask` (устанавливается вместе с пакетами
через `setup.py/pyproject.toml`). Он принимает путь к входным данным и
конфигурации с описанием окон `no_trade` и поддерживает два режима:

```bash
# удалить запрещённые интервалы
no-trade-mask --data data.csv --sandbox_config configs/legacy_sandbox.yaml --mode drop

# пометить строки train_weight=0.0, оставив их в датасете
no-trade-mask --data data.csv --sandbox_config configs/legacy_sandbox.yaml --mode weight
```

После выполнения утилита выводит процент заблокированных строк и сводку
`NoTradeConfig`. При указании `--histogram` дополнительно печатается
гистограмма длительностей блоков:

```bash
$ no-trade-mask --data data.csv --sandbox_config configs/legacy_sandbox.yaml --mode drop --histogram
Готово. Всего строк: 3. Запрещённых (no_trade): 2 (66.67%). Вышло: 1.
NoTradeConfig: {'funding_buffer_min': 5, 'daily_utc': ['00:00-00:05', '08:00-08:05', '16:00-16:05'], 'custom_ms': []}
Гистограмма длительностей блоков (минуты):
-0.5-0.5: 2
```

Загрузка настроек `no_trade` централизована: функция
`no_trade_config.get_no_trade_config()` считывает секцию `no_trade` из YAML-файла
и возвращает модель `NoTradeConfig`. Все модули используют её как единый
источник правды, исключая расхождения в трактовке конфигурации.
Подробнее о полях конфигурации и сценариях использования см. [docs/no_trade.md](docs/no_trade.md).

## Профили исполнения

В конфигурации можно описать несколько профилей исполнения. Каждый профиль
задаёт параметры симуляции и ожидаемое поведение выставляемых ордеров.

| Профиль       | `slippage_bps` | `offset_bps` | `ttl`, мс | `tif` | Поведение |
|---------------|----------------|--------------|-----------|-------|-----------|
| `conservative`| 5              | 2            | 5000      | GTC   | Пассивные лимитные заявки, ожидание исполнения |
| `balanced`    | 3              | 0            | 2000      | GTC   | Заявки около середины книги, умеренное ожидание |
| `aggressive`  | 1              | -1           | 500       | IOC   | Кроссует спред и быстро отменяет невыполненные заявки |

Пример YAML-конфига с переключением профиля:

```yaml
profile: balanced  # используется по умолчанию
profiles:
  conservative:
    slippage_bps: 5
    offset_bps: 2
    ttl: 5000
    tif: GTC
  balanced:
    slippage_bps: 3
    offset_bps: 0
    ttl: 2000
    tif: GTC
  aggressive:
    slippage_bps: 1
    offset_bps: -1
    ttl: 500
    tif: IOC
```

Скрипт `script_eval.py` позволяет выбрать конкретный профиль или
оценить все сразу:

```bash
python script_eval.py --config configs/config_eval.yaml --profile aggressive
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

Альтернативно режим оценки всех профилей можно включить непосредственно в
YAML-конфиге:

```yaml
all_profiles: true
input:
  trades_path: "logs/log_trades_<profile>.csv"
  equity_path: "logs/report_equity_<profile>.csv"
```

```bash
python script_eval.py --config configs/config_eval.yaml
```

При мульти-профильной оценке метрики (`Sharpe`, `PnL` и т.д.)
сохраняются отдельно для каждого профиля (`metrics_conservative.json`,
`metrics_balanced.json`, ...). Их следует интерпретировать как результаты
при соответствующих предположениях исполнения и сравнивать между
профилями.


## Проверка кривой проскальзывания

Скрипт `compare_slippage_curve.py` строит кривые `slippage_bps` по квантилям
размера ордера для исторических и симуляционных сделок и сравнивает их.
Если отклонение по каждой точке превышает допустимый порог, выполнение
заканчивается кодом ошибки.

```bash
python compare_slippage_curve.py hist.csv sim.csv --tolerance 5
```

Критерий акцептанса: абсолютное различие между средним `slippage_bps`
в соответствующих квантилях не должно превышать указанного порога в bps.

## Проверка PnL симулятора

`ExecutionSimulator` исполняет сделки по лучшим котировкам:
ордер `BUY` заполняется по цене `ask`, а ордер `SELL` -- по `bid`.
Незакрытые позиции помечаются по рынку (mark-to-market) также
по лучшим котировкам: для длинной позиции используется `bid`,
для короткой -- `ask`. Если в отчёте присутствует поле `mtm_price`,
оно переопределяет цену маркировки.

В отчётах симуляции присутствуют поля:

* `bid` и `ask` -- текущие лучшие котировки;
* `mtm_price` -- фактическая цена для mark-to-market
  (может отсутствовать/быть `0`, тогда используется `bid/ask`).

Проверочный скрипт пересчитывает `realized_pnl + unrealized_pnl`
по логу трейдов и указанным ценам. Пример пересчёта:

```python
from test_pnl_report_check import _recompute_total

trades = [
    {"side": "BUY", "price": 101.0, "qty": 1.0},
    {"side": "SELL", "price": 102.0, "qty": 1.0},
]
total = _recompute_total(trades, bid=102.0, ask=103.0, mtm_price=None)
# total == 1.0 (realized_pnl + unrealized_pnl)
```

Регрессионный тест для проверки PnL отчётов реализован
в файле `test_pnl_report_check.py`. Тест запускает
симулятор и сравнивает отчёт с пересчитанным результатом.

Запуск теста:

```bash
pytest test_pnl_report_check.py
```

## Проверка реалистичности симуляции

`scripts/sim_reality_check.py` сопоставляет метрики симуляции с
историческими данными и эталонной кривой капитала. Скрипт принимает пути к
логу сделок симуляции (`--trades`), историческому логу (`--historical-trades`),
опциональному файлу капитальной кривой (`--equity`), бенчмарку (`--benchmark`) и
JSON-файлу с допустимыми диапазонами KPI (`--kpi-thresholds`). Параметр
`--quantiles` задаёт число квантилей для построения статистики по размерам
ордеров.

Дополнительно доступны аргументы для анализа чувствительности:

- `--scenario-config` -- путь к JSON-файлу с определением сценариев;
- `--scenarios` -- список сценариев через запятую (по умолчанию используются все из конфигурации);
- `--sensitivity-threshold` -- относительное изменение KPI, при превышении которого сценарий помечается флагом «чрезмерная чувствительность».

Файл конфигурации сценариев задаёт множители комиссий и спреда, расширяя значения по умолчанию:

```json
{
  "Low":  {"fee_mult": 0.5, "spread_mult": 0.5},
  "Med":  {"fee_mult": 1.0, "spread_mult": 1.0},
  "High": {"fee_mult": 1.5, "spread_mult": 1.5}
}
```

Сценарий `Med` используется как базовый. Для каждого другого сценария рассчитывается изменение `pnl_total` относительно базы; если абсолютное изменение превышает `--sensitivity-threshold`, в отчёт добавляется флаг `scenario.<имя>: чрезмерная чувствительность`.

При запуске формируются отчёты `sim_reality_check.json` и
`sim_reality_check.md`, файлы `sim_reality_check_buckets.*`,
`sim_reality_check_degradation.*` и `sim_reality_check_scenarios.*`. Все они
сохраняются в каталог, где расположен файл `--trades`. Если значения KPI или
чувствительность выходят за пределы, список нарушений выводится в консоль и
попадает в отчёт.

```bash
# все KPI в пределах порогов
python scripts/sim_reality_check.py \
  --trades sim_trades.parquet \
  --historical-trades hist_trades.parquet \
  --equity sim_equity.parquet \
  --benchmark bench_equity.parquet \
  --scenario-config configs/scenarios.json \
  --scenarios Low,Med,High
Saved reports to run/sim_reality_check.json and run/sim_reality_check.md
Saved bucket stats to run/sim_reality_check_buckets.csv and run/sim_reality_check_buckets.png

# пример с нарушением порогов
python scripts/sim_reality_check.py \
  --trades sim_bad.parquet \
  --historical-trades hist_trades.parquet \
  --equity sim_equity.parquet \
  --benchmark bench_equity.parquet \
  --kpi-thresholds benchmarks/sim_kpi_thresholds.json \
  --sensitivity-threshold 0.25
Saved reports to run/sim_reality_check.json and run/sim_reality_check.md
Saved bucket stats to run/sim_reality_check_buckets.csv and run/sim_reality_check_buckets.png
Unrealistic KPIs detected:
 - equity.sharpe: нереалистично
```
