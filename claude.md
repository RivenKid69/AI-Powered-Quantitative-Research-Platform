# Claude Documentation - TradingBot2

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
| Изменить логику исполнения | `impl_sim_executor.py`, `execution_sim.py` | `pytest tests/test_execution*.py` |
| Настроить риск-менеджмент | `configs/risk.yaml`, `risk_guard.py` | Проверить `test_risk*.py` |
| Обновить модель PPO | `distributional_ppo.py` | Проверить все `test_distributional_ppo*.py` |
| Добавить новую метрику | `services/monitoring.py` | Обновить `metrics.json` schema |
| Калибровать параметры | `service_calibrate_*.py` | Запустить соответствующий script |
| Отладить training | `train_model_multi_patch.py` + logs | Проверить `tensorboard` logs |
| Проблемы с данными | `impl_offline_data.py`, `data_validation.py` | Проверить data degradation params |
| Live trading проблемы | `script_live.py` → `service_signal_runner.py` | Проверить ops_kill_switch, state_storage |

### 🔍 Quick File Reference

| Префикс | Слой | Зависимости | Примеры |
|---------|------|-------------|---------|
| `core_*` | Базовый | Нет | `core_config.py`, `core_models.py`, `core_strategy.py` |
| `impl_*` | Реализация | `core_` | `impl_sim_executor.py`, `impl_fees.py`, `impl_slippage.py` |
| `service_*` | Сервисы | `core_`, `impl_` | `service_backtest.py`, `service_train.py`, `service_eval.py` |
| `strategies/*` | Стратегии | Все предыдущие | `strategies/base.py`, `strategies/momentum.py` |
| `script_*` | CLI точки входа | Все | `script_backtest.py`, `script_live.py`, `script_eval.py` |

### ⚡ Критические команды (всегда под рукой)

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

# Обновление данных
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python scripts/refresh_fees.py
python -m services.universe --output data/universe/symbols.json

# Валидация
python check_feature_parity.py --data prices.csv --threshold 1e-6
python scripts/validate_seasonality.py --historical trades.csv --multipliers configs/liquidity_latency_seasonality.json
python scripts/sim_reality_check.py --trades sim.parquet --historical hist.parquet
```

### 🚨 Частые ошибки и их решения

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `AttributeError` в конфигах | Pydantic V2 API | Используйте `model_dump()` вместо `dict()` |
| Тесты падают после изменений | Не обновлены тесты | Найдите и обновите соответствующие тесты |
| Feature mismatch | Online/offline паритет | Запустите `check_feature_parity.py` |
| PBT state mismatch | VGS не синхронизирован | Проверьте `variance_gradient_scaler.py` state dict |
| Execution детерминизм нарушен | Изменён seed или порядок | Проверьте `test_execution_determinism.py` |
| Градиенты взрываются | UPGD noise слишком высок | Уменьшите `sigma` в optimizer config |

---

## 📊 СТАТУС ПРОЕКТА (2025-11-20)

### ✅ Последние обновления

- **Интеграция завершена**: UPGD + VGS + Twin Critics + PBT (100% тестов проходят) ✅
- **Pydantic V2**: Полная миграция завершена ✅
- **Security**: torch.load() security fix применён ✅
- **VGS + PBT**: State mismatch исправлен ✅
- **UPGD + VGS**: Adaptive noise scaling добавлен ✅

### 🎯 Активные возможности (Production Ready)

1. **AdaptiveUPGD Optimizer** — default optimizer для continual learning
2. **Twin Critics** — включено по умолчанию для снижения overestimation bias
3. **VGS (Variance Gradient Scaler)** — автоматическое масштабирование градиентов
4. **PBT (Population-Based Training)** — гиперпараметр optimization
5. **SA-PPO (State-Adversarial PPO)** — robust training против adversarial perturbations

### 📍 Текущая ветка

```
Branch: main
Recent commits:
- 7d83854 docs: Reorganize project documentation
- 45c45da test: Update integration tests for PBT API
- 2927e75 fix: Add adaptive noise scaling to UPGD
- 416cf11 fix: Fix VGS state mismatch during PBT
- 078a6c9 refactor: Migrate core_config to Pydantic V2
```

---

## О проекте

**TradingBot2** — высокочастотный торговый бот для криптовалют (преимущественно Binance spot), использующий reinforcement learning (PPO) для принятия торговых решений. Проект написан на Python с критичными к производительности модулями на Cython/C++ и включает в себя полноценный симулятор исполнения, модели проскальзывания, задержек и микроструктуры рынка.

### Основные характеристики

- **Язык**: Python 3.12 + Cython + C++
- **RL Framework**: Stable-Baselines3 (Distributional PPO with Twin Critics)
- **Optimizer**: AdaptiveUPGD (default) — continual learning with utility-based weight protection
- **Gradient Scaling**: VGS (Variance Gradient Scaler) — automatic per-layer gradient normalization
- **Training**: Population-Based Training (PBT) + State-Adversarial PPO (SA-PPO)
- **Биржа**: Binance (Spot/Futures)
- **Режимы работы**: Бэктест, Live trading, Обучение моделей
- **Архитектура**: Слоистая (layered) с dependency injection

---

## 🚀 Продвинутые возможности (2024-2025)

### 1. UPGD Optimizer (Utility-based Perturbed Gradient Descent)

**Статус**: ✅ Production Ready | **Default**: Enabled (AdaptiveUPGD)

**Описание**: Continual learning optimizer для предотвращения catastrophic forgetting и поддержания пластичности нейронных сетей.

**Ключевые файлы**:
- `optimizers/upgd_optimizer.py` — реализация UPGD/AdaptiveUPGD/UPGDW
- `docs/UPGD_INTEGRATION.md` — документация интеграции
- `tests/test_upgd*.py` — тесты

**Варианты**:
- **AdaptiveUPGD** (рекомендуется) — UPGD + Adam-style adaptive learning rates
- **UPGD** — базовый SGD с utility protection
- **UPGDW** — AdamW replacement с decoupled weight decay

**Конфигурация** (в `config_train.yaml`):
```yaml
model:
  optimizer_class: AdaptiveUPGD  # или UPGD, UPGDW
  optimizer_kwargs:
    lr: 1.0e-5
    weight_decay: 0.001
    beta_utility: 0.999  # EMA decay для utility
    sigma: 0.001         # Gaussian noise std (важно для VGS!)
    beta1: 0.9           # First moment (AdaptiveUPGD)
    beta2: 0.999         # Second moment (AdaptiveUPGD)
```

**Важно**: При использовании с VGS установите `sigma` в диапазоне 0.0005-0.001 для предотвращения amplification.

### 2. Twin Critics

**Статус**: ✅ Production Ready | **Default**: Enabled

**Описание**: Две независимые value networks для снижения overestimation bias (как в TD3/SAC).

**Ключевые файлы**:
- `distributional_ppo.py` — интеграция в PPO
- `docs/twin_critics.md` — полная документация
- `tests/test_twin_critics*.py` — тесты

**Архитектура**:
```
[Observation] → [Features] → [LSTM] → [MLP] → [Critic Head 1] → [Value 1]
                                              ↘ [Critic Head 2] → [Value 2]
Target Value = min(Value 1, Value 2)
```

**Конфигурация** (включено по умолчанию):
```yaml
arch_params:
  critic:
    distributional: true
    num_quantiles: 32
    huber_kappa: 1.0
    use_twin_critics: true  # Default (можно не указывать)
```

**Research Support**: PDPPO (2025), DNA (2022), TD3 (2018) показали улучшение производительности в 2x в стохастичных средах.

### 3. VGS (Variance Gradient Scaler)

**Статус**: ✅ Production Ready | **Default**: Enabled with UPGD

**Описание**: Автоматическое масштабирование градиентов per-layer на основе их вариации для стабилизации обучения.

**Ключевые файлы**:
- `variance_gradient_scaler.py` — реализация
- `distributional_ppo.py` — интеграция
- `tests/test_upgd_vgs*.py` — тесты интеграции

**Алгоритм**:
1. Накапливает градиенты за несколько backward passes
2. Вычисляет STD градиентов per-layer
3. Масштабирует градиенты: `grad_scaled = grad / (std + eps)`
4. Применяет масштабированные градиенты к оптимизатору

**Конфигурация**:
```yaml
model:
  vgs:
    enabled: true
    accumulation_steps: 4     # Шагов для накопления статистики
    warmup_steps: 10          # Warmup перед включением scaling
    eps: 1e-6                 # Numerical stability
    clip_threshold: 10.0      # Clipping для экстремальных scaling factors
```

**Важно**:
- VGS автоматически управляет своим state dict для PBT checkpointing
- При использовании с UPGD необходимо снизить `sigma` для предотвращения amplification градиентного шума

### 4. PBT (Population-Based Training)

**Статус**: ✅ Production Ready | **Config**: `config_pbt_adversarial.yaml`

**Описание**: Эволюционная оптимизация гиперпараметров через популяцию параллельных агентов.

**Ключевые файлы**:
- `adversarial/pbt_scheduler.py` — PBT scheduler
- `training_pbt_adversarial_integration.py` — интеграция с training loop
- `configs/config_pbt_adversarial.yaml` — конфигурация
- `tests/test_pbt*.py` — тесты

**Алгоритм**:
1. Запускает популяцию из N агентов с разными гиперпараметрами
2. Периодически оценивает производительность каждого агента
3. **Exploit**: Копирует веса от лучших агентов к худшим (truncation selection)
4. **Explore**: Применяет perturbation или resampling к гиперпараметрам

**Конфигурация**:
```yaml
pbt:
  enabled: true
  population_size: 8                 # Размер популяции
  perturbation_interval: 10          # Шагов между PBT операциями
  exploit_method: truncation         # 'truncation' или 'binary_tournament'
  explore_method: both               # 'perturb', 'resample', 'both'
  truncation_ratio: 0.25             # Top/bottom 25%
  metric_name: mean_reward           # Метрика для оптимизации
  metric_mode: max                   # 'max' или 'min'

  hyperparams:
    - name: learning_rate
      min_value: 1.0e-5
      max_value: 5.0e-4
      perturbation_factor: 1.2
      is_log_scale: true

    - name: adversarial_epsilon      # Для SA-PPO
      min_value: 0.01
      max_value: 0.15
      perturbation_factor: 1.15
```

**Запуск**:
```bash
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml
```

### 5. SA-PPO (State-Adversarial PPO)

**Статус**: ✅ Production Ready | **Config**: `config_pbt_adversarial.yaml`

**Описание**: Robust training через adversarial perturbations к state observations (PGD attack).

**Ключевые файлы**:
- `adversarial/sa_ppo.py` — SA-PPO реализация
- `training_pbt_adversarial_integration.py` — интеграция
- `configs/config_pbt_adversarial.yaml` — конфигурация

**Алгоритм**:
1. Для каждого batch создаёт adversarial examples через PGD:
   - Находит perturbation δ, максимизирующий loss: `max_δ L(s + δ)`
   - Ограничение: `||δ||_∞ ≤ ε` (L-inf norm)
2. Обучает policy/value на смеси clean и adversarial samples
3. Добавляет robust KL regularization между clean и adversarial policies

**Конфигурация**:
```yaml
adversarial:
  enabled: true
  perturbation:
    epsilon: 0.075              # L-inf norm constraint
    attack_steps: 3             # PGD iterations
    attack_lr: 0.03             # PGD step size
    random_init: true           # Random start для PGD
    norm_type: linf             # 'linf' или 'l2'
    attack_method: pgd          # 'pgd' или 'fgsm'

  adversarial_ratio: 0.5        # Ratio adversarial vs clean
  robust_kl_coef: 0.1           # Robust KL regularization
  warmup_updates: 10            # Updates перед включением
  attack_policy: true           # Attack policy loss
  attack_value: true            # Attack value loss
```

**Research Support**: SA-PPO показывает улучшенную robustness к distribution shift и noise в real-world environments.

---

## Архитектура проекта

Проект использует **строгую слоистую архитектуру** с префиксами имён файлов:

```
core_ → impl_ → service_ → strategies → script_
```

**КРИТИЧЕСКИ ВАЖНО**: Нарушение зависимостей между слоями приведёт к циклическим импортам и ошибкам!

### Слои (Layers)

#### 1. `core_*` — Базовый слой
Содержит базовые сущности, контракты (protocols), модели и константы. **Не зависит** от других слоёв.

**Ключевые файлы:**
- `core_config.py` — конфигурационные модели (CommonRunConfig, etc.) [Pydantic V2]
- `core_models.py` — TradeLogRow, EquityPoint, Decision
- `core_strategy.py` — Protocol для торговых стратегий
- `core_contracts.py` — Интерфейсы/контракты
- `core_events.py` — События системы
- `core_errors.py` — Кастомные исключения
- `core_constants.py` — Константы (сопоставление Cython и Python)

#### 2. `impl_*` — Слой реализации
Конкретные имплементации инфраструктуры и внешних зависимостей. Зависит **только от `core_`**.

**Ключевые файлы:**
- `impl_sim_executor.py` — Симулятор исполнения заявок
- `impl_fees.py` — Расчёт комиссий (с поддержкой BNB discount)
- `impl_slippage.py` — Модели проскальзывания
- `impl_latency.py` — Модели задержек (сезонные, волатильные)
- `impl_quantizer.py` — Квантование цен/объёмов по биржевым фильтрам
- `impl_offline_data.py` — Чтение исторических данных
- `impl_binance_public.py` — Публичные API Binance
- `impl_bar_executor.py` — Баровый исполнитель
- `impl_risk_basic.py` — Базовый риск-менеджмент

#### 3. `service_*` — Слой сервисов
Объединяет бизнес-логику. Может зависеть от `core_` и `impl_`.

**Ключевые файлы:**
- `service_backtest.py` — Сервис бэктестинга
- `service_train.py` — Сервис обучения моделей
- `service_eval.py` — Оценка моделей с разными профилями исполнения
- `service_signal_runner.py` — Запуск live trading
- `service_calibrate_tcost.py` / `service_calibrate_slippage.py` — Калибровка
- `service_fetch_exchange_specs.py` — Загрузка биржевых спецификаций
- `services/monitoring.py` — Мониторинг и метрики
- `services/rest_budget.py` — REST API rate limiting
- `services/ops_kill_switch.py` — Operational kill switch
- `services/state_storage.py` — Персистентность состояния
- `services/signal_bus.py` — Шина сигналов
- `services/universe.py` — Управление универсом символов

#### 4. `strategies/` — Торговые стратегии
Реализации алгоритмов принятия решений. Могут зависеть от всех предыдущих слоёв.

**Файлы:**
- `strategies/base.py` — Базовый класс Strategy
- `strategies/momentum.py` — Пример стратегии на моментуме

#### 5. `script_*` — CLI точки входа
Запускаемые скрипты. Используют DI контейнер и **не содержат бизнес-логику**.

**Основные скрипты:**
- `script_backtest.py` → ServiceBacktest
- `script_live.py` → ServiceSignalRunner
- `script_eval.py` → ServiceEval (поддержка `--all-profiles`)
- `script_compare_runs.py` → Сравнение метрик
- `script_calibrate_tcost.py`, `script_calibrate_slippage.py` → Калибровка
- `script_fetch_exchange_specs.py` → Загрузка exchange specs
- `train_model_multi_patch.py` → Обучение моделей (основной скрипт)

### Dependency Injection (DI)

Проект использует DI через модуль `di_registry.py`. Компоненты регистрируются и резолвятся динамически из YAML конфигураций.

Пример:
```yaml
components:
  market_data:
    target: impl_offline_data:OfflineCSVBarSource
    params: {paths: ["data/sample.csv"], timeframe: "1m"}
```

---

## Основные компоненты

### 1. Симулятор исполнения (ExecutionSimulator)

Находится в `execution_sim.py`. Включает:
- Симуляцию LOB (limit order book) через Cython модули
- Микроструктурный генератор (`micro_sim.pyx`, `cpp_microstructure_generator.cpp`)
- Модели проскальзывания (linear, sqrt, калиброванные)
- Учёт комиссий (maker/taker, BNB discount)
- TTL (time-to-live) для лимитных заявок
- TIF: GTC, IOC, FOK
- Алгоритмические исполнители: TWAP, POV, VWAP

### 2. Distributional PPO (`distributional_ppo.py`)

Кастомизированный PPO с:
- **Distributional value head** (quantile regression, 21-51 atoms)
- **Twin Critics** (default enabled) — две независимые value networks
- **Expected Value (EV) reserve sampling** для стабилизации обучения
- **EV batching** с приоритизацией редких событий
- **VGS (Variance Gradient Scaler)** — автоматическое per-layer gradient scaling
- **AdaptiveUPGD optimizer** (default) — continual learning
- **CVaR risk-aware learning** — focus на tail risk (worst 5% outcomes)
- Поддержка sampling mask для no-trade окон
- Отключённый PopArt (ранее использовался, теперь удалён)

**Критические параметры**:
```yaml
model:
  params:
    # Distributional value head
    num_atoms: 21               # Количество квантилей
    v_min: -10.0                # Минимальное значение support
    v_max: 10.0                 # Максимальное значение support
    v_range_ema_alpha: 0.005    # EMA для adaptive v_min/v_max

    # CVaR risk-aware learning
    cvar_alpha: 0.05            # Worst 5% tail
    cvar_weight: 0.15           # Weight для CVaR loss
    cvar_activation_threshold: 0.15

    # Value clipping (Twin Critics)
    clip_range_vf: 0.7          # Default: 0.7
    vf_clip_warmup_updates: 0   # Warmup disabled by default
```

### 3. Features Pipeline

- `feature_pipe.py` — Онлайн расчёт признаков
- `features_pipeline.py` — Оффлайн препроцессинг
- `feature_config.py` — Конфигурация фич (63 features current)
- `features/` — Директория с feature implementations
- Поддержка проверки паритета через `check_feature_parity.py`

**Feature Groups**:
- **Price features**: returns, log_returns, normalized_price
- **Volume features**: volume_ratio, quote_volume, taker_buy_ratio
- **Volatility features**: realized_vol, Parkinson, Yang-Zhang, GARCH
- **Momentum features**: RSI, MACD, momentum indicators
- **Market microstructure**: spread, depth, order flow imbalance
- **Technical indicators**: MA5, MA20, Bollinger Bands

### 4. Риск-менеджмент

- `risk_guard.py` — Гварды на позицию/PnL/дроудаун
- `risk_manager.pyx` — Cython модуль для быстрой проверки
- `dynamic_no_trade_guard.py` — Динамическое блокирование торговли
- `services/ops_kill_switch.py` — Операционный kill switch

**Risk Limits**:
```yaml
risk:
  max_position: 100000        # Максимальная позиция (USD)
  max_leverage: 1.0           # Максимальное плечо
  max_drawdown_pct: 0.10      # Max drawdown 10%
  stop_loss_pct: 0.05         # Stop loss 5%
  daily_loss_limit: 1000      # Дневной лимит убытка (USD)
```

### 5. No-Trade окна

- `no_trade.py`, `no_trade_config.py` — Управление запрещёнными окнами
- Поддержка funding windows, daily UTC windows, custom intervals
- Утилита: `no-trade-mask` (CLI)

**Funding Windows** (по умолчанию):
- 00:00 UTC ± 5 минут
- 08:00 UTC ± 5 минут
- 16:00 UTC ± 5 минут

### 6. Latency & Seasonality

- **Latency**: `latency.py`, `impl_latency.py` — моделирование задержек (mean, std, volatility)
- **Seasonality**: `utils_time.py`, `configs/liquidity_latency_seasonality.json`
  - 168 коэффициентов (24ч × 7 дней недели) для ликвидности, спреда, задержек
  - Валидация: `scripts/validate_seasonality.py`
  - Построение: `scripts/build_hourly_seasonality.py`

**Seasonality Structure**:
```json
{
  "liquidity_multipliers": [1.0, 0.95, ..., 1.1],  // 168 values
  "spread_multipliers": [1.0, 1.05, ..., 0.98],
  "latency_multipliers": [1.0, 1.02, ..., 0.97]
}
```

### 7. Fees & Quantization

- `fees.py`, `impl_fees.py` — Комиссии (BNB discount, maker/taker)
- `quantizer.py`, `impl_quantizer.py` — Квантование по биржевым фильтрам
- Auto-refresh фильтров: `scripts/fetch_binance_filters.py`
- Auto-refresh fees: `scripts/refresh_fees.py`

**Binance Fees** (typical):
- Spot Maker: 0.1% (0.075% with BNB)
- Spot Taker: 0.1% (0.075% with BNB)
- Futures Maker: 0.02%
- Futures Taker: 0.04%

### 8. Data Degradation

- `data_validation.py` — Моделирование пропусков, задержек, stale data
- Конфиг: `data_degradation` (stale_prob, drop_prob, dropout_prob, max_delay_ms)

**Параметры**:
```yaml
data_degradation:
  stale_prob: 0.01            # 1% chance to repeat previous bar
  drop_prob: 0.005            # 0.5% chance to drop bar
  dropout_prob: 0.02          # 2% chance of delay
  max_delay_ms: 500           # Max delay 500ms
```

### 9. Logging & Metrics

- `sim_logging.py` — Запись логов трейдов и equity
  - `logs/log_trades_<runid>.csv` (TradeLogRow)
  - `logs/report_equity_<runid>.csv` (EquityPoint)
- `services/monitoring.py` — Метрики (Sharpe, Sortino, MDD, CVaR, etc.)
- Агрегация через `aggregate_exec_logs.py`

**Ключевые метрики**:
- Sharpe Ratio, Sortino Ratio
- Max Drawdown (MDD)
- CVaR (Conditional Value at Risk)
- Hit Rate, Win Rate
- Total PnL, Turnover
- Average Latency

---

## Конфигурации (configs/)

### Основные конфиги

- **config_sim.yaml** — Симуляция (бэктест)
- **config_train.yaml** — Обучение модели (standard)
- **config_pbt_adversarial.yaml** — PBT + Adversarial training ⭐ NEW
- **config_live.yaml** — Live trading
- **config_eval.yaml** — Оценка модели
- **config_template.yaml** — Шаблон конфигурации

### Модульные конфиги (включаются через YAML anchors)

- **execution.yaml** — Параметры исполнения
- **fees.yaml** — Комиссии и округление
- **slippage.yaml** — Модели проскальзывания
- **risk.yaml** — Риск-менеджмент
- **no_trade.yaml** — No-trade окна
- **quantizer.yaml** — Квантование
- **timing.yaml** — Timing профили
- **runtime.yaml** / **runtime_trade.yaml** — Runtime параметры
- **state.yaml** — Персистентность состояния
- **monitoring.yaml** — Мониторинг
- **ops.yaml** / **ops.json** — Operational kill switch
- **rest_budget.yaml** — REST API rate limiting
- **offline.yaml** — Оффлайн datasets, сплиты

### Сезонность и режимы

- **liquidity_latency_seasonality.json** — 168 коэффициентов для ликвидности/латентности
- **market_regimes.json** — Рыночные режимы (trending, mean_reverting, volatile)

---

## CLI Примеры

### Бэктест
```bash
python script_backtest.py --config configs/config_sim.yaml
```

### Обучение (Standard)
```bash
python train_model_multi_patch.py \
  --config configs/config_train.yaml \
  --regime-config configs/market_regimes.json \
  --liquidity-seasonality configs/liquidity_latency_seasonality.json
```

### Обучение (PBT + Adversarial) ⭐ NEW
```bash
# Population-Based Training with State-Adversarial PPO
python train_model_multi_patch.py \
  --config configs/config_pbt_adversarial.yaml \
  --regime-config configs/market_regimes.json \
  --liquidity-seasonality configs/liquidity_latency_seasonality.json

# Monitor tensorboard for population metrics
tensorboard --logdir artifacts/pbt_checkpoints
```

### Live trading
```bash
python script_live.py --config configs/config_live.yaml
```

### Оценка модели (все профили)
```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

### Сравнение запусков
```bash
python script_compare_runs.py run1/ run2/ run3/ --csv compare.csv
```

### Обновление символов
```bash
python -m services.universe --output data/universe/symbols.json --liquidity-threshold 1e6
```

### Обновление биржевых фильтров
```bash
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
```

### Обновление комиссий
```bash
python scripts/refresh_fees.py
```

### Валидация сезонности
```bash
python scripts/validate_seasonality.py \
  --historical path/to/trades.csv \
  --multipliers data/latency/liquidity_latency_seasonality.json
```

### Проверка реалистичности симуляции
```bash
python scripts/sim_reality_check.py \
  --trades sim_trades.parquet \
  --historical-trades hist_trades.parquet \
  --equity sim_equity.parquet \
  --benchmark bench_equity.parquet \
  --kpi-thresholds benchmarks/sim_kpi_thresholds.json
```

---

## Cython/C++ модули

### Критичные к производительности компоненты

- **fast_lob.pyx / fast_lob.cpp** — Быстрая LOB
- **lob_state_cython.pyx** — Состояние LOB
- **micro_sim.pyx** — Микроструктурная симуляция
- **marketmarket_simulator_wrapper.pyx** — Обёртка C++ симулятора
- **obs_builder.pyx** — Построение наблюдений
- **reward.pyx** — Расчёт reward
- **risk_manager.pyx** — Риск-менеджмент
- **coreworkspace.pyx** — Рабочее пространство
- **execlob_book.pyx** — LOB для исполнения

### C++ компоненты

- **MarketSimulator.cpp/.h** — Основной симулятор рынка
- **OrderBook.cpp/.h** — Стакан заявок
- **cpp_microstructure_generator.cpp/.h** — Генератор микроструктуры

---

## Важные паттерны и концепции

### 1. Execution Profiles

Поддерживаются различные профили исполнения (conservative, balanced, aggressive) с разными:
- `slippage_bps` — проскальзывание
- `offset_bps` — смещение лимитной цены
- `ttl` — время жизни заявки (мс)
- `tif` — Time In Force (GTC/IOC/FOK)

### 2. Bar Execution Mode

Режим `execution.mode: bar` позволяет работать с агрегированными баровыми данными вместо tick-by-tick.

Параметры:
- `bar_price: close` — цена исполнения (open/high/low/close)
- `min_rebalance_step: 0.05` — минимальный шаг ребалансировки

Сигналы должны следовать формату [spot signal envelope](docs/bar_execution.md).

### 3. Intrabar Price Models

- **bridge** — Brownian bridge sampling (legacy)
- **reference** — Использование внешнего M1 reference feed для детерминированных fills

Настраивается через `execution.intrabar_price_model` в YAML.

### 4. Large Order Execution

Заявки с notional > `notional_threshold` разбиваются алгоритмически:
- **TWAP** — Time-Weighted Average Price
- **POV** — Percentage of Volume
- **VWAP** — Volume-Weighted Average Price

Параметры POV:
```yaml
pov:
  participation: 0.2       # 20% от наблюдаемого объёма
  child_interval_s: 1      # Интервал между дочерними заявками
  min_child_notional: 1000 # Минимальный размер дочерней заявки
```

### 5. Expected Value (EV) Reserve

Механизм в Distributional PPO для стабилизации обучения:
- Резервирует часть батча для редких/высоко-ценных событий
- Приоритизация через квантили EV
- Настраивается через `ev_reserve_*` параметры в конфиге

### 6. No-Trade Masks

Блокируют торговлю в определённые периоды:
- Funding windows (±5 минут от 00:00/08:00/16:00 UTC)
- Custom intervals (milliseconds)
- Daily UTC windows

Применяется через:
- Конфиг: `no_trade` секция
- Утилита: `no-trade-mask --mode drop/weight`

### 7. Data Degradation

Моделирование реальных проблем с данными:
- `stale_prob` — вероятность повторить предыдущий бар
- `drop_prob` — вероятность пропустить бар
- `dropout_prob` — вероятность задержки
- `max_delay_ms` — максимальная задержка

### 8. Kill Switch

Два типа:
- **Metric kill switch** — останавливает торговлю при плохих метриках
- **Operational kill switch** — останавливает при операционных проблемах

Восстановление:
```bash
python scripts/reset_kill_switch.py
```

---

## Data Pipeline

### 1. Ingestion (Загрузка данных)

```bash
python scripts/run_full_cycle.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1m,5m,15m \
  --start 2024-01-01 --end 2024-12-31
```

Модули:
- `ingest_orchestrator.py` — Оркестратор загрузки
- `ingest_klines.py` — Загрузка свечей
- `ingest_funding_mark.py` — Funding rates и mark prices
- `binance_public.py` — Публичное API Binance

### 2. Preprocessing

```bash
python prepare_and_run.py --config configs/feature_prepare.yaml
```

Модули:
- `prepare_events.py` — Подготовка событий
- `build_adv.py`, `build_adv_base.py` — ADV (Average Daily Volume)
- `make_features.py` — Создание признаков
- `make_prices_from_klines.py` — Извлечение цен из свечей

### 3. Training

```bash
python train_model_multi_patch.py --config configs/config_train.yaml
```

Создаёт модель (PPO policy) в формате Stable-Baselines3.

### 4. Evaluation

```bash
python script_eval.py --config configs/config_eval.yaml
```

Генерирует метрики в `metrics.json`.

### 5. Live Trading

```bash
python script_live.py --config configs/config_live.yaml
```

---

## Тестирование

Проект содержит **обширный набор тестов** (pytest):

### Категории тестов

- **Execution** — `test_execution_*.py` (детерминизм, профили, правила)
- **Fees** — `test_fees_*.py` (округление, BNB discount)
- **Latency** — `test_latency_*.py` (сезонность, волатильность)
- **Risk** — `test_risk_*.py` (exposure limits, kill switch)
- **Service** — `test_service_*.py` (бэктест, eval, signal runner)
- **No-trade** — `test_no_trade_*.py` (маски, окна)
- **Distributional PPO** — `test_distributional_ppo_*.py` (CVaR, outliers, EV reserve)
- **UPGD** — `test_upgd*.py` (optimizer, VGS integration) ⭐ NEW
- **Twin Critics** — `test_twin_critics*.py` (architecture, training) ⭐ NEW
- **PBT** — `test_pbt*.py` (scheduler, hyperparameter optimization) ⭐ NEW
- **Adversarial** — `test_*adversarial*.py` (SA-PPO, robust training) ⭐ NEW

### Запуск тестов

```bash
pytest tests/                          # Все тесты
pytest tests/test_execution_sim*.py    # Конкретная категория
pytest -k "test_fees"                  # По ключевому слову
pytest tests/test_upgd*.py -v          # UPGD тесты
pytest tests/test_pbt*.py -v           # PBT тесты
```

---

## Документация проекта (docs/)

### Основная документация

- **[DOCS_INDEX.md](DOCS_INDEX.md)** — Главный индекс всей документации проекта
- **[README.md](README.md)** — Обзор проекта и быстрый старт
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Архитектура системы
- **[CLAUDE.md](CLAUDE.md)** — Полная документация проекта (этот файл)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Руководство по участию в разработке
- **[CHANGELOG.md](CHANGELOG.md)** — История изменений
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** — Инструкции по сборке

### Продвинутые возможности (NEW 2024-2025)

- **[docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md)** — UPGD Optimizer интеграция ⭐
- **[docs/twin_critics.md](docs/twin_critics.md)** — Twin Critics архитектура ⭐
- **[docs/reports/upgd_vgs/](docs/reports/upgd_vgs/)** — UPGD + VGS отчеты ⭐
- **[docs/reports/twin_critics/](docs/reports/twin_critics/)** — Twin Critics отчеты ⭐

### Features & Components

- **[docs/pipeline.md](docs/pipeline.md)** — Decision pipeline architecture
- **[docs/bar_execution.md](docs/bar_execution.md)** — Bar execution mode
- **[docs/large_orders.md](docs/large_orders.md)** — Large order execution algorithms
- **[docs/moving_average.md](docs/moving_average.md)** — Moving average implementation
- **[docs/dynamic_spread.md](docs/dynamic_spread.md)** — Dynamic spread modeling

### Risk & Trading

- **[docs/no_trade.md](docs/no_trade.md)** — No-trade windows documentation
- **[docs/data_degradation.md](docs/data_degradation.md)** — Data degradation simulation
- **[docs/permissions.md](docs/permissions.md)** — Role-based access control

### Market Data & Seasonality

- **[docs/seasonality.md](docs/seasonality.md)** — Seasonality framework overview
- **[docs/seasonality_quickstart.md](docs/seasonality_quickstart.md)** — Quick start guide
- **[docs/seasonality_QA.md](docs/seasonality_QA.md)** — QA process for seasonality

### ML & Training

- **[docs/parallel.md](docs/parallel.md)** — Parallel environments and randomness
- **[docs/eval.md](docs/eval.md)** — Model evaluation framework

### Быстрые справочники

- **[QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)** — Быстрый старт
- **[FILE_REFERENCE.md](FILE_REFERENCE.md)** — Справочник по файлам
- **[VERIFICATION_INSTRUCTIONS.md](VERIFICATION_INSTRUCTIONS.md)** — Инструкции по верификации

### Отчеты и анализы

Все отчеты организованы в `docs/reports/`:

- **[docs/reports/integration/](docs/reports/integration/)** — Интеграция и миграция
  - **[INTEGRATION_SUCCESS_REPORT.md](docs/reports/integration/INTEGRATION_SUCCESS_REPORT.md)** ⭐
- **[docs/reports/bugs/](docs/reports/bugs/)** — Отчеты об ошибках
- **[docs/reports/audits/](docs/reports/audits/)** — Аудиты
- **[docs/reports/features/](docs/reports/features/)** — Feature mappings
- **[docs/reports/fixes/](docs/reports/fixes/)** — Исправления
- **[docs/reports/tests/](docs/reports/tests/)** — Тесты и верификация

---

## Важные переменные окружения

- `TB_FAIL_ON_STALE_FILTERS=1` — Фейлить при устаревших фильтрах
- `BINANCE_PUBLIC_FEES_DISABLE_AUTO=1` — Отключить автообновление комиссий
- `BINANCE_API_KEY`, `BINANCE_API_SECRET` — API ключи Binance
- `BINANCE_FEE_SNAPSHOT_CSV` — Путь к CSV с комиссиями
- `SYMS`, `LOOP`, `SLEEP_MIN` — Для `update_and_infer.py`

---

## Git & Collaboration

### Branching

Работа ведётся на feature branches с префиксом `claude/`:
```bash
git checkout -b claude/feature-name-SESSION_ID
```

### Commit Messages

Следуйте стилю из `git log`:
- Краткое описание (1-2 предложения)
- Фокус на "why", а не "what"
- Примеры:
  - "Add BNB fee settlement mode"
  - "Fix EV batch prioritization"
  - "Add adaptive noise scaling to UPGD to prevent VGS amplification"

### Pull Requests

Создание PR через `gh` CLI:
```bash
gh pr create --title "Feature: ..." --body "## Summary\n- ...\n\n## Test plan\n- ..."
```

---

## Debugging & Troubleshooting

### 1. Проверка паритета фич
```bash
python check_feature_parity.py --data prices.csv --threshold 1e-6
```

### 2. Проверка PnL
```bash
pytest tests/test_pnl_report_check.py
```

### 3. Проверка drift
```bash
python check_drift.py --baseline baseline.csv --current current.csv
```

### 4. Валидация кривой проскальзывания
```bash
python compare_slippage_curve.py hist.csv sim.csv --tolerance 5
```

### 5. Логи деградации
Ищите в выводе:
- `OfflineCSVBarSource degradation: ...`
- `BinanceWS degradation: ...`
- `LatencyQueue degradation: ...`

### 6. Отладка UPGD/VGS ⭐ NEW
```bash
# Проверка UPGD state dict
python -c "import torch; m=torch.load('model.zip'); print(m['optimizer']['state'].keys())"

# Проверка VGS state
python -c "from variance_gradient_scaler import VarianceGradientScaler; vgs=VGS(); print(vgs.state_dict())"

# Тесты UPGD + VGS интеграции
pytest tests/test_upgd_vgs*.py -v
```

### 7. Отладка PBT ⭐ NEW
```bash
# Проверка PBT checkpoints
ls -la artifacts/pbt_checkpoints/

# Проверка PBT scheduler state
python tests/test_pbt_scheduler.py -v

# Мониторинг PBT метрик
tensorboard --logdir artifacts/pbt_checkpoints
```

---

## Performance Tips

1. **Используйте Cython модули** — все критичные компоненты уже оптимизированы
2. **Параллельные окружения** — `shared_memory_vec_env.py` для multi-env training
3. **Кэширование REST** — настройте `rest_budget.cache` в `offline.yaml`
4. **Checkpointing** — используйте `checkpoint_path` для длительных запусков
5. **Offline режим** — используйте `--dry-run` для проверки без сетевых запросов
6. **UPGD optimizer** ⭐ — AdaptiveUPGD по умолчанию для continual learning
7. **VGS** ⭐ — автоматическое gradient scaling для стабильности
8. **Twin Critics** ⭐ — включено по умолчанию для лучших value estimates

---

## Частые задачи

### Добавить новый символ
1. Обновите `data/universe/symbols.json`
2. Перезагрузите фильтры: `python scripts/fetch_binance_filters.py --universe`
3. Загрузите исторические данные через `ingest_orchestrator.py`

### Изменить параметры риска
Отредактируйте `configs/risk.yaml` или передайте через CLI:
```bash
python script_backtest.py --config config.yaml --risk.max-position 100
```

### Добавить новую фичу
1. Реализуйте в `features/`
2. Зарегистрируйте в `features/registry.py`
3. Добавьте в `feature_config.py`
4. Проверьте паритет: `check_feature_parity.py`

### Калибровать slippage
```bash
python script_calibrate_slippage.py --config configs/slippage_calibrate.yaml
```

### Создать новую стратегию
1. Создайте файл в `strategies/`
2. Унаследуйте от `BaseStrategy`
3. Реализуйте метод `decide(ctx) -> list[Decision]`
4. Зарегистрируйте в DI (если нужно)

### Настроить UPGD optimizer ⭐ NEW
```yaml
model:
  optimizer_class: AdaptiveUPGD  # или UPGD, UPGDW
  optimizer_kwargs:
    lr: 1.0e-5
    weight_decay: 0.001
    beta_utility: 0.999
    sigma: 0.001       # Важно для VGS!
```

### Включить/отключить Twin Critics ⭐ NEW
```yaml
arch_params:
  critic:
    use_twin_critics: true  # Default enabled
```

### Настроить PBT ⭐ NEW
```yaml
pbt:
  enabled: true
  population_size: 8
  perturbation_interval: 10
  hyperparams:
    - name: learning_rate
      min_value: 1.0e-5
      max_value: 5.0e-4
```

---

## Ключевые метрики

При анализе результатов обращайте внимание на:

- **Sharpe Ratio** — скорректированная на риск доходность
- **Sortino Ratio** — учитывает только downside volatility
- **MDD (Max Drawdown)** — максимальная просадка
- **CVaR (Conditional Value at Risk)** — средний убыток в худших 5% случаев
- **Hit Rate** — процент прибыльных сделок
- **Win Rate** — процент прибыльных эпизодов ⭐ NEW
- **PnL Total** — суммарная прибыль/убыток
- **Turnover** — оборот
- **Avg Latency** — средняя задержка исполнения

---

## Production Checklist

Перед запуском в продакшн:

**Данные и конфигурация:**
- [ ] Обновлены фильтры (`fetch_binance_filters.py`)
- [ ] Обновлены комиссии (`refresh_fees.py`)
- [ ] Обновлены exchange specs (`script_fetch_exchange_specs.py`)
- [ ] Валидирована сезонность (`validate_seasonality.py`)
- [ ] Проверены risk limits (`risk.yaml`)
- [ ] Проверены no-trade окна (`no_trade.yaml`)

**Мониторинг и безопасность:**
- [ ] Настроен kill switch (`ops.yaml`)
- [ ] Настроен мониторинг (`monitoring.yaml`)
- [ ] Настроено сохранение состояния (`state.yaml`)
- [ ] Настроен REST rate limiting (`rest_budget.yaml`)

**Тестирование и валидация:**
- [ ] Проведён sim reality check (`sim_reality_check.py`)
- [ ] Все тесты проходят (`pytest tests/`)
- [ ] Проверен feature parity (`check_feature_parity.py`)
- [ ] Валидирован drift (`check_drift.py`)
- [ ] Проверена slippage curve (`compare_slippage_curve.py`)

**ML Модель (NEW 2025):** ⭐
- [ ] UPGD optimizer настроен (`optimizer_class: AdaptiveUPGD`)
- [ ] VGS enabled и warmup настроен (`vgs.warmup_steps`)
- [ ] Twin Critics enabled (`use_twin_critics: true`)
- [ ] CVaR параметры настроены (`cvar_alpha`, `cvar_weight`)
- [ ] Value clipping настроен (`clip_range_vf: 0.7`)
- [ ] PBT checkpoints проверены (если используется PBT)

**Live Trading:**
- [ ] API ключи настроены (`BINANCE_API_KEY`, `BINANCE_API_SECRET`)
- [ ] Режим live trading активирован (`config_live.yaml`)
- [ ] Signal bus настроен (`signal_bus.py`)
- [ ] State storage работает (`state_storage.py`)
- [ ] Kill switch протестирован (`reset_kill_switch.py`)

---

## Полезные ссылки

- **Documentation Index**: [DOCS_INDEX.md](DOCS_INDEX.md) — Главная навигация по документации
- **UPGD Integration**: [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) ⭐
- **Twin Critics**: [docs/twin_critics.md](docs/twin_critics.md) ⭐
- **Integration Success**: [docs/reports/integration/INTEGRATION_SUCCESS_REPORT.md](docs/reports/integration/INTEGRATION_SUCCESS_REPORT.md) ⭐
- **Issues**: Issues tracking (если есть)
- **Benchmarks**: `benchmarks/` — KPI thresholds
- **Artifacts**: `artifacts/` — Training artifacts, checkpoints
- **Data**: `data/` — Historical data, universes, specs
- **Logs**: `logs/` — Trade logs, equity curves

---

## Заключение

TradingBot2 — это сложная система с множеством компонентов. При работе с проектом:

### 🎯 Золотые правила

1. **Следуйте слоистой архитектуре** — не нарушайте зависимости между слоями
2. **Используйте DI** — регистрируйте компоненты через `di_registry`
3. **Пишите тесты** — особенно для критичной логики
4. **Проверяйте паритет** — онлайн и оффлайн фичи должны совпадать
5. **Мониторьте метрики** — используйте sim_reality_check
6. **Обновляйте конфиги** — фильтры, комиссии, сезонность устаревают

### ⭐ NEW: Продвинутые возможности (2025)

7. **Используйте AdaptiveUPGD** — default optimizer для continual learning
8. **Twin Critics enabled** — включено по умолчанию для лучших value estimates
9. **VGS для стабильности** — автоматическое gradient scaling
10. **PBT для hyperparameter tuning** — эволюционная оптимизация
11. **SA-PPO для robustness** — adversarial training против distribution shift

### 🔍 Когда что-то идёт не так

1. **Читайте ошибки внимательно** — stack trace укажет на проблему
2. **Проверьте тесты** — найдите тест, который покрывает проблемную область
3. **Используйте Glob/Grep** — быстро найдите определения классов/функций
4. **Проверьте конфиги** — многие проблемы связаны с неправильной конфигурацией
5. **Проверьте слойную архитектуру** — не нарушены ли зависимости
6. **Проверьте state dict** (для UPGD/VGS/PBT) — state должен быть синхронизирован
7. **Проверьте документацию** — [DOCS_INDEX.md](DOCS_INDEX.md) содержит всё

### 📚 Дальнейшее изучение

- **Начинающие**: Начните с [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)
- **Архитектура**: Изучите [ARCHITECTURE.md](ARCHITECTURE.md)
- **Продвинутые**: [UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) + [twin_critics.md](docs/twin_critics.md)
- **Production**: Следуйте Production Checklist выше
- **Отладка**: [VERIFICATION_INSTRUCTIONS.md](VERIFICATION_INSTRUCTIONS.md)

---

**Последнее обновление**: 2025-11-20
**Версия документации**: 2.0
**Статус**: ✅ Production Ready (UPGD + VGS + Twin Critics + PBT интеграция завершена)

Удачи в разработке! 🚀
