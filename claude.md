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

# Обновление данных
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json
python scripts/refresh_fees.py
python -m services.universe --output data/universe/symbols.json
```

---

## 🛡️ Критические правила (НЕ НАРУШАТЬ!)

1. **ActionProto.volume_frac = TARGET position, НЕ DELTA!**
   - ✅ `next_units = volume_frac * max_position`
   - ❌ `next_units = current_units + volume_frac * max_position` (удвоение!)

2. **Action space bounds: [-1, 1] ВЕЗДЕ**
   - ✅ `np.clip(action, -1.0, 1.0)`
   - ❌ `np.clip(action, 0.0, 1.0)` (потеря short/reduction)

3. **LongOnlyActionWrapper: mapping, НЕ clipping**
   - ✅ `mapped = (action + 1.0) / 2.0`
   - ❌ `clipped = max(0, action)` (теряет reduction сигналы)

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
| PBT deadlock (workers crash) | ready_percentage слишком высокий | `min_ready_members=2`, `ready_check_max_wait=10` |
| Non-monotonic quantiles | NN predictions без sorting | `critic.enforce_monotonicity=true` |
| Value loss не снижается | LSTM states не сбрасываются | Проверьте `_reset_lstm_states_for_done_envs` |
| External features = 0.0 | NaN → 0.0 silent conversion | `log_nan=True` для debugging |
| Градиенты взрываются | UPGD noise слишком высок | Уменьшите `sigma` (0.0005-0.001) |
| `AttributeError` в конфигах | Pydantic V2 API | `model_dump()` вместо `dict()` |
| Feature mismatch | Online/offline паритет | `check_feature_parity.py` |
| PBT state mismatch | VGS не синхронизирован | Проверьте `variance_gradient_scaler.py` state dict |

---

## ✅ FAQ: Закрытые вопросы (НЕ ПЕРЕОТКРЫВАТЬ!)

Эти вопросы были тщательно проанализированы. Подробности: [CRITICAL_ANALYSIS_THREE_PROBLEMS_2025_11_24.md](CRITICAL_ANALYSIS_THREE_PROBLEMS_2025_11_24.md)

| Вопрос | Ответ |
|--------|-------|
| "Look-ahead bias в индикаторах?" | ✅ **Исправлено 2025-11-23**. Все фичи сдвинуты. |
| "VGS недооценивает variance в N раз?" | ⚠️ **By design**. Var[mean(g)] валиден, работает в production. |
| "-10.0 bankruptcy penalty слишком резкий?" | ✅ **Стандартная практика RL**. Potential shaping даёт smooth gradient. |

---

## 📊 СТАТУС ПРОЕКТА (2025-11-25)

### ✅ Production Ready

Все критические исправления применены и протестированы. **200+ тестов** с 97%+ pass rate.

| Компонент | Статус | Тесты |
|-----------|--------|-------|
| AdaptiveUPGD Optimizer | ✅ Production | 119/121 |
| Twin Critics + VF Clipping | ✅ Production | 49/50 |
| VGS v3.1 | ✅ Production | 7/7 |
| PBT | ✅ Production | 14/14 |
| SA-PPO | ✅ Production | 16/16 |
| Data Leakage Prevention | ✅ Production | 46/47 |
| Technical Indicators | ✅ Production | 11/16 (C++ pending) |

### ⚠️ Требуется действие

**Переобучите модели**, если они обучены **до 2025-11-25**:
- Data leakage fix (2025-11-23) + close_orig fix (2025-11-25)
- RSI/CCI initialization fixes (2025-11-24)
- Twin Critics GAE fix (2025-11-21)
- LSTM state reset fix (2025-11-21)
- UPGD negative utility fix (2025-11-21)

---

## 📜 История критических исправлений

| Дата | Исправление | Влияние | Отчёт |
|------|-------------|---------|-------|
| **2025-11-25** | close_orig semantic conflict | Data leakage в pipeline | [CLOSE_ORIG_SEMANTIC_FIX_REPORT_2025_11_25.md](CLOSE_ORIG_SEMANTIC_FIX_REPORT_2025_11_25.md) |
| **2025-11-24** | Twin Critics loss aggregation | 25% underestimation | [CRITICAL_ANALYSIS_REPORT_2025_11_24.md](CRITICAL_ANALYSIS_REPORT_2025_11_24.md) |
| **2025-11-24** | RSI/CCI initialization | 5-20x error first 150 bars | [INDICATOR_INITIALIZATION_FIXES_SUMMARY.md](INDICATOR_INITIALIZATION_FIXES_SUMMARY.md) |
| **2025-11-23** | Data leakage (all features) | Look-ahead bias | [DATA_LEAKAGE_FIX_REPORT_2025_11_23.md](DATA_LEAKAGE_FIX_REPORT_2025_11_23.md) |
| **2025-11-23** | VGS v3.1 E[g²] computation | 10,000x underestimation | [VGS_E_G_SQUARED_BUG_REPORT.md](VGS_E_G_SQUARED_BUG_REPORT.md) |
| **2025-11-23** | SA-PPO epsilon + KL | Schedule + 10x faster | [SA_PPO_BUG_FIXES_REPORT_2025_11_23.md](SA_PPO_BUG_FIXES_REPORT_2025_11_23.md) |
| **2025-11-23** | GAE overflow protection | Float32 overflow | [GAE_OVERFLOW_PROTECTION_FIX_REPORT.md](GAE_OVERFLOW_PROTECTION_FIX_REPORT.md) |
| **2025-11-22** | PBT deadlock prevention | Indefinite wait | [BUG_FIXES_REPORT_2025_11_22.md](BUG_FIXES_REPORT_2025_11_22.md) |
| **2025-11-22** | Twin Critics VF Clipping | Independent critic updates | [TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md](TWIN_CRITICS_VF_CLIPPING_VERIFICATION_REPORT.md) |
| **2025-11-21** | Twin Critics GAE | min(Q1,Q2) not applied | [TWIN_CRITICS_GAE_FIX_REPORT.md](TWIN_CRITICS_GAE_FIX_REPORT.md) |
| **2025-11-21** | LSTM state reset | Temporal leakage 5-15% | [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md) |
| **2025-11-21** | UPGD negative utility | Inverted weight protection | [UPGD_NEGATIVE_UTILITY_FIX_REPORT.md](UPGD_NEGATIVE_UTILITY_FIX_REPORT.md) |
| **2025-11-21** | Action space (3 bugs) | Position doubling | [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md) |
| **2025-11-20** | Numerical stability (5 bugs) | Gradient explosions | [CRITICAL_FIXES_5_REPORT.md](CRITICAL_FIXES_5_REPORT.md) |
| **2025-11-20** | Feature engineering (3 bugs) | Volatility bias 1-5% | [CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md) |

---

## О проекте

**TradingBot2** — высокочастотный торговый бот для криптовалют (Binance spot/futures), использующий reinforcement learning (Distributional PPO) для принятия торговых решений.

### Основные характеристики

- **Язык**: Python 3.12 + Cython + C++
- **RL Framework**: Stable-Baselines3 (Distributional PPO with Twin Critics)
- **Optimizer**: AdaptiveUPGD (default) — continual learning
- **Gradient Scaling**: VGS v3.1 — automatic per-layer normalization
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
| Action Space | `test_critical_action_space_fixes.py` |
| LSTM | `test_lstm_episode_boundary_reset.py` |

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

Все отчёты в корневой директории (см. таблицу "История критических исправлений").

Архив: [docs/archive/verification_2025_11/](docs/archive/verification_2025_11/)

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

**Последнее обновление**: 2025-11-25
**Версия документации**: 3.0
**Статус**: ✅ Production Ready (все критические исправления применены)
