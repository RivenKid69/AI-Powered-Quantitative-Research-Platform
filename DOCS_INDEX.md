# TradingBot2 Documentation Index

> **Navigation Hub** для всей документации проекта

---

## 📊 Статус проекта (2025-11-27)

**Production Ready** - Все критические исправления применены и протестированы.

| Компонент | Статус | Версия |
|-----------|--------|--------|
| AdaptiveUPGD Optimizer | ✅ Production | — |
| Twin Critics + VF Clipping | ✅ Production | — |
| VGS | ✅ Production | v3.2 |
| PBT | ✅ Production | — |
| SA-PPO | ✅ Production | — |
| Data Leakage Prevention | ✅ Production | — |
| **Multi-Asset (Stocks)** | ✅ Production | Phase 3 |
| **Execution Providers** | ✅ Production | Phase 4 (L2) |
| **Live Trading Improvements** | ✅ Production | Phase 9 |

**⚠️ Переобучите модели**, если они обучены до 2025-11-26.

---

## 📚 Основная документация

### Ключевые файлы (корень проекта)

| Файл | Описание |
|------|----------|
| [CLAUDE.md](CLAUDE.md) | ⭐ **Master reference** - полная документация (RU) |
| [README.md](README.md) | Обзор проекта и quick start |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитектура системы |
| [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) | Инструкции по сборке |
| [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md) | Быстрый старт |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Руководство по контрибуции |
| [CHANGELOG.md](CHANGELOG.md) | История изменений |

---

## 📖 Техническая документация (docs/)

### Компоненты и функции

| Файл | Описание |
|------|----------|
| [docs/pipeline.md](docs/pipeline.md) | Decision pipeline architecture |
| [docs/bar_execution.md](docs/bar_execution.md) | Bar execution mode |
| [docs/eval.md](docs/eval.md) | Model evaluation framework |
| [docs/parallel.md](docs/parallel.md) | Parallel environments |
| [docs/data_degradation.md](docs/data_degradation.md) | Data degradation simulation |
| [docs/permissions.md](docs/permissions.md) | Role-based access control |
| [docs/no_trade.md](docs/no_trade.md) | No-trade windows |
| [docs/universe.md](docs/universe.md) | Trading universe management |

### ML и оптимизаторы

| Файл | Описание |
|------|----------|
| [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) | ⭐ UPGD optimizer integration |
| [docs/twin_critics.md](docs/twin_critics.md) | ⭐ Twin critics architecture |

### Multi-Asset Support (Phase 2-4, 9)

| Файл | Описание |
|------|----------|
| [CLAUDE.md#multi-exchange-support](CLAUDE.md) | ⭐ Multi-exchange adapters (Binance, Alpaca, Polygon) |
| [CLAUDE.md#stock-training-backtest](CLAUDE.md) | Stock training & backtest pipeline |
| [CLAUDE.md#execution-providers](CLAUDE.md) | Execution providers (L2 simulation) |
| [CLAUDE.md#live-trading-improvements](CLAUDE.md) | Live trading improvements (Phase 9) |

**Supported Assets:**
- **Crypto**: Binance Spot/Futures (24/7)
- **Stocks**: Alpaca/Polygon US Equities (market hours + extended)
- **ETFs**: SPY, QQQ, IWM, GLD, IAU, SGOL, SLV

### Seasonality Framework

| Файл | Описание |
|------|----------|
| [docs/seasonality.md](docs/seasonality.md) | Framework overview |
| [docs/seasonality_quickstart.md](docs/seasonality_quickstart.md) | Quick start guide |
| [docs/seasonality_api.md](docs/seasonality_api.md) | API reference |
| [docs/seasonality_example.md](docs/seasonality_example.md) | Usage examples |
| [docs/seasonality_checklist.md](docs/seasonality_checklist.md) | Deployment checklist |
| [docs/seasonality_QA.md](docs/seasonality_QA.md) | QA process |
| [docs/seasonality_data_policy.md](docs/seasonality_data_policy.md) | Data policy |
| [docs/seasonality_migration.md](docs/seasonality_migration.md) | Migration guide |
| [docs/seasonality_process.md](docs/seasonality_process.md) | Development process |
| [docs/seasonality_signoff.md](docs/seasonality_signoff.md) | Sign-off procedure |

---

## 🗄️ Архив документации

**Все исторические отчёты перемещены в `docs/archive/`**

### Структура архива

```
docs/archive/
├── reports_2025_11_27/           # Отчёты 27 ноября (EV analysis, Signal-Only)
├── reports_2025_11_25_cleanup/   # Основные архивированные отчёты
│   ├── root_reports/             # Критические исправления
│   ├── reports/
│   │   ├── analysis/             # Аналитические отчёты
│   │   ├── audits/               # Аудиты
│   │   ├── bugs/                 # Отчёты о багах
│   │   ├── features/             # Feature mappings
│   │   ├── fixes/                # Отчёты об исправлениях
│   │   ├── integration/          # Интеграционные отчёты
│   │   ├── self_review/          # Self-review отчёты
│   │   ├── summaries/            # Сводки
│   │   ├── tests/                # Тестовые отчёты
│   │   ├── twin_critics/         # Twin Critics отчёты
│   │   └── upgd_vgs/             # UPGD/VGS отчёты
│   └── ...
├── reports_2025_11/              # Отчёты ноябрь 2025
├── reports_2025_11_24/           # Отчёты 24 ноября
├── verification_2025_11/         # Верификация ноябрь
├── audits/                       # Исторические аудиты
├── twin_critics/                 # Twin Critics история
├── pbt/                          # PBT история
└── ...
```

### Ключевые архивные отчёты

Критические исправления (см. `docs/archive/reports_2025_11_25_cleanup/root_reports/`):

| Отчёт | Дата | Тема |
|-------|------|------|
| DATA_LEAKAGE_FIX_REPORT_2025_11_23.md | 2025-11-23 | Data leakage prevention |
| SA_PPO_BUG_FIXES_REPORT_2025_11_23.md | 2025-11-23 | SA-PPO fixes |
| GAE_OVERFLOW_PROTECTION_FIX_REPORT.md | 2025-11-23 | GAE overflow protection |
| TWIN_CRITICS_GAE_FIX_REPORT.md | 2025-11-21 | Twin Critics GAE |
| CRITICAL_LSTM_RESET_FIX_REPORT.md | 2025-11-21 | LSTM state reset |
| UPGD_NEGATIVE_UTILITY_FIX_REPORT.md | 2025-11-21 | UPGD negative utility |
| CRITICAL_FIXES_COMPLETE_REPORT.md | 2025-11-21 | Action space fixes |
| CRITICAL_FIXES_5_REPORT.md | 2025-11-20 | Numerical stability |
| CRITICAL_FIXES_REPORT.md | 2025-11-20 | Feature engineering |

---

## 🧪 Тестирование

### Тестовые файлы

```bash
pytest tests/                          # Все тесты
pytest tests/test_twin_critics*.py -v  # Twin Critics
pytest tests/test_upgd*.py -v          # UPGD
pytest tests/test_pbt*.py -v           # PBT
pytest tests/test_data_leakage*.py -v  # Data Leakage
```

### Статистика тестов

| Категория | Тесты |
|-----------|-------|
| Twin Critics | 49+ |
| UPGD | 119+ |
| VGS | 7+ |
| Data Leakage | 46+ |
| SA-PPO | 16+ |
| PBT | 14+ |

---

## 📍 Навигация

| Задача | Куда смотреть |
|--------|---------------|
| Новичок в проекте | [CLAUDE.md](CLAUDE.md) |
| Архитектура | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Быстрый старт | [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md) |
| Twin Critics | [docs/twin_critics.md](docs/twin_critics.md) |
| UPGD Optimizer | [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) |
| Multi-Asset (Stocks) | [CLAUDE.md](CLAUDE.md) (см. Phase 2-4, 9) |
| Live Trading | [CLAUDE.md](CLAUDE.md) (см. Phase 9) |
| Seasonality | [docs/seasonality.md](docs/seasonality.md) |
| Исторические отчёты | `docs/archive/` |

---

**Last Updated**: 2025-11-27
**Status**: ✅ Production Ready
**Version**: 4.0 (Multi-Asset Support + VGS v3.2)
