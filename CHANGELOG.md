# Changelog

## [Unreleased]

### Added
- **Seasonality Support**: Introduced hour-of-week seasonality multipliers to improve simulation fidelity.
  - **Required actions**:
    - Regenerate multipliers with the quick-start script.
    - Validate and update configurations before training or running simulations.
  - **Resources**:
    - [Seasonality overview](docs/seasonality.md)
    - [Quick start guide](docs/seasonality_quickstart.md)
    - [Process checklist](docs/seasonality_checklist.md)
    - [Example notebook](docs/seasonality_example.md)
    - [Migration guide](docs/seasonality_migration.md)
- **Dynamic spread builder**: Added `scripts/build_spread_seasonality.py` for generating
  hour-of-week spread profiles consumed by `slippage.dynamic`. The script
  supports custom output paths, rolling windows and warns when the source
  snapshot exceeds the configured `refresh_warn_days` threshold.
- **Fee settlement & rounding controls**: YAML-конфиги теперь содержат блоки
  `fees.rounding` и `fees.settlement` с безопасными значениями по умолчанию.
  `rounding` умеет использовать `commission_step` из биржевых фильтров и
  таблиц комиссий, а `settlement` описывает расчёт комиссий в альтернативном
  активе (например, BNB) с учётом скидок.
- **Daily turnover caps**: Added configuration fields, runtime enforcement, and
  monitoring visibility for daily USD/BPS turnover limits across per-symbol and
  portfolio aggregates. Includes persistence hooks and targeted pytest coverage
  ensuring partial/deferred execution when caps bind.

### Deprecated
- `LatencyImpl.dump_latency_multipliers` and
  `LatencyImpl.load_latency_multipliers` have been replaced by
  `dump_multipliers` and `load_multipliers`. The old names continue to work but
  emit `DeprecationWarning`. See the migration guide for details.

### Fixed
- **CRITICAL BUG #10: Temporal causality violation in stale data** (2025-11-20)
  - Fixed critical issue where stale bars were returned with PREVIOUS timestamp instead
    of CURRENT timestamp, violating temporal causality and corrupting model training
  - Root cause: `impl_offline_data.py` yielded `prev_bar` directly with old timestamp
  - Solution: Create new `Bar` with current timestamp but stale prices/volume
  - Files modified: `impl_offline_data.py`
  - Tests added: `tests/test_stale_bar_temporal_causality.py` (3 tests)
  - Impact: **CRITICAL** - Models trained with data degradation may have learned incorrect
    temporal patterns. Consider retraining if `stale_prob > 0` was used.
  - Reference: [CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md#problem-1-temporal-causality-violation)

- **CRITICAL BUG #11: Cross-symbol contamination in feature normalization** (2025-11-20)
  - Fixed critical issue where `shift()` applied after concatenating all symbols caused
    last row of Symbol1 to leak into first row of Symbol2, corrupting normalization stats
  - Root cause: `features_pipeline.py` applied `shift(1)` to concatenated DataFrame
  - Solution: Apply `shift()` per-symbol BEFORE concat, use `groupby()` in transform
  - Files modified: `features_pipeline.py`
  - Tests added: `tests/test_normalization_cross_symbol_contamination.py` (4 tests)
  - Impact: **CRITICAL** - Multi-symbol models may have contaminated features. Consider
    retraining if multiple symbols were used with normalization.
  - Reference: [CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md#problem-2-cross-symbol-contamination)

- **CRITICAL BUG #12: Inverted quantile loss formula** (2025-11-20)
  - Fixed critical mathematical error where quantile loss used `Q - T` instead of correct
    `T - Q` formula from Dabney et al. (2018), inverting asymmetric penalties
  - Root cause: `distributional_ppo.py` defaulted to legacy (incorrect) formula
  - Solution: Changed default to `_use_fixed_quantile_loss_asymmetry = True`
  - Files modified: `distributional_ppo.py`
  - Tests added: `tests/test_quantile_loss_formula_default.py` (3 tests)
  - Tests updated: `tests/test_quantile_loss_with_flag.py` (8 tests, all passing)
  - Impact: **CRITICAL** - Quantile critic models have suboptimal convergence and biased
    CVaR estimates. **STRONGLY RECOMMENDED** to retrain all quantile-based models.
  - Reference: [CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md#problem-3-inverted-quantile-loss)
  - Academic reference: Dabney et al. (2018) "Distributional RL with Quantile Regression"

- **Bug #9: VGS parameter tracking after model load** - Fixed critical issue where VGS
  (Variance Gradient Scaler) tracked stale parameter copies instead of actual policy
  parameters after `model.load()`, causing gradient scaling to have no effect on training
  after checkpoint restoration.
  - Root cause: VGS pickled parameter references that became stale after unpickling
  - Solution: Exclude `_parameters` from pickle state and relink via `update_parameters()`
    after load
  - Files modified: `variance_gradient_scaler.py`, `distributional_ppo.py`
  - Impact: Critical for production use of checkpointing with VGS enabled

- Ensured the explained-variance reserve path preserves training masks by
  default so no-trade windows and other zero-weight samples no longer depress
  EV metrics.
