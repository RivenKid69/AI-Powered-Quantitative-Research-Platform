# TradingBot2 Documentation Index

> **Navigation Hub** для всей документации проекта

---

## 🔥 CRITICAL - READ FIRST (2025-11-21)

**MAJOR UPDATE**: Multiple critical bugs discovered and fixed. All fixes are active by default.

### 🔴 Latest Critical Fixes (2025-11-21)

#### Numerical & Computational Fixes
- 🚨 [NUMERICAL_ISSUES_FIX_SUMMARY.md](NUMERICAL_ISSUES_FIX_SUMMARY.md) - **LSTM + NaN handling** (2 issues fixed)
  - **Issue #4**: LSTM states not reset on episode boundaries → **FIXED** (5-15% improvement expected)
  - **Issue #2**: NaN → 0.0 silent conversion → **IMPROVED** (logging added)
- 🚨 [CRITICAL_LSTM_RESET_FIX_REPORT.md](CRITICAL_LSTM_RESET_FIX_REPORT.md) - **Full LSTM reset documentation**
- 🚨 [FINAL_FIX_SUMMARY_2025_11_21.md](FINAL_FIX_SUMMARY_2025_11_21.md) - **Final comprehensive report**

#### Action Space Fixes
- 🚨 [CRITICAL_FIXES_COMPLETE_REPORT.md](CRITICAL_FIXES_COMPLETE_REPORT.md) - **3 critical action space bugs**
  - **Problem #1**: Sign convention mismatch in LongOnlyWrapper → **FIXED**
  - **Problem #2**: Position semantics DELTA→TARGET → **FIXED** (prevents position doubling!)
  - **Problem #3**: Action space range [0,1] vs [-1,1] → **FIXED**

#### Data & Critic Fixes (2025-11-20)
- 🚨 [CRITICAL_FIXES_REPORT.md](CRITICAL_FIXES_REPORT.md) - **3 critical data bugs**
  - **Problem #10**: Temporal causality violation in stale data → **FIXED**
  - **Problem #11**: Cross-symbol contamination in normalization → **FIXED**
  - **Problem #12**: Inverted quantile loss formula → **FIXED**

### 🛡️ Regression Prevention
- 📋 [REGRESSION_PREVENTION_CHECKLIST.md](REGRESSION_PREVENTION_CHECKLIST.md) - **Обязательный checklist для разработчиков**
- 📋 [docs/CRITICAL_BUGS_PREVENTION.md](docs/CRITICAL_BUGS_PREVENTION.md) - Prevention guide

**⚠️ Action Required**:
- LSTM models trained before 2025-11-21 → **RETRAIN RECOMMENDED** (5-15% improvement)
- Models with action space issues → **RETRAIN REQUIRED**
- Models with data bugs (2025-11-20) → **RETRAIN REQUIRED**

**Test Coverage**: 52+ new tests added (all passing ✅)

---

## 📚 Core Documentation

### Essential Documents
- [README.md](README.md) - Project overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design principles
- [CLAUDE.md](CLAUDE.md) - Complete project documentation (Russian) ⭐ **Updated with critical fixes**
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes ⭐ **Updated with bugs #10-12**
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) - Build and compilation instructions

### Quick References
- [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md) - Quick start guide
- [FILE_REFERENCE.md](FILE_REFERENCE.md) - File organization reference

## 📖 Technical Documentation (docs/)

### Features & Components
- [docs/pipeline.md](docs/pipeline.md) - Decision pipeline architecture
- [docs/bar_execution.md](docs/bar_execution.md) - Bar execution mode
- [docs/large_orders.md](docs/large_orders.md) - Large order execution algorithms
- [docs/moving_average.md](docs/moving_average.md) - Moving average implementation
- [docs/dynamic_spread.md](docs/dynamic_spread.md) - Dynamic spread modeling

### Risk & Trading
- [docs/no_trade.md](docs/no_trade.md) - No-trade windows documentation
- [docs/data_degradation.md](docs/data_degradation.md) - Data degradation simulation
- [docs/permissions.md](docs/permissions.md) - Role-based access control

### Market Data & Seasonality
- [docs/seasonality.md](docs/seasonality.md) - Seasonality framework overview
- [docs/seasonality_quickstart.md](docs/seasonality_quickstart.md) - Quick start guide
- [docs/seasonality_QA.md](docs/seasonality_QA.md) - QA process for seasonality
- [docs/seasonality_api.md](docs/seasonality_api.md) - Seasonality API reference
- [docs/seasonality_checklist.md](docs/seasonality_checklist.md) - Deployment checklist
- [docs/seasonality_data_policy.md](docs/seasonality_data_policy.md) - Data policy
- [docs/seasonality_example.md](docs/seasonality_example.md) - Usage examples
- [docs/seasonality_migration.md](docs/seasonality_migration.md) - Migration guide
- [docs/seasonality_process.md](docs/seasonality_process.md) - Development process
- [docs/seasonality_signoff.md](docs/seasonality_signoff.md) - Sign-off procedure

### ML & Training
- [docs/parallel.md](docs/parallel.md) - Parallel environments and randomness
- [docs/twin_critics.md](docs/twin_critics.md) - Twin critics architecture
- [docs/eval.md](docs/eval.md) - Model evaluation framework
- [docs/UPGD_INTEGRATION.md](docs/UPGD_INTEGRATION.md) - UPGD optimizer integration

### Algorithm-Specific Documentation
- [docs/parkinson_volatility.md](docs/parkinson_volatility.md) - Parkinson volatility estimator
- [docs/yang_zhang_volatility.md](docs/yang_zhang_volatility.md) - Yang-Zhang volatility estimator
- [docs/universe.md](docs/universe.md) - Trading universe management

### PPO & RL Fixes
- [docs/ppo_log_ratio_fix.md](docs/ppo_log_ratio_fix.md) - PPO log ratio clipping fix
- [docs/ppo_value_function_clipping_explained.md](docs/ppo_value_function_clipping_explained.md) - Value function clipping explained
- [docs/ppo_target_clipping_fix.md](docs/ppo_target_clipping_fix.md) - Target clipping fix
- [docs/distributional_vf_clipping.md](docs/distributional_vf_clipping.md) - Distributional VF clipping
- [docs/explained_variance_fix_ru.md](docs/explained_variance_fix_ru.md) - Explained variance fix (Russian)
- [docs/lagrangian_constraint_gradient_flow_fix.md](docs/lagrangian_constraint_gradient_flow_fix.md) - Lagrangian gradient flow fix
- [docs/price_validation_fix.md](docs/price_validation_fix.md) - Price validation fix

### Advanced Fixes
- [docs/ADVANTAGE_NORMALIZATION_FIX.md](docs/ADVANTAGE_NORMALIZATION_FIX.md) - Advantage normalization fix
- [docs/ADVANTAGE_STD_FLOOR_FIX.md](docs/ADVANTAGE_STD_FLOOR_FIX.md) - Advantage std floor fix
- [docs/ADVANTAGE_STD_FLOOR_FIX_V2.md](docs/ADVANTAGE_STD_FLOOR_FIX_V2.md) - Advantage std floor fix v2
- [docs/AWR_WEIGHTING.md](docs/AWR_WEIGHTING.md) - AWR weighting methodology
- [docs/CATEGORICAL_VF_CLIPPING_FIX.md](docs/CATEGORICAL_VF_CLIPPING_FIX.md) - Categorical VF clipping fix
- [docs/GRADIENT_FLOW_FIX_CATEGORICAL_PROJECTION.md](docs/GRADIENT_FLOW_FIX_CATEGORICAL_PROJECTION.md) - Gradient flow fix
- [docs/PPO_VF_CLIPPING_FIX.md](docs/PPO_VF_CLIPPING_FIX.md) - PPO VF clipping fix
- [docs/STD_DDOF_CORRECTION.md](docs/STD_DDOF_CORRECTION.md) - Standard deviation DDOF correction

### Validation & Verification
- [docs/ADVANTAGE_NORMALIZATION_VALIDATION_REPORT.md](docs/ADVANTAGE_NORMALIZATION_VALIDATION_REPORT.md) - Advantage normalization validation
- [docs/PREV_PRICE_VALIDATION_REPORT.md](docs/PREV_PRICE_VALIDATION_REPORT.md) - Previous price validation
- [docs/FINAL_VALIDATION_SUMMARY.md](docs/FINAL_VALIDATION_SUMMARY.md) - Final validation summary
- [docs/VF_CLIPPING_FIX_VERIFICATION.md](docs/VF_CLIPPING_FIX_VERIFICATION.md) - VF clipping verification
- [docs/FINAL_SOLUTION.md](docs/FINAL_SOLUTION.md) - Final solution summary
- [docs/advantage_normalization_analysis.md](docs/advantage_normalization_analysis.md) - Advantage normalization analysis

## 🐛 Bug Reports & Fixes (docs/reports/bugs/)

### Active Bug Reports
- [INTEGRATION_PROBLEMS_DETAILED_ANALYSIS.md](INTEGRATION_PROBLEMS_DETAILED_ANALYSIS.md) - ⚠️ Current integration issues
- [INTEGRATION_TESTING_SUMMARY.md](INTEGRATION_TESTING_SUMMARY.md) - Integration testing summary
- [INTEGRATION_SUCCESS_REPORT.md](INTEGRATION_SUCCESS_REPORT.md) - ✅ Integration success report

### Historical Bug Reports
*To be moved to docs/reports/bugs/*
- BUG_REPORT_RSI_NAN.md - RSI NaN bug
- BUG_REPORT_RETURNS_ZERO.md - Returns zero bug
- RSI_BUG_FIX_REPORT.md - RSI bug fix
- PARKINSON_ERROR_CORRECTION.md - Parkinson volatility error

### Bug Fix Summaries
- BUG_FIX_SUMMARY.md
- BUG_FIXES_SUMMARY.md
- CRITICAL_FIX_SUMMARY.md
- FIXES_SUMMARY.md

## 🔍 Audit Reports (docs/reports/audits/)

### Recent Audits
- DOCUMENTATION_AUDIT_2025-11-11.md - Latest documentation audit
- DEEP_INTEGRATION_AUDIT_REPORT.md - Deep integration audit
- TWIN_CRITICS_COMPREHENSIVE_AUDIT_REPORT.md - Twin critics audit

### Feature Audits
- FEATURE_AUDIT_REPORT.md - General feature audit
- FINAL_4H_AUDIT_REPORT.md - 4H timeframe audit
- DEEP_AUDIT_11_FEATURES_4H_REPORT.md - 11 features 4H audit
- MA5_AUDIT_FINAL_REPORT.md - MA5 indicator audit
- MA20_INDICATOR_AUDIT_REPORT.md - MA20 indicator audit
- SEASONALITY_AUDIT_REPORT.md - Seasonality audit

### System Audits
- AUDIT_REPORT.md - Main audit report
- AUDIT_VERIFICATION_REPORT.md - Verification report
- AUDIT_SELF_CHECK_REPORT.md - Self-check report
- AUDIT_SUMMARY.md - Audit summary
- AUDIT_CRITICAL_FINDINGS.md - Critical findings

## 🔧 Integration & Migration Reports (docs/reports/integration/)

### Integration Status
- ✅ INTEGRATION_SUCCESS_REPORT.md - **Current successful integration**
- INTEGRATION_PROBLEM_LOCALIZATION_REPORT.md - Problem localization
- INTEGRATION_BUGS_VERIFICATION_REPORT.md - Bugs verification
- FINAL_INTEGRATION_ANALYSIS_REPORT.md - Final analysis
- API_FIX_COMPLETED_REPORT.md - API fix completion

### Migration Guides
- MIGRATION_GUIDE_56_TO_62.md - 56→62 feature migration
- MIGRATION_GUIDE_62_TO_63.md - 62→63 feature migration
- PYDANTIC_V2_MIGRATION_SUMMARY.md - Pydantic v2 migration
- MIGRATION_4H_FINAL_REPORT.md - 4H timeframe migration
- MIGRATION_ANALYSIS_DETAILED.md - Detailed migration analysis

## ⚙️ Feature & Component Reports (docs/reports/features/)

### Feature Mappings
- FEATURE_MAPPING_56.md - 56 features mapping
- FEATURE_MAPPING_62.md - 62 features mapping
- FEATURE_MAPPING_63.md - 63 features mapping (current)
- CURRENT_FEATURE_MAPPING_56.md - Current 56 features mapping
- FULL_FEATURES_LIST.md - Complete features list
- OBSERVATION_MAPPING.md - Observation space mapping

### Feature Analysis
- ANALYSIS_56_FEATURES_FINAL_REPORT.md - 56 features final analysis
- ANALYSIS_FEATURES_STRUCTURE.md - Features structure analysis
- DETAILED_FEATURE_CORRUPTION_ANALYSIS.md - Feature corruption analysis
- FEATURE_ADAPTATION_4H_REPORT.md - 4H adaptation

### Specific Features
- GARCH_FEATURE.md - GARCH volatility feature
- GARCH_VERIFICATION.md - GARCH verification
- GARCH_FIX_DOCUMENTATION.md - GARCH fix documentation
- TAKER_BUY_RATIO_ANALYSIS_AND_FIXES.md - Taker buy ratio
- TBR_MOMENTUM_FIX.md - TBR momentum fix

## 📊 Analysis Reports (docs/reports/analysis/)

### Data Analysis
- ANALYSIS_DATA_DISTORTIONS_FULL.md - Data distortions analysis
- ANALYSIS_NAN_HANDLING.md - NaN handling analysis
- ANALYSIS_4H_TIMEFRAME.md - 4H timeframe analysis
- NORMALIZATION_ANALYSIS.md - Normalization analysis
- KL_DIVERGENCE_ANALYSIS.md - KL divergence analysis

### Algorithm Analysis
- PARKINSON_ANALYSIS_MATHEMATICAL.md - Parkinson mathematical analysis
- LAGRANGIAN_GRADIENT_FLOW_ANALYSIS.md - Lagrangian gradient flow
- VF_VARIANCE_DEEP_ANALYSIS.md - VF variance deep analysis
- VF_CLIPPING_ANALYSIS_REPORT.md - VF clipping analysis
- TRAINING_METRICS_ANALYSIS.md - Training metrics analysis
- TRAINING_PIPELINE_ANALYSIS.md - Training pipeline analysis

### System Analysis
- CODEBASE_STRUCTURE_ANALYSIS.md - Codebase structure
- PROJECT_STRUCTURE_ANALYSIS.md - Project structure
- SIZE_ANALYSIS.md - Code size analysis

## 🛠️ Fix Reports (docs/reports/fixes/)

### PPO & Value Function Fixes
- DISTRIBUTIONAL_VF_CLIPPING_FIX.md - Distributional VF clipping
- DISTRIBUTIONAL_VF_CLIPPING_SOLUTION.md - VF clipping solution
- DISTRIBUTIONAL_VF_VARIANCE_FIX.md - VF variance fix
- CATEGORICAL_VF_CLIPPING_FIX.md - Categorical VF clipping
- PER_QUANTILE_VF_CLIPPING_BUG_FIX.md - Per-quantile VF clipping
- VF_CLIPPING_FIX.md - General VF clipping fix

### Statistical Fixes
- COMPREHENSIVE_DDOF_FIX.md - DDOF comprehensive fix
- DDOF_FIX_SUMMARY.md - DDOF fix summary
- FINAL_DDOF_REPORT.md - DDOF final report

### Quantile & Loss Fixes
- QUANTILE_HUBER_LOSS_FIX.md - Quantile Huber loss fix
- QUANTILE_LOSS_FIX.md - Quantile loss fix
- QUANTILE_FIX_FINAL_REPORT.md - Quantile fix final report
- PER_QUANTILE_FIX_SUMMARY.md - Per-quantile fix summary

### Optimization Fixes
- CVAR_LAGRANGIAN_FIX.md - CVaR Lagrangian fix
- HPO_DATA_LEAKAGE_FIX.md - HPO data leakage fix
- FORWARD_LOOKING_BIAS_FIX_REPORT.md - Forward-looking bias fix

### Advantage & Normalization Fixes
- ADVANTAGE_STD_FLOOR_FIX_SUMMARY.md - Advantage std floor
- FINAL_SUMMARY_NAN_FIX.md - NaN fix summary
- NORMALIZATION_RECOMMENDATIONS.md - Normalization recommendations

### Data Fixes
- DATASET_FIX_README.md - Dataset fix documentation
- SEASONALITY_FIXES.md - Seasonality fixes
- YANG_ZHANG_FIX_SUMMARY.md - Yang-Zhang fix

### Integration-Specific Fixes
- TORCH_LOAD_SECURITY_FIX_REPORT.md - Torch load security fix
- ISSUE8_FIX_SUMMARY.md - Issue #8 fix
- BUG_LOCALIZATION_FINAL_REPORT.md - Bug localization

## 🧪 Test & Verification Reports (docs/reports/tests/)

### Test Coverage
- TEST_COVERAGE_REPORT.md - Test coverage analysis
- TEST_REPORT.md - General test report
- COMPREHENSIVE_TEST_VALIDATION_REPORT.md - Comprehensive validation

### Verification Reports
- VERIFICATION_REPORT.md - Main verification report
- FINAL_VERIFICATION_REPORT.md - Final verification
- VERIFICATION_SUMMARY_63.md - 63 features verification
- CRITICAL_BUGS_VERIFICATION_REPORT.md - Critical bugs verification
- CATEGORICAL_VF_CLIPPING_VERIFICATION.md - Categorical VF verification
- FINAL_VF_CLIPPING_VERIFICATION.md - Final VF clipping verification
- CRITICAL_FIX_VERIFICATION.md - Critical fix verification
- SELF_AUDIT_VERIFICATION.md - Self-audit verification

### Deep Verification
- DEEP_VALIDATION_SUMMARY.md - Deep validation summary
- DEEP_VERIFICATION_SUMMARY.md - Deep verification summary
- QUANTILE_HUBER_DEEP_VERIFICATION.md - Quantile Huber deep verification
- VERIFICATION_PER_QUANTILE_FIX.md - Per-quantile verification

### Component Testing
- PARKINSON_TESTING_SUMMARY.md - Parkinson testing
- UPGD_TEST_RESULTS_REPORT.md - UPGD test results
- UPGD_TEST_SUMMARY.md - UPGD test summary
- VGS_TEST_RESULTS.md - VGS test results

## 🔬 UPGD & VGS Reports (docs/reports/upgd_vgs/)

### UPGD Reports
- UPGD_TEST_SUITE_README.md - UPGD test suite documentation
- UPGD_VGS_FIX_DESIGN.md - UPGD-VGS fix design
- UPGD_VGS_PROBLEM4_FIX_SUMMARY.md - Problem #4 fix

### VGS Reports
- VGS_DEEP_ANALYSIS_REPORT.md - Deep analysis
- VGS_FINAL_REPORT.md - Final report
- vgs_param_fix_summary.md - Parameter fix summary
- VGS_PBT_FIX_SUMMARY.md - PBT fix summary

## 👥 Twin Critics Reports (docs/reports/twin_critics/)

- TWIN_CRITICS_COMPREHENSIVE_AUDIT_REPORT.md - Comprehensive audit
- TWIN_CRITICS_FINAL_REPORT.md - Final report
- TWIN_CRITICS_DEFAULT_ENABLED.md - Default enabled configuration
- TWIN_CRITICS_INTEGRATION_COMPLETE.md - Integration completion

## 📝 Self-Review & Critical Analysis

### Self-Reviews
- SELF_REVIEW_REPORT.md - Main self-review
- SELF_REVIEW_CRITICAL_BUGS_FOUND.md - Critical bugs found
- MA5_CRITICAL_SELF_AUDIT.md - MA5 critical audit
- MA5_CRITICAL_REANALYSIS.md - MA5 reanalysis

### Critical Reviews
- CRITICAL_REVIEW.md - Critical review of system
- VERDICT_KL_DIRECTION.md - KL divergence verdict

## 📦 Archive (docs/archive/)

### Deprecated Documentation
*Files to be moved here after review*

### Old Reports
*Historical reports that are no longer relevant*

## 🔗 External Resources

### Codex (AI Prompts)
- [docs/codex/README.md](docs/codex/README.md) - Codex documentation
- [docs/codex/prompt_02_autoexposure_ru.md](docs/codex/prompt_02_autoexposure_ru.md) - Auto-exposure prompt

### Build & Compilation
- COMPILATION_REPORT.md - Build compilation report
- BUG_09_QUICK_REFERENCE.md - Bug #9 quick reference

## 📋 Summary Documents

### Change Summaries
- CHANGES_SUMMARY.md - All changes summary
- DEEP_AUDIT_FIXES_SUMMARY.md - Deep audit fixes
- METRICS_FIXES_SUMMARY.md - Metrics fixes

### Documentation Summaries
- DOCUMENTATION_INDEX.md - This file
- DOCUMENTATION_VERIFICATION_REPORT.md - Documentation verification

### Verification Instructions
- VERIFICATION_INSTRUCTIONS.md - How to verify fixes

## 🎯 Current Status & Priority

### ⚠️ Active Issues
1. **Integration Testing** - See [INTEGRATION_TESTING_SUMMARY.md](INTEGRATION_TESTING_SUMMARY.md)
2. **Integration Problems** - See [INTEGRATION_PROBLEMS_DETAILED_ANALYSIS.md](INTEGRATION_PROBLEMS_DETAILED_ANALYSIS.md)

### ✅ Recently Completed
1. **Integration Success** - [INTEGRATION_SUCCESS_REPORT.md](INTEGRATION_SUCCESS_REPORT.md)
2. **API Fix** - [API_FIX_COMPLETED_REPORT.md](API_FIX_COMPLETED_REPORT.md)
3. **Torch Security** - [TORCH_LOAD_SECURITY_FIX_REPORT.md](TORCH_LOAD_SECURITY_FIX_REPORT.md)

### 🔄 In Progress
- Documentation reorganization
- Test coverage improvement
- Performance optimization

## 📍 Navigation Tips

1. **Looking for a specific feature?** → Check [Feature & Component Reports](#️-feature--component-reports-docsreportsfeatures)
2. **Found a bug?** → Check [Bug Reports](#-bug-reports--fixes-docsreportsbugs)
3. **Need to verify a fix?** → Check [Test & Verification Reports](#-test--verification-reports-docsreportstests)
4. **Understanding architecture?** → Start with [ARCHITECTURE.md](ARCHITECTURE.md)
5. **Quick start?** → See [QUICK_START_REFERENCE.md](QUICK_START_REFERENCE.md)
6. **Russian documentation?** → See [CLAUDE.md](CLAUDE.md)

## 🔄 Maintenance

This index should be updated when:
- New reports are added
- Documentation structure changes
- Major features are added/removed
- Bug fixes are completed

**Last Updated:** 2025-11-21
**Maintained by:** Claude Code
**Status:** ✅ Up to date (Version 2.1)