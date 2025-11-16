# MACD Indicator Validation Audit

**Date**: 2025-11-16
**Feature**: MACD (indices 8-9 in observation space)
**Status**: ⚠️ ISSUES FOUND - REQUIRES FIXES

---

## Executive Summary

Comprehensive audit of MACD (Moving Average Convergence Divergence) indicator implementation revealed:

- ✅ **NO look-ahead bias** - Implementation is mathematically correct
- ⚠️ **MISSING validity flags** - Unlike ma5/ma20, MACD lacks validation flags
- ⚠️ **Fallback masks problems** - 0.0 fallback indistinguishable from real MACD=0
- ⚠️ **No logging** - Silent fallback hides data quality issues

**Recommendation**: Add `macd_valid` and `macd_signal_valid` flags (like ma5/ma20)

---

## 1. Data Flow Analysis

### Path 1: MarketSimulator → Cython

```
MarketSimulator.cpp:330-339 (C++)
  ↓ MACD calculation
  ├─ ema12 = ema_step(ema12, closev, alpha12, ema12_init)
  ├─ ema26 = ema_step(ema26, closev, alpha26, ema26_init)
  ├─ macd = ema12 - ema26
  └─ v_macd[i] = macd  // ALWAYS writes value (even 0 on cold start)

  ↓
MarketSimulator.cpp:442 (getter)
  └─ return get_or_nan(v_macd, i)  // Returns NaN only if i >= v_macd.size()

  ↓
mediator.py:1100-1103 (Python wrapper)
  ├─ macd = float(sim.get_macd(row_idx))
  └─ except Exception: pass  // ⚠️ Silent fallback, no logging

  ↓
mediator.py:1304 (pass to obs_builder)
  └─ float(indicators["macd"])

  ↓
obs_builder.pyx:255-258 (Cython)
  ├─ out_features[8] = macd if not isnan(macd) else 0.0  // ⚠️ No validity flag
  └─ out_features[9] = macd_signal if not isnan(macd_signal) else 0.0
```

---

## 2. Mathematical Validation

### 2.1 MACD Formula (Standard)

```
MACD = EMA_12(close) - EMA_26(close)
Signal = EMA_9(MACD)

where:
  EMA_t = α × price_t + (1-α) × EMA_{t-1}
  α_12 = 2/(12+1) ≈ 0.154
  α_26 = 2/(26+1) ≈ 0.074
  α_9  = 2/(9+1)  = 0.200
```

### 2.2 Implementation Verification

**Source**: `MarketSimulator.cpp:330-339`

```cpp
// MACD(12,26) + signal(9) на close
const double alpha12 = 2.0 / (12.0 + 1.0);  // ✅ Correct: 0.154
const double alpha26 = 2.0 / (26.0 + 1.0);  // ✅ Correct: 0.074
const double alpha9  = 2.0 / ( 9.0 + 1.0);  // ✅ Correct: 0.200

ema12 = ema_step(ema12, closev, alpha12, ema12_init);
ema26 = ema_step(ema26, closev, alpha26, ema26_init);
double macd = ema12 - ema26;  // ✅ Correct formula
v_macd[i] = macd;

ema9  = ema_step(ema9, macd, alpha9, ema9_init);
v_macd_signal[i] = ema9;  // ✅ Signal line correct
```

**EMA Step Function** (`MarketSimulator.cpp:259-262`):

```cpp
static inline double ema_step(double prev, double x, double alpha, bool& init) {
    if (!init) { init = true; return x; }  // Cold start: use current value
    return alpha * x + (1.0 - alpha) * prev;  // ✅ Standard EMA formula
}
```

**Verification**:
- ✅ Alpha coefficients match standard MACD(12,26,9)
- ✅ EMA formula is textbook-correct
- ✅ Uses only current and past prices (no look-ahead)

---

## 3. Look-Ahead Bias Analysis

### 3.1 Data Access Pattern

**MarketSimulator.cpp:264-268**:
```cpp
void MarketSimulator::update_indicators(std::size_t i) {
    const double closev = m_close ? m_close[i] : m_last_close;  // ✅ Current bar
    const double highv  = m_high  ? m_high[i]  : m_last_high;
    const double lowv   = m_low   ? m_low[i]   : m_last_low;
    // ...
```

**Critical Check**:
- `closev = m_close[i]` uses **CURRENT** bar close (index `i`)
- EMA calculation uses `prev` (previous EMA) and `x` (current price)
- **NO access** to `m_close[i+1]` or future data

### 3.2 Temporal Causality

```
Time:     t-2        t-1        t (current)    t+1
         -----      -----       -----         -----
EMA_12:  EMA₋₂  →  EMA₋₁   →   EMA_t         [NOT accessed]
                                 ↑
                                uses: close_t, EMA₋₁
```

**Verification**:
- ✅ EMA at time `t` depends only on `close_t` and `EMA_{t-1}`
- ✅ No future data leakage
- ✅ Causal relationship preserved

**CONCLUSION**: ✅ **NO LOOK-AHEAD BIAS DETECTED**

---

## 4. Identified Issues

### 4.1 Missing Validity Flags (HIGH PRIORITY)

**Problem**: Unlike ma5/ma20, MACD lacks validity flags.

**Comparison**:

| Indicator | Value Index | Validity Flag Index | Fallback |
|-----------|-------------|---------------------|----------|
| ma5       | 3           | 4 (ma5_valid)       | 0.0      |
| ma20      | 5           | 6 (ma20_valid)      | 0.0      |
| **macd**  | **8**       | **❌ MISSING**      | **0.0**  |
| **macd_signal** | **9** | **❌ MISSING**    | **0.0**  |

**Code Evidence** (obs_builder.pyx):

```cython
# ma5 with validity flag (lines 237-241)
ma5_valid = not isnan(ma5)
out_features[feature_idx] = ma5 if ma5_valid else 0.0
feature_idx += 1
out_features[feature_idx] = 1.0 if ma5_valid else 0.0  # ✅ Validity flag
feature_idx += 1

# MACD WITHOUT validity flag (lines 254-258)
out_features[feature_idx] = macd if not isnan(macd) else 0.0
feature_idx += 1  # ⚠️ No validity flag!
out_features[feature_idx] = macd_signal if not isnan(macd_signal) else 0.0
feature_idx += 1  # ⚠️ No validity flag!
```

**Impact**:
- Agent cannot distinguish between:
  - MACD = 0.0 (real convergence, valid signal)
  - MACD = 0.0 (fallback due to missing data)
- Model may learn spurious patterns from fake zeros

### 4.2 Silent Fallback (MEDIUM PRIORITY)

**Problem**: Exception handling suppresses errors without logging.

**Code** (mediator.py:1100-1119):
```python
try:
    if hasattr(sim, "get_macd"):
        macd = float(sim.get_macd(row_idx))
    # ... other indicators ...
except Exception:
    pass  # ⚠️ Silent failure - no logging, no warning
```

**Impact**:
- Data quality issues go undetected
- Debugging is difficult when fallback occurs
- No monitoring of fallback frequency

### 4.3 Fallback Value Choice (LOW PRIORITY)

**Current**: Fallback to 0.0 for both MACD and signal.

**Analysis**:
- MACD = 0 means EMA_12 = EMA_26 (convergence)
- In reality, this is a VALID market state (neutral trend)
- Using 0.0 as fallback creates ambiguity

**Alternative considered**:
- Keep NaN to explicitly signal unavailability
- Requires model to handle NaN inputs

**Decision**: Keep 0.0 fallback BUT add validity flags (addresses ambiguity)

---

## 5. Cold Start Behavior

### 5.1 Initialization

**MarketSimulator.cpp:43-47**:
```cpp
// Vectors initialized with NaN
auto init_vec = [this](std::vector<double>& v) { v.assign(m_n, NAN); };
init_vec(v_macd); init_vec(v_macd_signal); // ... other indicators
```

### 5.2 First Bar (i=0)

**MarketSimulator.cpp:334-337**:
```cpp
ema12 = ema_step(ema12, closev, alpha12, ema12_init);  // returns closev (not init)
ema26 = ema_step(ema26, closev, alpha26, ema26_init);  // returns closev (not init)
double macd = ema12 - ema26;  // = closev - closev = 0.0
v_macd[i] = macd;  // ✅ ALWAYS writes (overwrites NaN)
```

**Result**:
- Bar 0: MACD = 0 (both EMAs equal close[0])
- Bar 1+: MACD calculated normally
- **v_macd NEVER contains NaN** after step() is called

**Conclusion**: Cold start is handled, but MACD=0 at i=0 is technically valid (not an error).

---

## 6. Research & Best Practices

### 6.1 Industry Standards

1. **Appel, Gerald (1979)**: "The Moving Average Convergence Divergence Trading Method"
   - MACD = EMA_12 - EMA_26 ✅ Matches our implementation
   - Signal = EMA_9(MACD) ✅ Matches our implementation

2. **Investopedia**: "MACD Calculation"
   - Standard parameters: (12, 26, 9) ✅ Confirmed

3. **QuantConnect**: "Handling Missing Technical Indicators"
   - **Recommendation**: Use validity flags for all indicators
   - **Reason**: Distinguishes missing data from neutral signals
   - ✅ Supports our proposal to add `macd_valid`/`macd_signal_valid`

### 6.2 Lag Analysis

**Theoretical Lag**:
- EMA_26 lag ≈ (26-1)/2 ≈ 12.5 bars
- MACD inherits EMA_26 lag ≈ 13 bars (natural, not a bug)

**Impact**: ⚠️ MEDIUM
- Lag is inherent to indicator design
- Not a data corruption issue
- Trader must account for lag in strategy

---

## 7. Recommendations

### 7.1 REQUIRED FIXES

1. **Add MACD Validity Flags** (CRITICAL)
   ```python
   # obs_builder.pyx
   macd_valid = not isnan(macd)
   out_features[8] = macd if macd_valid else 0.0
   out_features[9] = 1.0 if macd_valid else 0.0  # NEW: macd_valid flag

   macd_signal_valid = not isnan(macd_signal)
   out_features[10] = macd_signal if macd_signal_valid else 0.0
   out_features[11] = 1.0 if macd_signal_valid else 0.0  # NEW: macd_signal_valid flag
   ```

   **Impact**:
   - Observation space: 56 → 58 features
   - Requires model retraining (architecture change)

2. **Add Logging for Fallbacks** (HIGH)
   ```python
   # mediator.py
   except Exception as e:
       logging.getLogger(__name__).debug(
           f"MarketSimulator indicator fetch failed at row {row_idx}: {e}. "
           f"Using fallback values."
       )
   ```

3. **Update Documentation** (HIGH)
   - Update `FEATURE_MAPPING_56.md` → `FEATURE_MAPPING_58.md`
   - Document new indices 9 (macd_valid), 11 (macd_signal_valid)
   - Update observation space shape in configs

### 7.2 OPTIONAL IMPROVEMENTS

4. **Move MACD to transformers.py** (OPTIONAL)
   - **Pros**: Transparency, easier debugging, consistency with ma5/ma20
   - **Cons**: Requires refactoring, C++ implementation is faster
   - **Decision**: Keep in MarketSimulator for now (performance priority)

5. **Add Unit Tests** (RECOMMENDED)
   ```python
   def test_macd_no_lookahead():
       """Verify MACD uses only current and past prices."""
       sim = MarketSimulator(...)
       for i in range(len(prices)):
           macd = sim.get_macd(i)
           # Assert macd depends only on prices[0:i+1]
   ```

---

## 8. Validation Checklist

- [x] **Mathematical Correctness**: MACD formula matches industry standard
- [x] **Temporal Causality**: No look-ahead bias detected
- [x] **Code Review**: C++ implementation verified line-by-line
- [x] **Data Flow**: Traced from C++ → Python → Cython
- [ ] **Validity Flags**: NOT IMPLEMENTED (TO BE ADDED)
- [ ] **Logging**: NOT IMPLEMENTED (TO BE ADDED)
- [x] **Cold Start**: Handled (MACD=0 at i=0 is valid)
- [x] **NaN Handling**: Fallback to 0.0 (ambiguous without validity flags)

---

## 9. Final Verdict

### Критические находки:

| Issue | Severity | Status | Action Required |
|-------|----------|--------|-----------------|
| Missing validity flags | 🔴 HIGH | ❌ Not Fixed | Add `macd_valid`, `macd_signal_valid` |
| Silent fallback | 🟡 MEDIUM | ❌ Not Fixed | Add logging |
| Look-ahead bias | N/A | ✅ Not Present | No action |
| Incorrect formula | N/A | ✅ Not Present | No action |
| Cold start NaN | N/A | ✅ Handled | No action |

### Выводы анализа пользователя:

1. ✅ **NaN при холодном старте**: ЧАСТИЧНО - векторы инициализируются NaN, но перезаписываются при первом step()
2. ✅ **Fallback маскирует проблемы**: РЕАЛЬНО - без флага валидности нельзя различить реальный 0 и fallback
3. ✅ **Зависимость от внешнего модуля**: РЕАЛЬНО - но не проблема (архитектурное решение)
4. ✅ **Запаздывание**: РЕАЛЬНО - но это свойство индикатора (≈13 баров), не баг
5. ❌ **Неизвестная реализация**: ЛОЖНАЯ ТРЕВОГА - реализация проверена, корректна

### Итоговая рекомендация:

**ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ**:
1. Добавить флаги валидности (как для ma5/ma20)
2. Добавить логирование fallback случаев
3. Обновить документацию (56→58 признаков)

**НЕ ТРЕБУЕТСЯ**:
- Исправление формулы (корректна)
- Защита от look-ahead bias (отсутствует)
- Изменение fallback значения (0.0 приемлемо с флагами)

---

**Audit completed by**: Claude Code
**Next steps**: Implement required fixes (validity flags + logging)
