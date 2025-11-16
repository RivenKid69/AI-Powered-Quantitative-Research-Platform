# ДЕТАЛЬНЫЙ АНАЛИЗ ИСКАЖЕНИЙ ДАННЫХ ПО ПРИЗНАКАМ (ЭТАП 1/10)

## Признаки 0-5 (Блок Bar Level + MA Features)

---

### ПРИЗНАК 0: `price` (Текущая цена)

**Индекс:** 0
**Тип:** float32
**Диапазон:** > 0.0 (строго положительные значения)
**Источник:** `mediator.py:1260` → `_extract_market_data()` → `mark_price`
**Обработка:** `obs_builder.pyx:237` - прямое присвоение без нормализации

#### ПУТЬ ДАННЫХ:
1. **Входные данные** → DataFrame колонка `close` или `mark_price`
2. **Валидация P0** (mediator.py:1015-1053): `_validate_critical_price(mark_price)`
   - Проверка на NaN → ValueError
   - Проверка на Inf → ValueError
   - Проверка на price <= 0 → ValueError
3. **Передача** → `build_observation_vector()` parameter `price`
4. **Валидация P1** (obs_builder.pyx:633): `_validate_price(price, "price")`
   - Дублирующая проверка на NaN/Inf/<=0
5. **Запись** → `out_features[0] = price` (obs_builder.pyx:237)

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. NaN Propagation (КРИТИЧНО)**
- **Источник:** Пропущенные данные в исторических свечах
- **Вероятность:** НИЗКАЯ (строгая валидация на 2 уровнях)
- **Последствия:** Полный сбой модели (ValueError raises)
- **Митигация:**
  - Fail-fast validation (P0 + P1)
  - Explicit error messages
  - NO silent fallbacks (по дизайну)

**Ссылки на best practices:**
- Martin Fowler: "Fail-fast validation" - ловить ошибки рано
- IEEE 754: NaN propagation requires explicit handling
- Financial data standards: validation at ingestion

**2. Inf Values (КРИТИЧНО)**
- **Источник:** Арифметическое переполнение в upstream calculations
- **Вероятность:** ОЧЕНЬ НИЗКАЯ
- **Последствия:** ValueError с детальной диагностикой
- **Митигация:** Двухуровневая валидация с isfinite()

**3. Zero/Negative Prices (КРИТИЧНО)**
- **Источник:** Ошибки данных, неправильная денормализация
- **Вероятность:** НИЗКАЯ (биржевые данные всегда > 0)
- **Последствия:** ValueError
- **Митигация:** Строгая проверка `price > 0.0`

**4. Extreme Values (ВНИМАНИЕ)**
- **Источник:** Flash crashes, данные с разных бирж
- **Вероятность:** СРЕДНЯЯ
- **Последствия:** Модель видит нереалистичные цены
- **Митигация:** НЕТ встроенной (по дизайну - модель должна обучиться)
- **РЕКОМЕНДАЦИЯ:** Добавить sanity check (price в разумном диапазоне ±50% от MA)

**5. Precision Loss (НИЗКАЯ КРИТИЧНОСТЬ)**
- **Источник:** float32 вместо float64
- **Вероятность:** ГАРАНТИРОВАНА
- **Последствия:** Потеря точности ~7 значащих цифр (vs 15 для float64)
- **Для BTC ~$50,000:** Точность ~$0.005 (0.5 cent) - ПРИЕМЛЕМО
- **Митигация:** НЕТ (архитектурное решение для производительности)

---

### ПРИЗНАК 1: `log_volume_norm` (Нормализованный объем)

**Индекс:** 1
**Тип:** float32
**Диапазон:** [-1, 1] (после tanh)
**Источник:** `mediator.py:1063-1065` → `_extract_market_data()`
**Формула:** `tanh(log1p(quote_volume / 240e6))`
**Обработка:** `obs_builder.pyx:239` - прямое присвоение

#### ПУТЬ ДАННЫХ:
1. **Входные данные** → DataFrame `quote_asset_volume`
2. **Извлечение** (mediator.py:1059): `_get_safe_float(row, "quote_asset_volume", 1.0, min_value=0.0)`
   - Default: 1.0
   - min_value=0.0 → гарантирует volume >= 0
3. **Нормализация** (mediator.py:1063-1065):
   ```python
   if quote_volume > 0:
       log_volume_norm = float(np.tanh(np.log1p(quote_volume / 240e6)))
   else:
       log_volume_norm = 0.0
   ```
4. **Валидация P2** (obs_builder.pyx:639): `_validate_volume_metric(log_volume_norm, "log_volume_norm")`
5. **Запись** → `out_features[1] = log_volume_norm`

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Missing Volume Data (КРИТИЧНО)**
- **Источник:** Пропуски в данных биржи
- **Вероятность:** СРЕДНЯЯ (особенно старые данные)
- **Последствия:** Fallback к 1.0 → log_volume_norm ≈ -12.9 (занижение)
- **Митигация:**
  - Default 1.0 вместо 0.0 (избегает log1p(0) = 0)
  - if quote_volume > 0 check
- **ПРОБЛЕМА:** Default 1.0 СЛИШКОМ МАЛ для 4h таймфрейма!
  - Типичный 4h volume BTC: $50M-$500M
  - Default 1.0 → tanh(log1p(1/240e6)) ≈ tanh(-19.3) ≈ -1.0
  - **СИГНАЛ ИСКАЖЕНИЯ:** "минимальный объем" vs "пропущенные данные"

**BEST PRACTICE VIOLATION:**
- Cube Software: "Use domain-appropriate defaults"
- ТЕКУЩИЙ: 1.0 универсальный
- РЕКОМЕНДАЦИЯ: Использовать median volume за последние N баров как default

**2. Timeframe Mismatch (КРИТИЧНО)**
- **Источник:** Делитель 240e6 оптимизирован для 4h баров
- **Вероятность:** ГАРАНТИРОВАНА при смене таймфрейма
- **Последствия:** Неправильная нормализация
- **Пример:**
  - 1h bar: типичный volume ~$12.5M → делитель должен быть 60e6
  - Но используется 240e6 → все нормализуется к меньшим значениям
- **Митигация:** НЕТ (хардкод в коде)
- **РЕКОМЕНДАЦИЯ:** Параметризовать делитель через config

**3. log1p Domain Error (ЗАЩИЩЕНО)**
- **Источник:** Отрицательный volume (невозможен для log1p(x<-1))
- **Вероятность:** НУЛЕВАЯ (min_value=0.0 защита)
- **Митигация:** ИДЕАЛЬНАЯ (P0 защита с min_value=0.0)

**4. Saturation Effects (ВНИМАНИЕ)**
- **Источник:** Экстремально большой объем
- **Вероятность:** НИЗКАЯ (но возможна при листингах, новостях)
- **Последствия:** tanh saturation → все большие объемы выглядят одинаково
- **Пример:**
  - volume = 1B → tanh(log1p(1e9/240e6)) = tanh(1.42) ≈ 0.89
  - volume = 10B → tanh(log1p(10e9/240e6)) = tanh(3.72) ≈ 0.999
  - **ПОТЕРЯ ИНФОРМАЦИИ** о масштабе события
- **Митигация:** НЕТ
- **BEST PRACTICE:** "Feature Engineering for ML" (Google) - избегать преждевременной saturation

**5. Zero Volume Bars (ВНИМАНИЕ)**
- **Источник:** Пропуски торговли, деградация данных
- **Вероятность:** НИЗКАЯ (но возможна)
- **Последствия:** log_volume_norm = 0.0 (специальный сигнал)
- **Митигация:** if quote_volume > 0 check - ПРАВИЛЬНО
- **Альтернатива:** Использовать предыдущее значение (forward fill)

---

### ПРИЗНАК 2: `rel_volume` (Относительный объем)

**Индекс:** 2
**Тип:** float32
**Диапазон:** [-1, 1] (после tanh)
**Источник:** `mediator.py:1067-1069` → `_extract_market_data()`
**Формула:** `tanh(log1p(volume / 24000.0))`
**Обработка:** `obs_builder.pyx:241` - прямое присвоение

#### ПУТЬ ДАННЫХ:
1. **Входные данные** → DataFrame `volume`
2. **Извлечение** (mediator.py:1058): `_get_safe_float(row, "volume", 1.0, min_value=0.0)`
3. **Нормализация** (mediator.py:1067-1069):
   ```python
   if volume > 0:
       rel_volume = float(np.tanh(np.log1p(volume / 24000.0)))
   else:
       rel_volume = 0.0
   ```
4. **Валидация P2** (obs_builder.pyx:640): `_validate_volume_metric(rel_volume, "rel_volume")`
5. **Запись** → `out_features[2] = rel_volume`

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Default Value Distortion (КРИТИЧНО)**
- **Источник:** Пропущенные данные → default 1.0
- **Вероятность:** СРЕДНЯЯ
- **Последствия:**
  - tanh(log1p(1.0/24000.0)) = tanh(4.16e-5) ≈ 4.16e-5 (почти 0)
  - НО это отличается от true zero (volume = 0)
- **ПРОБЛЕМА:** Неразличимость "малый объем" vs "пропущенные данные"
- **Митигация:** Нет встроенной
- **РЕКОМЕНДАЦИЯ:** Добавить validity flag (как для индикаторов)

**2. Normalization Scale Mismatch (КРИТИЧНО)**
- **Источник:** Делитель 24000.0 не документирован
- **Вопрос:** Это contracts volume или base asset volume?
- **Для BTC/USDT 4h:**
  - Типичный volume: 1000-10000 BTC
  - 5000 BTC → tanh(log1p(5000/24000)) = tanh(0.19) ≈ 0.187
- **ПРОБЛЕМА:** Нет привязки к таймфрейму!
  - 1h bar: ~1250 BTC
  - 4h bar: ~5000 BTC
  - 24h bar: ~30000 BTC
  - Но делитель одинаковый!
- **Митигация:** НЕТ
- **BEST PRACTICE VIOLATION:** "Normalize based on distribution statistics" (sklearn)

**3. Inconsistency with log_volume_norm (АРХИТЕКТУРНАЯ ПРОБЛЕМА)**
- **log_volume_norm:** Использует `quote_asset_volume` (USDT)
- **rel_volume:** Использует `volume` (BTC)
- **ПРОБЛЕМА:** Разные единицы измерения!
- **Последствия:**
  - При росте цены BTC: quote_volume растет быстрее чем volume
  - Корреляция между признаками зависит от цены
- **Пример:**
  - BTC $50k: 1000 BTC = $50M quote
  - BTC $100k: 1000 BTC = $100M quote
  - log_volume_norm увеличится, rel_volume останется
- **РЕКОМЕНДАЦИЯ:** Использовать одну метрику или четко документировать различие

**4. Missing Validity Flag (СРЕДНЯЯ КРИТИЧНОСТЬ)**
- **Источник:** Нет индикатора достоверности данных
- **Сравнение с RSI/MACD:** Те имеют is_valid flags
- **ПРОБЛЕМА:** Модель не знает "это реальный малый объем или пропуск?"
- **Митигация:** НЕТ
- **BEST PRACTICE:** "ML with Missing Data" (OMSCS) - explicit missingness indicators

---

### ПРИЗНАК 3: `ma5` (5-периодная скользящая средняя)

**Индекс:** 3
**Тип:** float32
**Диапазон:** > 0.0 или 0.0 (fallback)
**Источник:** `mediator.py:1083` → `_extract_technical_indicators()`
**Название колонки:** `sma_1200` (для 4h таймфрейма = 5 bars × 240 min/bar)
**Обработка:** `obs_builder.pyx:244-246` - с проверкой валидности

#### ПУТЬ ДАННЫХ:
1. **Генерация** → `transformers.py` создает `sma_1200` (5 bars для 4h)
2. **Извлечение** (mediator.py:1083): `_get_safe_float(row, "sma_1200", float('nan'))`
   - Default: `float('nan')` - ПРАВИЛЬНО! (explicit missingness)
3. **Проверка валидности** (obs_builder.pyx:244):
   ```cython
   ma5_valid = not isnan(ma5)
   out_features[feature_idx] = ma5 if ma5_valid else 0.0
   feature_idx += 1
   out_features[feature_idx] = 1.0 if ma5_valid else 0.0  # validity flag
   ```
4. **Запись** →
   - `out_features[3] = ma5` (или 0.0)
   - `out_features[4] = is_ma5_valid` (1.0 или 0.0)

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Warmup Period (ОЖИДАЕМО)**
- **Источник:** Первые 4 бара недостаточно для SMA(5)
- **Вероятность:** ГАРАНТИРОВАНА в начале данных
- **Последствия:** ma5 = NaN → fallback 0.0
- **Митигация:** ИДЕАЛЬНАЯ!
  - Explicit NaN check
  - Validity flag сообщает модели "данных пока нет"
  - Fallback 0.0 безопасен (не путается с реальной ценой)
- **BEST PRACTICE COMPLIANCE:** ✅
  - "Handle warmup explicitly" (ta-lib documentation)
  - "Use validity indicators for time series" (Kaggle ML guide)

**2. Zero Fallback Ambiguity (МИНИМАЛЬНАЯ)**
- **Источник:** Fallback 0.0 при NaN
- **ПРОБЛЕМА:** Теоретически SMA никогда не 0.0 (цены > 0)
- **Следствие:** 0.0 = безопасный "нет данных" сигнал
- **Митигация:** Validity flag делает это явным - ОТЛИЧНО

**3. Timeframe Naming Confusion (ДОКУМЕНТАЦИЯ)**
- **Источник:** `sma_1200` название в МИНУТАХ, не барах
- **Для 4h:** 1200 минут = 5 bars × 240 min/bar - ПРАВИЛЬНО
- **Для 1h:** 1200 минут = 20 bars - НЕПРАВИЛЬНО!
- **ПРОБЛЕМА:** Хардкод названия в mediator.py:1083
- **Последствия:** При смене таймфрейма нужно менять код
- **Митигация:** НЕТ
- **РЕКОМЕНДАЦИЯ:** Использовать config-driven column mapping

**4. No Price Scale Normalization (ВНИМАНИЕ)**
- **Источник:** MA хранится в абсолютных ценах ($50,000)
- **Диапазон:** Не нормализован (в отличие от большинства признаков)
- **Последствия:**
  - BTC $50k: ma5 ≈ $50,000
  - BTC $100k: ma5 ≈ $100,000
  - Признак не инвариантен к масштабу цены
- **Альтернативы:**
  - (price - ma5) / price - относительное отклонение
  - (price / ma5) - 1.0 - процентное отклонение
  - tanh((price - ma5) / (0.01 * price)) - нормализованное отклонение
- **ТЕКУЩИЙ ПОДХОД:** Модель должна научиться сама
- **BEST PRACTICE:** Mixed
  - ✅ "Let model learn representations" (deep learning)
  - ❌ "Feature engineering reduces sample complexity" (classical ML)

**5. Insufficient Validation (НИЗКАЯ КРИТИЧНОСТЬ)**
- **Источник:** Нет проверки ma5 > 0 (если валидна)
- **Теория:** SMA цен всегда > 0
- **Практика:** Может быть NaN → 0.0, но не отрицательной
- **Риск:** НИЗКИЙ (данные биржи всегда положительные)
- **Митигация:** Неявная (upstream data guarantees)

---

### ПРИЗНАК 4: `is_ma5_valid` (Флаг валидности MA5)

**Индекс:** 4
**Тип:** float32
**Диапазон:** {0.0, 1.0} (бинарный)
**Источник:** `obs_builder.pyx:244-247`
**Формула:** `1.0 if not isnan(ma5) else 0.0`
**Обработка:** Прямая запись

#### ПУТЬ ДАННЫХ:
1. **Проверка** (obs_builder.pyx:244): `ma5_valid = not isnan(ma5)`
2. **Запись** (obs_builder.pyx:247): `out_features[4] = 1.0 if ma5_valid else 0.0`

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Perfect Implementation (ОТЛИЧНО)**
- **Источник:** Точная реализация best practice
- **BEST PRACTICE REFERENCE:**
  - "Missing data indicators" (sklearn.impute)
  - "Validity flags for ML" (Google MLOps)
  - OMSCS "Incomplete Data - Machine Learning Trading"
- **Преимущества:**
  - Модель явно знает когда данных нет
  - Устраняет ambiguity между "0.0 fallback" и "реальный 0.0"
  - Позволяет модели по-разному обрабатывать warmup vs stable periods

**2. No Intermediate States (АРХИТЕКТУРНОЕ ОГРАНИЧЕНИЕ)**
- **Текущее:** Бинарный флаг {0, 1}
- **Альтернатива:** Градиент надежности [0, 1]
  - 0.0: нет данных (первые 4 бара)
  - 0.5: частичные данные (бары 5-9, растущая достоверность)
  - 1.0: полная достоверность (10+ баров)
- **Пример:** SMA(5) на 5-м баре менее стабильна чем на 100-м
- **Текущая митигация:** НЕТ
- **BEST PRACTICE:** "Confidence-weighted features" (Bayesian ML)
- **Оценка:** MINOR enhancement opportunity

**3. Type Mismatch (ОПТИМИЗАЦИЯ)**
- **Источник:** Бинарный флаг хранится как float32
- **Затраты:** 32 бита для 1 бита информации
- **Альтернатива:** bool или uint8
- **Причина текущего:** Унификация observation vector (все float32)
- **Последствия:** 32× memory overhead для этого признака
- **Митигация:** Архитектурное решение для простоты
- **Оценка:** Acceptable trade-off (memory vs simplicity)

---

### ПРИЗНАК 5: `ma20` (20-периодная скользящая средняя)

**Индекс:** 5
**Тип:** float32
**Диапазон:** > 0.0 или 0.0 (fallback)
**Источник:** `mediator.py:1086` → `_extract_technical_indicators()`
**Название колонки:** `sma_5040` (для 4h таймфрейма = 21 bars × 240 min/bar)
**Обработка:** `obs_builder.pyx:250-253` - с проверкой валидности

#### ПУТЬ ДАННЫХ:
1. **Генерация** → `transformers.py` создает `sma_5040` (21 bars для 4h)
2. **Извлечение** (mediator.py:1086): `_get_safe_float(row, "sma_5040", float('nan'))`
3. **Проверка валидности** (obs_builder.pyx:250):
   ```cython
   ma20_valid = not isnan(ma20)
   out_features[feature_idx] = ma20 if ma20_valid else 0.0
   feature_idx += 1
   out_features[feature_idx] = 1.0 if ma20_valid else 0.0  # validity flag
   ```
4. **Запись** →
   - `out_features[5] = ma20` (или 0.0)
   - `out_features[6] = is_ma20_valid` (будет на след. этапе)

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Naming Inconsistency (КРИТИЧНО ДЛЯ ПОНИМАНИЯ)**
- **Проблема:** Признак назван `ma20`, но использует `sma_5040` (21 bars!)
- **Источник комментария** (mediator.py:1084-1086):
  ```python
  # NOTE: For 4h timeframe using sma_5040 (21 bars = 84h ≈ 3.5 days, weekly trend)
  # config_4h_timeframe.py specifies SMA_LOOKBACKS = [5, 21, 50] bars → [1200, 5040, 12000] minutes
  ma20 = self._get_safe_float(row, "sma_5040", float('nan'))
  ```
- **Объяснение:**
  - `ma20` = семантическое имя (20-period MA в классике)
  - `sma_5040` = техническое имя (5040 минут = 21 bars @ 4h)
  - Выбрано 21 вместо 20 для weekly alignment (3.5 days)
- **Последствия:**
  - Confusion для разработчиков
  - При переходе на другой таймфрейм нужно менять хардкод
- **BEST PRACTICE VIOLATION:**
  - "Use semantic names" (Clean Code)
  - "Config-driven feature engineering" (MLOps)
- **РЕКОМЕНДАЦИЯ:**
  - Переименовать в `ma21` ИЛИ
  - Сделать mapping через config

**2. Warmup Period (ОЖИДАЕМО)**
- **Источник:** Первые 20 баров недостаточно для SMA(21)
- **Вероятность:** ГАРАНТИРОВАНА
- **Длительность:** 21 bars = 84 часа = 3.5 дня
- **Митигация:** ИДЕАЛЬНАЯ (см. ma5 анализ)

**3. Correlation with MA5 (FEATURE REDUNDANCY)**
- **Источник:** MA5 и MA20 математически коррелированы
- **Pearson correlation:** Типично 0.95+ для цен
- **Последствия:**
  - Избыточность информации
  - Multicollinearity для линейных моделей (не проблема для NN)
- **Контраргумент:** Разница (MA5 - MA20) = trend signal
- **BEST PRACTICE:**
  - ✅ "Multiple timeframe MAs capture trend" (Technical Analysis)
  - ⚠️ "Decorrelate features when possible" (Feature Engineering)
- **Современный подход:** Пусть модель сама найдет комбинацию
- **Оценка:** ACCEPTABLE (neural networks handle correlation well)

**4. Absolute Price Scale (ПОВТОРЯЕТ MA5)**
- Те же проблемы что у MA5
- Нет нормализации к текущей цене
- Модель должна научиться инвариантности к масштабу

**5. Missing Cross-Feature Interaction (ENHANCEMENT OPPORTUNITY)**
- **Источник:** MA5 и MA20 хранятся отдельно
- **Полезная информация:**
  - `(ma5 - ma20) / price` - trend strength
  - `sign(ma5 - ma20)` - trend direction (golden/death cross)
- **Текущее:** Модель должна научиться сама
- **Альтернатива:** Добавить derived features
- **BEST PRACTICE:** Mixed
  - ✅ Deep learning: "Learn representations"
  - ❌ Classical ML: "Engineer cross-terms"
- **Оценка:** MINOR opportunity (low priority)

---

## СВОДКА ПО ЭТАПУ 1 (Признаки 0-5)

### КРИТИЧНЫЕ ПРОБЛЕМЫ:
1. **log_volume_norm default 1.0** - не подходит для 4h таймфрейма
2. **Normalization scale hardcoded** - не адаптируется к таймфрейму
3. **volume vs quote_volume inconsistency** - разные единицы измерения
4. **Timeframe-specific column names hardcoded** - нужен config-driven mapping

### ОТЛИЧНЫЕ РЕШЕНИЯ:
1. **Validity flags для MA** - best practice compliance
2. **Fail-fast price validation** - предотвращает NaN propagation
3. **Explicit NaN handling** - ясная семантика missing data
4. **Defense-in-depth validation** - P0, P1, P2 layers

### РЕКОМЕНДАЦИИ:
1. **Добавить validity flag для volume** признаков
2. **Параметризовать normalization scales** через config
3. **Унифицировать volume metrics** (или документировать различие)
4. **Добавить sanity checks для extreme values**
5. **Config-driven column name mapping** для multi-timeframe support

### ИСТОЧНИКИ И ЛУЧШИЕ ПРАКТИКИ:
- ✅ Martin Fowler: Fail-fast validation
- ✅ OWASP: Defense in depth
- ✅ IEEE 754: Explicit NaN handling
- ✅ sklearn: Missing data indicators
- ⚠️ Cube Software: Domain-appropriate defaults (частично)
- ⚠️ Google ML: Feature normalization (частично)

---

**ЭТАП 1 ЗАВЕРШЕН.**

---

# ЭТАП 2/10: Признаки 6-11 (Validity Flags + Technical Indicators)

## Признаки 6-11 (MA20 Validity + RSI + MACD Group)

---

### ПРИЗНАК 6: `is_ma20_valid` (Флаг валидности MA20)

**Индекс:** 6
**Тип:** float32
**Диапазон:** {0.0, 1.0} (бинарный)
**Источник:** `obs_builder.pyx:250-253`
**Формула:** `1.0 if not isnan(ma20) else 0.0`
**Обработка:** Прямая запись

#### ПУТЬ ДАННЫХ:
1. **Проверка** (obs_builder.pyx:250): `ma20_valid = not isnan(ma20)`
2. **Запись** (obs_builder.pyx:253): `out_features[6] = 1.0 if ma20_valid else 0.0`

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Identical to is_ma5_valid Implementation (КОНСИСТЕНТНОСТЬ)**
- **Источник:** Тот же подход что для MA5
- **Warmup period:** 21 bars (84 часа для 4h)
- **Митигация:** ИДЕАЛЬНАЯ - см. анализ is_ma5_valid
- **BEST PRACTICE COMPLIANCE:** ✅ Полное

**2. No Specific Issues**
- Все замечания из is_ma5_valid анализа применимы
- Консистентная реализация across всех MA validity flags

---

### ПРИЗНАК 7: `rsi14` (Relative Strength Index)

**Индекс:** 7
**Тип:** float32
**Диапазон:** [0, 100] (теоретически) или 50.0 (fallback)
**Источник:** `mediator.py:1087` → `_extract_technical_indicators()`
**Название колонки:** `rsi`
**Обработка:** `obs_builder.pyx:262-263` - с validity flag

#### ПУТЬ ДАННЫХ:
1. **Генерация** → `transformers.py` или другой источник создает `rsi`
2. **Извлечение** (mediator.py:1087): `_get_safe_float(row, "rsi", 50.0)`
   - Default: `50.0` (neutral RSI)
3. **Проверка валидности** (obs_builder.pyx:261-264):
   ```cython
   rsi_valid = not isnan(rsi14)
   out_features[feature_idx] = rsi14 if rsi_valid else 50.0
   feature_idx += 1
   out_features[feature_idx] = 1.0 if rsi_valid else 0.0  # validity flag
   ```
4. **Запись** →
   - `out_features[7] = rsi14` (или 50.0)
   - `out_features[8] = is_rsi14_valid` (следующий признак)

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Ambiguous Fallback Value 50.0 (КРИТИЧНО)**
- **Источник:** RSI = 50.0 имеет два значения:
  1. Реальная нейтральная зона (neither overbought nor oversold)
  2. Fallback для недостаточных данных
- **Вероятность:** ГАРАНТИРОВАНА в первые ~14 баров
- **Последствия:**
  - Модель видит RSI=50 но не знает "это реальный нейтрал или warmup?"
  - **КРИТИЧНО:** Validity flag РЕШАЕТ проблему!
- **Пример:**
  - Bar 10: RSI=NaN → rsi14=50.0, is_valid=0.0 (модель знает "fake 50")
  - Bar 20: RSI=50.1 → rsi14=50.1, is_valid=1.0 (модель знает "real neutral")
- **Митигация:** ОТЛИЧНАЯ (validity flag устраняет ambiguity)
- **BEST PRACTICE REFERENCE:**
  - obs_builder.pyx:258-260 комментарий: "Fallback 50.0 creates AMBIGUITY"
  - "Validity flag eliminates this: model can distinguish valid neutral from missing data"

**2. Warmup Period ~14 Bars (ОЖИДАЕМО)**
- **Источник:** RSI формула требует минимум 14 периодов
- **RSI Calculation:** Wilder's smoothing (EMA of gains/losses over 14 periods)
- **Для 4h таймфрейма:** 14 bars = 56 часов = 2.33 дня
- **Последствия:** Первые 2+ дня данных имеют is_rsi_valid=0.0
- **Митигация:** ИДЕАЛЬНАЯ (validity flag явно указывает на warmup)
- **BEST PRACTICE:** "Handle indicator warmup explicitly" (ta-lib)

**3. No Normalization (ВНИМАНИЕ)**
- **Источник:** RSI хранится в [0, 100] диапазоне
- **Проблема:** Не нормализован в [-1, 1] как большинство признаков
- **Сравнение:**
  - `log_volume_norm`: tanh → [-1, 1]
  - `ret_bar`: tanh → [-1, 1]
  - `rsi14`: raw → [0, 100]
- **Последствия:**
  - Разный масштаб признаков
  - Может замедлить обучение (gradient issues)
- **Контраргумент:**
  - RSI уже bounded [0, 100]
  - Neural networks могут адаптироваться через layer normalization
- **Альтернатива:** `(rsi14 - 50.0) / 50.0` → [-1, 1]
- **BEST PRACTICE:** Mixed
  - ✅ "Bounded indicators don't require normalization" (Technical Analysis)
  - ⚠️ "Uniform feature scales improve convergence" (Deep Learning)
- **Текущее решение:** Оставить как есть (модель адаптируется)

**4. RSI Extreme Values (ИНФОРМАТИВНО)**
- **Источник:** RSI может достичь 0 или 100 в экстремальных условиях
- **Вероятность:** НИЗКАЯ (но возможна при сильных трендах)
- **Примеры:**
  - RSI = 0: Только падения за последние 14 баров (крайне редко)
  - RSI = 100: Только росты за последние 14 баров (крайне редко)
- **Последствия:**
  - Сильный сигнал oversold/overbought
  - Модель должна научиться использовать эти экстремумы
- **Митигация:** НЕТ (это feature, не bug!)
- **BEST PRACTICE:** "Extreme RSI values are valid trading signals" (Technical Analysis)

**5. RSI Default 50.0 vs NaN Propagation (АРХИТЕКТУРНОЕ РЕШЕНИЕ)**
- **Вопрос:** Почему default=50.0 в mediator.py:1087, а не float('nan')?
- **Ответ:**
  ```python
  # mediator.py:1087
  rsi14 = self._get_safe_float(row, "rsi", 50.0)  # default 50.0
  # vs MA approach:
  ma5 = self._get_safe_float(row, "sma_1200", float('nan'))  # default NaN
  ```
- **Причина:**
  - Если `rsi` колонка отсутствует → return 50.0
  - Если `rsi` = NaN → return 50.0
  - В obs_builder.pyx: `rsi_valid = not isnan(rsi14)` будет True!
- **ПРОБЛЕМА:** Validity flag НЕ ЛОВИТ отсутствующую колонку!
- **Сценарий:**
  1. DataFrame не содержит колонку `rsi`
  2. `_get_safe_float` возвращает 50.0 (default)
  3. `rsi_valid = not isnan(50.0)` = True
  4. Модель думает что RSI=50 валиден, хотя данных нет!
- **Митигация:** ЧАСТИЧНАЯ
  - Работает если колонка есть но NaN
  - НЕ работает если колонка отсутствует
- **РЕКОМЕНДАЦИЯ:** Использовать float('nan') как default (как для MA)
- **BEST PRACTICE VIOLATION:** "Explicit is better than implicit" (Zen of Python)

---

### ПРИЗНАК 8: `is_rsi14_valid` (Флаг валидности RSI)

**Индекс:** 8
**Тип:** float32
**Диапазон:** {0.0, 1.0} (бинарный)
**Источник:** `obs_builder.pyx:261-265`
**Формула:** `1.0 if not isnan(rsi14) else 0.0`
**Обработка:** Прямая запись

#### ПУТЬ ДАННЫХ:
1. **Проверка** (obs_builder.pyx:261): `rsi_valid = not isnan(rsi14)`
2. **Запись** (obs_builder.pyx:264): `out_features[8] = 1.0 if rsi_valid else 0.0`

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Excellent Design Intent (ОТЛИЧНО)**
- **Комментарий из кода** (obs_builder.pyx:258-260):
  ```cython
  # CRITICAL: RSI requires ~14 bars for first valid value
  # Fallback 50.0 creates AMBIGUITY: neutral RSI (50) vs insufficient data (50)
  # Validity flag eliminates this: model can distinguish valid neutral from missing data
  ```
- **BEST PRACTICE COMPLIANCE:** ✅ Полное соответствие
- **Ссылки:**
  - "Missing data indicators" (sklearn)
  - "Incomplete Data - ML Trading" (OMSCS)

**2. Inherited Issue from RSI Default (КРИТИЧНО)**
- **Проблема:** См. RSI признак #7, issue #5
- **Если колонка `rsi` отсутствует:**
  - mediator возвращает 50.0 (не NaN)
  - is_rsi14_valid = 1.0 (ложно положительный!)
- **Последствия:** Модель не знает что RSI данных нет вообще
- **Митигация:** Изменить default в mediator.py на float('nan')
- **Текущий статус:** PARTIAL PROTECTION

**3. No Confidence Gradient (ENHANCEMENT OPPORTUNITY)**
- **Текущее:** Бинарный {0, 1}
- **Альтернатива:** Confidence score [0, 1]
  - Bars 0-13: confidence = 0.0 (warmup)
  - Bar 14: confidence = 0.5 (minimum data)
  - Bar 28+: confidence = 1.0 (stable RSI)
- **Обоснование:**
  - RSI на 14-м баре менее надежен чем на 100-м
  - Wilder's smoothing стабилизируется постепенно
- **BEST PRACTICE:** "Bayesian confidence estimation" (Probabilistic ML)
- **Оценка:** MINOR enhancement (low priority)

---

### ПРИЗНАК 9: `macd` (Moving Average Convergence Divergence)

**Индекс:** 9
**Тип:** float32
**Диапазон:** unbounded (может быть любым) или 0.0 (fallback)
**Источник:** `mediator.py:1090-1103` → `_extract_technical_indicators()`
**Обработка:** `obs_builder.pyx:272-273` - с validity flag

#### ПУТЬ ДАННЫХ:
1. **Инициализация** (mediator.py:1090): `macd = 0.0`
2. **Попытка извлечения из симулятора** (mediator.py:1100-1103):
   ```python
   if sim is not None and hasattr(sim, "get_macd"):
       try:
           if hasattr(sim, "get_macd"):
               macd = float(sim.get_macd(row_idx))
       except Exception:
           pass
   ```
3. **Проверка валидности** (obs_builder.pyx:271-274):
   ```cython
   macd_valid = not isnan(macd)
   out_features[feature_idx] = macd if macd_valid else 0.0
   feature_idx += 1
   out_features[feature_idx] = 1.0 if macd_valid else 0.0  # validity flag
   ```
4. **Запись** →
   - `out_features[9] = macd` (или 0.0)
   - `out_features[10] = is_macd_valid` (следующий признак)

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Dual-Source Architecture (КРИТИЧНО - АРХИТЕКТУРНАЯ ПРОБЛЕМА)**
- **Проблема:** MACD может приходить из двух источников:
  1. MarketSimulator (`sim.get_macd(row_idx)`)
  2. DataFrame колонка (НЕ РЕАЛИЗОВАНО в текущем коде!)
- **Текущая реализация:**
  - Только simulator source (mediator.py:1100-1103)
  - НЕТ fallback к DataFrame колонке
- **Последствия:**
  - Если simulator отсутствует → macd = 0.0 всегда
  - is_macd_valid = False всегда (NaN check)
- **Сравнение с RSI:**
  ```python
  # RSI: извлекается из DataFrame
  rsi14 = self._get_safe_float(row, "rsi", 50.0)
  # MACD: только из simulator
  macd = 0.0  # default
  if sim is not None and hasattr(sim, "get_macd"):
      macd = float(sim.get_macd(row_idx))
  ```
- **BEST PRACTICE VIOLATION:**
  - "Single Source of Truth" (DDD)
  - "Fallback chain for robustness" (Resilience Engineering)
- **РЕКОМЕНДАЦИЯ:**
  ```python
  # Try DataFrame first, then simulator
  macd = self._get_safe_float(row, "macd", float('nan'))
  if math.isnan(macd) and sim is not None:
      macd = float(sim.get_macd(row_idx))
  ```

**2. Silent Exception Swallowing (ОПАСНО)**
- **Источник:** (mediator.py:1118)
  ```python
  try:
      if hasattr(sim, "get_macd"):
          macd = float(sim.get_macd(row_idx))
  except Exception:
      pass  # ПРОБЛЕМА: Молча игнорирует ошибки!
  ```
- **Последствия:**
  - IndexError → macd=0.0 (молчание)
  - TypeError → macd=0.0 (молчание)
  - AttributeError → macd=0.0 (молчание)
  - НЕТ логирования ошибки!
- **Проблема:** Невозможно отладить проблемы с данными
- **BEST PRACTICE VIOLATION:**
  - "Never catch generic Exception" (Python best practices)
  - "Log all failures" (Observability)
- **РЕКОМЕНДАЦИЯ:**
  ```python
  try:
      macd = float(sim.get_macd(row_idx))
  except (IndexError, TypeError, ValueError) as e:
      logger.warning(f"Failed to get MACD: {e}")
      macd = float('nan')
  ```

**3. Warmup Period ~26 Bars (ОЖИДАЕМО)**
- **Источник:** MACD формула
  - EMA_fast = EMA(12)
  - EMA_slow = EMA(26)
  - MACD = EMA_fast - EMA_slow
- **Minimum bars:** 26 (для EMA_slow)
- **Для 4h таймфрейма:** 26 bars = 104 часа = 4.33 дня
- **Митигация:** ИДЕАЛЬНАЯ (validity flag)
- **BEST PRACTICE COMPLIANCE:** ✅

**4. Ambiguous Fallback 0.0 (КРИТИЧНО)**
- **Источник:** MACD = 0.0 имеет два значения:
  1. EMA_fast = EMA_slow (нет дивергенции - реальный сигнал)
  2. Fallback для недостаточных данных
- **Комментарий из кода** (obs_builder.pyx:268-270):
  ```cython
  # CRITICAL: MACD requires ~26 bars for first valid value (12+26 EMA periods)
  # Fallback 0.0 creates AMBIGUITY: no divergence (0) vs insufficient data (0)
  # Validity flag eliminates this: model can distinguish valid zero from missing data
  ```
- **Митигация:** ОТЛИЧНАЯ (validity flag решает проблему)
- **BEST PRACTICE COMPLIANCE:** ✅

**5. No Normalization (КРИТИЧНО)**
- **Источник:** MACD unbounded (может быть -1000 до +1000 для BTC)
- **Проблема:**
  - Не нормализован (в отличие от большинства признаков)
  - Зависит от абсолютной цены актива
- **Пример:**
  - BTC $50k: MACD = 100 (небольшой тренд)
  - BTC $100k: MACD = 200 (тот же % тренд, другой абсолют)
- **Последствия:**
  - Модель должна научиться масштабу для каждого актива
  - Не работает для multi-asset моделей
- **Альтернативы:**
  - `macd / price` - нормализация к цене
  - `tanh(macd / (price * 0.001))` - bounded нормализация
- **BEST PRACTICE VIOLATION:**
  - "Normalize features to similar scales" (sklearn)
  - "Price-invariant features for multi-asset" (Quantitative Finance)
- **Текущее решение:** Модель должна адаптироваться
- **Оценка:** СРЕДНЯЯ критичность (зависит от use case)

---

### ПРИЗНАК 10: `is_macd_valid` (Флаг валидности MACD)

**Индекс:** 10
**Тип:** float32
**Диапазон:** {0.0, 1.0} (бинарный)
**Источник:** `obs_builder.pyx:271-275`
**Формула:** `1.0 if not isnan(macd) else 0.0`
**Обработка:** Прямая запись

#### ПУТЬ ДАННЫХ:
1. **Проверка** (obs_builder.pyx:271): `macd_valid = not isnan(macd)`
2. **Запись** (obs_builder.pyx:274): `out_features[10] = 1.0 if macd_valid else 0.0`

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Inherited Issue from MACD Default (КРИТИЧНО)**
- **Проблема:** mediator.py инициализирует `macd = 0.0` (не NaN!)
- **Последствия:**
  - Если simulator отсутствует или failback → macd=0.0
  - `is_macd_valid = not isnan(0.0)` = True
  - **ЛОЖНО ПОЛОЖИТЕЛЬНЫЙ!** Модель думает MACD валиден
- **Сценарий:**
  1. No simulator → macd=0.0 (line 1090)
  2. is_macd_valid = True (потому что 0.0 не NaN)
  3. Модель видит "MACD=0 (no divergence)" вместо "нет данных"
- **BEST PRACTICE VIOLATION:** "Explicit missing data markers" (Data Engineering)
- **РЕКОМЕНДАЦИЯ:** Инициализировать `macd = float('nan')` в mediator.py
- **Текущий статус:** BROKEN (validity flag не работает как задумано!)

**2. Try-Except Breaks Validity Detection (КРИТИЧНО)**
- **Источник:** (mediator.py:1118) `except Exception: pass`
- **Проблема:**
  - Ошибка при `sim.get_macd()` → exception пойман молча
  - macd остается 0.0 (не меняется на NaN)
  - is_macd_valid = True (ложный сигнал!)
- **Правильный подход:**
  ```python
  macd = float('nan')  # default
  try:
      macd = float(sim.get_macd(row_idx))
  except Exception:
      pass  # macd уже NaN
  ```

**3. Excellent Design Intent (IF FIXED)**
- **Комментарий из кода** (obs_builder.pyx:268-270) - отличный дизайн
- **НО:** Реализация в mediator.py ломает замысел
- **Потенциал:** ВЫСОКИЙ (после исправления default на NaN)

---

### ПРИЗНАК 11: `macd_signal` (MACD Signal Line)

**Индекс:** 11
**Тип:** float32
**Диапазон:** unbounded или 0.0 (fallback)
**Источник:** `mediator.py:1091-1105` → `_extract_technical_indicators()`
**Обработка:** `obs_builder.pyx:282-283` - с validity flag

#### ПУТЬ ДАННЫХ:
1. **Инициализация** (mediator.py:1091): `macd_signal = 0.0`
2. **Попытка извлечения** (mediator.py:1104-1105):
   ```python
   if hasattr(sim, "get_macd_signal"):
       macd_signal = float(sim.get_macd_signal(row_idx))
   ```
3. **Проверка валидности** (obs_builder.pyx:281-284):
   ```cython
   macd_signal_valid = not isnan(macd_signal)
   out_features[feature_idx] = macd_signal if macd_signal_valid else 0.0
   feature_idx += 1
   out_features[feature_idx] = 1.0 if macd_signal_valid else 0.0  # validity flag (index 12)
   ```
4. **Запись** →
   - `out_features[11] = macd_signal` (или 0.0)
   - `out_features[12] = is_macd_signal_valid` (следующий этап)

#### ИСКАЖЕНИЯ И МИТИГАЦИИ:

**1. Identical Issues to MACD (КРИТИЧНО)**
- **Все проблемы MACD признака применимы:**
  1. Default 0.0 вместо NaN → ломает validity flag
  2. Silent exception swallowing
  3. No DataFrame fallback
  4. No normalization
- **Дополнительно:**
  - Warmup period: ~35 bars (26 для MACD + 9 для EMA signal)
  - Для 4h: 35 bars = 140 часов = 5.83 дня

**2. MACD vs Signal Correlation (ИНФОРМАТИВНО)**
- **Источник:** MACD Signal = EMA_9(MACD)
- **Correlation:** Очень высокая (0.9+)
- **Полезная информация:**
  - Histogram = MACD - Signal (divergence measure)
  - Текущее: модель должна вычислить сама
- **ENHANCEMENT OPPORTUNITY:**
  - Добавить `macd_histogram` как отдельный признак
  - Histogram пересекает 0 = trend change signal
- **BEST PRACTICE:** "Pre-compute derived features" (Feature Engineering)
- **Оценка:** MINOR priority (модель может научиться)

**3. Ambiguous Fallback 0.0 (КРИТИЧНО)**
- **Комментарий из кода** (obs_builder.pyx:278-280):
  ```cython
  # CRITICAL: MACD Signal requires ~35 bars (26 for MACD + 9 for signal line)
  # Fallback 0.0 creates AMBIGUITY: no signal (0) vs insufficient data (0)
  # Validity flag eliminates this: model can distinguish valid zero from missing data
  ```
- **Митигация:** WOULD BE EXCELLENT (если бы не broken validity flag)
- **Текущий статус:** BROKEN (см. MACD анализ)

---

## СВОДКА ПО ЭТАПУ 2 (Признаки 6-11)

### КРИТИЧНЫЕ ПРОБЛЕМЫ:
1. **MACD/MACD_signal default 0.0 вместо NaN** - ломает validity flags!
2. **Silent exception swallowing** - нет логирования ошибок
3. **No DataFrame fallback для MACD** - зависимость только от simulator
4. **No normalization для RSI/MACD** - разные масштабы признаков
5. **RSI default 50.0 может создать false validity** - если колонка отсутствует

### ОТЛИЧНЫЕ РЕШЕНИЯ:
1. **Validity flags design intent** - brilliant идея (если исправить)
2. **Explicit warmup handling** - комментарии в коде excellent
3. **Consistent pattern across indicators** - RSI, MACD, MACD_signal
4. **NaN-based missing data detection** - правильный подход

### РЕКОМЕНДАЦИИ (PRIORITY ORDER):
1. **CRITICAL:** Изменить defaults на float('nan'):
   - `macd = float('nan')` (не 0.0)
   - `macd_signal = float('nan')` (не 0.0)
   - `rsi14 = float('nan')` (не 50.0) в fallback случаях
2. **HIGH:** Добавить DataFrame fallback для MACD indicators
3. **MEDIUM:** Логировать exceptions вместо молчаливого pass
4. **LOW:** Рассмотреть нормализацию RSI/MACD к [-1, 1]
5. **ENHANCEMENT:** Добавить macd_histogram как derived feature

### ИСТОЧНИКИ И ЛУЧШИЕ ПРАКТИКИ:
- ✅ sklearn: Missing data indicators (design intent excellent)
- ✅ ta-lib: Indicator warmup handling
- ✅ OMSCS: Explicit missingness signals
- ❌ Python: Never catch generic Exception (нарушено)
- ❌ sklearn: Feature normalization (частично нарушено)
- ❌ DDD: Single Source of Truth (dual source без fallback)

### COMPARISON С ЭТАПОМ 1:
- **Этап 1:** Mostly excellent (minor issues)
- **Этап 2:** BROKEN validity flags для MACD (critical fix needed!)
- **Общее:** Отличный дизайн, но реализация mediator.py ломает его

---

**ЭТАП 2 ЗАВЕРШЕН.**

---

Дата анализа: 2025-11-16
Аналитик: Claude Code (Sonnet 4.5)
Версия кодовой базы: commit bc75c15
