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

**ЭТАП 1 ЗАВЕРШЕН. ГОТОВ К ПЕРЕХОДУ НА ЭТАП 2.**

Дата анализа: 2025-11-16
Аналитик: Claude Code (Sonnet 4.5)
Версия кодовой базы: commit bc75c15
