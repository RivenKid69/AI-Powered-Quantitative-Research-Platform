# РЕКОМЕНДУЕМЫЕ ИСПРАВЛЕНИЯ ДЛЯ PPO

Этот документ содержит конкретные исправления для проблем, выявленных в анализе.

---

## ИСПРАВЛЕНИЕ #1: Уменьшить max log_ratio (КРИТИЧНО)

### Текущий код (`distributional_ppo.py:7870-7871`)

```python
log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
ratio = torch.exp(log_ratio)
```

### Рекомендуемое исправление

```python
# CRITICAL FIX: Reduce max log_ratio to prevent numerical overflow
# exp(20) ≈ 485M causes numerical instability and violates PPO trust region assumption
# exp(10) ≈ 22k is more reasonable for "small" policy updates
log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
ratio = torch.exp(log_ratio)

# Optional: Add safety check and logging
if torch.any(torch.abs(log_ratio) > 5.0):
    self.logger.record("warn/large_log_ratio", float(torch.abs(log_ratio).max().item()))
```

### Обоснование
- exp(20) = 485,165,195 (слишком большое значение)
- exp(10) = 22,026 (более управляемое)
- exp(5) = 148 (консервативный вариант)

Рекомендуется начать с **±10** и экспериментально подобрать оптимальное значение.

---

## ИСПРАВЛЕНИЕ #2: Точная KL divergence для Gaussian

### Текущий код (`distributional_ppo.py:8003-8006`)

```python
# FIX: Use correct KL divergence formula for KL(old||new)
# Simple first-order approximation: KL(old||new) ≈ old_log_prob - new_log_prob
approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new
```

### Рекомендуемое исправление

```python
# Compute KL divergence
if isinstance(self.action_space, gym.spaces.Box):
    # For continuous actions (Gaussian), use exact KL formula
    # KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

    # Get distribution parameters
    inner_dist = getattr(dist, "distribution", None)
    old_inner_dist = None  # Need to store old distribution in rollout buffer

    if inner_dist is not None and old_inner_dist is not None:
        # Extract mean and std
        mean_new = getattr(inner_dist, "mean", None)
        if callable(mean_new):
            mean_new = mean_new()
        std_new = getattr(inner_dist, "stddev", None)
        if std_new is None:
            get_std = getattr(inner_dist, "get_std", None)
            std_new = get_std() if callable(get_std) else None

        mean_old = getattr(old_inner_dist, "mean", None)
        if callable(mean_old):
            mean_old = mean_old()
        std_old = getattr(old_inner_dist, "stddev", None)
        if std_old is None:
            get_std = getattr(old_inner_dist, "get_std", None)
            std_old = get_std() if callable(get_std) else None

        if mean_new is not None and std_new is not None and mean_old is not None and std_old is not None:
            # Exact KL for Gaussian
            var_ratio = (std_old / (std_new + 1e-8)) ** 2
            mean_diff_sq = (mean_old - mean_new) ** 2
            kl_per_dim = torch.log(std_new / (std_old + 1e-8)) + (var_ratio + mean_diff_sq / (std_new ** 2 + 1e-8)) / 2.0 - 0.5
            approx_kl_raw_tensor = kl_per_dim.sum(dim=-1)
        else:
            # Fallback to first-order approximation
            approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new
    else:
        # Fallback to first-order approximation
        approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new
else:
    # For discrete actions, first-order approximation is acceptable
    approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new

# Safety: Ensure KL is non-negative (for exact formula)
if isinstance(self.action_space, gym.spaces.Box):
    approx_kl_raw_tensor = torch.clamp(approx_kl_raw_tensor, min=0.0)
```

### Альтернатива: Использовать PyTorch distributions

```python
if isinstance(self.action_space, gym.spaces.Box):
    # Use PyTorch's built-in KL divergence
    from torch.distributions.kl import kl_divergence
    try:
        kl_div = kl_divergence(old_dist, dist)
        if kl_div.ndim > 1:
            kl_div = kl_div.sum(dim=-1)
        approx_kl_raw_tensor = kl_div
    except NotImplementedError:
        # Fallback to approximation
        approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new
else:
    approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new
```

**Примечание:** Для этого исправления нужно сохранять старое распределение (не только log_prob) в rollout buffer.

---

## ИСПРАВЛЕНИЕ #3: Улучшенная нормализация advantages

### Вариант A: Нормализация внутри каждой группы (рекомендуется для экспериментов)

```python
# Compute group keys for samples
group_keys_array = self._resolve_group_keys_for_training_batch(rollout_data, valid_indices, value_valid_indices)

# Normalize advantages within each group separately
unique_groups = torch.unique(group_keys_array)

for group_id in unique_groups:
    group_mask = (group_keys_array == group_id)
    group_advantages = advantages_flat[group_mask]

    if group_advantages.numel() > 1:  # Need at least 2 samples for std
        group_mean = group_advantages.mean()
        group_std = group_advantages.std().clamp(min=1e-8)

        advantages_flat[group_mask] = (group_advantages - group_mean) / group_std
    else:
        # Single sample: just center to 0
        advantages_flat[group_mask] = 0.0

# Use normalized advantages
advantages = advantages_flat.view_as(advantages)
advantages_selected = advantages_flat[valid_indices] if valid_indices is not None else advantages_flat
```

### Вариант B: Робастная нормализация (устойчива к outliers)

```python
# Use median and IQR instead of mean and std
# More robust to outliers
advantages_flat = advantages.reshape(-1)

if valid_indices is not None:
    advantages_subset = advantages_flat[valid_indices]
else:
    advantages_subset = advantages_flat

# Compute robust statistics
median = torch.median(advantages_subset)
q75 = torch.quantile(advantages_subset, 0.75)
q25 = torch.quantile(advantages_subset, 0.25)
iqr = q75 - q25

# Normalize using robust statistics
iqr_safe = torch.clamp(iqr, min=1e-8)
advantages_flat = (advantages_flat - median) / iqr_safe

advantages = advantages_flat.view_as(advantages)
advantages_selected = advantages_flat[valid_indices] if valid_indices is not None else advantages_flat
```

### Вариант C: Не нормализовать (как в оригинальном PPO)

```python
# Option: Disable advantage normalization
# Original PPO paper doesn't require advantage normalization
if getattr(self, "normalize_advantage", False):
    # Current normalization logic
    advantages_flat = advantages.reshape(-1)
    advantages_normalized_flat = (advantages_flat - group_adv_mean) / group_adv_std_clamped
    advantages = advantages_normalized_flat.view_as(advantages)
else:
    # No normalization - use raw advantages
    pass

advantages_selected = advantages.reshape(-1)[valid_indices] if valid_indices is not None else advantages.reshape(-1)
```

**Рекомендация:** Начать с **Варианта C** (отключить нормализацию) и сравнить с текущей реализацией. Затем экспериментировать с Вариантом A или B.

---

## ДОПОЛНИТЕЛЬНЫЕ УЛУЧШЕНИЯ

### Добавить early stopping на основе KL divergence

```python
# In train() method, after computing approx_kl
target_kl = getattr(self, "target_kl", 0.01)  # Default from PPO paper

mean_kl = kl_raw_sum / kl_raw_count if kl_raw_count > 0 else 0.0

if mean_kl > target_kl * 1.5:  # 1.5x threshold for safety
    self.logger.record("train/early_stop_kl", mean_kl)
    self.logger.record("train/early_stop_epoch", epoch)
    break  # Stop training epoch early
```

### Добавить мониторинг соотношения BC/PPO loss

```python
# After computing losses
bc_ratio = abs(policy_loss_bc_weighted_value) / (abs(policy_loss_ppo_value) + 1e-8)
self.logger.record("train/bc_ppo_ratio", bc_ratio)

if bc_ratio > 2.0:  # BC loss dominates
    self.logger.record("warn/bc_dominates", bc_ratio)

# Optional: Adaptive bc_coef
if getattr(self, "adaptive_bc_coef", False):
    max_bc_ratio = 1.0
    if bc_ratio > max_bc_ratio:
        # Reduce bc_coef to maintain balance
        reduction_factor = max_bc_ratio / bc_ratio
        self.bc_coef = max(self.bc_coef * reduction_factor, 0.01)  # Min 0.01
        self.logger.record("train/bc_coef_adjusted", self.bc_coef)
```

### Добавить проверку на численные проблемы

```python
# After computing ratio
if torch.any(~torch.isfinite(ratio)):
    non_finite_count = (~torch.isfinite(ratio)).sum().item()
    self.logger.record("error/non_finite_ratio_count", float(non_finite_count))
    self.logger.record("error/non_finite_ratio_frac", float(non_finite_count) / ratio.numel())

    # Replace non-finite with 1.0 (neutral)
    ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))

# Check for extreme values
max_ratio = ratio.max().item()
min_ratio = ratio.min().item()

if max_ratio > 100.0 or min_ratio < 0.01:
    self.logger.record("warn/extreme_ratio_max", max_ratio)
    self.logger.record("warn/extreme_ratio_min", min_ratio)
```

---

## ПЛАН ВНЕДРЕНИЯ

### Фаза 1: Критическое исправление (немедленно)
1. ✅ Изменить `max` в `torch.clamp(log_ratio, ...)` с 20 на 10
2. ✅ Добавить мониторинг extreme ratio values
3. ✅ Запустить тесты и сравнить стабильность

### Фаза 2: Улучшение KL (на следующей неделе)
1. Реализовать точную KL для Gaussian
2. Сохранять старое распределение в rollout buffer (если требуется)
3. Сравнить качество KL approximation vs exact

### Фаза 3: Эксперименты с нормализацией (исследование)
1. Тестировать три варианта нормализации advantages:
   - Без нормализации (baseline)
   - Групповая нормализация (текущая)
   - Внутригрупповая нормализация (новая)
2. Сравнить:
   - Стабильность обучения
   - Скорость сходимости
   - Итоговое качество политики
3. Выбрать лучший подход

### Фаза 4: Дополнительные улучшения (опционально)
1. Early stopping на основе KL
2. Adaptive BC coefficient
3. Improved monitoring

---

## ТЕСТИРОВАНИЕ

### Тест 1: Проверка стабильности после уменьшения max_log_ratio

```bash
# Before fix
python train.py --config baseline_config.yaml --seed 42

# After fix
python train.py --config baseline_config.yaml --seed 42

# Compare:
# - Training stability (loss variance)
# - Final performance
# - Number of NaN/Inf incidents
```

### Тест 2: Сравнение KL approximation

```python
# Add logging in code
self.logger.record("train/kl_approx", approx_kl_mean)
self.logger.record("train/kl_exact", exact_kl_mean)
self.logger.record("train/kl_error", abs(approx_kl_mean - exact_kl_mean))
```

### Тест 3: A/B тест нормализации advantages

```bash
# Run 3 seeds for each variant
for seed in 42 43 44; do
    # No normalization
    python train.py --normalize_advantage=false --seed=$seed --tag=no_norm

    # Global normalization (current)
    python train.py --normalize_advantage=true --group_norm=true --seed=$seed --tag=global_norm

    # Per-group normalization
    python train.py --normalize_advantage=true --group_norm=false --seed=$seed --tag=group_norm
done

# Compare results
python compare_runs.py --tags no_norm global_norm group_norm
```

---

## ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

### После исправления #1 (log_ratio clamp)
- ✅ Меньше численных ошибок (NaN/Inf)
- ✅ Более стабильное обучение (меньше variance)
- ✅ Возможно немного медленнее (более консервативные обновления)

### После исправления #2 (exact KL)
- ✅ Более точная оценка policy divergence
- ✅ Лучший early stopping (если используется)
- ⚠️ Небольшое увеличение compute (вычисление exact KL)

### После исправления #3 (advantage normalization)
- ❓ Зависит от задачи - требуются эксперименты
- Возможно улучшение для задач с гетерогенными группами
- Возможно ухудшение, если текущая нормализация работает хорошо

---

## МОНИТОРИНГ ПОСЛЕ ВНЕДРЕНИЯ

### Ключевые метрики для отслеживания

1. **Стабильность:**
   - `warn/ratio_all_nonfinite` (должна уменьшиться)
   - `error/non_finite_ratio_count` (должна быть 0)
   - Loss variance (должна уменьшиться)

2. **KL divergence:**
   - `train/approx_kl` (должна быть в пределах [0, target_kl])
   - `train/kl_error` (для exact KL, должна быть малой)

3. **Policy updates:**
   - `train/clip_fraction` (должна быть 0.1-0.3, оптимально ~0.2)
   - `train/entropy` (должна медленно уменьшаться)
   - `warn/extreme_ratio_max` (должна быть < 10 после fix)

4. **Loss components:**
   - `train/bc_ppo_ratio` (должна быть < 1.0)
   - `train/policy_loss_ppo` vs `train/policy_loss_bc_weighted`

### Когда откатывать изменения

Откатить исправление, если:
1. Performance значительно ухудшается (>10% drop)
2. Появляются новые численные проблемы
3. Обучение становится слишком медленным (>2x slowdown)

---

## ИТОГОВЫЕ РЕКОМЕНДАЦИИ

1. **Обязательно внедрить:**
   - Уменьшение max_log_ratio до 10 (критично для стабильности)

2. **Сильно рекомендуется:**
   - Точная KL для Gaussian (улучшает оценку divergence)
   - Мониторинг BC/PPO ratio (предотвращает дисбаланс)

3. **Экспериментальное:**
   - Альтернативная нормализация advantages (зависит от задачи)
   - Early stopping на основе KL (для безопасности)

4. **Опционально:**
   - Adaptive BC coefficient
   - Adaptive clipping range

---

**Следующие шаги:**
1. Создать feature branch для исправлений
2. Внедрить критическое исправление #1
3. Запустить regression тесты
4. Сравнить результаты с baseline
5. Постепенно внедрять остальные улучшения

**Контакт для вопросов:** См. PPO_CRITICAL_ANALYSIS_REPORT.md
