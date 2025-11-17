"""
Глубокий анализ математической корректности реализации PPO.

Этот тест проверяет:
1. Корректность PPO clipped objective
2. Корректность value function clipping
3. Корректность advantage estimation и нормализации
4. Корректность градиентного потока
5. Соответствие лучшим практикам из исследований
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import pytest


class TestPPOClippedObjective:
    """Проверка математической корректности PPO clipped objective."""

    def test_ppo_loss_symmetry(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: PPO loss должна быть симметрична относительно
        положительных и отрицательных advantages.

        Согласно оригинальной статье Schulman et al. 2017:
        L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

        Проблема: При A < 0, min выбирает МАКСИМАЛЬНО отрицательное значение,
        что может привести к асимметрии в обновлениях.
        """
        clip_range = 0.2

        # Тест 1: Положительные advantages
        advantages_pos = torch.tensor([1.0, 2.0, 3.0])
        ratio_pos = torch.tensor([1.5, 1.1, 0.9])  # различные ratio

        loss_1_pos = advantages_pos * ratio_pos
        loss_2_pos = advantages_pos * torch.clamp(ratio_pos, 1 - clip_range, 1 + clip_range)
        ppo_loss_pos = -torch.min(loss_1_pos, loss_2_pos).mean()

        # Тест 2: Отрицательные advantages (той же абсолютной величины)
        advantages_neg = -advantages_pos
        ratio_neg = ratio_pos  # те же ratio

        loss_1_neg = advantages_neg * ratio_neg
        loss_2_neg = advantages_neg * torch.clamp(ratio_neg, 1 - clip_range, 1 + clip_range)
        ppo_loss_neg = -torch.min(loss_1_neg, loss_2_neg).mean()

        # ОЖИДАНИЕ: losses должны быть равны по абсолютной величине
        print(f"\nPPO Loss (positive adv): {ppo_loss_pos.item():.6f}")
        print(f"PPO Loss (negative adv): {ppo_loss_neg.item():.6f}")
        print(f"Difference: {abs(ppo_loss_pos.item() + ppo_loss_neg.item()):.6f}")

        # Проверка симметрии
        assert torch.allclose(ppo_loss_pos, -ppo_loss_neg, atol=1e-6), \
            "PPO loss не симметрична для положительных и отрицательных advantages"

    def test_ppo_clipping_behavior(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: Clipping должен срабатывать правильно.

        Проблема: При большом ratio и положительном advantage, clipping должен
        ограничивать обновление. При большом ratio и отрицательном advantage,
        clipping НЕ должен срабатывать (агент должен избегать плохих действий).
        """
        clip_range = 0.2

        # Случай 1: Большой ratio, положительный advantage
        # ОЖИДАНИЕ: Clipping ограничивает обновление до (1 + clip_range) * advantage
        advantage = torch.tensor([2.0])
        ratio = torch.tensor([2.0])  # Очень большой ratio

        loss_1 = advantage * ratio  # = 4.0
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)  # = 2.0 * 1.2 = 2.4
        ppo_loss = -torch.min(loss_1, loss_2)  # min(4.0, 2.4) = 2.4, затем -2.4

        expected_clipped = -advantage * (1 + clip_range)
        print(f"\nCase 1 - Large ratio, positive advantage:")
        print(f"  PPO loss: {ppo_loss.item():.6f}")
        print(f"  Expected (clipped): {expected_clipped.item():.6f}")
        assert torch.allclose(ppo_loss, expected_clipped, atol=1e-6), \
            "Clipping не работает правильно для большого ratio и положительного advantage"

        # Случай 2: Маленький ratio, положительный advantage
        # ОЖИДАНИЕ: Clipping ограничивает обновление до (1 - clip_range) * advantage
        advantage = torch.tensor([2.0])
        ratio = torch.tensor([0.5])  # Очень маленький ratio

        loss_1 = advantage * ratio  # = 1.0
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)  # = 2.0 * 0.8 = 1.6
        ppo_loss = -torch.min(loss_1, loss_2)  # min(1.0, 1.6) = 1.0, затем -1.0

        expected_unclipped = -advantage * ratio
        print(f"\nCase 2 - Small ratio, positive advantage:")
        print(f"  PPO loss: {ppo_loss.item():.6f}")
        print(f"  Expected (unclipped): {expected_unclipped.item():.6f}")
        assert torch.allclose(ppo_loss, expected_unclipped, atol=1e-6), \
            "Для маленького ratio с положительным advantage должен использоваться unclipped loss"

        # Случай 3: Большой ratio, отрицательный advantage
        # ОЖИДАНИЕ: НЕТ clipping (агент должен сильно избегать плохих действий)
        advantage = torch.tensor([-2.0])
        ratio = torch.tensor([2.0])

        loss_1 = advantage * ratio  # = -4.0
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)  # = -2.0 * 1.2 = -2.4
        ppo_loss = -torch.min(loss_1, loss_2)  # min(-4.0, -2.4) = -4.0, затем 4.0

        expected_unclipped = -advantage * ratio
        print(f"\nCase 3 - Large ratio, negative advantage:")
        print(f"  PPO loss: {ppo_loss.item():.6f}")
        print(f"  Expected (unclipped): {expected_unclipped.item():.6f}")
        assert torch.allclose(ppo_loss, expected_unclipped, atol=1e-6), \
            "Для большого ratio с отрицательным advantage не должно быть clipping"

    def test_ppo_gradient_flow(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: Градиенты должны течь правильно через PPO loss.

        Проблема: При использовании min(), градиент течет только через меньшую ветку.
        Это может привести к нестабильности, если ветки часто переключаются.
        """
        clip_range = 0.2

        # Создаем параметр, требующий градиент
        log_prob_ratio = torch.tensor([0.5], requires_grad=True)
        old_log_prob = torch.tensor([0.0])
        advantage = torch.tensor([1.0])

        # Вычисляем PPO loss
        ratio = torch.exp(log_prob_ratio - old_log_prob)
        loss_1 = advantage * ratio
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        ppo_loss = -torch.min(loss_1, loss_2).mean()

        # Вычисляем градиент
        ppo_loss.backward()

        print(f"\nGradient flow test:")
        print(f"  Ratio: {ratio.item():.6f}")
        print(f"  Loss 1 (unclipped): {loss_1.item():.6f}")
        print(f"  Loss 2 (clipped): {loss_2.item():.6f}")
        print(f"  PPO loss: {ppo_loss.item():.6f}")
        print(f"  Gradient: {log_prob_ratio.grad.item():.6f}")

        assert log_prob_ratio.grad is not None, "Градиент не вычислен"
        assert torch.isfinite(log_prob_ratio.grad), "Градиент не конечен"


class TestValueFunctionClipping:
    """Проверка корректности value function clipping."""

    def test_vf_clipping_formula(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: VF clipping должна использовать правильную формулу.

        Согласно оригинальной статье PPO (Schulman et al. 2017), правильная формула:
        L^VF = max(L_unclipped, L_clipped)
        где:
        - L_unclipped = (V_θ(s) - V_target)^2
        - L_clipped = (clip(V_θ(s), V_old - ε, V_old + ε) - V_target)^2

        ВАЖНО: Целевое значение V_target НЕ должно клиппироваться!
        """
        clip_range_vf = 0.2

        # Сценарий 1: Новое значение сильно отличается от старого
        v_old = torch.tensor([5.0])
        v_new = torch.tensor([8.0])  # Сильно больше старого
        v_target = torch.tensor([6.0])

        # Unclipped loss
        loss_unclipped = (v_new - v_target) ** 2  # (8 - 6)^2 = 4

        # Clipped value
        v_clipped = torch.clamp(v_new, v_old - clip_range_vf, v_old + clip_range_vf)  # clip(8, 4.8, 5.2) = 5.2
        loss_clipped = (v_clipped - v_target) ** 2  # (5.2 - 6)^2 = 0.64

        # VF loss (максимум из двух)
        vf_loss = torch.max(loss_unclipped, loss_clipped)  # max(4, 0.64) = 4

        print(f"\nVF Clipping test:")
        print(f"  V_old: {v_old.item():.2f}")
        print(f"  V_new: {v_new.item():.2f}")
        print(f"  V_target: {v_target.item():.2f}")
        print(f"  V_clipped: {v_clipped.item():.2f}")
        print(f"  Loss unclipped: {loss_unclipped.item():.6f}")
        print(f"  Loss clipped: {loss_clipped.item():.6f}")
        print(f"  VF loss (max): {vf_loss.item():.6f}")

        # ОЖИДАНИЕ: VF loss должна быть равна unclipped loss (так как она больше)
        assert torch.allclose(vf_loss, loss_unclipped, atol=1e-6), \
            "VF loss должна выбирать максимум между clipped и unclipped loss"

        # КРИТИЧЕСКАЯ ПРОВЕРКА: Target не должен клиппироваться
        # Ошибочная реализация:
        v_target_clipped_wrong = torch.clamp(v_target, v_old - clip_range_vf, v_old + clip_range_vf)
        loss_wrong = (v_new - v_target_clipped_wrong) ** 2

        print(f"\n  WRONG implementation (clipped target): {loss_wrong.item():.6f}")

        # Убедимся, что правильная реализация отличается от неправильной
        assert not torch.allclose(vf_loss, loss_wrong, atol=1e-6), \
            "VF clipping не должна клиппировать target!"

    def test_vf_clipping_with_distributional_critic(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: VF clipping для distributional critic.

        Для квантильного критика нужно клиппировать ПРЕДСКАЗАНИЯ, но не ЦЕЛИ.
        Проблема: При клиппировании всего распределения квантилей, нужно
        сдвигать все квантили одинаково, чтобы сохранить форму распределения.
        """
        clip_range_vf = 0.2
        num_quantiles = 5

        # Старые и новые квантили
        quantiles_old_mean = 5.0
        quantiles_new = torch.tensor([[4.0, 4.5, 5.0, 5.5, 6.5]])  # mean ≈ 5.1
        v_old = torch.tensor([[quantiles_old_mean]])

        # Target (цель)
        target = torch.tensor([[6.0]])

        # Вычисляем среднее новых квантилей
        v_new_mean = quantiles_new.mean(dim=1, keepdim=True)  # ≈ 5.1

        # Клиппируем СРЕДНЕЕ
        v_clipped_mean = torch.clamp(
            v_new_mean,
            v_old - clip_range_vf,
            v_old + clip_range_vf
        )  # clip(5.1, 4.8, 5.2) = 5.1

        # Сдвигаем все квантили на разницу
        delta = v_clipped_mean - v_new_mean  # 5.1 - 5.1 = 0
        quantiles_clipped = quantiles_new + delta

        print(f"\nDistributional VF Clipping test:")
        print(f"  Old mean: {v_old.item():.2f}")
        print(f"  New quantiles: {quantiles_new.numpy()}")
        print(f"  New mean: {v_new_mean.item():.2f}")
        print(f"  Clipped mean: {v_clipped_mean.item():.2f}")
        print(f"  Clipped quantiles: {quantiles_clipped.numpy()}")
        print(f"  Delta shift: {delta.item():.6f}")

        # ОЖИДАНИЕ: Все квантили должны сдвинуться на одинаковую величину
        expected_shift = v_clipped_mean - v_new_mean
        actual_shifts = quantiles_clipped - quantiles_new
        assert torch.allclose(actual_shifts, expected_shift, atol=1e-6), \
            "Все квантили должны сдвигаться на одинаковую величину"

        # ОЖИДАНИЕ: Среднее клиппированных квантилей должно равняться клиппированному среднему
        clipped_quantiles_mean = quantiles_clipped.mean(dim=1, keepdim=True)
        assert torch.allclose(clipped_quantiles_mean, v_clipped_mean, atol=1e-6), \
            "Среднее клиппированных квантилей должно совпадать с клиппированным средним"


class TestAdvantageEstimation:
    """Проверка корректности вычисления advantages."""

    def test_gae_computation(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: GAE (Generalized Advantage Estimation) должна
        вычисляться правильно.

        Формула GAE (Schulman et al. 2016):
        A_t = Σ_(l=0)^∞ (γλ)^l δ_(t+l)
        где δ_t = r_t + γV(s_(t+1)) - V(s_t)

        Проблема: Неправильная обработка episode boundaries может привести
        к утечке информации между эпизодами.
        """
        gamma = 0.99
        gae_lambda = 0.95

        # Простой случай: 1 эпизод, 3 шага
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([5.0, 6.0, 7.0])
        next_value = 0.0  # Эпизод завершен
        dones = np.array([False, False, True])

        # Вычисляем TD errors
        next_values = np.append(values[1:], next_value)
        next_non_terminal = 1.0 - dones.astype(float)

        deltas = rewards + gamma * next_values * next_non_terminal - values

        print(f"\nGAE Computation test:")
        print(f"  Rewards: {rewards}")
        print(f"  Values: {values}")
        print(f"  Next values: {next_values}")
        print(f"  Deltas: {deltas}")

        # Вычисляем GAE вручную (backward pass)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal_t = 1.0 - dones[t]
            else:
                next_non_terminal_t = 1.0 - dones[t]

            delta = deltas[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal_t * last_gae
            advantages[t] = last_gae

        print(f"  Advantages: {advantages}")

        # ОЖИДАНИЕ: Advantage на последнем шаге должен быть равен delta
        # (так как эпизод завершился)
        expected_last_adv = deltas[-1]
        print(f"  Expected last advantage: {expected_last_adv:.6f}")
        print(f"  Actual last advantage: {advantages[-1]:.6f}")

        assert np.allclose(advantages[-1], expected_last_adv, atol=1e-6), \
            "Advantage на последнем шаге должен равняться delta"

        # Returns должны быть advantages + values
        returns = advantages + values
        print(f"  Returns: {returns}")

        # ОЖИДАНИЕ: Return на последнем шаге должен быть reward + gamma * next_value
        expected_last_return = rewards[-1] + gamma * next_value * (1.0 - dones[-1])
        assert np.allclose(returns[-1], expected_last_return, atol=1e-6), \
            "Return на последнем шаге должен быть r + γV_next"

    def test_advantage_normalization(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: Нормализация advantages должна быть корректной.

        Проблема 1: Нормализация по всему батчу может привести к проблемам
        при использовании mini-batch gradient descent.

        Проблема 2: Групповая нормализация (по подгруппам) может сломать
        относительную важность разных групп.
        """
        # Тест 1: Обычная нормализация
        advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages_normalized = (advantages - adv_mean) / (adv_std + 1e-8)

        print(f"\nAdvantage Normalization test:")
        print(f"  Original advantages: {advantages}")
        print(f"  Mean: {adv_mean:.6f}, Std: {adv_std:.6f}")
        print(f"  Normalized advantages: {advantages_normalized}")

        # ОЖИДАНИЕ: Среднее нормализованных advantages должно быть ~0
        assert np.allclose(advantages_normalized.mean(), 0.0, atol=1e-6), \
            "Среднее нормализованных advantages должно быть 0"

        # ОЖИДАНИЕ: Стандартное отклонение должно быть ~1
        # (используем ddof=0, так как это population std)
        assert np.allclose(advantages_normalized.std(ddof=0), 1.0, atol=1e-6), \
            "Стандартное отклонение нормализованных advantages должно быть 1"

        # Тест 2: Групповая нормализация
        # Проблема: Если группы имеют разные размеры, нормализация может
        # исказить их относительную важность
        group_1 = np.array([1.0, 2.0, 3.0])  # 3 элемента, mean=2, std≈0.816
        group_2 = np.array([10.0])  # 1 элемент

        # Глобальная статистика
        all_values = np.concatenate([group_1, group_2])
        global_mean = all_values.mean()  # (1+2+3+10)/4 = 4
        global_std = all_values.std()  # ≈3.674

        # Нормализация каждой группы по глобальной статистике
        group_1_norm = (group_1 - global_mean) / (global_std + 1e-8)
        group_2_norm = (group_2 - global_mean) / (global_std + 1e-8)

        print(f"\n  Group normalization test:")
        print(f"  Group 1 (size 3): {group_1}")
        print(f"  Group 2 (size 1): {group_2}")
        print(f"  Global mean: {global_mean:.6f}, std: {global_std:.6f}")
        print(f"  Group 1 normalized: {group_1_norm}")
        print(f"  Group 2 normalized: {group_2_norm}")

        # ОЖИДАНИЕ: Объединенные нормализованные значения должны иметь mean≈0, std≈1
        all_normalized = np.concatenate([group_1_norm, group_2_norm])
        assert np.allclose(all_normalized.mean(), 0.0, atol=1e-6), \
            "Среднее всех нормализованных значений должно быть 0"
        assert np.allclose(all_normalized.std(ddof=0), 1.0, atol=1e-6), \
            "Std всех нормализованных значений должно быть 1"


class TestKLDivergence:
    """Проверка корректности вычисления KL divergence."""

    def test_kl_approximation(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: Аппроксимация KL divergence в PPO.

        В коде (строка 8006):
        approx_kl_raw_tensor = old_log_prob_raw - log_prob_raw_new

        Это первого порядка аппроксимация:
        KL(π_old || π_new) ≈ E[log π_old(a|s) - log π_new(a|s)]

        Эта аппроксимация корректна для малых изменений политики, но
        может быть неточной для больших изменений.

        Правильная формула KL для гауссовых распределений:
        KL(N(μ1,σ1) || N(μ2,σ2)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        """
        # Тест 1: Первого порядка аппроксимация
        old_log_prob = torch.tensor([-2.0, -3.0, -1.5])
        new_log_prob = torch.tensor([-2.1, -2.9, -1.6])

        # Аппроксимация из кода
        approx_kl = old_log_prob - new_log_prob

        print(f"\nKL Divergence Approximation test:")
        print(f"  Old log prob: {old_log_prob.numpy()}")
        print(f"  New log prob: {new_log_prob.numpy()}")
        print(f"  Approx KL: {approx_kl.numpy()}")
        print(f"  Mean approx KL: {approx_kl.mean().item():.6f}")

        # ОЖИДАНИЕ: Для малых изменений, approx KL должна быть небольшой
        assert torch.all(torch.abs(approx_kl) < 1.0), \
            "Для малых изменений политики, KL должна быть небольшой"

        # Тест 2: Проверка знака
        # KL divergence всегда неотрицательна
        # Но аппроксимация может быть отрицательной!
        old_log_prob_high = torch.tensor([-1.0])
        new_log_prob_low = torch.tensor([-2.0])

        approx_kl_case = old_log_prob_high - new_log_prob_low  # = 1.0 (положительная)

        old_log_prob_low = torch.tensor([-2.0])
        new_log_prob_high = torch.tensor([-1.0])

        approx_kl_case2 = old_log_prob_low - new_log_prob_high  # = -1.0 (отрицательная!)

        print(f"\n  Sign test:")
        print(f"  Case 1 (old high, new low): {approx_kl_case.item():.6f}")
        print(f"  Case 2 (old low, new high): {approx_kl_case2.item():.6f}")

        # ПРЕДУПРЕЖДЕНИЕ: Аппроксимация KL может быть отрицательной!
        # Это нормально для первого порядка аппроксимации, но нужно учитывать.
        print(f"  WARNING: First-order KL approximation can be negative!")


class TestEntropyBonus:
    """Проверка корректности entropy bonus."""

    def test_entropy_gradient_flow(self):
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: Entropy bonus должна поощрять исследование.

        Loss = policy_loss + c * value_loss - α * entropy

        Знак минус перед entropy важен: мы МАКСИМИЗИРУЕМ entropy.

        Проблема: Если entropy_loss = -entropy (как в коде, строка 8018),
        то нужно добавлять с плюсом: + α * entropy_loss
        """
        # Создаем простое распределение
        logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        probs = F.softmax(logits, dim=-1)

        # Вычисляем entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # В коде используется entropy_loss = -entropy
        entropy_loss = -entropy

        # Вклад в итоговый loss (с положительным коэффициентом)
        ent_coef = 0.01
        total_loss_contribution = ent_coef * entropy_loss

        print(f"\nEntropy Bonus test:")
        print(f"  Logits: {logits.detach().numpy()}")
        print(f"  Probs: {probs.detach().numpy()}")
        print(f"  Entropy: {entropy.item():.6f}")
        print(f"  Entropy loss (negated): {entropy_loss.item():.6f}")
        print(f"  Total contribution: {total_loss_contribution.item():.6f}")

        # Вычисляем градиент
        total_loss_contribution.mean().backward()

        print(f"  Gradient w.r.t. logits: {logits.grad.numpy()}")

        # ОЖИДАНИЕ: Градиент должен УМЕНЬШАТЬ вероятность наиболее вероятного действия
        # (чтобы распределение стало более uniform, т.е. entropy увеличилась)
        max_prob_idx = probs.argmax()
        assert logits.grad is not None

        # При минимизации (ent_coef * entropy_loss), где entropy_loss = -entropy,
        # мы фактически МАКСИМИЗИРУЕМ entropy
        # Это означает, что градиент должен делать распределение более равномерным
        print(f"  Max prob action index: {max_prob_idx.item()}")
        print(f"  Gradient pushes probability: {'down' if logits.grad[0, max_prob_idx] < 0 else 'up'}")


class TestCriticalIssues:
    """Тесты для выявления критических проблем в реализации."""

    def test_ratio_clipping_overflow(self):
        """
        КРИТИЧЕСКАЯ ПРОБЛЕМА: Переполнение при вычислении ratio.

        В коде (строка 7870-7871):
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        Проблема: exp(20) ≈ 485,165,195 - это огромное число!
        При max=20, ratio может достигать ~485M, что может вызвать:
        1. Численную нестабильность
        2. Большие градиенты
        3. NaN/Inf в вычислениях

        Рекомендация: Использовать более консервативный предел, например ±10
        (exp(10) ≈ 22,026 - все еще большой, но более управляемый)
        """
        # Текущая реализация
        log_ratio_extreme = torch.tensor([20.0, -20.0])
        log_ratio_clamped = torch.clamp(log_ratio_extreme, min=-20.0, max=20.0)
        ratio_current = torch.exp(log_ratio_clamped)

        print(f"\nRatio Overflow test:")
        print(f"  Log ratio (clamped): {log_ratio_clamped.numpy()}")
        print(f"  Ratio (current impl): {ratio_current.numpy()}")
        print(f"  Max ratio: {ratio_current.max().item():.2e}")

        # ПРОБЛЕМА: ratio может быть экстремально большой
        assert ratio_current.max() < 1e6, \
            "CRITICAL: Ratio может достигать очень больших значений (>1M), что может вызвать численную нестабильность"

        # Рекомендуемая реализация
        log_ratio_recommended = torch.clamp(log_ratio_extreme, min=-10.0, max=10.0)
        ratio_recommended = torch.exp(log_ratio_recommended)

        print(f"\n  Recommended clamp range: [-10, 10]")
        print(f"  Ratio (recommended): {ratio_recommended.numpy()}")
        print(f"  Max ratio: {ratio_recommended.max().item():.2e}")

    def test_advantage_normalization_bias(self):
        """
        ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА: Групповая нормализация advantages.

        В коде (строки 7819-7826) используется групповая нормализация:
        advantages_normalized = (advantages - group_mean) / group_std

        Проблема: Если группы имеют разные характеристики (например,
        разные уровни риска или доходности), глобальная нормализация
        может исказить их относительную важность.

        Вопрос: Должны ли мы нормализовать каждую группу отдельно или
        использовать глобальную статистику?
        """
        # Симулируем две группы с разными характеристиками
        group_1_advantages = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Низкая волатильность
        group_2_advantages = np.array([5.0, 10.0, 15.0])  # Высокая волатильность

        # Глобальная нормализация (как в коде)
        all_advantages = np.concatenate([group_1_advantages, group_2_advantages])
        global_mean = all_advantages.mean()
        global_std = all_advantages.std()

        group_1_normalized_global = (group_1_advantages - global_mean) / global_std
        group_2_normalized_global = (group_2_advantages - global_mean) / global_std

        print(f"\nAdvantage Normalization Bias test:")
        print(f"  Group 1 advantages: {group_1_advantages}")
        print(f"  Group 2 advantages: {group_2_advantages}")
        print(f"  Global mean: {global_mean:.3f}, std: {global_std:.3f}")
        print(f"  Group 1 normalized (global): {group_1_normalized_global}")
        print(f"  Group 2 normalized (global): {group_2_normalized_global}")

        # Альтернатива: Нормализация каждой группы отдельно
        group_1_mean = group_1_advantages.mean()
        group_1_std = group_1_advantages.std()
        group_1_normalized_local = (group_1_advantages - group_1_mean) / (group_1_std + 1e-8)

        group_2_mean = group_2_advantages.mean()
        group_2_std = group_2_advantages.std()
        group_2_normalized_local = (group_2_advantages - group_2_mean) / (group_2_std + 1e-8)

        print(f"\n  Group 1 normalized (local): {group_1_normalized_local}")
        print(f"  Group 2 normalized (local): {group_2_normalized_local}")

        # ВОПРОС: Какой подход лучше?
        # Глобальная нормализация сохраняет относительную важность между группами,
        # но может дать очень разные распределения для разных групп.
        # Локальная нормализация дает каждой группе mean=0, std=1, но
        # теряет информацию о том, какая группа "важнее".

        print(f"\n  Question: Should we use global or local normalization?")
        print(f"  Global: Preserves relative importance between groups")
        print(f"  Local: Each group has mean=0, std=1, but loses inter-group info")


if __name__ == "__main__":
    print("="*80)
    print("ГЛУБОКИЙ АНАЛИЗ МАТЕМАТИЧЕСКОЙ КОРРЕКТНОСТИ PPO")
    print("="*80)

    # Запускаем все тесты
    test_classes = [
        TestPPOClippedObjective(),
        TestValueFunctionClipping(),
        TestAdvantageEstimation(),
        TestKLDivergence(),
        TestEntropyBonus(),
        TestCriticalIssues(),
    ]

    for test_obj in test_classes:
        print(f"\n{'='*80}")
        print(f"Running: {test_obj.__class__.__name__}")
        print(f"{'='*80}")

        for method_name in dir(test_obj):
            if method_name.startswith("test_"):
                method = getattr(test_obj, method_name)
                print(f"\n{'-'*80}")
                print(f"Test: {method_name}")
                print(f"{'-'*80}")
                try:
                    method()
                    print(f"✓ PASSED")
                except AssertionError as e:
                    print(f"✗ FAILED: {e}")
                except Exception as e:
                    print(f"✗ ERROR: {e}")

    print(f"\n{'='*80}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*80}")
