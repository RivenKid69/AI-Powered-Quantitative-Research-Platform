"""
Tests for State-Adversarial PPO (SA-PPO).

Target: 100% code coverage
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

from adversarial.sa_ppo import SAPPOConfig, StateAdversarialPPO
from adversarial.state_perturbation import PerturbationConfig


class TestSAPPOConfig:
    """Tests for SAPPOConfig."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = SAPPOConfig()
        assert config.enabled is True
        assert config.perturbation is not None
        assert isinstance(config.perturbation, PerturbationConfig)
        assert config.adversarial_ratio == 0.5
        assert config.robust_kl_coef == 0.1

    def test_custom_initialization(self):
        """Test custom configuration."""
        perturbation = PerturbationConfig(epsilon=0.2)
        config = SAPPOConfig(
            enabled=False,
            perturbation=perturbation,
            adversarial_ratio=0.7,
            robust_kl_coef=0.2,
        )
        assert config.enabled is False
        assert config.perturbation.epsilon == 0.2
        assert config.adversarial_ratio == 0.7

    def test_validation_invalid_adversarial_ratio(self):
        """Test invalid adversarial_ratio raises error."""
        with pytest.raises(ValueError, match="adversarial_ratio must be in"):
            SAPPOConfig(adversarial_ratio=1.5)

    def test_validation_negative_robust_kl_coef(self):
        """Test negative robust_kl_coef raises error."""
        with pytest.raises(ValueError, match="robust_kl_coef must be >= 0"):
            SAPPOConfig(robust_kl_coef=-0.1)

    def test_validation_negative_warmup(self):
        """Test negative warmup_updates raises error."""
        with pytest.raises(ValueError, match="warmup_updates must be >= 0"):
            SAPPOConfig(warmup_updates=-1)

    def test_validation_invalid_epsilon_schedule(self):
        """Test invalid epsilon_schedule raises error."""
        with pytest.raises(ValueError, match="epsilon_schedule must be"):
            SAPPOConfig(epsilon_schedule="invalid")


class TestStateAdversarialPPO:
    """Tests for StateAdversarialPPO."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PPO model."""
        model = MagicMock()
        policy = MagicMock()
        model.policy = policy
        return model

    @pytest.fixture
    def config(self):
        """Default SA-PPO configuration."""
        return SAPPOConfig(
            enabled=True,
            adversarial_ratio=0.5,
            robust_kl_coef=0.1,
            warmup_updates=5,
        )

    @pytest.fixture
    def sa_ppo(self, config, mock_model):
        """SA-PPO instance."""
        return StateAdversarialPPO(config, mock_model)

    def test_initialization(self, config, mock_model):
        """Test SA-PPO initialization."""
        sa_ppo = StateAdversarialPPO(config, mock_model)
        assert sa_ppo.config == config
        assert sa_ppo.model == mock_model
        assert sa_ppo._update_count == 0
        assert not sa_ppo._adversarial_enabled

    def test_on_training_start(self, sa_ppo):
        """Test on_training_start callback."""
        sa_ppo.on_training_start()
        assert sa_ppo._adversarial_enabled

    def test_is_adversarial_enabled_before_warmup(self, sa_ppo):
        """Test adversarial training is disabled during warmup."""
        sa_ppo.on_training_start()
        sa_ppo._update_count = 3  # Less than warmup_updates=5
        assert not sa_ppo.is_adversarial_enabled

    def test_is_adversarial_enabled_after_warmup(self, sa_ppo):
        """Test adversarial training enabled after warmup."""
        sa_ppo.on_training_start()
        sa_ppo._update_count = 10  # Greater than warmup_updates=5
        assert sa_ppo.is_adversarial_enabled

    def test_on_update_start(self, sa_ppo):
        """Test on_update_start increments counter."""
        initial_count = sa_ppo._update_count
        sa_ppo.on_update_start()
        assert sa_ppo._update_count == initial_count + 1

    def test_reset_stats(self, sa_ppo):
        """Test reset_stats clears all statistics."""
        sa_ppo._total_adversarial_samples = 100
        sa_ppo._total_clean_samples = 50
        sa_ppo._total_robust_kl_penalty = 5.0

        sa_ppo.reset_stats()

        assert sa_ppo._total_adversarial_samples == 0
        assert sa_ppo._total_clean_samples == 0
        assert sa_ppo._total_robust_kl_penalty == 0.0

    def test_get_stats(self, sa_ppo):
        """Test get_stats returns correct statistics."""
        sa_ppo._update_count = 10
        sa_ppo._total_adversarial_samples = 100
        sa_ppo._total_clean_samples = 100
        sa_ppo._total_robust_kl_penalty = 10.0

        stats = sa_ppo.get_stats()

        assert "sa_ppo/update_count" in stats
        assert stats["sa_ppo/update_count"] == 10
        assert stats["sa_ppo/adversarial_samples"] == 100
        assert stats["sa_ppo/clean_samples"] == 100
        assert stats["sa_ppo/adversarial_ratio"] == 0.5

    def test_compute_adversarial_loss_disabled(self, sa_ppo, mock_model):
        """Test standard loss when adversarial training disabled."""
        sa_ppo._adversarial_enabled = False

        states = torch.randn(4, 10)
        actions = torch.randn(4, 2)
        advantages = torch.randn(4)
        returns = torch.randn(4)
        old_log_probs = torch.randn(4)

        # Mock model outputs
        dist_mock = MagicMock()
        dist_mock.log_prob.return_value = torch.randn(4)
        mock_model.policy.get_distribution.return_value = dist_mock
        mock_model.policy.predict_values.return_value = torch.randn(4)

        loss, info = sa_ppo.compute_adversarial_loss(
            states, actions, advantages, returns, old_log_probs
        )

        assert isinstance(loss, torch.Tensor)
        assert "sa_ppo/policy_loss" in info
        assert "sa_ppo/value_loss" in info

    def test_get_current_epsilon_constant(self, sa_ppo):
        """Test epsilon with constant schedule."""
        sa_ppo.config.adaptive_epsilon = False
        sa_ppo.config.perturbation.epsilon = 0.1

        epsilon = sa_ppo._get_current_epsilon()
        assert epsilon == 0.1

    def test_get_current_epsilon_linear(self, sa_ppo):
        """Test epsilon with linear schedule."""
        sa_ppo.config.adaptive_epsilon = True
        sa_ppo.config.epsilon_schedule = "linear"
        sa_ppo.config.perturbation.epsilon = 0.1
        sa_ppo.config.epsilon_final = 0.05
        sa_ppo._update_count = 500  # 50% progress with max_updates=1000

        epsilon = sa_ppo._get_current_epsilon()
        # Should be halfway between 0.1 and 0.05
        assert 0.05 <= epsilon <= 0.1

    def test_get_current_epsilon_cosine(self, sa_ppo):
        """Test epsilon with cosine schedule."""
        sa_ppo.config.adaptive_epsilon = True
        sa_ppo.config.epsilon_schedule = "cosine"
        sa_ppo.config.perturbation.epsilon = 0.1
        sa_ppo.config.epsilon_final = 0.05
        sa_ppo._update_count = 1000

        epsilon = sa_ppo._get_current_epsilon()
        # Should be final epsilon
        assert abs(epsilon - 0.05) < 1e-3

    def test_update_epsilon_schedule(self, sa_ppo):
        """Test epsilon schedule update."""
        sa_ppo.config.adaptive_epsilon = True
        sa_ppo.config.epsilon_schedule = "linear"  # Use linear schedule
        sa_ppo.config.perturbation.epsilon = 0.1
        sa_ppo.config.epsilon_final = 0.05
        sa_ppo._update_count = 500  # Halfway through (max_updates=1000)

        initial_epsilon = sa_ppo.config.perturbation.epsilon
        sa_ppo._update_epsilon_schedule()

        # With linear schedule at 50% progress: 0.1 + (0.05 - 0.1) * 0.5 = 0.075
        expected_epsilon = 0.075
        assert abs(sa_ppo.config.perturbation.epsilon - expected_epsilon) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
