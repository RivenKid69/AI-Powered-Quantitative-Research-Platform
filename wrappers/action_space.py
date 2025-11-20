from __future__ import annotations

"""Utilities for enforcing the score-based action space."""

from dataclasses import replace
from typing import Any

import numpy as np
from gymnasium import ActionWrapper, spaces

from action_proto import ActionProto


SCORE_LOW: float = 0.0
SCORE_HIGH: float = 1.0
SCORE_SHAPE: tuple[int, ...] = (1,)


class ScoreActionWrapper(ActionWrapper):
    """Project all outgoing actions to the ``[0, 1]`` score interval."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            low=SCORE_LOW,
            high=SCORE_HIGH,
            shape=SCORE_SHAPE,
            dtype=np.float32,
        )
        self.observation_space = env.observation_space

    def action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise ValueError(
                f"ScoreActionWrapper expects a single scalar action, got shape {arr.shape}"
            )
        score = float(arr[0])
        if not np.isfinite(score):
            raise ValueError(f"Received non-finite score action: {score}")
        clipped = np.clip(score, SCORE_LOW, SCORE_HIGH)
        return np.asarray([clipped], dtype=np.float32)


class LongOnlyActionWrapper(ActionWrapper):
    """
    Transform actions to enforce long-only constraint.

    CRITICAL FIX (2025-11-21):
    - Maps policy outputs from [-1, 1] to [0, 1] for long-only trading
    - Preserves position reduction signals (negative → reduce to zero)
    - -1.0 → 0.0 (full exit), 0.0 → 0.5 (50% long), +1.0 → 1.0 (100% long)

    Rationale:
    - Long-only prevents SHORT positions, not position reductions
    - Policy needs to express "reduce position" via negative outputs
    - Linear mapping preserves information: a' = (a + 1) / 2
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)

    @staticmethod
    def _map_to_long_only(value: float) -> float:
        """Map from [-1, 1] to [0, 1] preserving reduction signals.

        Args:
            value: Action in [-1, 1] range (policy output)

        Returns:
            Mapped value in [0, 1] (long-only position target)

        Examples:
            -1.0 → 0.0 (full exit to cash)
            -0.5 → 0.25 (reduce to 25% long)
             0.0 → 0.5 (50% long position)
             0.5 → 0.75 (75% long position)
             1.0 → 1.0 (100% long position)
        """
        # Linear transformation: [-1, 1] → [0, 1]
        mapped = (value + 1.0) / 2.0
        # Clamp to ensure bounds (handles numeric errors)
        return float(np.clip(mapped, 0.0, 1.0))

    def action(self, action: Any) -> Any:
        if action is None:
            return action
        if isinstance(action, np.ndarray):
            if action.size == 0:
                return action
            # Map each element from [-1, 1] to [0, 1]
            arr = action.astype(np.float32, copy=False)
            mapped = (arr + 1.0) / 2.0
            clipped = np.clip(mapped, SCORE_LOW, SCORE_HIGH)
            return clipped
        if isinstance(action, (list, tuple)):
            arr = np.asarray(action, dtype=np.float32)
            mapped = (arr + 1.0) / 2.0
            return np.clip(mapped, SCORE_LOW, SCORE_HIGH)
        if isinstance(action, ActionProto):
            mapped = self._map_to_long_only(action.volume_frac)
            if abs(mapped - action.volume_frac) < 1e-9:
                return action  # No change needed
            return replace(action, volume_frac=mapped)
        try:
            value = float(action)
        except (TypeError, ValueError):
            return action
        if not np.isfinite(value):
            raise ValueError(f"Non-finite long-only score: {value}")
        return self._map_to_long_only(value)

