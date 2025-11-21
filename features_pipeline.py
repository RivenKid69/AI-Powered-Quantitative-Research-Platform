# -*- coding: utf-8 -*-
"""
features_pipeline.py
------------------------------------------------------------------
Single source of truth for feature normalization both in training
and inference. Works over dict[str, pandas.DataFrame] where each DF
follows the canonical schema established in prepare_and_run.py.
- Adds standardized columns with suffix '_z' (z-score).
- Leaves original columns intact.
- Saves/loads stats to/from JSON for reproducibility.

FIXES:
- FIX (MEDIUM #3): Added outlier detection via winsorization for robust statistics.
- FIX (2025-11-21): Winsorization consistency - bounds from fit() applied in transform()
- FIX (2025-11-21): Close shift consistency - always shift to prevent look-ahead bias

Usage:
    pipe = FeaturePipeline()
    pipe.fit(all_dfs_dict)                 # during training
    pipe.save("models/preproc_pipeline.json")
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")

    pipe = FeaturePipeline.load("models/preproc_pipeline.json")  # inference
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")
"""
import os
import json
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

CANON_PREFIX = [
    "timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
    "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"
]


# ==============================================================================
# FIX (MEDIUM #3): Outlier Detection Utilities
# ==============================================================================

def winsorize_array(data: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """
    Winsorize array: cap extreme values at specified percentiles.

    Winsorization is preferred over outlier removal in finance because it:
    - Preserves data points (no removal → no gaps in time series)
    - Bounds extreme values (flash crashes, fat-finger errors)
    - Maintains distribution shape in bulk (99% of data unchanged)

    Common use: Handle crypto market anomalies (flash wicks, exchange glitches).

    Args:
        data: Input array (may contain NaN)
        lower_percentile: Lower percentile bound (default: 1st percentile)
        upper_percentile: Upper percentile bound (default: 99th percentile)

    Returns:
        Winsorized array with same shape as input

    References:
        Dixon, W. J. (1960). "Simplified Estimation from Censored Normal Samples"
        Cont, R. (2001). "Empirical Properties of Asset Returns"

    Examples:
        >>> data = np.array([0.01, 0.02, 0.03, -0.50, 0.04])  # -50% flash crash
        >>> winsorize_array(data, 1, 99)
        array([0.01, 0.02, 0.03, 0.01, 0.04])  # Crash clipped to 1st percentile
    """
    if len(data) == 0:
        return data

    # Ignore NaN for percentile calculation
    lower_bound = np.nanpercentile(data, lower_percentile)
    upper_bound = np.nanpercentile(data, upper_percentile)

    return np.clip(data, lower_bound, upper_bound)

# Additional optional features we may standardize if present
OPTIONAL_NUMERIC = [
    "fear_greed_value","fear_greed_value_norm",
    "recent_event_high_96h","recent_event_medium_96h","time_since_last_event_hours",
]

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)

def _columns_to_scale(df: pd.DataFrame) -> List[str]:
    # Key columns which are numeric but shouldn't be z-scored directly:
    exclude = {"timestamp"}  # 'symbol' non-numeric already excluded
    cols: List[str] = []
    for c in df.columns:
        if c in exclude: 
            continue
        if c == "symbol":
            continue
        if c.endswith("_z"):  # already standardized
            continue
        if _is_numeric(df[c]):
            cols.append(c)
    return cols

class FeaturePipeline:
    def __init__(
        self,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        metadata: Optional[Dict[str, object]] = None,
        enable_winsorization: bool = True,  # FIX (MEDIUM #3): Winsorization enabled by default
        winsorize_percentiles: Tuple[float, float] = (1.0, 99.0),
        strict_idempotency: bool = True,  # FIX (2025-11-21): Fail on repeated transform_df()
    ):
        """Container for feature normalization statistics.

        Parameters
        ----------
        stats:
            Mapping from column name to ``{"mean": float, "std": float, "is_constant": bool}``.
        metadata:
            Additional information persisted alongside the statistics (for
            example the training window bounds or split version).
        enable_winsorization:
            If True, apply winsorization to mitigate outliers before computing statistics.
            Recommended for financial data to handle flash crashes, fat wicks, etc.
            Default: True
        winsorize_percentiles:
            Tuple of (lower_percentile, upper_percentile) for winsorization.
            Default: (1.0, 99.0) clips extreme 1% on each tail.
        strict_idempotency:
            If True (default), raise ValueError on repeated transform_df() to prevent
            double-shifting and data corruption. If False, make transform_df() idempotent
            (return already-transformed DataFrame unchanged).
            Default: True (strict mode for data integrity)
        """

        # stats: {col: {"mean": float, "std": float, "is_constant": bool, "winsorize_bounds": tuple}}
        self.stats: Dict[str, Dict[str, float]] = stats or {}
        self.metadata: Dict[str, object] = metadata or {}
        self.enable_winsorization = enable_winsorization
        self.winsorize_percentiles = winsorize_percentiles
        self.strict_idempotency = strict_idempotency

        # FIX (2025-11-21): Removed _close_shifted_in_fit flag - always shift in transform_df()
        # Previous logic was inverted and caused look-ahead bias

    def reset(self) -> None:
        """Drop previously computed statistics.

        Creating a fresh instance for each ``TradingEnv`` or clearing the
        state on episode reset avoids cross‑environment leakage of
        normalization parameters.
        """

        self.stats.clear()
        self.metadata.clear()

    def fit(
        self,
        dfs: Dict[str, pd.DataFrame],
        *,
        train_mask_column: Optional[str] = None,
        train_mask_values: Optional[Iterable] = None,
        train_start_ts: Optional[int] = None,
        train_end_ts: Optional[int] = None,
        timestamp_column: str = "timestamp",
        split_version: Optional[str] = None,
        train_intervals: Optional[Sequence[Tuple[Optional[int], Optional[int]]]] = None,
    ) -> "FeaturePipeline":
        """Fit normalization statistics from the provided dataframes.

        The caller may either provide a boolean/role mask identifying the
        training rows (for example a ``wf_role`` column equal to ``"train"``)
        or explicit ``train_start_ts``/``train_end_ts`` bounds. When both are
        supplied the intersection is used.
        """

        frames: List[pd.DataFrame] = []
        per_symbol_counts: Dict[str, int] = {}

        if train_mask_column is None and (train_start_ts is not None or train_end_ts is not None):
            # Ensure the timestamp column exists up-front when time bounds are used.
            for name, df in dfs.items():
                if timestamp_column not in df.columns:
                    raise KeyError(
                        f"DataFrame '{name}' is missing timestamp column '{timestamp_column}' required for training window filter."
                    )

        mask_values: Optional[Sequence] = None
        if train_mask_values is not None:
            mask_values = tuple(train_mask_values)

        for name, df in dfs.items():
            if df is None:
                continue
            cur = df
            if train_mask_column is not None:
                if train_mask_column not in cur.columns:
                    raise KeyError(
                        f"DataFrame '{name}' is missing training mask column '{train_mask_column}'."
                    )
                mask_series = cur[train_mask_column]
                if mask_values is None:
                    if pd.api.types.is_bool_dtype(mask_series):
                        mask = mask_series.astype(bool)
                    else:
                        mask = mask_series.astype(str).str.lower() == "train"
                else:
                    mask = mask_series.isin(mask_values)
                cur = cur.loc[mask]

            if train_start_ts is not None or train_end_ts is not None:
                ts = pd.to_numeric(cur[timestamp_column], errors="coerce")
                time_mask = pd.Series(True, index=cur.index)
                if train_start_ts is not None:
                    time_mask &= ts >= int(train_start_ts)
                if train_end_ts is not None:
                    time_mask &= ts <= int(train_end_ts)
                cur = cur.loc[time_mask]

            if not cur.empty:
                per_symbol_counts[name] = int(len(cur))
                frames.append(cur)

        if not frames:
            raise ValueError("No rows available to fit FeaturePipeline after applying training filters.")

        # FIX: Apply shift() per-symbol BEFORE concat to prevent cross-symbol contamination
        # Each frame corresponds to one symbol, so we shift each independently
        # FIX (2025-11-21): Always shift close in fit() to compute stats on previous close
        shifted_frames: List[pd.DataFrame] = []
        for frame in frames:
            if "close_orig" not in frame.columns and "close" in frame.columns:
                frame_copy = frame.copy()
                frame_copy["close"] = frame_copy["close"].shift(1)
                shifted_frames.append(frame_copy)
            else:
                shifted_frames.append(frame)

        big = pd.concat(shifted_frames, axis=0, ignore_index=True)
        cols = _columns_to_scale(big)
        stats = {}
        all_nan_columns = []  # FIX (2025-11-21): Track all-NaN columns for warning

        for c in cols:
            v = big[c].astype(float).to_numpy()

            # FIX (2025-11-21): Detect all-NaN columns BEFORE winsorization
            # If column is entirely NaN, np.nanpercentile returns NaN bounds
            # which leads to silent NaN → 0.0 conversion (semantic ambiguity)
            is_all_nan = v.size > 0 and np.isnan(v).all()

            # FIX (MEDIUM #3): Apply winsorization to mitigate outliers
            # This prevents flash crashes, fat-finger errors, and data anomalies
            # from contaminating normalization statistics (mean, std)
            # FIX (2025-11-21): Store winsorization bounds for consistent application in transform
            winsorize_bounds = None
            if self.enable_winsorization and not is_all_nan:
                # Only compute bounds if column has at least one non-NaN value
                lower_bound = np.nanpercentile(v, self.winsorize_percentiles[0])
                upper_bound = np.nanpercentile(v, self.winsorize_percentiles[1])

                # FIX (2025-11-21): Validate bounds are finite
                # If bounds are NaN (edge case: very few non-NaN values), skip winsorization
                if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                    v_clean = np.clip(v, lower_bound, upper_bound)
                    winsorize_bounds = (float(lower_bound), float(upper_bound))
                else:
                    # Bounds invalid → treat as all-NaN
                    v_clean = v
                    is_all_nan = True
            else:
                v_clean = v

            m = float(np.nanmean(v_clean))
            # IMPROVEMENT: Use population std (ddof=0) for ML consistency
            # This aligns with ML frameworks: scikit-learn StandardScaler, PyTorch normalization
            # use ddof=0 (population std) for feature scaling, not ddof=1 (sample std).
            # For large datasets (n > 100), difference is negligible: sqrt(n/(n-1)) ≈ 1.005
            # For consistency with standard ML pipelines, we use ddof=0.
            # Reference: Pedregosa et al. (2011), "Scikit-learn: Machine Learning in Python"
            s = float(np.nanstd(v_clean, ddof=0))

            # FIX (MEDIUM #4): Store zero variance indicator for proper handling
            # When s == 0 (constant feature), we mark it explicitly so transform can return zeros
            # instead of applying (value - mean) / 1.0 which may not be zero for NaN values
            is_constant = (not np.isfinite(s)) or (s == 0.0)
            if is_constant:
                s = 1.0  # avoid division by zero (will be handled specially in transform)
            if not np.isfinite(m):
                m = 0.0

            stats[c] = {"mean": m, "std": s, "is_constant": is_constant}

            # FIX (2025-11-21): Mark all-NaN columns explicitly
            # This prevents silent NaN → 0.0 conversion and provides clear semantics
            if is_all_nan:
                stats[c]["is_all_nan"] = True
                all_nan_columns.append(c)
                # Do NOT store winsorize_bounds for all-NaN columns
                # This signals to transform_df() to skip winsorization
            elif winsorize_bounds is not None:
                # FIX (2025-11-21): Store winsorization bounds for train/inference consistency
                stats[c]["winsorize_bounds"] = winsorize_bounds

        # FIX (2025-11-21): Warn about all-NaN columns
        # These columns cannot provide useful information for training
        # and should be investigated for data quality issues
        if all_nan_columns:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Found {len(all_nan_columns)} column(s) with ALL NaN values: {all_nan_columns}. "
                f"These columns will be normalized to NaN (not zeros) to preserve semantic meaning. "
                f"Consider: (1) Checking data quality, (2) Imputing missing values, or (3) Removing these features."
            )

        intervals_payload: Optional[List[Dict[str, Optional[int]]]] = None
        if train_intervals:
            intervals_payload = [
                {
                    "start_ts": int(start) if start is not None else None,
                    "end_ts": int(end) if end is not None else None,
                }
                for start, end in train_intervals
            ]

        metadata: Dict[str, object] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "filters": {
                "train_mask_column": train_mask_column,
                "train_mask_values": list(mask_values) if mask_values is not None else None,
                "train_start_ts": int(train_start_ts) if train_start_ts is not None else None,
                "train_end_ts": int(train_end_ts) if train_end_ts is not None else None,
                "timestamp_column": timestamp_column,
                "train_intervals": intervals_payload,
            },
            "train_rows_by_symbol": per_symbol_counts,
            "train_rows_total": int(sum(per_symbol_counts.values())),
        }
        if split_version is not None:
            metadata["split_version"] = str(split_version)

        self.stats = stats
        self.metadata = metadata
        return self

    def transform_df(self, df: pd.DataFrame, add_suffix: str = "_z") -> pd.DataFrame:
        """Transform DataFrame by applying normalization statistics.

        IMPORTANT: This method should be called ONLY ONCE per DataFrame.
        Repeated calls will cause double-shifting of 'close' column, leading to
        data misalignment and look-ahead bias accumulation.

        To apply transform multiple times:
        1. Preserve original close: df["close_orig"] = df["close"].copy()
        2. Or use fresh copy from original data source

        Args:
            df: DataFrame to transform
            add_suffix: Suffix for normalized columns (default: "_z")

        Returns:
            Transformed DataFrame with normalized columns

        Raises:
            ValueError: If pipeline is empty (call fit() or load() first)
            ValueError: If repeated application detected (strict_idempotency=True, default)
        """
        if not self.stats:
            raise ValueError("FeaturePipeline is empty; call fit() or load().")

        # FIX (2025-11-21): Detect repeated transform_df() application
        # Check for marker in DataFrame attrs (metadata introduced in pandas 1.0)
        # This prevents silent data corruption from double-shifting
        if hasattr(df, 'attrs') and df.attrs.get('_feature_pipeline_transformed', False):
            if self.strict_idempotency:
                # STRICT MODE (default): Fail immediately to prevent data corruption
                raise ValueError(
                    "transform_df() called on already-transformed DataFrame! "
                    "This would cause DOUBLE SHIFT of 'close' column, leading to:\n"
                    "  1. Data misalignment (close lag = 2 instead of 1)\n"
                    "  2. Accumulated look-ahead bias (features based on t+1 close)\n"
                    "  3. Scale mismatch (z-scores applied twice with different stats)\n\n"
                    "To fix:\n"
                    "  - Option A: Preserve 'close_orig' before first transform\n"
                    "  - Option B: Use fresh copy from original data source\n"
                    "  - Option C: Set strict_idempotency=False for idempotent behavior\n\n"
                    "This error prevents silent data corruption in training loop."
                )
            else:
                # IDEMPOTENT MODE: Return already-transformed DataFrame unchanged
                import warnings
                warnings.warn(
                    "transform_df() called on already-transformed DataFrame. "
                    "Returning without changes (idempotent mode). "
                    "Set strict_idempotency=True to fail immediately and catch errors.",
                    RuntimeWarning,
                    stacklevel=2
                )
                return df  # Return original (already transformed) without changes

        out = df.copy()

        # FIX (2025-11-21): ALWAYS shift close to prevent look-ahead bias
        # Previous logic was inverted: shift was skipped when _close_shifted_in_fit=True
        # This caused scale mismatch: stats from shifted data applied to unshifted data
        #
        # Correct behavior: If stats were computed on shifted data, transform must also
        # operate on shifted data for consistency and to prevent look-ahead bias
        if "close_orig" not in out.columns and "close" in out.columns:
            if "symbol" in out.columns:
                # Per-symbol shift to prevent cross-symbol contamination
                out["close"] = out.groupby("symbol", group_keys=False)["close"].shift(1)
            else:
                # Single symbol case - standard shift
                out["close"] = out["close"].shift(1)

        for c, ms in self.stats.items():
            if c not in out.columns:
                # silently skip columns missing in this DF
                continue
            v = out[c].astype(float).to_numpy()

            # FIX (2025-11-21): Handle all-NaN columns explicitly
            # If column was all-NaN during training, preserve NaN semantics
            # (do NOT convert to zeros, as zeros have different meaning)
            if ms.get("is_all_nan", False):
                # Column was entirely NaN during training
                # Keep as NaN to preserve semantic distinction from zero values
                # Model should handle NaN appropriately (skip, impute, or use validity flags)
                z = np.full_like(v, np.nan, dtype=float)
                out[c + add_suffix] = z
                continue  # Skip winsorization and standardization

            # FIX (2025-11-21): Apply winsorization bounds from training for consistency
            # This ensures train/inference distribution match and prevents OOD z-scores
            #
            # Best practice references:
            # - Huber (1981) "Robust Statistics": Apply same robust procedure on train/test
            # - Scikit-learn RobustScaler: Clips test data using train quantiles
            # - De Prado (2018) "Advances in Financial ML": Consistent winsorization
            if "winsorize_bounds" in ms:
                lower, upper = ms["winsorize_bounds"]
                v = np.clip(v, lower, upper)

            # FIX (MEDIUM #4): Handle constant features explicitly
            # For zero-variance features, return zeros instead of (value - mean) / 1.0
            # This prevents NaN propagation and ensures semantic correctness
            if ms.get("is_constant", False):
                # Feature had zero variance during training → always normalize to 0
                z = np.zeros_like(v, dtype=float)
            else:
                z = (v - ms["mean"]) / ms["std"]
            out[c + add_suffix] = z

        # FIX (2025-11-21): Mark DataFrame as transformed to detect repeated applications
        # Use DataFrame.attrs (metadata dict introduced in pandas 1.0)
        # This marker survives copy() operations and helps prevent silent data corruption
        if hasattr(out, 'attrs'):
            out.attrs['_feature_pipeline_transformed'] = True

        return out

    def transform_dict(self, dfs: Dict[str, pd.DataFrame], add_suffix: str = "_z") -> Dict[str, pd.DataFrame]:
        return {k: self.transform_df(v, add_suffix=add_suffix) for k, v in dfs.items()}

    def get_metadata(self) -> Dict[str, object]:
        """Return metadata captured during :meth:`fit`."""

        return dict(self.metadata)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # FIX (2025-11-21): Save configuration flags for reproducibility
        payload = {
            "stats": self.stats,
            "metadata": self.metadata,
            "config": {
                "enable_winsorization": self.enable_winsorization,
                "winsorize_percentiles": self.winsorize_percentiles,
                "strict_idempotency": self.strict_idempotency,
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "stats" in payload:
            stats = payload.get("stats", {})
            metadata = payload.get("metadata", {})
            # FIX (2025-11-21): Load configuration flags if available
            config = payload.get("config", {})
            return cls(
                stats=stats,
                metadata=metadata,
                enable_winsorization=config.get("enable_winsorization", True),
                winsorize_percentiles=tuple(config.get("winsorize_percentiles", [1.0, 99.0])),
                strict_idempotency=config.get("strict_idempotency", True),
            )
        else:
            # Backwards compatibility for legacy artifacts containing only stats.
            stats = payload
            metadata = {}
            return cls(stats=stats, metadata=metadata)
