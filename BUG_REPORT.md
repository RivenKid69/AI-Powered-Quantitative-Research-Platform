# Critical Bug Report

## 1. Timestamp Misalignment causing Feature Lag

**Severity:** Critical
**Component:** Data Pipeline (`prepare_and_run.py`, `features_pipeline.py`)

### Description
The `prepare_and_run.py` script was recently updated (around 2025-11-25) to standardize the `timestamp` column to use **CLOSE TIME**. However, the `features_pipeline.py` logic still applies a 1-step shift to all feature columns (`out[col] = out[col].shift(1)`).

### Impact
When `timestamp` represents **CLOSE TIME** $t$:
1. Row $t$ contains market data for the candle ending at $t$ (Open, High, Low, Close).
2. `features_pipeline` shifts features, so Row $t$ receives features from Row $t-1$.
3. Effectively, at time $t$ (when candle $t$ is closed and fully available), the model is fed features from time $t-1$.
4. This introduces a **1-step lag** (using stale data from the previous candle), ignoring the most recent market information available for decision making.

### Conflict with Requirements
This behavior contradicts the project requirement stated in memory: "Internal data timestamps ... must be standardized to the candle **Open Time**". If timestamps were Open Time ($t_{open}$), shifting by 1 would be correct (Row $t_{open}$ gets features from $t_{open}-1$, which is the most recent closed candle relative to the open).

### Verification
A reproduction script `verify_timestamp_lag.py` was created to simulate the pipeline.
- Input: DataFrame with timestamps $[T_0, T_1, \dots]$.
- Transformation: `FeaturePipeline` shifted features.
- Result: At timestamp $T_1$, the feature values corresponded to data from $T_0$.
- Conclusion: Since $T_1$ is Close Time, data from $T_1$ is available but ignored.

---

## Verified Correct Implementations

The following potential issues were checked and found to be **CORRECT** in the current codebase:

1.  **Twin Critics VF Clipping:** `distributional_ppo.py` correctly implements independent clipping for each critic (`loss_c1_clipped` vs `loss_c1_unclipped`) and aggregates using `max(clipped, unclipped)` per critic before averaging.
2.  **Quantile Levels:** `custom_policy_patch1.py` correctly implements the midpoint formula for quantile levels ($\tau_i = (i + 0.5) / N$), matching the assumptions in `_cvar_from_quantiles`.
3.  **Bankruptcy Reward:** `reward.pyx` correctly returns a fixed negative penalty (`-10.0`) instead of `NAN` when net worth drops to zero.
