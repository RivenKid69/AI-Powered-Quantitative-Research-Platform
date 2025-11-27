# L3 LOB Simulator - Configuration Reference

## Overview

The L3 simulator uses Pydantic-based configuration with YAML support. All configuration can be done programmatically or via YAML files.

**File**: `lob/config.py` (~765 lines)

## Quick Start

```python
from lob.config import L3ExecutionConfig, create_l3_config

# Use preset
config = create_l3_config("equity")

# Or load from YAML
config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")

# Create provider with config
from execution_providers import create_execution_provider, AssetClass

provider = create_execution_provider(
    AssetClass.EQUITY,
    level="L3",
    config=config,
)
```

## Configuration Presets

| Preset | Use Case | Features Enabled |
|--------|----------|------------------|
| `equity` | US equity simulation | All features |
| `crypto` | Crypto simulation | Impact, no dark pools |
| `minimal` | Fast simulation | Matching only |
| `equity_hft` | HFT simulation | Low latency, no dark pools |
| `equity_retail` | Retail simulation | High latency |

```python
# Available presets
config = create_l3_config("equity")
config = create_l3_config("crypto")
config = create_l3_config("minimal")
```

## Full Configuration Reference

### L3ExecutionConfig (Root)

```python
from lob.config import L3ExecutionConfig

config = L3ExecutionConfig(
    enabled=True,                    # Master switch
    lob_data=LOBDataConfig(...),     # Data source
    latency=L3LatencyConfig(...),    # Latency simulation
    fill_probability=FillProbabilityConfig(...),
    queue_value=L3QueueValueConfig(...),
    market_impact=MarketImpactConfig(...),
    hidden_liquidity=HiddenLiquidityConfig(...),
    dark_pools=DarkPoolsConfig(...),
    queue_tracking=QueueTrackingConfig(...),
    event_scheduling=EventSchedulingConfig(...),
)
```

### LOBDataConfig

```yaml
lob_data:
  source: internal           # internal, lobster, itch, synthetic
  reconstruction_mode: full  # full or incremental
  max_depth: 20              # Max levels to maintain
  snapshot_interval_ms: 1000 # Snapshot frequency
  message_path: null         # Path to LOB data files
```

```python
from lob.config import LOBDataConfig, LOBDataSourceType

config = LOBDataConfig(
    source=LOBDataSourceType.INTERNAL,
    reconstruction_mode="full",
    max_depth=20,
    snapshot_interval_ms=1000,
    message_path=None,
)
```

### LatencyConfig

```yaml
latency:
  enabled: true
  profile: institutional  # colocated, proximity, retail, institutional, custom

  feed_latency:
    enabled: true
    distribution: lognormal  # constant, uniform, lognormal, pareto, gamma
    mean_us: 200.0
    std_us: 50.0
    min_us: 50.0
    max_us: 2000.0
    spike_prob: 0.001
    spike_mult: 10.0

  order_latency:
    enabled: true
    distribution: lognormal
    mean_us: 300.0
    std_us: 80.0
    min_us: 100.0
    max_us: 3000.0

  exchange_latency:
    enabled: true
    distribution: lognormal
    mean_us: 100.0
    std_us: 30.0

  fill_latency:
    enabled: true
    distribution: lognormal
    mean_us: 150.0
    std_us: 40.0

  time_of_day_adjustment: true
  volatility_adjustment: true
```

```python
from lob.config import (
    L3LatencyConfig,
    LatencyComponentConfig,
    LatencyDistributionType,
    LatencyProfileType,
)

config = L3LatencyConfig(
    enabled=True,
    profile=LatencyProfileType.INSTITUTIONAL,
    feed_latency=LatencyComponentConfig(
        enabled=True,
        distribution=LatencyDistributionType.LOGNORMAL,
        mean_us=200.0,
        std_us=50.0,
        min_us=50.0,
        max_us=2000.0,
        spike_prob=0.001,
        spike_mult=10.0,
    ),
    order_latency=LatencyComponentConfig(...),
    exchange_latency=LatencyComponentConfig(...),
    fill_latency=LatencyComponentConfig(...),
    time_of_day_adjustment=True,
    volatility_adjustment=True,
)
```

### FillProbabilityConfig

```yaml
fill_probability:
  enabled: true
  model: queue_reactive  # poisson, queue_reactive, historical, distance_based
  base_rate: 100.0       # Base volume rate (qty/sec)
  queue_decay_alpha: 0.01
  spread_sensitivity_beta: 0.5
  imbalance_factor: 0.3
  volatility_factor: 0.2
  confidence_threshold: 0.5
```

```python
from lob.config import FillProbabilityConfig, L3FillProbabilityModelType

config = FillProbabilityConfig(
    enabled=True,
    model=L3FillProbabilityModelType.QUEUE_REACTIVE,
    base_rate=100.0,
    queue_decay_alpha=0.01,
    spread_sensitivity_beta=0.5,
    imbalance_factor=0.3,
    volatility_factor=0.2,
    confidence_threshold=0.5,
)
```

### QueueValueConfig

```yaml
queue_value:
  enabled: true
  hold_threshold: 0.001
  cancel_threshold: -0.002
  adverse_selection_lambda: 0.1
  time_horizon_sec: 60.0
  discount_rate: 0.0
```

```python
from lob.config import L3QueueValueConfig

config = L3QueueValueConfig(
    enabled=True,
    hold_threshold=0.001,
    cancel_threshold=-0.002,
    adverse_selection_lambda=0.1,
    time_horizon_sec=60.0,
    discount_rate=0.0,
)
```

### MarketImpactConfig

```yaml
market_impact:
  enabled: true
  model: almgren_chriss  # kyle, almgren_chriss, gatheral, linear
  eta: 0.05              # Temporary impact coefficient
  gamma: 0.03            # Permanent impact coefficient
  delta: 0.5             # Impact exponent
  tau_ms: 30000          # Decay time constant
  beta: 1.5              # Power-law decay exponent
  decay_type: power_law  # exponential, power_law, linear, none
  apply_to_lob: true     # Apply effects to LOB state
  momentum_detection: true
```

```python
from lob.config import MarketImpactConfig, L3ImpactModelType, DecayFunctionType

config = MarketImpactConfig(
    enabled=True,
    model=L3ImpactModelType.ALMGREN_CHRISS,
    eta=0.05,
    gamma=0.03,
    delta=0.5,
    tau_ms=30000,
    beta=1.5,
    decay_type=DecayFunctionType.POWER_LAW,
    apply_to_lob=True,
    momentum_detection=True,
)
```

### HiddenLiquidityConfig

```yaml
hidden_liquidity:
  enabled: true
  iceberg:
    enabled: true
    min_refills_to_confirm: 2
    lookback_window_sec: 60.0
    min_display_size: 10.0
    hidden_ratio_estimate: 0.15
  default_hidden_ratio: 0.15
  use_historical_estimates: true
```

```python
from lob.config import HiddenLiquidityConfig, IcebergConfig

config = HiddenLiquidityConfig(
    enabled=True,
    iceberg=IcebergConfig(
        enabled=True,
        min_refills_to_confirm=2,
        lookback_window_sec=60.0,
        min_display_size=10.0,
        hidden_ratio_estimate=0.15,
    ),
    default_hidden_ratio=0.15,
    use_historical_estimates=True,
)
```

### DarkPoolsConfig

```yaml
dark_pools:
  enabled: true
  max_dark_fill_pct: 0.5
  routing_strategy: sequential  # sequential, parallel, smart

  venues:
    - venue_id: sigma_x
      venue_type: midpoint_cross
      enabled: true
      min_order_size: 100.0
      max_order_size: 0  # 0 = unlimited
      base_fill_probability: 0.30
      size_penalty_factor: 0.5
      info_leakage_probability: 0.10
      latency_ms: 5.0

    - venue_id: iex_dark
      venue_type: midpoint_cross
      enabled: true
      min_order_size: 50.0
      base_fill_probability: 0.25

    - venue_id: liquidnet
      venue_type: negotiation
      enabled: true
      min_order_size: 10000.0
      base_fill_probability: 0.15
```

```python
from lob.config import DarkPoolsConfig, DarkPoolVenueConfig, DarkPoolVenueType

config = DarkPoolsConfig(
    enabled=True,
    max_dark_fill_pct=0.5,
    routing_strategy="sequential",
    venues=[
        DarkPoolVenueConfig(
            venue_id="sigma_x",
            venue_type=DarkPoolVenueType.MIDPOINT_CROSS,
            enabled=True,
            min_order_size=100.0,
            max_order_size=0,
            base_fill_probability=0.30,
            size_penalty_factor=0.5,
            info_leakage_probability=0.10,
            latency_ms=5.0,
        ),
        # ... more venues
    ],
)
```

### QueueTrackingConfig

```yaml
queue_tracking:
  enabled: true
  estimation_method: pessimistic  # pessimistic, probabilistic, mbo
  use_mbo_when_available: true
  max_tracked_orders: 10000
  cleanup_interval_sec: 60.0
```

### EventSchedulingConfig

```yaml
event_scheduling:
  enabled: true
  race_condition_buffer_us: 100.0
  max_events_per_step: 1000
  deterministic_ordering: true
```

## Loading Configuration

### From YAML File

```python
from lob.config import L3ExecutionConfig

# Load from file
config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")

# With environment variable substitution
# In YAML: api_key: ${API_KEY}
config = L3ExecutionConfig.from_yaml(
    "configs/execution_l3.yaml",
    env_override=True,
)
```

### From Dictionary

```python
config_dict = {
    "enabled": True,
    "latency": {
        "enabled": True,
        "profile": "institutional",
    },
    "market_impact": {
        "enabled": True,
        "model": "almgren_chriss",
    },
}

config = L3ExecutionConfig.model_validate(config_dict)
```

### Programmatic Construction

```python
config = L3ExecutionConfig(
    enabled=True,
    latency=L3LatencyConfig(
        enabled=True,
        profile=LatencyProfileType.INSTITUTIONAL,
    ),
    market_impact=MarketImpactConfig(
        enabled=True,
        model=L3ImpactModelType.ALMGREN_CHRISS,
        eta=0.05,
        gamma=0.03,
    ),
)
```

## Modifying Configuration

```python
# Load base config
config = create_l3_config("equity")

# Modify
config.latency.enabled = False
config.dark_pools.enabled = False
config.market_impact.eta = 0.08

# Or create modified copy
new_config = config.model_copy(deep=True)
new_config.latency.profile = LatencyProfileType.COLOCATED
```

## Saving Configuration

```python
# To YAML
config.to_yaml("my_config.yaml")

# To dict
config_dict = config.model_dump()

# To JSON
import json
json.dumps(config.model_dump())
```

## Validation

Configuration is validated on construction:

```python
try:
    config = L3ExecutionConfig(
        latency=L3LatencyConfig(
            profile="invalid_profile",  # Will fail
        ),
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Environment Variables

```yaml
# configs/execution_l3.yaml
latency:
  feed_latency:
    mean_us: ${FEED_LATENCY_US:200.0}  # Default 200.0
```

```python
import os
os.environ["FEED_LATENCY_US"] = "150.0"

config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")
print(config.latency.feed_latency.mean_us)  # 150.0
```

## Integration with Main Config

```yaml
# configs/config_backtest.yaml
execution:
  level: L3
  l3_config_path: configs/execution_l3.yaml

# Or inline
execution:
  level: L3
  l3_config:
    enabled: true
    latency:
      profile: institutional
```

```python
# script_backtest.py
if config.get("execution", {}).get("level") == "L3":
    l3_config_path = config["execution"].get("l3_config_path")
    if l3_config_path:
        l3_config = L3ExecutionConfig.from_yaml(l3_config_path)
    elif "l3_config" in config["execution"]:
        l3_config = L3ExecutionConfig.model_validate(
            config["execution"]["l3_config"]
        )
    else:
        l3_config = create_l3_config("equity")
```

## Related Documentation

- [Overview](overview.md) - Architecture overview
- [Migration Guide](../L3_MIGRATION_GUIDE.md) - L2 to L3 migration
- [YAML Config File](../../configs/execution_l3.yaml) - Full example
