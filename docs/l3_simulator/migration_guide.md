# L3 LOB Simulator - Migration Guide

This document provides a reference to the main migration guide for upgrading from L2 (Statistical) to L3 (LOB Simulation) execution providers.

## Main Migration Guide

**For the complete migration guide, see**: [L3_MIGRATION_GUIDE.md](../L3_MIGRATION_GUIDE.md)

## Quick Migration Summary

### 1. Update Factory Calls

```python
# Before (L2)
provider = create_execution_provider(AssetClass.EQUITY)

# After (L3)
provider = create_execution_provider(AssetClass.EQUITY, level="L3")
```

### 2. Add L3 Configuration

```yaml
# configs/config_backtest.yaml
execution:
  level: L3
  l3_config_path: configs/execution_l3.yaml
```

### 3. Configure L3 Features

```yaml
# configs/execution_l3.yaml
enabled: true
latency:
  enabled: true
  profile: institutional
market_impact:
  enabled: true
  model: almgren_chriss
```

## Key Differences Summary

| Aspect | L2 | L3 |
|--------|----|----|
| Slippage | √participation | Full LOB + Impact |
| Fills | OHLCV bar-based | Matching engine |
| Queue Position | Not tracked | FIFO simulation |
| Market Impact | Simple Almgren-Chriss | Kyle/AC/Gatheral |
| Latency | Not simulated | Realistic profiles |
| Dark Pools | Not supported | Multi-venue |
| Performance | >50,000/sec | >1,000/sec (full) |

## Backward Compatibility

- **Crypto unchanged**: Default L2
- **L2 available**: Explicit `level="L2"`
- **API preserved**: Same function signatures

## Related Documentation

- [Overview](overview.md) - L3 architecture
- [Configuration](configuration.md) - Config reference
- [Deployment](deployment.md) - Deployment guide
- [Full Migration Guide](../L3_MIGRATION_GUIDE.md) - Complete details
