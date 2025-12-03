# Operations Runbook

This document provides operational procedures for running the TradingBot2 platform in simulation, training, and live modes.

---

## Table of Contents

1. [Pre-Flight Checks](#pre-flight-checks)
2. [Simulation Mode](#simulation-mode)
3. [Training Mode](#training-mode)
4. [Live Trading Mode](#live-trading-mode)
5. [Monitoring](#monitoring)
6. [Emergency Procedures](#emergency-procedures)
7. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
8. [Maintenance Tasks](#maintenance-tasks)

---

## Pre-Flight Checks

**Always run the doctor script before any operation:**

```bash
python scripts/doctor.py --verbose
```

### Required Checks

| Check | Command | Expected |
|-------|---------|----------|
| Environment | `python scripts/doctor.py` | All checks pass |
| API Keys | Check env vars set | Non-empty |
| Filters | Check file dates | < 30 days old |
| Disk Space | OS command | > 10GB free |
| Network | Ping exchange | < 500ms |

### Environment Variables

**Crypto (Binance):**
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

**Stocks (Alpaca):**
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

---

## Simulation Mode

Simulation (backtest) mode tests strategies against historical data without any exchange connectivity.

### Quick Start

```bash
# Crypto backtest
python script_backtest.py --config configs/config_sim.yaml

# Stock backtest
python script_backtest.py --config configs/config_backtest_stocks.yaml
```

### Configuration

Key settings in your config file:

```yaml
mode: sim
execution:
  mode: bar
  enabled: true

# Enable for realistic simulation
use_seasonality: true
latency:
  use_seasonality: true
  base_ms: 250
```

### Output

Results are saved to:
- `logs/` - Execution logs
- `artifacts/` - Performance metrics, equity curves
- Console - Summary statistics

### Validation Checklist

- [ ] Data files exist and are not corrupted
- [ ] Date range is valid (train_start_ts < train_end_ts)
- [ ] Symbols exist in data files
- [ ] Fee configuration matches intended exchange

---

## Training Mode

Training mode uses reinforcement learning to optimize trading strategies.

### Quick Start

```bash
# Standard training (crypto)
python train_model_multi_patch.py --config configs/config_train.yaml

# Stock training
python train_model_multi_patch.py --config configs/config_train_stocks.yaml

# PBT + Adversarial training
python train_model_multi_patch.py --config configs/config_pbt_adversarial.yaml
```

### Key Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `learning_rate` | 1e-4 | Base learning rate |
| `gamma` | 0.99 | Discount factor |
| `total_timesteps` | 1M-10M | Total training steps |

### Monitoring Training

```bash
# TensorBoard (in separate terminal)
tensorboard --logdir logs/tensorboard/

# Tail training logs
tail -f logs/training.log
```

### Training Checkpoints

Checkpoints are saved to `artifacts/`:
- `best_model.zip` - Best performing model
- `checkpoint_*.zip` - Periodic checkpoints
- `final_model.zip` - Final model

### Training Validation

After training, evaluate the model:

```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

---

## Live Trading Mode

**CAUTION: Live trading involves real money. Always test thoroughly first.**

### Pre-Live Checklist

1. [ ] Run `python scripts/doctor.py --verbose`
2. [ ] Test with `--dry-run` flag first
3. [ ] Verify API key permissions (no withdrawal access!)
4. [ ] Set conservative position limits
5. [ ] Ensure kill switch is accessible
6. [ ] Have emergency contact ready

### Dry Run (No Real Trades)

```bash
# Crypto dry run
python script_live.py --config configs/config_live.yaml --dry-run

# Stock dry run
python script_live.py --config configs/config_live_alpaca.yaml --dry-run
```

### Paper Trading (Alpaca)

```bash
# Paper trading uses Alpaca sandbox
python script_live.py --config configs/config_live_alpaca.yaml
# Ensure config has: paper: true
```

### Live Trading

```bash
# Crypto live (Binance)
python script_live.py --config configs/config_live.yaml

# Stock live (Alpaca)
python script_live.py --config configs/config_live_alpaca.yaml
# Ensure config has: paper: false
```

### Risk Limits

Configure conservative limits in your config:

```yaml
risk:
  enabled: true                    # ALWAYS enable!
  max_abs_position_notional: 1000  # Max position size
  max_order_notional: 500          # Max order size
  daily_loss_limit: 100            # Daily loss limit
  max_orders_per_min: 10           # Rate limit
```

### Monitoring Live Trading

```bash
# Watch logs in real-time
tail -f logs/live-trading.log

# Check healthcheck endpoint (if enabled)
curl http://localhost:8080/health

# Monitor positions (Binance)
# Check exchange dashboard

# Monitor positions (Alpaca)
# Check https://app.alpaca.markets
```

---

## Monitoring

### Log Locations

| Log | Path | Purpose |
|-----|------|---------|
| Main | `logs/<run_id>.log` | Primary execution log |
| Trades | `logs/trades.log` | Trade history |
| Errors | `logs/errors.log` | Error-only log |
| TensorBoard | `logs/tensorboard/` | Training metrics |

### Key Metrics to Monitor

**Training:**
- Policy loss (should decrease)
- Value loss (should stabilize)
- Entropy (should decrease slowly)
- Episode reward (should increase)
- KL divergence (should stay < 0.1)

**Live Trading:**
- P&L (total and daily)
- Position sizes
- Order fill rates
- Latency (< 500ms typical)
- Error rates

### Healthcheck Endpoint

If healthcheck is enabled:

```bash
# Health status
curl http://localhost:8080/health

# Detailed metrics
curl http://localhost:8080/metrics
```

Response codes:
- `200` - Healthy
- `503` - Unhealthy (check logs)

---

## Emergency Procedures

### Kill Switch

**Method 1: Flag File (Fastest)**
```bash
touch state/kill_switch.flag
```

**Method 2: Ctrl+C**
- Press Ctrl+C once for graceful shutdown
- Wait for "Shutdown complete" message
- Press Ctrl+C again only if stuck

**Method 3: Manual Position Close**
- Log into exchange dashboard
- Close all positions manually
- Then kill the process

**Method 4: API Key Revocation (Last Resort)**
- Revoke API keys on exchange
- This stops ALL API access immediately

### Recovery After Emergency Stop

1. Remove kill switch flag:
   ```bash
   rm state/kill_switch.flag
   ```

2. Check state files:
   ```bash
   ls -la state/
   ```

3. Verify no orphan positions:
   - Check exchange dashboard
   - Reconcile with local state

4. Review logs for root cause:
   ```bash
   grep -i error logs/*.log | tail -100
   ```

5. Run doctor before resuming:
   ```bash
   python scripts/doctor.py --verbose
   ```

---

## Common Issues & Troubleshooting

### Connection Issues

**Symptom:** "Connection refused" or timeout errors

**Solutions:**
1. Check network connectivity
2. Verify API endpoint URLs
3. Check if exchange is under maintenance
4. Try increasing timeout values

```yaml
latency:
  timeout_ms: 5000  # Increase from default
  retries: 3
```

### Clock Drift

**Symptom:** "Timestamp outside recv window" errors

**Solutions:**
1. Sync system clock:
   ```bash
   # Windows
   w32tm /resync

   # Linux
   sudo ntpdate pool.ntp.org
   ```

2. Configure clock sync in config:
   ```yaml
   clock_sync:
     refresh_sec: 60      # More frequent sync
     warn_threshold_ms: 200
     kill_threshold_ms: 1000
   ```

### Rate Limiting

**Symptom:** "Too many requests" or 429 errors

**Solutions:**
1. Reduce signal frequency:
   ```yaml
   max_signals_per_sec: 2.0  # Lower value
   ```

2. Increase backoff:
   ```yaml
   backoff_base_s: 5.0
   max_backoff_s: 120.0
   ```

### Out of Memory

**Symptom:** Process killed, "MemoryError"

**Solutions:**
1. Reduce batch sizes
2. Reduce number of symbols
3. Use shorter history windows
4. Increase system swap

### Model Loading Errors

**Symptom:** "Model not found" or checkpoint errors

**Solutions:**
1. Verify checkpoint path exists
2. Check model was saved correctly
3. Ensure compatible Python/library versions

```bash
# Check checkpoint
python -c "import zipfile; zipfile.ZipFile('artifacts/best_model.zip').namelist()"
```

### Stale Filter Errors

**Symptom:** "Filters are stale" or quantizer errors

**Solutions:**
1. Update filters:
   ```bash
   python scripts/fetch_binance_filters.py --out data/binance_filters.json
   ```

2. Update fees:
   ```bash
   python scripts/refresh_fees.py
   ```

---

## Maintenance Tasks

### Daily

- [ ] Check logs for errors
- [ ] Verify positions match expected
- [ ] Monitor P&L
- [ ] Check disk space

### Weekly

- [ ] Update exchange filters
- [ ] Review and archive old logs
- [ ] Check for library updates
- [ ] Run full test suite

```bash
# Update filters
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json

# Run tests
pytest tests/ -v
```

### Monthly

- [ ] Rotate API keys
- [ ] Review and update risk limits
- [ ] Evaluate model performance
- [ ] Consider retraining if performance degrades

### Data Refresh Commands

```bash
# Binance filters
python scripts/fetch_binance_filters.py --universe --out data/binance_filters.json

# Fee schedules
python scripts/refresh_fees.py

# Universe updates
python -m services.universe --output data/universe/symbols.json

# Alpaca universe
python scripts/fetch_alpaca_universe.py --output data/universe/alpaca_symbols.json
```

---

## Quick Reference Commands

### Simulation
```bash
python script_backtest.py --config configs/config_sim.yaml
```

### Training
```bash
python train_model_multi_patch.py --config configs/config_train.yaml
```

### Evaluation
```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

### Live (Dry Run)
```bash
python script_live.py --config configs/config_live.yaml --dry-run
```

### Live (Real)
```bash
python script_live.py --config configs/config_live.yaml
```

### Doctor Check
```bash
python scripts/doctor.py --verbose
```

### Emergency Stop
```bash
touch state/kill_switch.flag
```

---

## Support

- **Documentation:** See `docs/` directory
- **Issues:** Check `CLAUDE.md` troubleshooting section
- **Tests:** Run `pytest tests/` to verify system health

---

*Last updated: 2025-12-03*
