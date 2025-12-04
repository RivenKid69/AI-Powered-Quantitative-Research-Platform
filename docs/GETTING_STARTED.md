# Getting Started Guide

Welcome! This guide will take you from zero to running your first profitable strategy in under 30 minutes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Backtest (5 minutes)](#your-first-backtest)
4. [Understanding Results](#understanding-results)
5. [Training Your First Model](#training-your-first-model)
6. [Going Live](#going-live)
7. [Next Steps](#next-steps)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or 3.12 (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Knowledge Requirements
- Basic Python knowledge (functions, classes)
- Understanding of trading concepts (long/short, profit/loss)
- No ML experience required (we'll guide you!)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd TradingBot2
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python scripts/quickstart.py check crypto_momentum
```

**Expected output:**
```
✓ Python version: 3.12.0
✓ Dependencies installed
✓ Data directory exists
✓ Configuration valid
✓ Ready to run!
```

---

## Your First Backtest

### Option A: One-Command Quick Start

```bash
python scripts/quickstart.py run crypto_momentum
```

This runs a pre-configured momentum strategy on BTC/ETH historical data.

### Option B: Step-by-Step

#### 1. Check Available Presets

```bash
python scripts/quickstart.py list
```

**Output:**
```
Available Presets:
┌─────────────────┬─────────────┬───────────────────────┬────────────┐
│ Preset          │ Asset Class │ Strategy              │ Difficulty │
├─────────────────┼─────────────┼───────────────────────┼────────────┤
│ crypto_momentum │ Crypto Spot │ Momentum (BTC, ETH)   │ ⭐⭐       │
│ equity_swing    │ US Equity   │ Mean-Reversion        │ ⭐⭐       │
│ forex_carry     │ Forex OTC   │ Carry + Momentum      │ ⭐⭐⭐     │
│ crypto_perp     │ Futures     │ Funding Arbitrage     │ ⭐⭐⭐⭐   │
│ cme_index       │ CME Futures │ Index Momentum        │ ⭐⭐⭐⭐⭐ │
└─────────────────┴─────────────┴───────────────────────┴────────────┘
```

#### 2. Get Preset Info

```bash
python scripts/quickstart.py info crypto_momentum
```

**Output:**
```
Preset: crypto_momentum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Description:
  Simple momentum strategy on major cryptocurrencies.
  Buys when price is above 20-day SMA, sells when below.

Assets: BTC/USDT, ETH/USDT
Timeframe: 4 hours
Data Period: 2023-01-01 to 2024-01-01

Expected Performance:
  - Annual Return: 30-50%
  - Sharpe Ratio: 1.5-2.0
  - Max Drawdown: 15-25%

Requirements:
  - No API keys needed (uses historical data)
  - ~500MB disk space for data
```

#### 3. Run Backtest

```bash
python scripts/quickstart.py run crypto_momentum
```

**Output:**
```
Loading data... ████████████████████ 100%
Running backtest...

=== Backtest Results ===
Period: 2023-01-01 to 2024-01-01
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics:
  Total Return:     +42.3%
  Annual Return:    +42.3%
  Sharpe Ratio:     1.85
  Sortino Ratio:    2.41
  Max Drawdown:     -12.4%
  Win Rate:         58.2%
  Profit Factor:    1.73

Trade Statistics:
  Total Trades:     156
  Avg Trade:        +0.27%
  Best Trade:       +8.4%
  Worst Trade:      -4.2%
  Avg Duration:     18.5 hours

Results saved to: results/crypto_momentum_2024-12-04/
```

---

## Understanding Results

### Key Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Total Return** | Overall profit/loss | > 0% |
| **Sharpe Ratio** | Risk-adjusted return | > 1.5 |
| **Max Drawdown** | Largest peak-to-trough drop | < 20% |
| **Win Rate** | % of profitable trades | > 50% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |

### Reading the Results Directory

```
results/crypto_momentum_2024-12-04/
├── summary.json          # Key metrics
├── trades.csv            # All trade details
├── equity_curve.png      # Portfolio value over time
├── drawdown.png          # Drawdown visualization
├── monthly_returns.png   # Monthly performance
└── config.yaml           # Configuration used
```

### Equity Curve Example

```
Portfolio Value ($)
     │
10000│━━━━┓
     │    ┃
 8000│    ┗━━━━┓
     │         ┃
 6000│         ┗━━━┓
     │             ┗━━━━━━━━━━━━━━━━━━━━━━━━━
     │
     └────────────────────────────────────────
        Jan   Mar   May   Jul   Sep   Nov
```

---

## Training Your First Model

### Why Train?

Pre-built strategies use fixed rules. Training creates an AI model that:
- Adapts to market conditions
- Learns optimal position sizing
- Discovers non-obvious patterns

### Step 1: Prepare Training Data

Data is automatically downloaded for quick start presets. For custom data:

```bash
# Crypto
python scripts/fetch_binance_filters.py --universe

# Stocks
python scripts/download_stock_data.py --symbols AAPL MSFT GOOGL --start 2020-01-01
```

### Step 2: Start Training

```bash
python scripts/quickstart.py train crypto_momentum
```

**Output:**
```
Starting training...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Episode 100/1000  │████░░░░░░░░░░░░░░░░│ 10%
  Avg Reward: 0.42
  Sharpe: 1.23
  Loss: 0.0234

Episode 200/1000  │████████░░░░░░░░░░░░│ 20%
  Avg Reward: 0.58
  Sharpe: 1.45
  Loss: 0.0189

... (training continues) ...

Training complete!
Model saved to: models/crypto_momentum_v1.pt

Validation Results:
  Sharpe Ratio: 1.92 (+0.47 vs baseline)
  Max Drawdown: -10.2% (improved from -12.4%)
```

### Step 3: Backtest Trained Model

```bash
python script_backtest.py --config configs/config_sim.yaml --model models/crypto_momentum_v1.pt
```

### Training Tips

| Tip | Explanation |
|-----|-------------|
| **Start simple** | Use default settings first |
| **Monitor overfitting** | Watch for validation divergence |
| **Save checkpoints** | Training auto-saves every 100 episodes |
| **Use GPU** | 5-10x faster with CUDA |

---

## Going Live

### Prerequisites

1. **Exchange Account**
   - Binance (crypto)
   - Alpaca (stocks)
   - OANDA (forex)
   - Interactive Brokers (futures/options)

2. **API Keys**
   - Generate API keys from your exchange
   - Store securely (never commit to git!)

### Step 1: Configure API Keys

Create `.env` file (never commit this!):

```bash
# Crypto (Binance)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Stocks (Alpaca)
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here

# Forex (OANDA)
OANDA_API_KEY=your_key_here
OANDA_ACCOUNT_ID=your_account_id
```

### Step 2: Paper Trading (Recommended!)

Always test with paper trading first:

```bash
# Crypto (Binance Testnet)
python script_live.py --config configs/config_live.yaml --paper

# Stocks (Alpaca Paper)
python script_live.py --config configs/config_live_alpaca.yaml --paper
```

### Step 3: Live Trading

**Warning**: Real money at risk. Start with small positions!

```bash
# Crypto
python script_live.py --config configs/config_live.yaml

# Stocks
python script_live.py --config configs/config_live_alpaca.yaml --live
```

### Safety Features

| Feature | Description |
|---------|-------------|
| **Kill Switch** | Emergency stop via `services/ops_kill_switch.py` |
| **Position Limits** | Max position per symbol and total |
| **Drawdown Guard** | Auto-stop if drawdown exceeds limit |
| **Error Throttling** | Pause on repeated errors |

---

## Next Steps

### Level 1: Customize Strategies

1. Modify parameters in preset configs:
   ```yaml
   # configs/quickstart/crypto_momentum.yaml
   strategy:
     lookback_period: 20  # Try 10, 30, 50
     entry_threshold: 0.02  # Try 0.01, 0.03
   ```

2. Run backtest with modified config:
   ```bash
   python script_backtest.py --config configs/quickstart/crypto_momentum.yaml
   ```

### Level 2: Create Custom Strategies

1. Create strategy file:
   ```python
   # strategies/my_strategy.py
   from strategies.base import BaseStrategy

   class MyStrategy(BaseStrategy):
       def generate_signal(self, data):
           # Your logic here
           return signal
   ```

2. Register in config and run

### Level 3: Advanced Training

1. Try different asset classes
2. Experiment with PBT (Population-Based Training)
3. Add adversarial training for robustness

### Level 4: Multi-Asset Portfolio

1. Combine multiple strategies
2. Implement portfolio-level risk management
3. Cross-asset correlation analysis

---

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError"

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### "Data not found"

```bash
# Download required data
python scripts/quickstart.py check crypto_momentum
# Follow any missing data instructions
```

#### "API connection failed"

1. Check API keys in `.env`
2. Verify exchange status
3. Check network connection
4. For Binance: ensure IP is whitelisted

#### "Out of memory"

```yaml
# Reduce batch size in config
training:
  batch_size: 32  # Reduce from 64
  n_envs: 2       # Reduce from 4
```

### Getting Help

1. **Check logs**: `logs/` directory contains detailed logs
2. **Run tests**: `pytest tests/` to verify installation
3. **Documentation**: See [DOCS_INDEX.md](DOCS_INDEX.md)

---

## Quick Reference

### Essential Commands

```bash
# List presets
python scripts/quickstart.py list

# Check environment
python scripts/quickstart.py check <preset>

# Run backtest
python scripts/quickstart.py run <preset>

# Train model
python scripts/quickstart.py train <preset>

# Live trading (paper)
python script_live.py --config <config> --paper

# Run all tests
pytest tests/
```

### Directory Structure

```
TradingBot2/
├── configs/           # Configuration files
│   ├── quickstart/    # Pre-built strategy configs
│   └── *.yaml         # Main configs
├── data/              # Market data
├── models/            # Trained models
├── results/           # Backtest results
├── scripts/           # Utility scripts
├── strategies/        # Strategy implementations
├── tests/             # Automated tests
└── logs/              # Application logs
```

### Configuration Hierarchy

```
Default Config (core_config.py)
    └── Asset Class Config (asset_class_defaults.yaml)
        └── Strategy Config (quickstart/*.yaml)
            └── Command Line Arguments
```

---

## Congratulations!

You've completed the getting started guide. You now know how to:

- [x] Run backtests with pre-built strategies
- [x] Understand performance metrics
- [x] Train your own AI models
- [x] Deploy to paper and live trading

### What's Next?

| Goal | Resource |
|------|----------|
| Deep dive into architecture | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| Understand execution models | [docs/bar_execution.md](bar_execution.md) |
| Advanced training options | [docs/UPGD_INTEGRATION.md](UPGD_INTEGRATION.md) |
| Full API reference | [CLAUDE.md](../CLAUDE.md) |

---

*Need help? Open an issue on GitHub or check [DOCS_INDEX.md](DOCS_INDEX.md) for more resources.*
