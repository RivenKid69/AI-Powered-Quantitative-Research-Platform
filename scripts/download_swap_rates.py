# -*- coding: utf-8 -*-
"""
scripts/download_swap_rates.py
Download historical swap/financing rates for forex backtesting.

Swap rates (rollover rates) are the interest paid/received when holding
positions overnight. They're critical for accurate forex backtesting as
they can significantly impact returns on carry trades.

Sources:
- OANDA API: /v3/accounts/{id}/instruments (financing field)
- Cache locally for historical backtest

Features:
- Daily swap rate history
- Long and short rates separately
- Multiple currency pairs
- Interest differential calculation
- Output compatible with backtesting pipeline

Formula for overnight swap:
    Swap = (Contract Size × Swap Rate × Days Held) / 365
    Where Swap Rate is in pips per lot per day

Usage:
    # Download swap rates for major pairs
    python scripts/download_swap_rates.py --majors --start 2020-01-01

    # Download for specific pairs
    python scripts/download_swap_rates.py --pairs EUR_USD GBP_USD --start 2020-01-01

    # Output directory
    python scripts/download_swap_rates.py --pairs EUR_USD --output data/forex/swaps/

Output:
    data/forex/swaps/EUR_USD_swaps.parquet
    data/forex/swaps/GBP_USD_swaps.parquet

Columns:
    - date: Date (YYYY-MM-DD)
    - pair: Currency pair
    - long_swap: Swap rate for long positions (pips/lot/day)
    - short_swap: Swap rate for short positions (pips/lot/day)
    - long_swap_pct: Annual % rate for long positions
    - short_swap_pct: Annual % rate for short positions
    - source: Data source ("oanda" or "interpolated")

Note: OANDA provides current financing rates but not historical.
For historical rates, this script uses interest rate differentials
from FRED to estimate historical swap rates.

Author: AI-Powered Quantitative Research Platform Team
Date: 2025-11-30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================

@dataclass
class SwapDownloadConfig:
    """Configuration for swap rate download."""

    # OANDA credentials (for current rates)
    api_key: Optional[str] = None
    account_id: Optional[str] = None
    practice: bool = True

    # Pairs to download
    pairs: List[str] = field(default_factory=list)

    # Date range
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD

    # Output
    output_dir: str = "data/forex/swaps"
    output_format: str = "parquet"

    # Interest rate data (for historical estimation)
    interest_rate_dir: str = "data/forex/rates"

    # Skip existing
    skip_existing: bool = True


# =========================
# Major Pairs
# =========================

MAJOR_PAIRS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]

# Currency to interest rate series mapping
# These are the central bank policy rates from FRED
CURRENCY_RATE_SERIES = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "GBP": "IUDSOIA",
    "JPY": "IRSTCI01JPM156N",
    "CHF": "IRSTCI01CHM156N",
    "AUD": "RBATCTR",
    "CAD": "IRSTCB01CAM156N",
    "NZD": "RBNZCTR",
}

# Typical broker markup on interest differential (bps)
BROKER_SPREAD_BPS = 25  # 0.25% typical retail markup


# =========================
# Swap Rate Estimation
# =========================

def estimate_swap_from_interest_rates(
    pair: str,
    base_rate: float,
    quote_rate: float,
    spot_price: float,
    contract_size: int = 100000,
    broker_spread_bps: int = BROKER_SPREAD_BPS,
) -> Tuple[float, float]:
    """
    Estimate swap rates from interest rate differential.

    Swap rate = (Interest Rate Differential + Broker Spread) / 365

    For a long position in base currency:
        - You pay quote currency interest
        - You receive base currency interest
        - Net = base_rate - quote_rate - broker_spread

    Args:
        pair: Currency pair (e.g., "EUR_USD")
        base_rate: Base currency interest rate (annual %)
        quote_rate: Quote currency interest rate (annual %)
        spot_price: Current spot price
        contract_size: Contract size (default: 100,000)
        broker_spread_bps: Broker markup in basis points

    Returns:
        (long_swap_pips, short_swap_pips) per day per lot
    """
    # Interest rate differential
    diff = base_rate - quote_rate

    # Convert broker spread to percentage
    broker_spread_pct = broker_spread_bps / 100

    # Daily rates (divided by 365)
    long_rate_daily = (diff - broker_spread_pct) / 365  # Long base = receive base, pay quote
    short_rate_daily = (-diff - broker_spread_pct) / 365  # Short base = pay base, receive quote

    # Convert to pips per lot
    # Pip value depends on pair type
    if "JPY" in pair:
        pip_value = 0.01  # JPY pairs
    else:
        pip_value = 0.0001  # Standard pairs

    # Swap in pips = (Daily Rate % × Spot Price × Lot Size) / Pip Value
    long_swap_pips = (long_rate_daily / 100) * spot_price * contract_size / pip_value / contract_size
    short_swap_pips = (short_rate_daily / 100) * spot_price * contract_size / pip_value / contract_size

    return long_swap_pips, short_swap_pips


def load_interest_rates(
    currency: str,
    rate_dir: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.Series]:
    """
    Load interest rate series from cached FRED data.

    Args:
        currency: Currency code (e.g., "USD")
        rate_dir: Directory with interest rate files
        start_date: Start date
        end_date: End date

    Returns:
        pandas Series with date index and rate values, or None if not found
    """
    rate_path = Path(rate_dir) / f"{currency}_rates.parquet"

    if not rate_path.exists():
        # Try CSV fallback
        rate_path = Path(rate_dir) / f"{currency}_rates.csv"

    if not rate_path.exists():
        logger.warning(f"Interest rate file not found: {rate_path}")
        return None

    try:
        if str(rate_path).endswith(".parquet"):
            df = pd.read_parquet(rate_path)
        else:
            df = pd.read_csv(rate_path)

        # Ensure date column
        date_col = "date" if "date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])

        # Get rate column
        rate_col = "rate" if "rate" in df.columns else df.columns[1]

        # Set index and filter
        df = df.set_index(date_col)
        df = df.loc[start_date:end_date]

        return df[rate_col]

    except Exception as e:
        logger.error(f"Error loading interest rates for {currency}: {e}")
        return None


def fetch_current_swaps_oanda(
    pair: str,
    api_key: str,
    account_id: str,
    practice: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Fetch current swap rates from OANDA API.

    Args:
        pair: Currency pair
        api_key: OANDA API key
        account_id: OANDA account ID
        practice: Use practice environment

    Returns:
        Dict with long_swap and short_swap, or None if failed
    """
    try:
        import requests

        base_url = (
            "https://api-fxpractice.oanda.com" if practice
            else "https://api-fxtrade.oanda.com"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}/v3/accounts/{account_id}/instruments"
        params = {"instruments": pair}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        instruments = data.get("instruments", [])

        if not instruments:
            return None

        instrument = instruments[0]
        financing = instrument.get("financing", {})

        # OANDA returns financing rates as daily rates
        long_rate = financing.get("longRate", 0)
        short_rate = financing.get("shortRate", 0)

        return {
            "long_swap_pct": float(long_rate) * 365 * 100,  # Convert to annual %
            "short_swap_pct": float(short_rate) * 365 * 100,
        }

    except Exception as e:
        logger.warning(f"Failed to fetch current swaps from OANDA: {e}")
        return None


# =========================
# Historical Swap Estimation
# =========================

def build_historical_swaps(
    pair: str,
    config: SwapDownloadConfig,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Build historical swap rate estimates for a currency pair.

    Uses interest rate differentials from FRED data to estimate
    historical swap rates.

    Args:
        pair: Currency pair (e.g., "EUR_USD")
        config: Download configuration

    Returns:
        (pair, dataframe, error_message)
    """
    try:
        # Parse pair
        parts = pair.split("_")
        if len(parts) != 2:
            return pair, None, f"Invalid pair format: {pair}"

        base_currency, quote_currency = parts

        # Date range
        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=365 * 3)

        logger.info(f"Building swap history for {pair}: {start_dt.date()} to {end_dt.date()}")

        # Load interest rates
        base_rates = load_interest_rates(base_currency, config.interest_rate_dir, start_dt, end_dt)
        quote_rates = load_interest_rates(quote_currency, config.interest_rate_dir, start_dt, end_dt)

        if base_rates is None or quote_rates is None:
            # Fall back to synthetic data with typical rates
            logger.warning(f"Interest rate data not found for {pair}, using synthetic estimates")
            return _build_synthetic_swaps(pair, start_dt, end_dt)

        # Align rates to common dates
        combined = pd.DataFrame({
            "base_rate": base_rates,
            "quote_rate": quote_rates,
        }).ffill().dropna()

        if combined.empty:
            return pair, None, "No overlapping interest rate data"

        # Estimate swap rates
        records = []
        for date, row in combined.iterrows():
            # Estimate spot price (simplified - use 1.0 for most pairs)
            spot = 1.0
            if "JPY" in pair:
                spot = 110.0  # Approximate USD/JPY

            long_swap, short_swap = estimate_swap_from_interest_rates(
                pair=pair,
                base_rate=row["base_rate"],
                quote_rate=row["quote_rate"],
                spot_price=spot,
            )

            records.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                "pair": pair,
                "long_swap": long_swap,
                "short_swap": short_swap,
                "long_swap_pct": (row["base_rate"] - row["quote_rate"]) - BROKER_SPREAD_BPS / 100,
                "short_swap_pct": (row["quote_rate"] - row["base_rate"]) - BROKER_SPREAD_BPS / 100,
                "base_rate": row["base_rate"],
                "quote_rate": row["quote_rate"],
                "source": "estimated",
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        logger.info(f"Built {len(df)} swap rate records for {pair}")
        return pair, df, None

    except Exception as e:
        logger.exception(f"Error building swaps for {pair}")
        return pair, None, str(e)


def _build_synthetic_swaps(
    pair: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Build synthetic swap rates when interest rate data is unavailable.

    Uses typical swap rate patterns based on historical averages.
    """
    # Typical swap rates (pips per lot per day) as of 2024
    # These are approximations based on typical broker rates
    TYPICAL_SWAPS = {
        "EUR_USD": (-0.5, -0.3),   # Long pays, short pays (both negative due to spreads)
        "USD_JPY": (0.8, -1.2),    # Long receives (positive carry)
        "GBP_USD": (-0.4, -0.4),
        "USD_CHF": (0.6, -1.0),
        "AUD_USD": (0.3, -0.7),
        "USD_CAD": (0.4, -0.8),
        "NZD_USD": (0.5, -0.9),
    }

    long_swap, short_swap = TYPICAL_SWAPS.get(pair, (-0.5, -0.5))

    # Generate daily records
    dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
    records = []

    for date in dates:
        # Add some variation (±20%)
        noise = np.random.uniform(0.8, 1.2)

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "pair": pair,
            "long_swap": long_swap * noise,
            "short_swap": short_swap * noise,
            "long_swap_pct": long_swap * noise * 365 * 0.0001,  # Rough conversion
            "short_swap_pct": short_swap * noise * 365 * 0.0001,
            "base_rate": np.nan,
            "quote_rate": np.nan,
            "source": "synthetic",
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    logger.warning(f"Built {len(df)} SYNTHETIC swap records for {pair}")
    return pair, df, None


# =========================
# File I/O
# =========================

def save_swaps(
    df: pd.DataFrame,
    pair: str,
    config: SwapDownloadConfig,
) -> str:
    """Save swap rate DataFrame to file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_pair = pair.replace("/", "_")

    if config.output_format == "parquet":
        filepath = output_dir / f"{safe_pair}_swaps.parquet"
        df.to_parquet(filepath, index=False)
    else:
        filepath = output_dir / f"{safe_pair}_swaps.csv"
        df.to_csv(filepath, index=False)

    return str(filepath)


# =========================
# Main Runner
# =========================

def download_all_swaps(config: SwapDownloadConfig) -> Dict[str, Any]:
    """
    Download/estimate swap rates for all configured pairs.

    Args:
        config: Download configuration

    Returns:
        Summary dict with success/failed pairs
    """
    pairs = config.pairs if config.pairs else MAJOR_PAIRS
    pairs = [p.upper().replace("/", "_") for p in pairs]

    logger.info(f"Processing {len(pairs)} pairs for swap rates")

    # Check existing files
    output_dir = Path(config.output_dir)
    if config.skip_existing:
        existing = set()
        for f in output_dir.glob("*_swaps.*"):
            existing.add(f.stem.replace("_swaps", ""))
        pairs = [p for p in pairs if p not in existing]

    if not pairs:
        logger.info("No pairs to process (all exist)")
        return {"success": [], "failed": []}

    success = []
    failed = []

    for pair in pairs:
        pair, df, error = build_historical_swaps(pair, config)

        if error:
            logger.error(f"Failed for {pair}: {error}")
            failed.append((pair, error))
        elif df is not None and not df.empty:
            filepath = save_swaps(df, pair, config)
            logger.info(f"Saved {pair} swaps to {filepath}")
            success.append(pair)
        else:
            failed.append((pair, "No data"))

    logger.info(f"Swap download complete: {len(success)} success, {len(failed)} failed")
    return {"success": success, "failed": failed}


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical swap rates for forex pairs",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Currency pairs (e.g., EUR_USD GBP_USD)",
    )
    parser.add_argument(
        "--majors",
        action="store_true",
        help="Process major pairs only",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="data/forex/swaps",
        help="Output directory",
    )
    parser.add_argument(
        "--rate-dir",
        dest="interest_rate_dir",
        default="data/forex/rates",
        help="Interest rate data directory",
    )
    parser.add_argument(
        "--api-key",
        help="OANDA API key for current rates",
    )
    parser.add_argument(
        "--account-id",
        help="OANDA account ID",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if files exist",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = SwapDownloadConfig(
        api_key=args.api_key,
        account_id=args.account_id,
        pairs=args.pairs if args.pairs else (MAJOR_PAIRS if args.majors else []),
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        interest_rate_dir=args.interest_rate_dir,
        skip_existing=not args.force,
    )

    if not config.pairs:
        config.pairs = MAJOR_PAIRS

    try:
        summary = download_all_swaps(config)
        return 0 if not summary["failed"] else 1
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
