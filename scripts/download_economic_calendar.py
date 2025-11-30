# -*- coding: utf-8 -*-
"""
scripts/download_economic_calendar.py
Download economic calendar events for forex trading.

Economic events (NFP, CPI, central bank decisions) create volatility spikes
and are critical for:
1. No-trade windows around high-impact events
2. Volatility regime features for ML models
3. Event-driven position sizing

Sources:
- Primary: OANDA Labs Calendar API (free with OANDA account)
- Backup: Investing.com / ForexFactory (requires scraping)

Features:
- Historical and future events
- Impact classification (High/Medium/Low)
- Multiple currencies
- Event categories (NFP, CPI, GDP, Interest Rate, etc.)
- Actual vs forecast vs previous values

Usage:
    # Download calendar for major currencies
    python scripts/download_economic_calendar.py --currencies USD EUR GBP JPY --start 2020-01-01

    # Download high-impact events only
    python scripts/download_economic_calendar.py --currencies USD --high-impact-only

    # Output directory
    python scripts/download_economic_calendar.py --output data/forex/calendar/

Output:
    data/forex/calendar/economic_calendar.parquet

Columns:
    - datetime: Event datetime (UTC)
    - date: Date (YYYY-MM-DD)
    - time: Time (HH:MM)
    - currency: Currency code (USD, EUR, etc.)
    - event: Event name
    - impact: Impact level (high, medium, low)
    - actual: Actual value (if released)
    - forecast: Market forecast
    - previous: Previous period value
    - source: Data source

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
from enum import Enum
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

class EventImpact(Enum):
    """Economic event impact level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CalendarDownloadConfig:
    """Configuration for economic calendar download."""

    # Currencies to include
    currencies: List[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])

    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: int = 365 * 3

    # Filtering
    high_impact_only: bool = False
    include_forecasts: bool = True

    # Output
    output_dir: str = "data/forex/calendar"
    output_format: str = "parquet"

    # Skip existing
    skip_existing: bool = True


# =========================
# High-Impact Economic Events
# =========================

# High-impact events that typically move forex markets significantly
HIGH_IMPACT_EVENTS = {
    "USD": [
        "Non-Farm Payrolls",
        "FOMC Rate Decision",
        "CPI",
        "Core CPI",
        "Retail Sales",
        "GDP",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "ADP Employment",
        "Initial Jobless Claims",
        "Durable Goods Orders",
        "Trade Balance",
        "Consumer Confidence",
        "PCE Price Index",
        "Core PCE Price Index",
        "Fed Chair Powell Speech",
    ],
    "EUR": [
        "ECB Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "PMI Manufacturing",
        "PMI Services",
        "ZEW Economic Sentiment",
        "German CPI",
        "German GDP",
        "Trade Balance",
        "ECB President Lagarde Speech",
    ],
    "GBP": [
        "BOE Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "Retail Sales",
        "PMI Manufacturing",
        "PMI Services",
        "Employment Change",
        "Trade Balance",
        "BOE Governor Bailey Speech",
    ],
    "JPY": [
        "BOJ Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "Trade Balance",
        "Tankan Survey",
        "Industrial Production",
        "Retail Sales",
        "BOJ Governor Speech",
    ],
    "CHF": [
        "SNB Rate Decision",
        "CPI",
        "GDP",
        "Trade Balance",
    ],
    "AUD": [
        "RBA Rate Decision",
        "CPI",
        "GDP",
        "Employment Change",
        "Trade Balance",
        "Retail Sales",
    ],
    "CAD": [
        "BOC Rate Decision",
        "CPI",
        "Core CPI",
        "GDP",
        "Employment Change",
        "Retail Sales",
        "Trade Balance",
    ],
    "NZD": [
        "RBNZ Rate Decision",
        "CPI",
        "GDP",
        "Employment Change",
        "Trade Balance",
    ],
}

# Event schedule patterns (approximate release times in UTC)
# These are used to generate synthetic historical data
EVENT_SCHEDULES = {
    "Non-Farm Payrolls": {
        "frequency": "monthly",
        "day": "first_friday",
        "time_utc": "13:30",
        "currency": "USD",
    },
    "FOMC Rate Decision": {
        "frequency": "6_weeks",
        "time_utc": "19:00",
        "currency": "USD",
    },
    "ECB Rate Decision": {
        "frequency": "6_weeks",
        "time_utc": "12:45",
        "currency": "EUR",
    },
    "BOE Rate Decision": {
        "frequency": "6_weeks",
        "time_utc": "12:00",
        "currency": "GBP",
    },
    "CPI": {
        "frequency": "monthly",
        "currency": "multiple",
    },
}


# =========================
# OANDA Calendar API
# =========================

def fetch_calendar_oanda(
    config: CalendarDownloadConfig,
    api_key: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch economic calendar from OANDA Labs API.

    Note: OANDA's calendar API may have limited historical data.
    This function attempts to fetch what's available.

    Args:
        config: Download configuration
        api_key: OANDA API key
        account_id: OANDA account ID

    Returns:
        DataFrame with calendar events or None if failed
    """
    try:
        import requests

        api_key = api_key or os.environ.get("OANDA_API_KEY")
        if not api_key:
            logger.warning("OANDA API key not provided, using synthetic data")
            return None

        # OANDA Labs calendar endpoint
        # Note: This is the public calendar, not requiring authentication
        url = "https://www.oanda.com/labsapi/calendar"

        # Date range
        end_dt = datetime.now(timezone.utc)
        if config.end_date:
            end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if config.start_date:
            start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = end_dt - timedelta(days=config.lookback_days)

        params = {
            "period": f"{start_dt.strftime('%Y%m%d')}/{end_dt.strftime('%Y%m%d')}",
        }

        response = requests.get(url, params=params, timeout=60)

        if response.status_code != 200:
            logger.warning(f"OANDA calendar API returned {response.status_code}")
            return None

        data = response.json()

        if not data:
            return None

        # Parse events
        records = []
        for event in data:
            currency = event.get("currency", "")
            if config.currencies and currency not in config.currencies:
                continue

            impact = _classify_impact(event.get("title", ""), currency)
            if config.high_impact_only and impact != "high":
                continue

            records.append({
                "datetime": event.get("timestamp"),
                "date": event.get("timestamp", "")[:10],
                "time": event.get("timestamp", "")[11:16],
                "currency": currency,
                "event": event.get("title", ""),
                "impact": impact,
                "actual": event.get("actual"),
                "forecast": event.get("forecast"),
                "previous": event.get("previous"),
                "source": "oanda",
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch OANDA calendar: {e}")
        return None


def _classify_impact(event_name: str, currency: str) -> str:
    """Classify event impact level based on event name."""
    event_lower = event_name.lower()

    # Check if it's a known high-impact event
    high_impact_keywords = [
        "rate decision", "non-farm", "payroll", "cpi", "inflation",
        "gdp", "employment", "retail sales", "pmi",
        "fomc", "ecb", "boe", "boj", "rba", "rbnz", "snb", "boc",
    ]

    for keyword in high_impact_keywords:
        if keyword in event_lower:
            return "high"

    medium_impact_keywords = [
        "trade balance", "industrial", "housing", "consumer",
        "sentiment", "confidence", "speech", "testimony",
    ]

    for keyword in medium_impact_keywords:
        if keyword in event_lower:
            return "medium"

    return "low"


# =========================
# Synthetic Calendar Generation
# =========================

def generate_synthetic_calendar(
    config: CalendarDownloadConfig,
) -> pd.DataFrame:
    """
    Generate synthetic economic calendar based on known schedules.

    This is used when API data is unavailable or for historical backfill.
    Uses typical event patterns and schedules.

    Args:
        config: Download configuration

    Returns:
        DataFrame with synthetic calendar events
    """
    logger.info("Generating synthetic economic calendar")

    end_dt = datetime.now(timezone.utc)
    if config.end_date:
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if config.start_date:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=config.lookback_days)

    records = []

    for currency in config.currencies:
        high_impact_events = HIGH_IMPACT_EVENTS.get(currency, [])

        for event_name in high_impact_events:
            # Generate monthly events
            current_date = start_dt
            while current_date <= end_dt:
                # Determine event time (approximate)
                if "Rate Decision" in event_name:
                    # Central bank decisions typically on specific weeks
                    event_hour = 12 if currency in ["EUR", "GBP"] else 19
                    event_date = _get_meeting_date(current_date, currency)
                elif "Non-Farm" in event_name:
                    # First Friday of month, 8:30 AM ET = 13:30 UTC
                    event_date = _get_first_friday(current_date)
                    event_hour = 13
                else:
                    # Other events typically mid-month
                    event_date = current_date.replace(day=min(15, 28))
                    event_hour = 13

                if event_date and start_dt <= event_date <= end_dt:
                    event_dt = event_date.replace(hour=event_hour, minute=30)

                    records.append({
                        "datetime": event_dt.isoformat(),
                        "date": event_dt.strftime("%Y-%m-%d"),
                        "time": event_dt.strftime("%H:%M"),
                        "currency": currency,
                        "event": event_name,
                        "impact": "high" if event_name in high_impact_events[:5] else "medium",
                        "actual": None,  # No actual values for synthetic
                        "forecast": None,
                        "previous": None,
                        "source": "synthetic",
                    })

                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

    df = pd.DataFrame(records)

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Generated {len(df)} synthetic calendar events")
    return df


def _get_first_friday(dt: datetime) -> datetime:
    """Get first Friday of the month."""
    first_day = dt.replace(day=1)
    # Days until Friday (4 = Friday)
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_friday)


def _get_meeting_date(dt: datetime, currency: str) -> Optional[datetime]:
    """
    Get approximate central bank meeting date for the month.

    Central banks typically meet every 6-8 weeks on predetermined schedules.
    This is a simplified approximation.
    """
    # Typical meeting patterns (day of month)
    meeting_patterns = {
        "USD": [15, 16],  # FOMC mid-month Wed/Thu
        "EUR": [10, 12],  # ECB typically early month
        "GBP": [5, 8],    # BOE typically first week
        "JPY": [15, 20],  # BOJ mid-month
        "AUD": [3, 5],    # RBA first Tuesday
        "CAD": [8, 10],   # BOC
        "NZD": [10, 12],  # RBNZ
        "CHF": [15, 18],  # SNB
    }

    days = meeting_patterns.get(currency, [15])

    # Central banks meet ~8 times per year, so skip some months
    # Simplified: assume meetings in specific months
    meeting_months = [1, 3, 5, 6, 7, 9, 11, 12]  # Approximate

    if dt.month in meeting_months:
        return dt.replace(day=days[0])
    return None


# =========================
# File I/O
# =========================

def save_calendar(
    df: pd.DataFrame,
    config: CalendarDownloadConfig,
) -> str:
    """Save calendar DataFrame to file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "parquet":
        filepath = output_dir / "economic_calendar.parquet"
        df.to_parquet(filepath, index=False)
    else:
        filepath = output_dir / "economic_calendar.csv"
        df.to_csv(filepath, index=False)

    return str(filepath)


def load_existing_calendar(config: CalendarDownloadConfig) -> Optional[pd.DataFrame]:
    """Load existing calendar file if present."""
    output_dir = Path(config.output_dir)

    for ext in ["parquet", "csv"]:
        filepath = output_dir / f"economic_calendar.{ext}"
        if filepath.exists():
            if ext == "parquet":
                return pd.read_parquet(filepath)
            else:
                return pd.read_csv(filepath)

    return None


# =========================
# Main Runner
# =========================

def download_calendar(config: CalendarDownloadConfig) -> Dict[str, Any]:
    """
    Download or generate economic calendar.

    Args:
        config: Download configuration

    Returns:
        Summary dict
    """
    # Check existing
    if config.skip_existing:
        existing = load_existing_calendar(config)
        if existing is not None and len(existing) > 0:
            logger.info(f"Using existing calendar with {len(existing)} events")
            return {"events": len(existing), "source": "existing"}

    # Try OANDA first
    df = fetch_calendar_oanda(config)

    if df is None or df.empty:
        # Fall back to synthetic
        logger.info("Using synthetic calendar generation")
        df = generate_synthetic_calendar(config)

    if df.empty:
        logger.warning("No calendar events generated")
        return {"events": 0, "source": "none"}

    # Save
    filepath = save_calendar(df, config)
    logger.info(f"Saved {len(df)} events to {filepath}")

    return {
        "events": len(df),
        "source": df["source"].iloc[0] if "source" in df.columns else "unknown",
        "filepath": filepath,
    }


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download economic calendar for forex trading",
    )

    parser.add_argument(
        "--currencies",
        nargs="+",
        default=["USD", "EUR", "GBP", "JPY"],
        help="Currency codes (default: USD EUR GBP JPY)",
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
        default="data/forex/calendar",
        help="Output directory",
    )
    parser.add_argument(
        "--high-impact-only",
        action="store_true",
        help="Only include high-impact events",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if file exists",
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

    config = CalendarDownloadConfig(
        currencies=[c.upper() for c in args.currencies],
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        high_impact_only=args.high_impact_only,
        skip_existing=not args.force,
    )

    try:
        summary = download_calendar(config)
        logger.info(f"Calendar download complete: {summary}")
        return 0
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
