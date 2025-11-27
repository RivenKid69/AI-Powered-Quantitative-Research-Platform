# -*- coding: utf-8 -*-
"""
services/corporate_actions.py
Service layer for corporate actions management.

This service provides:
- Caching of corporate actions data with disk persistence
- Batch operations for multiple symbols
- Price adjustment computation for backtesting
- Earnings calendar integration
- Gap analysis features for pre-market trading

Architecture:
    Adapter (Yahoo/Polygon) → Service (caching, batch) → Data Loader → Features

Usage:
    from services.corporate_actions import CorporateActionsService

    service = CorporateActionsService()

    # Get adjusted prices
    adj_df = service.adjust_prices(df, symbol="AAPL")

    # Get upcoming dividends
    upcoming = service.get_upcoming_dividends(["AAPL", "MSFT"], days_ahead=30)

    # Compute earnings features
    features = service.compute_earnings_features("AAPL", "2024-01-15")

Best Practices:
    - Always use split-adjusted prices for training
    - Consider total return (dividend-adjusted) for long-term backtests
    - Be aware of survivorship bias (delisted stocks)
    - Use ex-date for timing (price adjusts on ex-date)

References:
    - CRSP adjustment methodology
    - Bloomberg corporate actions handling
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default cache directory
DEFAULT_CACHE_DIR = Path("data/cache/corporate_actions")


@dataclass
class CorporateActionsConfig:
    """Configuration for CorporateActionsService."""
    cache_dir: Path = DEFAULT_CACHE_DIR
    cache_ttl_hours: int = 24  # Cache validity in hours
    vendor: str = "yahoo"  # Default data vendor
    adjust_dividends: bool = False  # Whether to compute dividend-adjusted prices
    adjust_splits: bool = True  # Whether to apply split adjustments


class CorporateActionsService:
    """
    Service for managing corporate actions data.

    Provides caching, batch operations, and price adjustment
    for dividends and splits.

    Attributes:
        config: Service configuration
        _adapter: Underlying data adapter
        _cache: In-memory cache
    """

    def __init__(
        self,
        config: Optional[CorporateActionsConfig] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize corporate actions service.

        Args:
            config: Service configuration
            adapter_config: Configuration passed to adapter
        """
        self.config = config or CorporateActionsConfig()
        self._adapter_config = adapter_config or {}

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded adapter
        self._adapter = None

        # In-memory cache
        self._dividends_cache: Dict[str, List[Dict]] = {}
        self._splits_cache: Dict[str, List[Dict]] = {}
        self._earnings_cache: Dict[str, List[Dict]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    @property
    def adapter(self):
        """Get or create the corporate actions adapter."""
        if self._adapter is None:
            from adapters.registry import create_corporate_actions_adapter
            self._adapter = create_corporate_actions_adapter(
                self.config.vendor,
                self._adapter_config,
            )
        return self._adapter

    @property
    def earnings_adapter(self):
        """Get or create the earnings adapter."""
        if not hasattr(self, '_earnings_adapter') or self._earnings_adapter is None:
            from adapters.registry import create_earnings_adapter
            self._earnings_adapter = create_earnings_adapter(
                self.config.vendor,
                self._adapter_config,
            )
        return self._earnings_adapter

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _get_cache_path(self, symbol: str, data_type: str) -> Path:
        """Get cache file path for a symbol and data type."""
        return self.config.cache_dir / f"{symbol.upper()}_{data_type}.json"

    def _is_cache_valid(self, symbol: str, data_type: str) -> bool:
        """Check if cache is still valid."""
        cache_key = f"{symbol}_{data_type}"

        # Check in-memory timestamp
        if cache_key in self._cache_timestamps:
            age = datetime.now() - self._cache_timestamps[cache_key]
            if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                return True

        # Check file cache
        cache_path = self._get_cache_path(symbol, data_type)
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age = datetime.now() - mtime
            if age.total_seconds() < self.config.cache_ttl_hours * 3600:
                return True

        return False

    def _load_from_cache(
        self,
        symbol: str,
        data_type: str,
    ) -> Optional[List[Dict]]:
        """Load data from cache."""
        cache_key = f"{symbol}_{data_type}"

        # Try in-memory cache first
        if data_type == "dividends" and symbol in self._dividends_cache:
            return self._dividends_cache[symbol]
        elif data_type == "splits" and symbol in self._splits_cache:
            return self._splits_cache[symbol]
        elif data_type == "earnings" and symbol in self._earnings_cache:
            return self._earnings_cache[symbol]

        # Try file cache
        cache_path = self._get_cache_path(symbol, data_type)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)

                # Populate in-memory cache
                if data_type == "dividends":
                    self._dividends_cache[symbol] = data
                elif data_type == "splits":
                    self._splits_cache[symbol] = data
                elif data_type == "earnings":
                    self._earnings_cache[symbol] = data

                self._cache_timestamps[cache_key] = datetime.now()
                return data

            except Exception as e:
                logger.warning(f"Error loading cache for {symbol}/{data_type}: {e}")

        return None

    def _save_to_cache(
        self,
        symbol: str,
        data_type: str,
        data: List[Dict],
    ) -> None:
        """Save data to cache."""
        cache_key = f"{symbol}_{data_type}"

        # Update in-memory cache
        if data_type == "dividends":
            self._dividends_cache[symbol] = data
        elif data_type == "splits":
            self._splits_cache[symbol] = data
        elif data_type == "earnings":
            self._earnings_cache[symbol] = data

        self._cache_timestamps[cache_key] = datetime.now()

        # Save to file
        cache_path = self._get_cache_path(symbol, data_type)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error saving cache for {symbol}/{data_type}: {e}")

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            # Clear specific symbol
            for data_type in ["dividends", "splits", "earnings"]:
                cache_path = self._get_cache_path(symbol, data_type)
                if cache_path.exists():
                    cache_path.unlink()

            self._dividends_cache.pop(symbol, None)
            self._splits_cache.pop(symbol, None)
            self._earnings_cache.pop(symbol, None)
        else:
            # Clear all
            for f in self.config.cache_dir.glob("*.json"):
                f.unlink()

            self._dividends_cache.clear()
            self._splits_cache.clear()
            self._earnings_cache.clear()
            self._cache_timestamps.clear()

    # =========================================================================
    # Data Fetching
    # =========================================================================

    def get_dividends(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Get dividend history for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            force_refresh: Bypass cache

        Returns:
            List of dividend records as dicts
        """
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and self._is_cache_valid(symbol, "dividends"):
            cached = self._load_from_cache(symbol, "dividends")
            if cached is not None:
                # Apply date filters
                result = cached
                if start_date:
                    result = [d for d in result if d.get("ex_date", "") >= start_date]
                if end_date:
                    result = [d for d in result if d.get("ex_date", "") <= end_date]
                return result

        # Fetch from adapter
        try:
            from adapters.models import Dividend
            dividends = self.adapter.get_dividends(
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
            data = [d.to_dict() for d in dividends]

            # Cache full result (without date filters)
            self._save_to_cache(symbol, "dividends", data)

            return data

        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return []

    def get_splits(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Get split history for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            force_refresh: Bypass cache

        Returns:
            List of split records as dicts
        """
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and self._is_cache_valid(symbol, "splits"):
            cached = self._load_from_cache(symbol, "splits")
            if cached is not None:
                result = cached
                if start_date:
                    result = [s for s in result if s.get("ex_date", "") >= start_date]
                if end_date:
                    result = [s for s in result if s.get("ex_date", "") <= end_date]
                return result

        # Fetch from adapter
        try:
            splits = self.adapter.get_splits(
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
            data = [s.to_dict() for s in splits]
            self._save_to_cache(symbol, "splits", data)
            return data

        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            return []

    def get_earnings(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Get earnings history for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            force_refresh: Bypass cache

        Returns:
            List of earnings records as dicts
        """
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and self._is_cache_valid(symbol, "earnings"):
            cached = self._load_from_cache(symbol, "earnings")
            if cached is not None:
                result = cached
                if start_date:
                    result = [e for e in result if e.get("report_date", "") >= start_date]
                if end_date:
                    result = [e for e in result if e.get("report_date", "") <= end_date]
                return result

        # Fetch from adapter
        try:
            earnings = self.earnings_adapter.get_earnings_history(
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
            data = [e.to_dict() for e in earnings]
            self._save_to_cache(symbol, "earnings", data)
            return data

        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return []

    def get_upcoming_dividends(
        self,
        symbols: Sequence[str],
        days_ahead: int = 30,
    ) -> Dict[str, List[Dict]]:
        """
        Get upcoming ex-dividend dates for multiple symbols.

        Args:
            symbols: List of symbols
            days_ahead: Days to look ahead

        Returns:
            Dict mapping symbol to list of upcoming dividends
        """
        today = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        result: Dict[str, List[Dict]] = {}

        for symbol in symbols:
            divs = self.get_dividends(symbol, start_date=today, end_date=end)
            if divs:
                result[symbol.upper()] = divs

        return result

    def get_upcoming_earnings(
        self,
        symbols: Sequence[str],
        days_ahead: int = 30,
    ) -> Dict[str, List[Dict]]:
        """
        Get upcoming earnings dates for multiple symbols.

        Args:
            symbols: List of symbols
            days_ahead: Days to look ahead

        Returns:
            Dict mapping symbol to list of upcoming earnings
        """
        try:
            upcoming = self.earnings_adapter.get_upcoming_earnings(
                symbols,
                days_ahead=days_ahead,
            )
            return {k: [e.to_dict() for e in v] for k, v in upcoming.items()}

        except Exception as e:
            logger.error(f"Error fetching upcoming earnings: {e}")
            return {}

    # =========================================================================
    # Price Adjustment
    # =========================================================================

    def compute_split_factors(
        self,
        symbol: str,
        dates: Sequence[str],
    ) -> Dict[str, float]:
        """
        Compute cumulative split adjustment factors for dates.

        Args:
            symbol: Stock symbol
            dates: List of dates (ISO format)

        Returns:
            Dict mapping date to cumulative split factor
        """
        splits = self.get_splits(symbol)

        if not splits:
            return {d: 1.0 for d in dates}

        # Sort splits by date
        splits_sorted = sorted(splits, key=lambda s: s["ex_date"])

        # Build cumulative factor from earliest date forward
        split_factor_at_date: Dict[str, float] = {}
        cumulative = 1.0

        all_dates = sorted(set(dates) | {s["ex_date"] for s in splits_sorted})

        split_idx = 0
        for date in all_dates:
            # Apply any splits on this date
            while split_idx < len(splits_sorted) and splits_sorted[split_idx]["ex_date"] <= date:
                factor = splits_sorted[split_idx].get("adjustment_factor", 1.0)
                cumulative *= factor
                split_idx += 1

            split_factor_at_date[date] = cumulative

        # Return only requested dates
        return {d: split_factor_at_date.get(d, 1.0) for d in dates}

    def adjust_prices(
        self,
        df: pd.DataFrame,
        symbol: str,
        adjust_splits: Optional[bool] = None,
        adjust_dividends: Optional[bool] = None,
        price_cols: Sequence[str] = ("open", "high", "low", "close"),
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Apply split and dividend adjustments to price data.

        Creates adjusted price columns while preserving originals.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            adjust_splits: Apply split adjustment (default from config)
            adjust_dividends: Apply dividend adjustment (default from config)
            price_cols: Columns to adjust
            volume_col: Volume column name

        Returns:
            DataFrame with adjusted prices (adds _adjusted suffix columns)
        """
        df = df.copy()

        if adjust_splits is None:
            adjust_splits = self.config.adjust_splits
        if adjust_dividends is None:
            adjust_dividends = self.config.adjust_dividends

        # Get date column
        if "timestamp" in df.columns:
            # Convert timestamp to date string
            df["_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["_date"] = df["date"].astype(str)
        else:
            logger.warning("No timestamp or date column found, cannot adjust prices")
            return df

        dates = df["_date"].unique().tolist()

        # Compute split factors
        if adjust_splits:
            split_factors = self.compute_split_factors(symbol, dates)

            # Apply split adjustment (multiply prices, divide volume)
            df["_split_factor"] = df["_date"].map(split_factors).fillna(1.0)

            for col in price_cols:
                if col in df.columns:
                    df[f"{col}_adjusted"] = df[col] * df["_split_factor"]

            if volume_col in df.columns:
                # Volume is inverse-adjusted
                df[f"{volume_col}_adjusted"] = df[volume_col] / df["_split_factor"]

            df.drop(columns=["_split_factor"], inplace=True)

        else:
            # No adjustment, just copy columns
            for col in price_cols:
                if col in df.columns:
                    df[f"{col}_adjusted"] = df[col]
            if volume_col in df.columns:
                df[f"{volume_col}_adjusted"] = df[volume_col]

        # Clean up
        df.drop(columns=["_date"], inplace=True)

        return df

    def get_total_return_factor(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> float:
        """
        Compute total return factor including dividends.

        Total return = price return + dividend return

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Total return factor (1.0 = no return)
        """
        dividends = self.get_dividends(symbol, start_date=start_date, end_date=end_date)

        total_dividend = sum(
            float(d.get("amount", 0))
            for d in dividends
        )

        # This is a simplified calculation - proper total return
        # needs price data to compute daily reinvestment
        return 1.0 + total_dividend / 100.0  # Approximate

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def compute_earnings_features(
        self,
        symbol: str,
        as_of_date: str,
    ) -> Dict[str, Any]:
        """
        Compute earnings-related features for a given date.

        Features:
        - days_to_next_earnings: Days until next earnings
        - days_since_last_earnings: Days since last earnings
        - last_surprise_pct: EPS surprise from last quarter
        - beat_streak: Consecutive quarters beating estimates

        Args:
            symbol: Stock symbol
            as_of_date: Reference date

        Returns:
            Dict of feature values
        """
        try:
            return self.earnings_adapter.compute_earnings_features(symbol, as_of_date)
        except Exception as e:
            logger.warning(f"Error computing earnings features: {e}")
            return {
                "days_to_next_earnings": None,
                "days_since_last_earnings": None,
                "last_surprise_pct": None,
                "beat_streak": 0,
            }

    def compute_dividend_features(
        self,
        symbol: str,
        as_of_date: str,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute dividend-related features.

        Features:
        - days_to_ex_div: Days until next ex-dividend date
        - trailing_dividend_yield: 12-month trailing yield (requires price)
        - dividend_streak: Consecutive quarters with dividend

        Args:
            symbol: Stock symbol
            as_of_date: Reference date
            price: Current price (for yield calculation)

        Returns:
            Dict of feature values
        """
        features: Dict[str, Any] = {
            "days_to_ex_div": None,
            "trailing_dividend_yield": None,
            "dividend_streak": 0,
        }

        try:
            # Get recent dividends
            start = (
                datetime.fromisoformat(as_of_date) - timedelta(days=400)
            ).strftime("%Y-%m-%d")
            end = (
                datetime.fromisoformat(as_of_date) + timedelta(days=60)
            ).strftime("%Y-%m-%d")

            dividends = self.get_dividends(symbol, start_date=start, end_date=end)

            if not dividends:
                return features

            as_of_dt = datetime.fromisoformat(as_of_date)

            # Split into past and future
            past_divs = [
                d for d in dividends
                if datetime.fromisoformat(d["ex_date"]) <= as_of_dt
            ]
            future_divs = [
                d for d in dividends
                if datetime.fromisoformat(d["ex_date"]) > as_of_dt
            ]

            # Days to next ex-dividend
            if future_divs:
                next_ex = min(
                    d["ex_date"] for d in future_divs
                )
                next_dt = datetime.fromisoformat(next_ex)
                features["days_to_ex_div"] = (next_dt - as_of_dt).days

            # Trailing 12-month yield
            if price and price > 0:
                one_year_ago = as_of_dt - timedelta(days=365)
                trailing_divs = [
                    float(d.get("amount", 0))
                    for d in past_divs
                    if datetime.fromisoformat(d["ex_date"]) >= one_year_ago
                ]
                trailing_sum = sum(trailing_divs)
                features["trailing_dividend_yield"] = (trailing_sum / price) * 100

            # Dividend streak (simplified)
            features["dividend_streak"] = len(past_divs)

        except Exception as e:
            logger.warning(f"Error computing dividend features: {e}")

        return features

    # =========================================================================
    # Gap Analysis (Pre-market)
    # =========================================================================

    def compute_gap_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pre-market gap features.

        Gap = (today_open - yesterday_close) / yesterday_close

        Features added:
        - gap_pct: Gap percentage
        - gap_abs: Absolute gap (log)
        - gap_direction: 1 for up, -1 for down, 0 for flat
        - gap_filled: Whether gap was filled during day
        - gap_magnitude: Categorized gap size (small/medium/large)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with gap features added
        """
        df = df.copy()

        if "close" not in df.columns or "open" not in df.columns:
            logger.warning("Missing close/open columns for gap analysis")
            return df

        # Compute previous close
        prev_close = df["close"].shift(1)

        # Gap percentage
        df["gap_pct"] = ((df["open"] - prev_close) / prev_close * 100).fillna(0)

        # Absolute gap (log-scaled for normalization)
        df["gap_abs"] = np.log1p(np.abs(df["gap_pct"])) * np.sign(df["gap_pct"])

        # Gap direction
        df["gap_direction"] = np.sign(df["gap_pct"])

        # Gap magnitude category
        def categorize_gap(pct: float) -> int:
            """
            Categorize gap:
            0: No gap (< 0.5%)
            1: Small (0.5% - 2%)
            2: Medium (2% - 5%)
            3: Large (> 5%)
            """
            abs_pct = abs(pct)
            if abs_pct < 0.5:
                return 0
            elif abs_pct < 2.0:
                return 1
            elif abs_pct < 5.0:
                return 2
            else:
                return 3

        df["gap_magnitude"] = df["gap_pct"].apply(categorize_gap)

        # Gap filled (price crossed previous close during day)
        df["gap_filled"] = 0
        up_gap_mask = df["gap_pct"] > 0
        down_gap_mask = df["gap_pct"] < 0

        # Up gap filled if low goes below prev close
        df.loc[up_gap_mask & (df["low"] <= prev_close), "gap_filled"] = 1

        # Down gap filled if high goes above prev close
        df.loc[down_gap_mask & (df["high"] >= prev_close), "gap_filled"] = 1

        return df

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def prefetch_data(
        self,
        symbols: Sequence[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """
        Prefetch and cache corporate actions for multiple symbols.

        Use this to warm up cache before backtesting.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
        """
        logger.info(f"Prefetching corporate actions for {len(symbols)} symbols")

        for symbol in symbols:
            try:
                self.get_dividends(symbol, start_date=start_date, end_date=end_date)
                self.get_splits(symbol, start_date=start_date, end_date=end_date)
                self.get_earnings(symbol, start_date=start_date, end_date=end_date)
            except Exception as e:
                logger.warning(f"Error prefetching {symbol}: {e}")

        logger.info("Prefetch complete")

    def get_corporate_actions_summary(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of corporate actions for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Summary dict with counts and totals
        """
        dividends = self.get_dividends(symbol, start_date=start_date, end_date=end_date)
        splits = self.get_splits(symbol, start_date=start_date, end_date=end_date)
        earnings = self.get_earnings(symbol, start_date=start_date, end_date=end_date)

        total_dividends = sum(float(d.get("amount", 0)) for d in dividends)

        return {
            "symbol": symbol.upper(),
            "dividend_count": len(dividends),
            "total_dividends": round(total_dividends, 4),
            "split_count": len(splits),
            "splits": [
                f"{s.get('ratio', [1,1])[0]}:{s.get('ratio', [1,1])[1]}"
                for s in splits
            ],
            "earnings_count": len(earnings),
            "last_dividend_date": dividends[-1]["ex_date"] if dividends else None,
            "last_split_date": splits[-1]["ex_date"] if splits else None,
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_service: Optional[CorporateActionsService] = None


def get_service() -> CorporateActionsService:
    """Get default CorporateActionsService instance."""
    global _default_service
    if _default_service is None:
        _default_service = CorporateActionsService()
    return _default_service


def adjust_prices_for_splits(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Convenience function to adjust prices for splits."""
    return get_service().adjust_prices(df, symbol, adjust_splits=True, adjust_dividends=False)


def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to compute gap features."""
    return get_service().compute_gap_features(df)
