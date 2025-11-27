# -*- coding: utf-8 -*-
"""
adapters/binance/trading_hours.py
Binance trading hours adapter.

Crypto markets trade 24/7, so this is a simplified implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from adapters.base import TradingHoursAdapter
from adapters.models import (
    CRYPTO_CONTINUOUS_SESSION,
    ExchangeVendor,
    MarketCalendar,
    MarketType,
    SessionType,
    TradingSession,
    create_crypto_calendar,
)

logger = logging.getLogger(__name__)


class BinanceTradingHoursAdapter(TradingHoursAdapter):
    """
    Binance trading hours adapter.

    Crypto markets are 24/7, so most methods return constant values.
    This adapter exists for API consistency with equity markets.

    Configuration:
        market_type: Market type (default: CRYPTO_SPOT)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Parse market type from config
        market_type_str = self._config.get("market_type", "CRYPTO_SPOT")
        try:
            self._market_type = MarketType(market_type_str)
        except ValueError:
            self._market_type = MarketType.CRYPTO_SPOT

        # Create calendar
        self._calendar = create_crypto_calendar(vendor)

    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """
        Check if market is open.

        Crypto markets are always open (24/7).

        Args:
            ts: Unix timestamp in milliseconds
            session_type: Session type (ignored for crypto)

        Returns:
            True (always for crypto)
        """
        # Crypto is always open
        if self._market_type.is_crypto:
            return True

        # For non-crypto types (shouldn't happen with Binance), defer to calendar
        return True

    def next_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market open.

        For 24/7 markets, returns current timestamp (already open).

        Args:
            ts: Current timestamp in milliseconds
            session_type: Session type (ignored for crypto)

        Returns:
            Current timestamp (market is already open)
        """
        if self._market_type.is_crypto:
            return ts  # Already open

        return ts

    def next_close(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market close.

        For 24/7 markets, returns far future timestamp (never closes).

        Args:
            ts: Current timestamp in milliseconds
            session_type: Session type (ignored for crypto)

        Returns:
            Far future timestamp (~10 years from ts)
        """
        if self._market_type.is_crypto:
            # Return 10 years from now as "never closes"
            return ts + (10 * 365 * 24 * 60 * 60 * 1000)

        return ts + (10 * 365 * 24 * 60 * 60 * 1000)

    def get_calendar(self) -> MarketCalendar:
        """
        Get market calendar.

        Returns:
            MarketCalendar for crypto (24/7 continuous session)
        """
        return self._calendar

    def is_holiday(self, ts: int) -> bool:
        """
        Check if date is a holiday.

        Crypto markets have no holidays.

        Args:
            ts: Timestamp in milliseconds

        Returns:
            False (crypto has no holidays)
        """
        return False

    def get_session(self, ts: int) -> Optional[TradingSession]:
        """
        Get current trading session.

        For crypto, always returns continuous session.

        Args:
            ts: Timestamp in milliseconds

        Returns:
            Continuous 24/7 trading session
        """
        if self._market_type.is_crypto:
            return CRYPTO_CONTINUOUS_SESSION
        return CRYPTO_CONTINUOUS_SESSION

    def time_to_close(self, ts: int) -> Optional[int]:
        """
        Get milliseconds until market close.

        For 24/7 markets, returns None (never closes).

        Args:
            ts: Current timestamp

        Returns:
            None (market never closes)
        """
        if self._market_type.is_crypto:
            return None  # Never closes
        return None

    def supports_extended_hours(self) -> bool:
        """
        Check if exchange supports extended hours trading.

        Crypto is always "extended hours" (24/7).

        Returns:
            True
        """
        return True

    def get_maintenance_windows(self) -> list:
        """
        Get scheduled maintenance windows.

        Binance may have occasional maintenance, but this is typically
        announced dynamically. This method returns empty list.

        Returns:
            Empty list (no scheduled maintenance)
        """
        return []
