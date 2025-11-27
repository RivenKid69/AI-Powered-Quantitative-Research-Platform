# -*- coding: utf-8 -*-
"""
services/session_router.py
Session-aware order routing for US equity markets.

Phase 9: Live Trading Improvements (2025-11-27)

This module provides intelligent order routing based on market session:
- Regular hours (9:30 AM - 4:00 PM ET)
- Pre-market (4:00 AM - 9:30 AM ET)
- After-hours (4:00 PM - 8:00 PM ET)

Features:
- Session detection
- Order type validation for each session
- Spread adjustment for extended hours
- Routing decisions for optimal execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Protocol
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone
ET = ZoneInfo("America/New_York")


# =============================================================================
# Session Types and Constants
# =============================================================================


class TradingSession(str, Enum):
    """Trading session types."""

    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"


@dataclass
class SessionInfo:
    """Information about current trading session."""

    session: TradingSession
    is_open: bool
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    time_to_close_ms: Optional[int] = None
    time_to_next_open_ms: Optional[int] = None
    is_half_day: bool = False


# Session characteristics
SESSION_CHARACTERISTICS: Dict[TradingSession, Dict[str, Any]] = {
    TradingSession.CLOSED: {
        "accepts_market_orders": False,
        "accepts_limit_orders": False,
        "typical_spread_multiplier": None,  # N/A
        "typical_volume_fraction": 0.0,
        "description": "Market closed",
    },
    TradingSession.PRE_MARKET: {
        "accepts_market_orders": False,  # Must use limit orders
        "accepts_limit_orders": True,
        "typical_spread_multiplier": 2.5,  # 2-3x regular spread
        "typical_volume_fraction": 0.05,  # ~5% of daily volume
        "description": "Pre-market session (4:00 AM - 9:30 AM ET)",
    },
    TradingSession.REGULAR: {
        "accepts_market_orders": True,
        "accepts_limit_orders": True,
        "typical_spread_multiplier": 1.0,
        "typical_volume_fraction": 0.90,  # ~90% of daily volume
        "description": "Regular trading hours (9:30 AM - 4:00 PM ET)",
    },
    TradingSession.AFTER_HOURS: {
        "accepts_market_orders": False,  # Must use limit orders
        "accepts_limit_orders": True,
        "typical_spread_multiplier": 2.0,  # 2x regular spread
        "typical_volume_fraction": 0.05,  # ~5% of daily volume
        "description": "After-hours session (4:00 PM - 8:00 PM ET)",
    },
}


# =============================================================================
# Protocols
# =============================================================================


class TradingHoursProvider(Protocol):
    """Protocol for trading hours providers."""

    def is_market_open(self, ts: int, *, session_type: Optional[str] = None) -> bool:
        """Check if market is open."""
        ...

    def next_open(self, ts: int) -> int:
        """Get next market open timestamp."""
        ...

    def next_close(self, ts: int) -> int:
        """Get next market close timestamp."""
        ...


class OrderExecutionProvider(Protocol):
    """Protocol for order execution providers."""

    def submit_order(self, order: Any) -> Any:
        """Submit order."""
        ...

    def submit_extended_hours_order(self, order: Any, session: str) -> Any:
        """Submit order with extended hours handling."""
        ...


# =============================================================================
# Session Detection
# =============================================================================


def get_current_session(
    ts_ms: Optional[int] = None,
    trading_hours: Optional[TradingHoursProvider] = None,
) -> SessionInfo:
    """
    Detect current trading session.

    Args:
        ts_ms: Timestamp in milliseconds (None = current time)
        trading_hours: Optional trading hours provider

    Returns:
        SessionInfo with current session details
    """
    if ts_ms is None:
        dt = datetime.now(ET)
        ts_ms = int(dt.timestamp() * 1000)
    else:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=ET)

    # Check if trading day (Monday-Friday, no holidays)
    # Basic check - use trading_hours provider for full holiday support
    if dt.weekday() >= 5:  # Weekend
        return SessionInfo(
            session=TradingSession.CLOSED,
            is_open=False,
        )

    # Get time in minutes from midnight
    minutes = dt.hour * 60 + dt.minute

    # Pre-market: 4:00 AM - 9:30 AM (240 - 570 minutes)
    if 240 <= minutes < 570:
        return SessionInfo(
            session=TradingSession.PRE_MARKET,
            is_open=True,
            session_start=dt.replace(hour=4, minute=0, second=0, microsecond=0),
            session_end=dt.replace(hour=9, minute=30, second=0, microsecond=0),
            time_to_close_ms=int((570 - minutes) * 60 * 1000),
        )

    # Regular hours: 9:30 AM - 4:00 PM (570 - 960 minutes)
    if 570 <= minutes < 960:
        return SessionInfo(
            session=TradingSession.REGULAR,
            is_open=True,
            session_start=dt.replace(hour=9, minute=30, second=0, microsecond=0),
            session_end=dt.replace(hour=16, minute=0, second=0, microsecond=0),
            time_to_close_ms=int((960 - minutes) * 60 * 1000),
        )

    # After-hours: 4:00 PM - 8:00 PM (960 - 1200 minutes)
    if 960 <= minutes < 1200:
        return SessionInfo(
            session=TradingSession.AFTER_HOURS,
            is_open=True,
            session_start=dt.replace(hour=16, minute=0, second=0, microsecond=0),
            session_end=dt.replace(hour=20, minute=0, second=0, microsecond=0),
            time_to_close_ms=int((1200 - minutes) * 60 * 1000),
        )

    # Closed (overnight: 8:00 PM - 4:00 AM)
    if minutes >= 1200:
        # Time until next pre-market (4:00 AM next day)
        time_to_open = (24 * 60 - minutes + 240) * 60 * 1000
    else:
        # Time until pre-market (4:00 AM same day)
        time_to_open = (240 - minutes) * 60 * 1000

    return SessionInfo(
        session=TradingSession.CLOSED,
        is_open=False,
        time_to_next_open_ms=int(time_to_open),
    )


# =============================================================================
# Order Routing
# =============================================================================


@dataclass
class RoutingDecision:
    """Decision for how to route an order."""

    should_submit: bool
    use_extended_hours: bool
    order_type_override: Optional[str] = None
    limit_price_adjustment: float = 0.0  # In basis points
    recommended_tif: str = "DAY"
    reason: str = ""
    warnings: list = None  # type: ignore

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SessionRouter:
    """
    Session-aware order router for US equity markets.

    Determines optimal routing based on:
    - Current session (pre-market, regular, after-hours)
    - Order type (market vs limit)
    - Size and urgency
    - Symbol liquidity

    Usage:
        router = SessionRouter(trading_hours=alpaca_hours_adapter)
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="BUY",
            qty=100,
            order_type="market",
        )
        if decision.should_submit:
            # Submit order with decision parameters
            ...
    """

    def __init__(
        self,
        trading_hours: Optional[TradingHoursProvider] = None,
        allow_extended_hours: bool = True,
        extended_hours_spread_multiplier: float = 2.0,
    ) -> None:
        """
        Initialize session router.

        Args:
            trading_hours: Trading hours provider
            allow_extended_hours: Whether to allow extended hours trading
            extended_hours_spread_multiplier: Spread multiplier for extended hours
        """
        self._trading_hours = trading_hours
        self._allow_extended = allow_extended_hours
        self._spread_mult = extended_hours_spread_multiplier

    @property
    def current_session(self) -> SessionInfo:
        """Get current trading session."""
        return get_current_session(trading_hours=self._trading_hours)

    def get_routing_decision(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        limit_price: Optional[float] = None,
        urgent: bool = False,
        ts_ms: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Get routing decision for an order.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Order quantity
            order_type: Order type ("market" or "limit")
            limit_price: Limit price (if limit order)
            urgent: Whether order is urgent
            ts_ms: Timestamp (None = current time)

        Returns:
            RoutingDecision with routing parameters
        """
        session_info = get_current_session(ts_ms)
        session_chars = SESSION_CHARACTERISTICS[session_info.session]
        warnings: list = []

        # Market closed
        if session_info.session == TradingSession.CLOSED:
            return RoutingDecision(
                should_submit=False,
                use_extended_hours=False,
                reason="Market is closed",
                warnings=["Market is currently closed. Order will be queued."],
            )

        # Extended hours
        if session_info.session in (TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS):
            if not self._allow_extended:
                return RoutingDecision(
                    should_submit=False,
                    use_extended_hours=False,
                    reason="Extended hours trading disabled",
                    warnings=["Extended hours trading is disabled in configuration."],
                )

            # Market orders not allowed in extended hours
            if order_type.lower() == "market":
                if limit_price is None:
                    return RoutingDecision(
                        should_submit=False,
                        use_extended_hours=True,
                        order_type_override="limit",
                        reason="Market orders not allowed in extended hours - need limit price",
                        warnings=[
                            "Market orders are not allowed during extended hours. "
                            "Please specify a limit price."
                        ],
                    )
                else:
                    # Convert to limit order
                    warnings.append(
                        "Converting market order to limit order for extended hours."
                    )
                    return RoutingDecision(
                        should_submit=True,
                        use_extended_hours=True,
                        order_type_override="limit",
                        recommended_tif="DAY",
                        reason="Extended hours - converted to limit order",
                        warnings=warnings,
                    )

            # Limit order in extended hours - add spread adjustment
            spread_adj = (self._spread_mult - 1.0) * 100  # Convert to bps
            warnings.append(
                f"Extended hours: spreads typically {self._spread_mult:.1f}x wider. "
                f"Consider {spread_adj:.0f} bps price adjustment."
            )

            return RoutingDecision(
                should_submit=True,
                use_extended_hours=True,
                limit_price_adjustment=spread_adj,
                recommended_tif="DAY",
                reason=f"Routing to {session_info.session.value} session",
                warnings=warnings,
            )

        # Regular hours - all order types allowed
        return RoutingDecision(
            should_submit=True,
            use_extended_hours=False,
            recommended_tif="DAY",
            reason="Regular trading hours - standard routing",
            warnings=warnings,
        )

    def adjust_limit_price_for_session(
        self,
        base_price: float,
        side: str,
        session: Optional[TradingSession] = None,
    ) -> float:
        """
        Adjust limit price for session spread characteristics.

        Args:
            base_price: Original limit price
            side: Order side ("buy" or "sell")
            session: Trading session (None = current)

        Returns:
            Adjusted limit price
        """
        if session is None:
            session = self.current_session.session

        multiplier = SESSION_CHARACTERISTICS[session]["typical_spread_multiplier"]
        if multiplier is None or multiplier == 1.0:
            return base_price

        # Calculate adjustment (half the spread widening)
        adjustment_factor = (multiplier - 1.0) / 2.0

        if side.lower() == "buy":
            # Buy: reduce price by spread adjustment (more aggressive)
            return base_price * (1 - adjustment_factor / 100)
        else:
            # Sell: increase price by spread adjustment (more aggressive)
            return base_price * (1 + adjustment_factor / 100)

    def get_session_volume_estimate(
        self,
        daily_volume: float,
        session: Optional[TradingSession] = None,
    ) -> float:
        """
        Estimate volume for a session.

        Args:
            daily_volume: Average daily volume
            session: Trading session (None = current)

        Returns:
            Estimated session volume
        """
        if session is None:
            session = self.current_session.session

        fraction = SESSION_CHARACTERISTICS[session]["typical_volume_fraction"]
        return daily_volume * fraction

    def should_wait_for_regular(
        self,
        order_qty: float,
        daily_adv: float,
        urgency: str = "normal",
    ) -> bool:
        """
        Determine if order should wait for regular hours.

        For large orders, waiting for regular hours may provide better
        execution due to higher liquidity.

        Args:
            order_qty: Order quantity
            daily_adv: Average daily volume
            urgency: Order urgency ("low", "normal", "high")

        Returns:
            True if order should wait for regular hours
        """
        session = self.current_session

        # Already in regular hours
        if session.session == TradingSession.REGULAR:
            return False

        # Urgent orders should execute now
        if urgency == "high":
            return False

        # Calculate participation rate
        session_volume = self.get_session_volume_estimate(daily_adv)
        if session_volume == 0:
            return True

        participation = order_qty / session_volume

        # If participation > 10% in extended hours, consider waiting
        if participation > 0.10:
            logger.info(
                f"Large order ({participation:.1%} of session volume) - "
                f"consider waiting for regular hours"
            )
            return urgency == "low"

        return False


# =============================================================================
# Factory Functions
# =============================================================================


def create_session_router(
    trading_hours: Optional[TradingHoursProvider] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SessionRouter:
    """
    Create a session router with configuration.

    Args:
        trading_hours: Trading hours provider
        config: Optional configuration dict

    Returns:
        Configured SessionRouter
    """
    config = config or {}

    return SessionRouter(
        trading_hours=trading_hours,
        allow_extended_hours=config.get("allow_extended_hours", True),
        extended_hours_spread_multiplier=config.get("extended_hours_spread_multiplier", 2.0),
    )


def create_alpaca_session_router(
    alpaca_adapter: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SessionRouter:
    """
    Create a session router for Alpaca.

    Args:
        alpaca_adapter: AlpacaTradingHoursAdapter instance
        config: Optional configuration dict

    Returns:
        SessionRouter configured for Alpaca
    """
    return create_session_router(
        trading_hours=alpaca_adapter,
        config=config,
    )
