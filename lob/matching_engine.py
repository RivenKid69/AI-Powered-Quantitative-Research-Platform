"""
FIFO Matching Engine with Price-Time Priority.

Implements CME Globex-style matching algorithm:
- Price-Time Priority (FIFO)
- Aggressive vs Passive order detection
- Self-Trade Prevention (STP) logic
- Partial fill handling with correct remaining qty
- Walk-through simulation for market orders

Reference:
    CME Globex Matching Algorithm
    https://www.cmegroup.com/confluence/display/EPICSANDBOX/CME+Globex+Matching+Algorithm+Steps

Performance Target: <10us per match operation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderBook,
    OrderType,
    PriceLevel,
    Side,
    Trade,
)


# ==============================================================================
# Enums and Constants
# ==============================================================================


class STPAction(IntEnum):
    """Self-Trade Prevention action modes."""

    CANCEL_NEWEST = 1  # Cancel the incoming (aggressive) order
    CANCEL_OLDEST = 2  # Cancel the resting (passive) order
    CANCEL_BOTH = 3  # Cancel both orders
    DECREMENT_AND_CANCEL = 4  # Decrement qty and cancel remaining


class OrderStatus(IntEnum):
    """Order status enumeration."""

    NEW = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4


class MatchType(IntEnum):
    """Type of match that occurred."""

    MAKER = 1  # Passive order (added liquidity)
    TAKER = 2  # Aggressive order (removed liquidity)


# ==============================================================================
# Result Data Structures
# ==============================================================================


@dataclass
class MatchResult:
    """
    Result of a match operation.

    Attributes:
        fills: List of fills generated
        resting_order: Remaining order to add to book (if limit order not fully filled)
        cancelled_orders: Orders cancelled due to STP
        total_filled_qty: Total quantity filled
        avg_fill_price: Volume-weighted average fill price
        is_complete: Whether the incoming order was fully filled
        match_type: MAKER or TAKER
    """

    fills: List[Fill]
    resting_order: Optional[LimitOrder] = None
    cancelled_orders: List[LimitOrder] = field(default_factory=list)
    total_filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    is_complete: bool = False
    match_type: MatchType = MatchType.TAKER

    @classmethod
    def empty(cls) -> "MatchResult":
        """Create empty result."""
        return cls(fills=[], total_filled_qty=0.0, is_complete=False)

    @classmethod
    def from_fills(
        cls,
        fills: List[Fill],
        original_qty: float,
        resting_order: Optional[LimitOrder] = None,
        cancelled_orders: Optional[List[LimitOrder]] = None,
    ) -> "MatchResult":
        """Create MatchResult from list of fills."""
        total_qty = sum(f.total_qty for f in fills)
        total_notional = sum(f.notional for f in fills)
        avg_price = total_notional / total_qty if total_qty > 0 else 0.0

        return cls(
            fills=fills,
            resting_order=resting_order,
            cancelled_orders=cancelled_orders or [],
            total_filled_qty=total_qty,
            avg_fill_price=avg_price,
            is_complete=total_qty >= original_qty,
            match_type=MatchType.TAKER if total_qty > 0 else MatchType.MAKER,
        )


@dataclass
class STPResult:
    """Result of Self-Trade Prevention check."""

    triggered: bool = False
    cancel_incoming: bool = False
    cancel_resting: bool = False
    incoming_reduction: float = 0.0
    resting_reduction: float = 0.0


# ==============================================================================
# Matching Engine
# ==============================================================================


class MatchingEngine:
    """
    FIFO Matching Engine with Price-Time Priority.

    Implements a full matching engine for limit order books:
    - Price-time priority (FIFO) matching
    - Market order execution with walk-through
    - Limit order matching (aggressive portion) and resting
    - Self-trade prevention
    - Partial fill handling

    Reference:
        CME Globex Matching Algorithm Steps
        https://www.cmegroup.com/confluence/display/EPICSANDBOX/CME+Globex+Matching+Algorithm+Steps

    Usage:
        engine = MatchingEngine()

        # Match market order
        result = engine.match_market_order(Side.BUY, 100.0, book)

        # Match limit order
        result = engine.match_limit_order(order, book)

    Performance:
        Target: <10us per match operation
    """

    def __init__(
        self,
        stp_action: STPAction = STPAction.CANCEL_NEWEST,
        enable_stp: bool = True,
        on_trade: Optional[Callable[[Trade], None]] = None,
        on_stp: Optional[Callable[[LimitOrder, LimitOrder], None]] = None,
    ) -> None:
        """
        Initialize matching engine.

        Args:
            stp_action: Self-trade prevention action mode
            enable_stp: Whether to enable self-trade prevention
            on_trade: Callback for each trade executed
            on_stp: Callback when STP triggers (incoming, resting)
        """
        self._stp_action = stp_action
        self._enable_stp = enable_stp
        self._on_trade = on_trade
        self._on_stp = on_stp

        # Statistics
        self._match_count = 0
        self._trade_count = 0
        self._stp_count = 0

    # ==========================================================================
    # Market Order Matching
    # ==========================================================================

    def match_market_order(
        self,
        side: Side,
        qty: float,
        order_book: OrderBook,
        taker_order_id: Optional[str] = None,
        taker_participant_id: Optional[str] = None,
    ) -> MatchResult:
        """
        Match market order against resting liquidity.

        Walks through price levels in price-time priority order,
        filling at each level until order is complete or book is exhausted.

        Args:
            side: BUY (lifts asks) or SELL (hits bids)
            qty: Quantity to execute
            order_book: OrderBook to match against
            taker_order_id: ID of the taker order
            taker_participant_id: Participant ID for STP checking

        Returns:
            MatchResult with fills and execution details
        """
        self._match_count += 1

        if qty <= 0:
            return MatchResult.empty()

        # Select opposite side of the book
        if side == Side.BUY:
            levels = order_book._asks
            get_price = lambda key: key  # Ascending
        else:
            levels = order_book._bids
            get_price = lambda key: -key  # Stored as negative

        remaining_qty = qty
        all_trades: List[Trade] = []
        cancelled_orders: List[LimitOrder] = []
        levels_to_remove: List[float] = []

        # Walk through price levels in priority order
        for key in list(levels.keys()):
            if remaining_qty <= 0:
                break

            level = levels[key]
            price = get_price(key)

            # Match at this level
            trades, cancelled, filled_qty = self._match_at_level(
                level=level,
                price=price,
                qty=remaining_qty,
                aggressor_side=side,
                taker_order_id=taker_order_id,
                taker_participant_id=taker_participant_id,
            )

            all_trades.extend(trades)
            cancelled_orders.extend(cancelled)
            remaining_qty -= filled_qty

            # Track empty levels for cleanup
            if level.is_empty:
                levels_to_remove.append(key)

            # Update order book index for filled orders
            self._cleanup_filled_orders(order_book, trades)

        # Remove empty levels
        for key in levels_to_remove:
            if key in levels:
                del levels[key]

        # Create fill result
        fills = self._create_fills_from_trades(
            trades=all_trades,
            taker_order_id=taker_order_id or "",
            original_qty=qty,
        )

        return MatchResult.from_fills(
            fills=fills,
            original_qty=qty,
            cancelled_orders=cancelled_orders,
        )

    # ==========================================================================
    # Limit Order Matching
    # ==========================================================================

    def match_limit_order(
        self,
        order: LimitOrder,
        order_book: OrderBook,
    ) -> MatchResult:
        """
        Match limit order against resting liquidity.

        First matches the aggressive portion (if order crosses the spread),
        then returns the resting portion to add to the book.

        Args:
            order: LimitOrder to match
            order_book: OrderBook to match against

        Returns:
            MatchResult with fills and resting order (if any)
        """
        self._match_count += 1

        if order.remaining_qty <= 0:
            return MatchResult.empty()

        # Check if order is aggressive (crosses the spread)
        is_aggressive = self._is_aggressive_order(order, order_book)

        if not is_aggressive:
            # Passive order - goes directly to book
            return MatchResult(
                fills=[],
                resting_order=order,
                total_filled_qty=0.0,
                is_complete=False,
                match_type=MatchType.MAKER,
            )

        # Aggressive order - match against opposite side
        remaining_qty = order.remaining_qty
        all_trades: List[Trade] = []
        cancelled_orders: List[LimitOrder] = []

        # Select opposite side
        if order.side == Side.BUY:
            levels = order_book._asks
            get_price = lambda key: key
            price_limit = order.price  # Can buy at or below limit
            price_check = lambda p: p <= price_limit
        else:
            levels = order_book._bids
            get_price = lambda key: -key
            price_limit = order.price  # Can sell at or above limit
            price_check = lambda p: p >= price_limit

        levels_to_remove: List[float] = []

        # Walk through price levels
        for key in list(levels.keys()):
            if remaining_qty <= 0:
                break

            level = levels[key]
            price = get_price(key)

            # Check price limit
            if not price_check(price):
                break

            # Match at this level
            trades, cancelled, filled_qty = self._match_at_level(
                level=level,
                price=price,
                qty=remaining_qty,
                aggressor_side=order.side,
                taker_order_id=order.order_id,
                taker_participant_id=order.participant_id,
            )

            all_trades.extend(trades)
            cancelled_orders.extend(cancelled)
            remaining_qty -= filled_qty

            # Track empty levels
            if level.is_empty:
                levels_to_remove.append(key)

            # Update order book
            self._cleanup_filled_orders(order_book, trades)

        # Remove empty levels
        for key in levels_to_remove:
            if key in levels:
                del levels[key]

        # Create fills
        fills = self._create_fills_from_trades(
            trades=all_trades,
            taker_order_id=order.order_id,
            original_qty=order.remaining_qty,
        )

        # Calculate remaining order
        total_filled = sum(f.total_qty for f in fills)
        resting_order = None

        if remaining_qty > 0:
            # Create resting order with remaining qty
            resting_order = LimitOrder(
                order_id=order.order_id,
                price=order.price,
                qty=order.qty,
                remaining_qty=remaining_qty,
                timestamp_ns=order.timestamp_ns,
                side=order.side,
                hidden_qty=order.hidden_qty,
                display_qty=min(order.display_qty, remaining_qty),
                is_own=order.is_own,
                order_type=order.order_type,
                participant_id=order.participant_id,
            )

        return MatchResult.from_fills(
            fills=fills,
            original_qty=order.remaining_qty,
            resting_order=resting_order,
            cancelled_orders=cancelled_orders,
        )

    # ==========================================================================
    # Internal Matching Logic
    # ==========================================================================

    def _match_at_level(
        self,
        level: PriceLevel,
        price: float,
        qty: float,
        aggressor_side: Side,
        taker_order_id: Optional[str] = None,
        taker_participant_id: Optional[str] = None,
    ) -> Tuple[List[Trade], List[LimitOrder], float]:
        """
        Match quantity at a single price level (FIFO order).

        Args:
            level: PriceLevel to match at
            price: Price of this level
            qty: Quantity to match
            aggressor_side: Side of the aggressor
            taker_order_id: ID of taker order
            taker_participant_id: Participant ID for STP

        Returns:
            Tuple of (trades, cancelled_orders, total_filled)
        """
        trades: List[Trade] = []
        cancelled_orders: List[LimitOrder] = []
        remaining = qty
        orders_to_remove: List[str] = []

        # Iterate through orders in FIFO order
        for order in list(level.orders):
            if remaining <= 0:
                break

            # Check self-trade prevention
            if self._enable_stp and taker_participant_id:
                stp_result = self._check_stp(
                    incoming_participant=taker_participant_id,
                    resting_order=order,
                )

                if stp_result.triggered:
                    self._stp_count += 1

                    if self._on_stp:
                        self._on_stp(
                            LimitOrder(
                                order_id=taker_order_id or "",
                                price=price,
                                qty=qty,
                                remaining_qty=remaining,
                                timestamp_ns=time.time_ns(),
                                side=aggressor_side,
                            ),
                            order,
                        )

                    if stp_result.cancel_incoming:
                        # Cancel incoming order - stop matching
                        break

                    if stp_result.cancel_resting:
                        # Cancel resting order
                        cancelled_orders.append(order)
                        orders_to_remove.append(order.order_id)
                        continue

            # Calculate fill amount
            fill_amount = min(remaining, order.remaining_qty)

            if fill_amount > 0:
                # Create trade
                trade = Trade(
                    price=price,
                    qty=fill_amount,
                    maker_order_id=order.order_id,
                    taker_order_id=taker_order_id,
                    maker_is_own=order.is_own,
                    taker_is_own=False,  # Caller can update
                    timestamp_ns=time.time_ns(),
                    aggressor_side=aggressor_side,
                )
                trades.append(trade)
                self._trade_count += 1

                # Callback
                if self._on_trade:
                    self._on_trade(trade)

                # Fill the order
                order.fill(fill_amount)
                remaining -= fill_amount

                # Track filled orders for removal
                if order.is_filled:
                    orders_to_remove.append(order.order_id)

        # Remove filled/cancelled orders from level
        for order_id in orders_to_remove:
            level.remove_order(order_id)

        return trades, cancelled_orders, qty - remaining

    def _is_aggressive_order(
        self,
        order: LimitOrder,
        order_book: OrderBook,
    ) -> bool:
        """
        Check if limit order is aggressive (crosses the spread).

        Args:
            order: LimitOrder to check
            order_book: OrderBook to check against

        Returns:
            True if order crosses the spread
        """
        if order.side == Side.BUY:
            best_ask = order_book.best_ask
            if best_ask is None:
                return False
            return order.price >= best_ask
        else:
            best_bid = order_book.best_bid
            if best_bid is None:
                return False
            return order.price <= best_bid

    def _check_stp(
        self,
        incoming_participant: str,
        resting_order: LimitOrder,
    ) -> STPResult:
        """
        Check for self-trade and determine action.

        Args:
            incoming_participant: Participant ID of incoming order
            resting_order: Resting order in the book

        Returns:
            STPResult with triggered flag and actions
        """
        # No STP if no participant IDs
        if not incoming_participant or not resting_order.participant_id:
            return STPResult(triggered=False)

        # No self-trade if different participants
        if incoming_participant != resting_order.participant_id:
            return STPResult(triggered=False)

        # Self-trade detected - determine action
        if self._stp_action == STPAction.CANCEL_NEWEST:
            return STPResult(triggered=True, cancel_incoming=True)
        elif self._stp_action == STPAction.CANCEL_OLDEST:
            return STPResult(triggered=True, cancel_resting=True)
        elif self._stp_action == STPAction.CANCEL_BOTH:
            return STPResult(triggered=True, cancel_incoming=True, cancel_resting=True)
        else:
            # DECREMENT_AND_CANCEL - cancel smaller, decrement larger
            return STPResult(triggered=True, cancel_incoming=True)

    def _cleanup_filled_orders(
        self,
        order_book: OrderBook,
        trades: List[Trade],
    ) -> None:
        """Remove filled orders from order book index."""
        for trade in trades:
            order_id = trade.maker_order_id
            if order_id in order_book._orders:
                order, _ = order_book._orders[order_id]
                if order.is_filled:
                    del order_book._orders[order_id]
                    order_book._order_count -= 1

    def _create_fills_from_trades(
        self,
        trades: List[Trade],
        taker_order_id: str,
        original_qty: float,
    ) -> List[Fill]:
        """Create Fill objects from trades."""
        if not trades:
            return []

        # Group trades by price for reporting
        return [
            Fill.from_trades(
                order_id=taker_order_id,
                original_qty=original_qty,
                trades=trades,
            )
        ]

    # ==========================================================================
    # Simulation / Walk-Through
    # ==========================================================================

    def simulate_market_order(
        self,
        side: Side,
        qty: float,
        order_book: OrderBook,
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        Simulate market order execution without modifying book.

        Walks through the book to estimate execution price
        without actually matching.

        Args:
            side: BUY or SELL
            qty: Quantity to simulate
            order_book: OrderBook to simulate against

        Returns:
            Tuple of (avg_price, total_filled, fills_by_level)
        """
        return order_book.walk_book(side, qty)

    def estimate_market_impact(
        self,
        side: Side,
        qty: float,
        order_book: OrderBook,
    ) -> Optional[float]:
        """
        Estimate market impact in basis points.

        Args:
            side: BUY or SELL
            qty: Quantity to estimate impact for
            order_book: OrderBook

        Returns:
            Estimated market impact in basis points, or None if insufficient liquidity
        """
        mid = order_book.mid_price
        if mid is None or mid == 0:
            return None

        avg_price, total_filled, _ = self.simulate_market_order(side, qty, order_book)

        if total_filled < qty:
            return None  # Insufficient liquidity

        # Impact = |execution_price - mid| / mid * 10000
        impact_bps = abs(avg_price - mid) / mid * 10000.0
        return impact_bps

    # ==========================================================================
    # Statistics
    # ==========================================================================

    @property
    def match_count(self) -> int:
        """Total match operations performed."""
        return self._match_count

    @property
    def trade_count(self) -> int:
        """Total trades executed."""
        return self._trade_count

    @property
    def stp_count(self) -> int:
        """Number of self-trade prevention triggers."""
        return self._stp_count

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self._match_count = 0
        self._trade_count = 0
        self._stp_count = 0


# ==============================================================================
# Pro-Rata Matching Engine (Optional Extension)
# ==============================================================================


class ProRataMatchingEngine(MatchingEngine):
    """
    Pro-Rata Matching Engine.

    Some exchanges use pro-rata allocation instead of FIFO at certain
    price levels. This is common in options and some futures markets.

    Allocation formula:
        allocation_i = order_qty_i / total_level_qty * fill_qty

    Note: This is provided for completeness but most equity markets use FIFO.
    """

    def __init__(
        self,
        min_allocation: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initialize pro-rata matching engine.

        Args:
            min_allocation: Minimum allocation per order (rounds down to 0 if below)
            **kwargs: Arguments passed to MatchingEngine
        """
        super().__init__(**kwargs)
        self._min_allocation = min_allocation

    def _match_at_level(
        self,
        level: PriceLevel,
        price: float,
        qty: float,
        aggressor_side: Side,
        taker_order_id: Optional[str] = None,
        taker_participant_id: Optional[str] = None,
    ) -> Tuple[List[Trade], List[LimitOrder], float]:
        """
        Match using pro-rata allocation at this level.

        Each resting order receives a proportional share of the fill.
        """
        trades: List[Trade] = []
        cancelled_orders: List[LimitOrder] = []

        total_level_qty = level.total_visible_qty + level.total_hidden_qty
        if total_level_qty <= 0:
            return trades, cancelled_orders, 0.0

        # Calculate fill amount (can't fill more than available)
        fill_qty = min(qty, total_level_qty)

        # Allocate pro-rata to each order
        remaining_fill = fill_qty
        orders_to_remove: List[str] = []

        for order in list(level.orders):
            if remaining_fill <= 0:
                break

            # STP check
            if self._enable_stp and taker_participant_id:
                stp_result = self._check_stp(taker_participant_id, order)
                if stp_result.triggered:
                    self._stp_count += 1
                    if stp_result.cancel_resting:
                        cancelled_orders.append(order)
                        orders_to_remove.append(order.order_id)
                        continue

            # Pro-rata allocation
            allocation = (order.remaining_qty / total_level_qty) * fill_qty

            # Apply minimum allocation
            if allocation < self._min_allocation:
                allocation = 0.0
            else:
                allocation = min(allocation, order.remaining_qty, remaining_fill)

            if allocation > 0:
                trade = Trade(
                    price=price,
                    qty=allocation,
                    maker_order_id=order.order_id,
                    taker_order_id=taker_order_id,
                    maker_is_own=order.is_own,
                    timestamp_ns=time.time_ns(),
                    aggressor_side=aggressor_side,
                )
                trades.append(trade)
                self._trade_count += 1

                if self._on_trade:
                    self._on_trade(trade)

                order.fill(allocation)
                remaining_fill -= allocation

                if order.is_filled:
                    orders_to_remove.append(order.order_id)

        # Remove filled orders
        for order_id in orders_to_remove:
            level.remove_order(order_id)

        return trades, cancelled_orders, fill_qty - remaining_fill


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_matching_engine(
    algorithm: str = "fifo",
    **kwargs,
) -> MatchingEngine:
    """
    Factory function to create matching engine.

    Args:
        algorithm: Matching algorithm ("fifo" or "pro_rata")
        **kwargs: Arguments for engine constructor

    Returns:
        MatchingEngine instance
    """
    if algorithm.lower() == "fifo":
        return MatchingEngine(**kwargs)
    elif algorithm.lower() in ("pro_rata", "prorata"):
        return ProRataMatchingEngine(**kwargs)
    else:
        raise ValueError(f"Unknown matching algorithm: {algorithm}")
