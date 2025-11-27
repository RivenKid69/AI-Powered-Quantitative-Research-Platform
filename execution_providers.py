# -*- coding: utf-8 -*-
"""
execution_providers.py
Multi-asset execution provider interfaces and implementations.

This module provides a clean abstraction layer for execution simulation,
supporting both crypto (Binance) and equities (Alpaca/Polygon) markets.

Architecture:
    Protocol-based interfaces (SlippageProvider, FillProvider, FeeProvider)
    allow pluggable implementations for different markets and fidelity levels.

Levels of Fidelity:
    L1: Simple constant spread/fee model
    L2: Statistical models (√participation impact, OHLCV fills) - DEFAULT
    L3: Full LOB simulation (future - requires order book data)

Design Principles:
    - Protocol-based for flexibility and testability
    - Backward compatible with existing crypto infrastructure
    - Asset-class agnostic interfaces with specialized implementations
    - Research-backed models (Almgren-Chriss for market impact)

References:
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    - Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
"""

from __future__ import annotations

import enum
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AssetClass(enum.Enum):
    """Asset class enumeration for execution providers."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"


class OrderSide(enum.Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(enum.Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class LiquidityRole(enum.Enum):
    """Liquidity role (maker/taker) for fee calculation."""
    MAKER = "maker"
    TAKER = "taker"
    UNKNOWN = "unknown"


# Default constants
_DEFAULT_SPREAD_BPS = 5.0
_DEFAULT_IMPACT_COEF = 0.1  # Almgren-Chriss style
_DEFAULT_VOLATILITY_SCALE = 1.0
_MIN_PARTICIPATION = 1e-12
_MAX_SLIPPAGE_BPS = 500.0  # Safety cap


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class MarketState:
    """
    Market snapshot for execution decisions.

    Captures the current market state including quotes, liquidity,
    and optional order book depth (L3).

    Attributes:
        timestamp: Unix timestamp in milliseconds
        bid: Best bid price
        ask: Best ask price
        bid_size: Size at best bid
        ask_size: Size at best ask
        last_price: Last traded price
        mid_price: Mid-market price (computed if None)
        spread_bps: Spread in basis points (computed if None)
        adv: Average daily volume (optional)
        volatility: Current volatility estimate (optional)
        bid_depth: L3 bid depth [(price, size), ...]
        ask_depth: L3 ask depth [(price, size), ...]
    """
    timestamp: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    last_price: Optional[float] = None
    mid_price: Optional[float] = None
    spread_bps: Optional[float] = None
    adv: Optional[float] = None
    volatility: Optional[float] = None
    # L3 extension points
    bid_depth: Optional[List[Tuple[float, float]]] = None
    ask_depth: Optional[List[Tuple[float, float]]] = None

    def get_mid_price(self) -> Optional[float]:
        """Get mid-price, computing from bid/ask if not provided."""
        if self.mid_price is not None and math.isfinite(self.mid_price):
            return self.mid_price
        if self.bid is not None and self.ask is not None:
            if math.isfinite(self.bid) and math.isfinite(self.ask):
                return (self.bid + self.ask) / 2.0
        if self.last_price is not None and math.isfinite(self.last_price):
            return self.last_price
        return None

    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points, computing if not provided."""
        if self.spread_bps is not None and math.isfinite(self.spread_bps):
            return self.spread_bps
        mid = self.get_mid_price()
        if mid is None or mid <= 0:
            return None
        if self.bid is not None and self.ask is not None:
            if math.isfinite(self.bid) and math.isfinite(self.ask):
                spread = self.ask - self.bid
                return (spread / mid) * 10000.0
        return None

    def get_reference_price(self, side: Union[str, OrderSide]) -> Optional[float]:
        """Get reference price for a given side (bid for sell, ask for buy)."""
        side_str = side.value if isinstance(side, OrderSide) else str(side).upper()
        if side_str == "BUY":
            if self.ask is not None and math.isfinite(self.ask):
                return self.ask
        elif side_str == "SELL":
            if self.bid is not None and math.isfinite(self.bid):
                return self.bid
        return self.get_mid_price()


@dataclass(frozen=True)
class Order:
    """
    Order for execution.

    Represents an order to be executed in the simulation.

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
        side: Order side ("BUY" or "SELL")
        qty: Quantity to trade
        order_type: Order type ("MARKET" or "LIMIT")
        limit_price: Limit price (required for LIMIT orders)
        notional: Optional notional value (computed if not provided)
        asset_class: Asset class for this order
        client_order_id: Optional client-provided order ID
        time_in_force: Time in force (GTC, IOC, FOK)
    """
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    order_type: str  # "MARKET" | "LIMIT"
    limit_price: Optional[float] = None
    notional: Optional[float] = None
    asset_class: AssetClass = AssetClass.CRYPTO
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"

    def __post_init__(self) -> None:
        # Validation (for frozen dataclass, we just log warnings)
        if self.qty <= 0:
            logger.warning("Order qty should be positive: %s", self.qty)
        if self.order_type == "LIMIT" and self.limit_price is None:
            logger.warning("LIMIT order missing limit_price")

    def get_notional(self, price: float) -> float:
        """Compute notional value at given price."""
        if self.notional is not None and self.notional > 0:
            return self.notional
        return abs(self.qty) * price

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return str(self.side).upper() == "BUY"


@dataclass
class Fill:
    """
    Execution result (fill).

    Represents the result of an order execution attempt.

    Attributes:
        price: Fill price
        qty: Filled quantity
        fee: Fee amount
        slippage_bps: Slippage in basis points
        liquidity: Liquidity role ("maker" or "taker")
        timestamp: Fill timestamp (optional)
        notional: Trade notional value
        fee_breakdown: Detailed fee breakdown (optional)
        metadata: Additional execution metadata
    """
    price: float
    qty: float
    fee: float
    slippage_bps: float
    liquidity: str  # "maker" | "taker"
    timestamp: Optional[int] = None
    notional: Optional[float] = None
    fee_breakdown: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.notional is None:
            self.notional = self.price * self.qty

    @property
    def total_cost(self) -> float:
        """Total execution cost (slippage + fee)."""
        notional = self.notional or (self.price * self.qty)
        slippage_cost = notional * self.slippage_bps / 10000.0
        return slippage_cost + self.fee

    @property
    def total_cost_bps(self) -> float:
        """Total execution cost in basis points."""
        notional = self.notional or (self.price * self.qty)
        if notional <= 0:
            return 0.0
        return (self.total_cost / notional) * 10000.0


@dataclass
class BarData:
    """
    OHLCV bar data for fill simulation.

    Used by L2 providers for intrabar price interpolation.

    Attributes:
        open: Bar open price
        high: Bar high price
        low: Bar low price
        close: Bar close price
        volume: Bar volume
        timestamp: Bar start timestamp
        timeframe_ms: Bar timeframe in milliseconds
    """
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    timestamp: Optional[int] = None
    timeframe_ms: Optional[int] = None

    def contains_price(self, price: float, tolerance: float = 0.0) -> bool:
        """Check if price is within bar's high-low range."""
        return (self.low - tolerance) <= price <= (self.high + tolerance)

    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def bar_range(self) -> float:
        """Bar price range (high - low)."""
        return self.high - self.low


# =============================================================================
# Protocol Definitions (Interfaces)
# =============================================================================

@runtime_checkable
class SlippageProvider(Protocol):
    """
    Abstract slippage computation protocol.

    Implementations compute expected slippage based on order details,
    market state, and participation ratio.

    L2 Example: √participation model (Almgren-Chriss)
    L3 Example: LOB walk-through simulation
    """

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute expected slippage in basis points.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / ADV (or bar volume)

        Returns:
            Expected slippage in basis points (always non-negative)
        """
        ...


@runtime_checkable
class FillProvider(Protocol):
    """
    Abstract fill logic protocol.

    Implementations determine if/how an order fills based on
    order type, market state, and bar data.

    L2 Example: OHLCV bar-based fills (check if limit touched)
    L3 Example: Matching engine simulation with queue position
    """

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt to fill order.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data (OHLCV)

        Returns:
            Fill if order executed, None if not filled
        """
        ...


@runtime_checkable
class FeeProvider(Protocol):
    """
    Abstract fee computation protocol.

    Implementations compute trading fees based on notional,
    side, liquidity role, and asset-specific rules.

    Examples:
    - Crypto: Maker/taker fee tiers (0.02%/0.04%)
    - Equity: Commission-free with regulatory fees (SEC, TAF)
    """

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute fee amount.

        Args:
            notional: Trade notional value (price * qty)
            side: Trade side ("BUY" or "SELL")
            liquidity: Liquidity role ("maker" or "taker")
            qty: Trade quantity

        Returns:
            Fee amount in quote currency
        """
        ...


@runtime_checkable
class ExecutionProvider(Protocol):
    """
    Combined execution provider protocol.

    High-level interface that combines slippage, fill, and fee
    computation into a single execution workflow.
    """

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Execute an order with full slippage/fee calculation.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data

        Returns:
            Fill with slippage and fees, or None if not filled
        """
        ...

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        ...


# =============================================================================
# L2 Implementations (Statistical Models)
# =============================================================================

class StatisticalSlippageProvider:
    """
    L2: Statistical slippage model (Almgren-Chriss style).

    Implements square-root market impact model:
        slippage = half_spread + impact_coef * sqrt(participation) * volatility_scale

    Based on empirical research showing market impact scales with
    the square root of participation ratio.

    References:
        - Almgren & Chriss (2001): sqrt impact scaling
        - Kyle (1985): Lambda model for market impact
        - Gatheral (2010): Transient vs permanent impact

    Attributes:
        impact_coef: Market impact coefficient (default: 0.1)
        spread_bps: Default half-spread in basis points
        volatility_scale: Volatility scaling factor
        min_slippage_bps: Minimum slippage floor
        max_slippage_bps: Maximum slippage cap
    """

    def __init__(
        self,
        impact_coef: float = _DEFAULT_IMPACT_COEF,
        spread_bps: float = _DEFAULT_SPREAD_BPS,
        volatility_scale: float = _DEFAULT_VOLATILITY_SCALE,
        min_slippage_bps: float = 0.0,
        max_slippage_bps: float = _MAX_SLIPPAGE_BPS,
    ) -> None:
        """
        Initialize statistical slippage provider.

        Args:
            impact_coef: Market impact coefficient (k in k*sqrt(v))
            spread_bps: Default half-spread in basis points
            volatility_scale: Volatility adjustment factor
            min_slippage_bps: Minimum slippage floor
            max_slippage_bps: Maximum slippage cap (safety limit)
        """
        self.impact_coef = float(impact_coef)
        self.spread_bps = float(spread_bps)
        self.volatility_scale = float(volatility_scale)
        self.min_slippage_bps = float(min_slippage_bps)
        self.max_slippage_bps = float(max_slippage_bps)

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute expected slippage using √participation model.

        Formula:
            slippage_bps = half_spread + k * sqrt(participation) * vol_scale * 10000

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / reference volume

        Returns:
            Expected slippage in basis points
        """
        # Get spread from market or use default
        spread = market.get_spread_bps()
        if spread is None or not math.isfinite(spread) or spread < 0:
            half_spread = self.spread_bps / 2.0
        else:
            half_spread = spread / 2.0

        # Sanitize participation ratio
        participation = max(_MIN_PARTICIPATION, abs(participation_ratio))

        # Volatility adjustment
        vol_factor = self.volatility_scale
        if market.volatility is not None and math.isfinite(market.volatility):
            vol_factor *= market.volatility

        # Square-root market impact (Almgren-Chriss)
        impact_bps = self.impact_coef * math.sqrt(participation) * vol_factor * 10000.0

        # Total slippage
        total_slippage = half_spread + impact_bps

        # Apply bounds
        total_slippage = max(self.min_slippage_bps, total_slippage)
        total_slippage = min(self.max_slippage_bps, total_slippage)

        return float(total_slippage)

    def estimate_impact_cost(
        self,
        notional: float,
        adv: float,
        volatility: float = 0.02,
    ) -> Dict[str, float]:
        """
        Estimate market impact cost for a given trade size.

        Useful for pre-trade analytics and optimal execution planning.

        Args:
            notional: Trade notional value
            adv: Average daily volume (in quote currency)
            volatility: Annualized volatility (default: 2%)

        Returns:
            Dict with impact breakdown
        """
        if adv <= 0:
            return {"participation": 0.0, "impact_bps": 0.0, "impact_cost": 0.0}

        participation = notional / adv
        impact_bps = self.compute_slippage_bps(
            Order("", "BUY", 1.0, "MARKET"),
            MarketState(0, volatility=volatility),
            participation,
        )
        impact_cost = notional * impact_bps / 10000.0

        return {
            "participation": participation,
            "impact_bps": impact_bps,
            "impact_cost": impact_cost,
        }


class OHLCVFillProvider:
    """
    L2: Fill provider based on OHLCV bar data.

    Determines order fills based on whether the bar's price range
    would trigger the order:
    - MARKET: Always fills at reference price + slippage
    - LIMIT BUY: Fills if bar_low <= limit_price
    - LIMIT SELL: Fills if bar_high >= limit_price

    Uses pluggable SlippageProvider and FeeProvider for cost calculation.

    Attributes:
        slippage: SlippageProvider for slippage calculation
        fees: FeeProvider for fee calculation
        fill_at_limit: If True, limit orders fill at limit price; else at touch
        partial_fills: If True, allow partial fills based on liquidity
    """

    def __init__(
        self,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        fill_at_limit: bool = True,
        partial_fills: bool = False,
    ) -> None:
        """
        Initialize OHLCV fill provider.

        Args:
            slippage_provider: Provider for slippage calculation
            fee_provider: Provider for fee calculation
            fill_at_limit: Fill limit orders at limit price (True) or touch price
            partial_fills: Allow partial fills based on available liquidity
        """
        self.slippage = slippage_provider or StatisticalSlippageProvider()
        self.fees = fee_provider or ZeroFeeProvider()
        self.fill_at_limit = fill_at_limit
        self.partial_fills = partial_fills

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt to fill order based on bar data.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data (OHLCV)

        Returns:
            Fill if order executed, None if not filled
        """
        if order.qty <= 0:
            return None

        is_buy = order.is_buy
        order_type = str(order.order_type).upper()

        # Determine fill price and whether order fills
        filled = False
        fill_price: Optional[float] = None
        liquidity_role = LiquidityRole.TAKER

        if order_type == "MARKET":
            # Market orders always fill
            filled = True
            liquidity_role = LiquidityRole.TAKER
            # Get reference price from market
            ref_price = market.get_reference_price(order.side)
            if ref_price is None:
                ref_price = bar.open  # Fallback to bar open
            fill_price = ref_price

        elif order_type == "LIMIT":
            limit_price = order.limit_price
            if limit_price is None:
                logger.warning("LIMIT order missing limit_price")
                return None

            # First check: Immediate execution (crossing spread) → TAKER fill
            # A buy limit above the ask crosses the spread and fills immediately
            # A sell limit below the bid crosses the spread and fills immediately
            ref_price = market.get_reference_price(order.side)
            if ref_price is not None:
                if is_buy and limit_price >= ref_price:
                    # Buy limit at or above ask → immediate taker fill
                    filled = True
                    liquidity_role = LiquidityRole.TAKER
                    fill_price = ref_price
                elif not is_buy and limit_price <= ref_price:
                    # Sell limit at or below bid → immediate taker fill
                    filled = True
                    liquidity_role = LiquidityRole.TAKER
                    fill_price = ref_price

            # Second check: Passive fill based on bar range → MAKER fill
            # Only if not already filled as taker
            if not filled:
                if is_buy:
                    # Buy limit fills if bar_low <= limit_price
                    if bar.low <= limit_price:
                        filled = True
                        liquidity_role = LiquidityRole.MAKER
                        fill_price = limit_price if self.fill_at_limit else bar.low
                else:
                    # Sell limit fills if bar_high >= limit_price
                    if bar.high >= limit_price:
                        filled = True
                        liquidity_role = LiquidityRole.MAKER
                        fill_price = limit_price if self.fill_at_limit else bar.high

        if not filled or fill_price is None:
            return None

        # Calculate participation ratio for slippage
        participation = self._compute_participation(order, market, bar)

        # Calculate slippage
        slippage_bps = self.slippage.compute_slippage_bps(order, market, participation)

        # Apply slippage to fill price
        if liquidity_role == LiquidityRole.TAKER:
            slippage_mult = 1.0 + (slippage_bps / 10000.0) * (1 if is_buy else -1)
            adjusted_price = fill_price * slippage_mult
        else:
            # Maker orders have reduced/no slippage
            adjusted_price = fill_price
            slippage_bps = 0.0  # Maker fills at limit price

        # Clip to bar range (sanity check)
        adjusted_price = max(bar.low, min(bar.high, adjusted_price))

        # Calculate fill quantity (with partial fill support)
        fill_qty = order.qty
        if self.partial_fills and market.bid_size is not None and market.ask_size is not None:
            available = market.ask_size if is_buy else market.bid_size
            fill_qty = min(fill_qty, available)

        # Calculate notional and fee
        notional = adjusted_price * fill_qty
        fee = self.fees.compute_fee(
            notional=notional,
            side=order.side,
            liquidity=liquidity_role.value,
            qty=fill_qty,
        )

        return Fill(
            price=adjusted_price,
            qty=fill_qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity=liquidity_role.value,
            timestamp=market.timestamp,
            notional=notional,
            metadata={
                "original_price": fill_price,
                "participation": participation,
                "bar_range": (bar.low, bar.high),
            },
        )

    def _compute_participation(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> float:
        """Compute participation ratio for slippage calculation."""
        # Try ADV first
        if market.adv is not None and market.adv > 0:
            ref_price = market.get_mid_price() or bar.typical_price
            order_notional = order.get_notional(ref_price)
            return order_notional / market.adv

        # Fallback to bar volume
        if bar.volume is not None and bar.volume > 0:
            return order.qty / bar.volume

        # Default small participation
        return 0.001


class ZeroFeeProvider:
    """
    Fee provider returning zero fees.

    Useful as default/fallback or for testing.
    """

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """Return zero fee."""
        return 0.0


class CryptoFeeProvider:
    """
    L2: Crypto exchange fee provider (Binance-style).

    Implements tiered maker/taker fee structure common to
    cryptocurrency exchanges.

    Attributes:
        maker_bps: Maker fee in basis points (default: 2.0 = 0.02%)
        taker_bps: Taker fee in basis points (default: 4.0 = 0.04%)
        discount_rate: BNB discount rate (default: 0.75 = 25% off)
        use_discount: Whether to apply discount
    """

    def __init__(
        self,
        maker_bps: float = 2.0,
        taker_bps: float = 4.0,
        discount_rate: float = 0.75,
        use_discount: bool = False,
    ) -> None:
        """
        Initialize crypto fee provider.

        Args:
            maker_bps: Maker fee in basis points
            taker_bps: Taker fee in basis points
            discount_rate: Discount multiplier (e.g., 0.75 for 25% off)
            use_discount: Whether to apply discount (e.g., BNB payment)
        """
        self.maker_bps = float(maker_bps)
        self.taker_bps = float(taker_bps)
        self.discount_rate = float(discount_rate)
        self.use_discount = bool(use_discount)

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute trading fee based on liquidity role.

        Args:
            notional: Trade notional value
            side: Trade side (not used for crypto)
            liquidity: "maker" or "taker"
            qty: Trade quantity (not used for crypto)

        Returns:
            Fee amount in quote currency
        """
        liquidity_norm = str(liquidity).lower()

        if liquidity_norm == "maker":
            rate_bps = self.maker_bps
        else:
            rate_bps = self.taker_bps

        if self.use_discount:
            rate_bps *= self.discount_rate

        fee = abs(notional) * rate_bps / 10000.0
        return float(fee)


class EquityFeeProvider:
    """
    L2: US Equity fee provider (Alpaca-style).

    Implements commission-free trading with regulatory fees:
    - SEC fee: ~$0.0000278 per dollar of sale proceeds
    - TAF fee: $0.000166 per share sold (max $8.30)

    These fees only apply to SELL orders.

    Attributes:
        sec_fee_rate: SEC fee per dollar (sell only)
        taf_fee_rate: TAF fee per share (sell only)
        taf_max_fee: Maximum TAF fee per trade
        include_regulatory: Whether to include regulatory fees
    """

    # Regulatory fee rates (2024 values)
    SEC_FEE_RATE = 0.0000278
    TAF_FEE_RATE = 0.000166
    TAF_MAX_FEE = 8.30

    def __init__(
        self,
        sec_fee_rate: Optional[float] = None,
        taf_fee_rate: Optional[float] = None,
        taf_max_fee: Optional[float] = None,
        include_regulatory: bool = True,
    ) -> None:
        """
        Initialize equity fee provider.

        Args:
            sec_fee_rate: SEC fee per dollar (default: current rate)
            taf_fee_rate: TAF fee per share (default: current rate)
            taf_max_fee: Maximum TAF fee (default: $8.30)
            include_regulatory: Whether to include regulatory fees
        """
        self.sec_fee_rate = sec_fee_rate if sec_fee_rate is not None else self.SEC_FEE_RATE
        self.taf_fee_rate = taf_fee_rate if taf_fee_rate is not None else self.TAF_FEE_RATE
        self.taf_max_fee = taf_max_fee if taf_max_fee is not None else self.TAF_MAX_FEE
        self.include_regulatory = include_regulatory

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute trading fee (regulatory fees on sells).

        Alpaca is commission-free, but regulatory fees apply to sales.

        Args:
            notional: Trade notional value
            side: Trade side ("BUY" or "SELL")
            liquidity: Liquidity role (not used for equity)
            qty: Number of shares

        Returns:
            Fee amount in USD
        """
        # Commission-free for buys
        if str(side).upper() != "SELL":
            return 0.0

        if not self.include_regulatory:
            return 0.0

        fee = 0.0

        # SEC fee (on sale proceeds)
        sec_fee = abs(notional) * self.sec_fee_rate
        fee += sec_fee

        # TAF fee (per share sold)
        taf_fee = min(abs(qty) * self.taf_fee_rate, self.taf_max_fee)
        fee += taf_fee

        return round(fee, 4)

    def estimate_regulatory_breakdown(
        self,
        notional: float,
        qty: float,
    ) -> Dict[str, float]:
        """
        Get detailed regulatory fee breakdown.

        Args:
            notional: Trade notional value
            qty: Number of shares

        Returns:
            Dict with fee breakdown
        """
        sec_fee = abs(notional) * self.sec_fee_rate
        taf_fee = min(abs(qty) * self.taf_fee_rate, self.taf_max_fee)

        return {
            "sec_fee": round(sec_fee, 4),
            "taf_fee": round(taf_fee, 4),
            "total": round(sec_fee + taf_fee, 4),
        }


# =============================================================================
# L2 Combined Execution Provider
# =============================================================================

class L2ExecutionProvider:
    """
    L2: Combined execution provider using statistical models.

    Combines SlippageProvider, FillProvider, and FeeProvider into
    a complete execution simulation workflow.

    Supports both crypto and equity asset classes with appropriate
    default providers.

    Attributes:
        asset_class: Asset class (CRYPTO or EQUITY)
        slippage: SlippageProvider instance
        fill: FillProvider instance (typically OHLCVFillProvider)
        fees: FeeProvider instance
    """

    def __init__(
        self,
        asset_class: AssetClass = AssetClass.CRYPTO,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize L2 execution provider.

        Args:
            asset_class: Asset class to configure defaults for
            slippage_provider: Custom slippage provider
            fee_provider: Custom fee provider
            **kwargs: Additional configuration
        """
        self._asset_class = asset_class

        # Default slippage provider
        if slippage_provider is not None:
            self.slippage = slippage_provider
        else:
            # Asset-class specific defaults
            if asset_class == AssetClass.EQUITY:
                # Equities typically have tighter spreads
                self.slippage = StatisticalSlippageProvider(
                    impact_coef=0.05,
                    spread_bps=2.0,
                )
            else:
                # Crypto has wider spreads
                self.slippage = StatisticalSlippageProvider(
                    impact_coef=0.1,
                    spread_bps=5.0,
                )

        # Default fee provider
        if fee_provider is not None:
            self.fees = fee_provider
        else:
            if asset_class == AssetClass.EQUITY:
                self.fees = EquityFeeProvider()
            else:
                self.fees = CryptoFeeProvider()

        # Fill provider with injected slippage and fees
        self.fill = OHLCVFillProvider(
            slippage_provider=self.slippage,
            fee_provider=self.fees,
            fill_at_limit=kwargs.get("fill_at_limit", True),
            partial_fills=kwargs.get("partial_fills", False),
        )

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        return self._asset_class

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Execute an order with full slippage/fee calculation.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data

        Returns:
            Fill with slippage and fees, or None if not filled
        """
        return self.fill.try_fill(order, market, bar)

    def estimate_execution_cost(
        self,
        notional: float,
        adv: float,
        side: str = "BUY",
        volatility: float = 0.02,
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation.

        Useful for optimal execution planning.

        Args:
            notional: Planned trade notional
            adv: Average daily volume
            side: Trade side
            volatility: Expected volatility

        Returns:
            Dict with cost breakdown
        """
        # Participation
        participation = notional / adv if adv > 0 else 0.01

        # Slippage estimate
        dummy_order = Order("", side, 1.0, "MARKET")
        dummy_market = MarketState(0, volatility=volatility, adv=adv)
        slippage_bps = self.slippage.compute_slippage_bps(
            dummy_order, dummy_market, participation
        )
        slippage_cost = notional * slippage_bps / 10000.0

        # Fee estimate (taker worst case)
        fee = self.fees.compute_fee(
            notional=notional,
            side=side,
            liquidity="taker",
            qty=notional / 100.0,  # Approximate qty
        )

        return {
            "participation": participation,
            "slippage_bps": slippage_bps,
            "slippage_cost": slippage_cost,
            "fee": fee,
            "total_cost": slippage_cost + fee,
            "total_bps": (slippage_cost + fee) / notional * 10000.0 if notional > 0 else 0.0,
        }


# =============================================================================
# L3 Stubs (Future LOB-based implementations)
# =============================================================================

class LOBSlippageProvider:
    """
    L3: Order book based slippage provider (FUTURE).

    Will simulate walking through the order book to compute
    actual execution price based on available liquidity.

    Features (planned):
    - Walk-through LOB simulation
    - Queue position estimation
    - Hidden liquidity modeling
    - Cross-venue routing optimization

    Status: STUB - not yet implemented
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LOB slippage provider (stub)."""
        self._config = kwargs
        logger.warning(
            "LOBSlippageProvider is a stub. Use StatisticalSlippageProvider for now."
        )

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute slippage from order book walk-through.

        STUB: Falls back to simple spread-based estimate.
        """
        # Check if L3 data available
        if market.bid_depth is None or market.ask_depth is None:
            logger.debug("LOB depth not available, using spread estimate")
            spread = market.get_spread_bps() or 10.0
            return spread / 2.0

        # TODO: Implement LOB walk-through
        # For now, return spread-based estimate
        spread = market.get_spread_bps() or 10.0
        return spread / 2.0


class LOBFillProvider:
    """
    L3: Matching engine simulation (FUTURE).

    Will simulate a full matching engine with queue position
    tracking and time priority.

    Features (planned):
    - Time-price priority matching
    - Queue position tracking
    - Partial fill simulation
    - Hidden order modeling

    Status: STUB - not yet implemented
    """

    def __init__(
        self,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize LOB fill provider (stub)."""
        self.slippage = slippage_provider or StatisticalSlippageProvider()
        self.fees = fee_provider or ZeroFeeProvider()
        self._config = kwargs
        logger.warning(
            "LOBFillProvider is a stub. Use OHLCVFillProvider for now."
        )

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt fill with matching engine simulation.

        STUB: Delegates to OHLCV fill logic.
        """
        # Fall back to OHLCV fill provider
        ohlcv_fill = OHLCVFillProvider(
            slippage_provider=self.slippage,
            fee_provider=self.fees,
        )
        return ohlcv_fill.try_fill(order, market, bar)


# =============================================================================
# Factory Functions
# =============================================================================

def create_slippage_provider(
    level: str = "L2",
    asset_class: AssetClass = AssetClass.CRYPTO,
    **kwargs: Any,
) -> SlippageProvider:
    """
    Factory function to create slippage provider.

    Args:
        level: Fidelity level ("L1", "L2", "L3")
        asset_class: Asset class for defaults
        **kwargs: Provider-specific configuration

    Returns:
        SlippageProvider instance
    """
    level_upper = str(level).upper()

    if level_upper == "L3":
        return LOBSlippageProvider(**kwargs)

    # L1/L2: Statistical model
    if asset_class == AssetClass.EQUITY:
        defaults = {"impact_coef": 0.05, "spread_bps": 2.0}
    else:
        defaults = {"impact_coef": 0.1, "spread_bps": 5.0}

    defaults.update(kwargs)
    return StatisticalSlippageProvider(**defaults)


def create_fee_provider(
    asset_class: AssetClass = AssetClass.CRYPTO,
    **kwargs: Any,
) -> FeeProvider:
    """
    Factory function to create fee provider.

    Args:
        asset_class: Asset class for fee structure
        **kwargs: Provider-specific configuration

    Returns:
        FeeProvider instance
    """
    if asset_class == AssetClass.EQUITY:
        return EquityFeeProvider(**kwargs)
    elif asset_class == AssetClass.CRYPTO:
        return CryptoFeeProvider(**kwargs)
    else:
        return ZeroFeeProvider()


def create_fill_provider(
    level: str = "L2",
    asset_class: AssetClass = AssetClass.CRYPTO,
    slippage_provider: Optional[SlippageProvider] = None,
    fee_provider: Optional[FeeProvider] = None,
    **kwargs: Any,
) -> FillProvider:
    """
    Factory function to create fill provider.

    Args:
        level: Fidelity level ("L2", "L3")
        asset_class: Asset class for defaults
        slippage_provider: Custom slippage provider
        fee_provider: Custom fee provider
        **kwargs: Provider-specific configuration

    Returns:
        FillProvider instance
    """
    level_upper = str(level).upper()

    # Create default providers if not provided
    if slippage_provider is None:
        slippage_provider = create_slippage_provider(level, asset_class)
    if fee_provider is None:
        fee_provider = create_fee_provider(asset_class)

    if level_upper == "L3":
        return LOBFillProvider(
            slippage_provider=slippage_provider,
            fee_provider=fee_provider,
            **kwargs,
        )

    # L2: OHLCV-based fills
    return OHLCVFillProvider(
        slippage_provider=slippage_provider,
        fee_provider=fee_provider,
        **kwargs,
    )


def create_execution_provider(
    asset_class: AssetClass = AssetClass.CRYPTO,
    level: str = "L2",
    **kwargs: Any,
) -> ExecutionProvider:
    """
    Factory function to create combined execution provider.

    Args:
        asset_class: Asset class (CRYPTO or EQUITY)
        level: Fidelity level ("L2", "L3")
        **kwargs: Provider-specific configuration

    Returns:
        ExecutionProvider instance
    """
    return L2ExecutionProvider(
        asset_class=asset_class,
        slippage_provider=kwargs.pop("slippage_provider", None),
        fee_provider=kwargs.pop("fee_provider", None),
        **kwargs,
    )


# =============================================================================
# Backward Compatibility: Integration with existing execution_sim.py
# =============================================================================

def wrap_legacy_slippage_config(config: Any) -> StatisticalSlippageProvider:
    """
    Create SlippageProvider from legacy SlippageConfig.

    Provides backward compatibility with existing slippage configuration.

    Args:
        config: Legacy SlippageConfig or dict

    Returns:
        StatisticalSlippageProvider instance
    """
    if config is None:
        return StatisticalSlippageProvider()

    # Extract parameters from legacy config
    if hasattr(config, "k"):
        impact_coef = float(getattr(config, "k", 0.1))
    elif isinstance(config, Mapping) and "k" in config:
        impact_coef = float(config.get("k", 0.1))
    else:
        impact_coef = 0.1

    if hasattr(config, "default_spread_bps"):
        spread_bps = float(getattr(config, "default_spread_bps", 5.0))
    elif isinstance(config, Mapping) and "default_spread_bps" in config:
        spread_bps = float(config.get("default_spread_bps", 5.0))
    else:
        spread_bps = 5.0

    return StatisticalSlippageProvider(
        impact_coef=impact_coef,
        spread_bps=spread_bps,
    )


def wrap_legacy_fees_model(model: Any) -> FeeProvider:
    """
    Create FeeProvider from legacy FeesModel.

    Provides backward compatibility with existing fees configuration.

    Args:
        model: Legacy FeesModel or dict

    Returns:
        FeeProvider instance
    """
    if model is None:
        return CryptoFeeProvider()

    # Extract parameters from legacy model
    if hasattr(model, "maker_rate_bps"):
        maker_bps = float(getattr(model, "maker_rate_bps", 2.0))
    elif isinstance(model, Mapping) and "maker_rate_bps" in model:
        maker_bps = float(model.get("maker_rate_bps", 2.0))
    else:
        maker_bps = 2.0

    if hasattr(model, "taker_rate_bps"):
        taker_bps = float(getattr(model, "taker_rate_bps", 4.0))
    elif isinstance(model, Mapping) and "taker_rate_bps" in model:
        taker_bps = float(model.get("taker_rate_bps", 4.0))
    else:
        taker_bps = 4.0

    return CryptoFeeProvider(
        maker_bps=maker_bps,
        taker_bps=taker_bps,
    )
