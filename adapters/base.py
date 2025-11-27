# -*- coding: utf-8 -*-
"""
adapters/base.py
Abstract base classes for multi-exchange adapter framework.

This module defines the contracts (interfaces) that all exchange adapters must implement.
The design follows the Strategy pattern, allowing easy switching between exchanges
(Binance, Alpaca, etc.) without changing business logic.

Design Principles:
- ABC-based interfaces for clear contracts
- Async-first design with sync wrappers where needed
- Type hints for better IDE support and runtime checking
- Minimal coupling to exchange-specific details
- Compatible with existing core_contracts.py protocols

Architecture:
    BaseAdapter (abstract)
    ├── MarketDataAdapter      # OHLCV bars, ticks, real-time data
    ├── FeeAdapter             # Fee computation
    ├── TradingHoursAdapter    # Market schedule, holidays
    ├── OrderExecutionAdapter  # Order placement, fills, positions
    └── ExchangeInfoAdapter    # Symbols, filters, rules
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from core_models import Bar, Tick, Order, ExecReport, Position, Side, Liquidity

from .models import (
    AccountInfo,
    ExchangeRule,
    ExchangeVendor,
    FeeSchedule,
    MarketCalendar,
    MarketType,
    SessionType,
    SymbolInfo,
    TradingSession,
)


# =========================
# Type Variables
# =========================

T = TypeVar("T")
BarType = TypeVar("BarType", bound=Bar)


# =========================
# Base Adapter
# =========================

class BaseAdapter(ABC):
    """
    Base class for all exchange adapters.

    Provides common functionality:
    - Vendor identification
    - Connection state management
    - Configuration handling
    - Logging hooks

    Subclasses should implement specific adapter interfaces.
    """

    def __init__(
        self,
        vendor: ExchangeVendor,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize base adapter.

        Args:
            vendor: Exchange vendor identifier
            config: Adapter-specific configuration
        """
        self._vendor = vendor
        self._config = dict(config) if config else {}
        self._is_connected = False
        self._last_error: Optional[str] = None

    @property
    def vendor(self) -> ExchangeVendor:
        """Exchange vendor identifier."""
        return self._vendor

    @property
    def config(self) -> Dict[str, Any]:
        """Current configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Whether adapter is connected to exchange."""
        return self._is_connected

    @property
    def last_error(self) -> Optional[str]:
        """Last error message if any."""
        return self._last_error

    def connect(self) -> bool:
        """
        Establish connection to exchange.

        Returns:
            True if connection successful
        """
        try:
            self._do_connect()
            self._is_connected = True
            self._last_error = None
            return True
        except Exception as e:
            self._is_connected = False
            self._last_error = str(e)
            return False

    def disconnect(self) -> None:
        """Disconnect from exchange."""
        try:
            self._do_disconnect()
        finally:
            self._is_connected = False

    def _do_connect(self) -> None:
        """Implementation-specific connection logic. Override in subclass."""
        pass

    def _do_disconnect(self) -> None:
        """Implementation-specific disconnection logic. Override in subclass."""
        pass

    def __enter__(self) -> "BaseAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()


# =========================
# Market Data Adapter
# =========================

class MarketDataAdapter(BaseAdapter):
    """
    Abstract adapter for market data (OHLCV bars, ticks).

    Implementations:
    - BinanceMarketDataAdapter: Binance REST/WebSocket
    - AlpacaMarketDataAdapter: Alpaca REST/WebSocket

    Usage:
        adapter = BinanceMarketDataAdapter(config)
        bars = adapter.get_bars("BTCUSDT", "1h", limit=100)
        for bar in adapter.stream_bars(["BTCUSDT"], interval_ms=3600000):
            process(bar)
    """

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Fetch historical OHLCV bars.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d")
            limit: Maximum number of bars to return
            start_ts: Start timestamp in milliseconds (inclusive)
            end_ts: End timestamp in milliseconds (exclusive)

        Returns:
            List of Bar objects, oldest first
        """
        ...

    @abstractmethod
    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Latest Bar or None if unavailable
        """
        ...

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get current tick (BBO/last price) for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current Tick or None if unavailable
        """
        ...

    @abstractmethod
    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Stream real-time bars.

        Args:
            symbols: List of symbols to stream
            interval_ms: Bar interval in milliseconds

        Yields:
            Bar objects as they become available
        """
        ...

    @abstractmethod
    def stream_ticks(
        self,
        symbols: Sequence[str],
    ) -> Iterator[Tick]:
        """
        Stream real-time ticks (BBO updates).

        Args:
            symbols: List of symbols to stream

        Yields:
            Tick objects as they become available
        """
        ...

    def get_bars_multi(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Dict[str, List[Bar]]:
        """
        Fetch historical bars for multiple symbols.

        Default implementation calls get_bars sequentially.
        Override for parallel/batch fetching.

        Args:
            symbols: List of trading symbols
            timeframe: Bar timeframe
            limit: Maximum bars per symbol
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            Dict mapping symbol to list of bars
        """
        result: Dict[str, List[Bar]] = {}
        for symbol in symbols:
            result[symbol] = self.get_bars(
                symbol, timeframe, limit=limit, start_ts=start_ts, end_ts=end_ts
            )
        return result

    def get_ticks_multi(self, symbols: Sequence[str]) -> Dict[str, Optional[Tick]]:
        """
        Get current ticks for multiple symbols.

        Default implementation calls get_tick sequentially.
        Override for parallel/batch fetching.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbol to Tick (or None)
        """
        return {symbol: self.get_tick(symbol) for symbol in symbols}


# =========================
# Fee Adapter
# =========================

class FeeAdapter(BaseAdapter):
    """
    Abstract adapter for fee computation.

    Implementations handle exchange-specific fee structures:
    - Binance: Percentage-based with BNB discount, VIP tiers
    - Alpaca: Commission-free for stocks (some fees for options)

    Usage:
        adapter = BinanceFeeAdapter(config)
        fee = adapter.compute_fee(
            notional=1000.0,
            side=Side.BUY,
            liquidity="taker",
            symbol="BTCUSDT"
        )
    """

    @abstractmethod
    def compute_fee(
        self,
        notional: float,
        side: Side,
        liquidity: Union[str, Liquidity],
        *,
        symbol: Optional[str] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        """
        Compute trading fee for a trade.

        Args:
            notional: Trade value (price * quantity)
            side: Trade direction (BUY/SELL)
            liquidity: "maker" or "taker"
            symbol: Trading symbol (for symbol-specific fees)
            qty: Trade quantity (for per-share fees)
            price: Trade price (optional)

        Returns:
            Fee amount in quote currency
        """
        ...

    @abstractmethod
    def get_fee_schedule(self, symbol: Optional[str] = None) -> FeeSchedule:
        """
        Get fee schedule for symbol or exchange default.

        Args:
            symbol: Optional symbol for symbol-specific fees

        Returns:
            FeeSchedule with fee rates and structure
        """
        ...

    @abstractmethod
    def get_effective_rates(
        self,
        symbol: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Get effective maker/taker rates in basis points.

        Args:
            symbol: Optional symbol

        Returns:
            (maker_bps, taker_bps)
        """
        ...

    def expected_fee(
        self,
        notional: float,
        *,
        p_maker: float = 0.0,
        symbol: Optional[str] = None,
    ) -> float:
        """
        Compute expected fee given maker probability.

        Args:
            notional: Trade notional value
            p_maker: Probability of maker fill (0-1)
            symbol: Trading symbol

        Returns:
            Expected fee amount
        """
        maker_bps, taker_bps = self.get_effective_rates(symbol)
        expected_bps = p_maker * maker_bps + (1 - p_maker) * taker_bps
        return notional * expected_bps / 10_000


# =========================
# Trading Hours Adapter
# =========================

class TradingHoursAdapter(BaseAdapter):
    """
    Abstract adapter for trading hours and market schedule.

    Handles:
    - Market open/close times
    - Pre-market/after-hours sessions
    - Holidays and half-days
    - Timezone conversions

    Critical for:
    - Order timing (avoid submission during closed hours)
    - Backtest accuracy (proper handling of market hours)
    - Live trading decisions

    Usage:
        adapter = AlpacaTradingHoursAdapter(config)
        if adapter.is_market_open(current_ts):
            submit_order()
        else:
            next_open = adapter.next_open(current_ts)
    """

    @abstractmethod
    def is_market_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> bool:
        """
        Check if market is open at given timestamp.

        Args:
            ts: Unix timestamp in milliseconds
            session_type: Specific session to check (None = any tradable session)

        Returns:
            True if market is open
        """
        ...

    @abstractmethod
    def next_open(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market open.

        Args:
            ts: Current timestamp in milliseconds
            session_type: Specific session type (None = regular)

        Returns:
            Timestamp of next open in milliseconds
        """
        ...

    @abstractmethod
    def next_close(
        self,
        ts: int,
        *,
        session_type: Optional[SessionType] = None,
    ) -> int:
        """
        Get timestamp of next market close.

        Args:
            ts: Current timestamp in milliseconds
            session_type: Specific session type (None = regular)

        Returns:
            Timestamp of next close in milliseconds
        """
        ...

    @abstractmethod
    def get_calendar(self) -> MarketCalendar:
        """
        Get market calendar with sessions and holidays.

        Returns:
            MarketCalendar object
        """
        ...

    @abstractmethod
    def is_holiday(self, ts: int) -> bool:
        """
        Check if date is a market holiday.

        Args:
            ts: Timestamp in milliseconds

        Returns:
            True if date is a holiday
        """
        ...

    def get_session(self, ts: int) -> Optional[TradingSession]:
        """
        Get current trading session at timestamp.

        Args:
            ts: Timestamp in milliseconds

        Returns:
            Current TradingSession or None if market closed
        """
        calendar = self.get_calendar()
        for session in calendar.sessions:
            if self.is_market_open(ts, session_type=session.session_type):
                return session
        return None

    def time_to_close(self, ts: int) -> Optional[int]:
        """
        Get milliseconds until market close.

        Args:
            ts: Current timestamp

        Returns:
            Milliseconds until close, or None if market closed
        """
        if not self.is_market_open(ts):
            return None
        return self.next_close(ts) - ts

    def time_to_open(self, ts: int) -> Optional[int]:
        """
        Get milliseconds until market open.

        Args:
            ts: Current timestamp

        Returns:
            Milliseconds until open, or None if market already open
        """
        if self.is_market_open(ts):
            return 0
        return self.next_open(ts) - ts


# =========================
# Order Execution Adapter
# =========================

@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: str = ""
    filled_qty: Decimal = Decimal("0")
    filled_price: Optional[Decimal] = None
    fee: Decimal = Decimal("0")
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.raw_response is None:
            self.raw_response = {}


class OrderExecutionAdapter(BaseAdapter):
    """
    Abstract adapter for order execution.

    Handles:
    - Order submission (market, limit)
    - Order cancellation
    - Position tracking
    - Fill notifications

    Usage:
        adapter = AlpacaOrderExecutionAdapter(config, api_key, secret)
        result = adapter.submit_order(order)
        if result.success:
            print(f"Order {result.order_id} submitted")
    """

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order to exchange.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status and fill info
        """
        ...

    @abstractmethod
    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        ...

    @abstractmethod
    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            ExecReport with current status, or None if not found
        """
        ...

    @abstractmethod
    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        ...

    @abstractmethod
    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Position]:
        """
        Get current positions.

        Args:
            symbols: Filter by symbols (optional, None = all)

        Returns:
            Dict mapping symbol to Position
        """
        ...

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status
        """
        ...

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            Number of orders cancelled
        """
        orders = self.get_open_orders(symbol)
        cancelled = 0
        for order in orders:
            if self.cancel_order(client_order_id=order.client_order_id):
                cancelled += 1
        return cancelled


# =========================
# Exchange Info Adapter
# =========================

class ExchangeInfoAdapter(BaseAdapter):
    """
    Abstract adapter for exchange metadata and symbol info.

    Handles:
    - Symbol listing and filtering
    - Trading rules (tick size, lot size, min notional)
    - Exchange status

    Usage:
        adapter = BinanceExchangeInfoAdapter(config)
        symbols = adapter.get_symbols(filters={"quote_asset": "USDT"})
        rules = adapter.get_exchange_rules("BTCUSDT")
    """

    @abstractmethod
    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable symbols.

        Args:
            filters: Optional filters:
                - quote_asset: Filter by quote currency
                - base_asset: Filter by base currency
                - market_type: Filter by market type
                - is_tradable: Only tradable symbols
                - min_volume: Minimum 24h volume

        Returns:
            List of symbol strings
        """
        ...

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information.

        Args:
            symbol: Trading symbol

        Returns:
            SymbolInfo or None if symbol not found
        """
        ...

    @abstractmethod
    def get_exchange_rules(self, symbol: str) -> Optional[ExchangeRule]:
        """
        Get trading rules for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            ExchangeRule or None if symbol not found
        """
        ...

    @abstractmethod
    def refresh(self) -> bool:
        """
        Refresh cached exchange info.

        Returns:
            True if refresh successful
        """
        ...

    def get_symbols_info(
        self,
        symbols: Sequence[str],
    ) -> Dict[str, Optional[SymbolInfo]]:
        """
        Get info for multiple symbols.

        Default implementation calls get_symbol_info sequentially.
        Override for batch fetching.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to SymbolInfo (or None)
        """
        return {symbol: self.get_symbol_info(symbol) for symbol in symbols}

    def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol is valid and tradable.

        Args:
            symbol: Trading symbol

        Returns:
            (is_valid, error_message)
        """
        info = self.get_symbol_info(symbol)
        if info is None:
            return False, f"Symbol {symbol} not found"
        if not info.is_tradable:
            return False, f"Symbol {symbol} is not tradable"
        return True, None


# =========================
# Combined Exchange Adapter
# =========================

class ExchangeAdapter(
    MarketDataAdapter,
    FeeAdapter,
    TradingHoursAdapter,
    OrderExecutionAdapter,
    ExchangeInfoAdapter,
):
    """
    Combined adapter implementing all exchange interfaces.

    This is a convenience class for exchanges that want to provide
    all functionality in a single adapter. Individual adapters can
    also be used separately for more modular design.

    Usage:
        adapter = BinanceExchangeAdapter(config, api_key, secret)
        # Use any interface methods
        bars = adapter.get_bars("BTCUSDT", "1h")
        fee = adapter.compute_fee(1000.0, Side.BUY, "taker")
        adapter.submit_order(order)
    """

    pass


# =========================
# Protocols for Duck Typing
# =========================

@runtime_checkable
class SupportsMarketData(Protocol):
    """Protocol for classes supporting market data."""

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
    ) -> List[Bar]:
        ...

    def stream_bars(
        self,
        symbols: Sequence[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        ...


@runtime_checkable
class SupportsFees(Protocol):
    """Protocol for classes supporting fee computation."""

    def compute_fee(
        self,
        notional: float,
        side: Side,
        liquidity: Union[str, Liquidity],
    ) -> float:
        ...


@runtime_checkable
class SupportsTradingHours(Protocol):
    """Protocol for classes supporting trading hours."""

    def is_market_open(self, ts: int) -> bool:
        ...

    def next_open(self, ts: int) -> int:
        ...


# =========================
# Factory Protocol
# =========================

class AdapterFactory(Protocol):
    """Protocol for adapter factories."""

    def create_market_data_adapter(
        self,
        vendor: ExchangeVendor,
        config: Optional[Mapping[str, Any]] = None,
    ) -> MarketDataAdapter:
        ...

    def create_fee_adapter(
        self,
        vendor: ExchangeVendor,
        config: Optional[Mapping[str, Any]] = None,
    ) -> FeeAdapter:
        ...

    def create_trading_hours_adapter(
        self,
        vendor: ExchangeVendor,
        config: Optional[Mapping[str, Any]] = None,
    ) -> TradingHoursAdapter:
        ...

    def create_order_execution_adapter(
        self,
        vendor: ExchangeVendor,
        config: Optional[Mapping[str, Any]] = None,
    ) -> OrderExecutionAdapter:
        ...

    def create_exchange_info_adapter(
        self,
        vendor: ExchangeVendor,
        config: Optional[Mapping[str, Any]] = None,
    ) -> ExchangeInfoAdapter:
        ...
