# -*- coding: utf-8 -*-
"""
LOB Data Adapters for L3 Simulation.

Provides unified interface for loading order book data from various formats:
- LOBSTER: Academic format from NASDAQ OMX
- ITCH: NASDAQ TotalView-ITCH 5.0 binary format
- Binance: Cryptocurrency depth stream
- Alpaca: US equities market data

Each adapter converts vendor-specific data into OrderBook structure
for use in L3 simulation.

Architecture:
    Raw Data → Adapter → LOBSnapshot / OrderBook → L3 Simulation

Note:
    Crypto adapters (Binance) return L2-level data (aggregated by price).
    LOBSTER/ITCH adapters return L3-level data (individual orders).
    Alpaca returns L2 quotes + trades.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

from lob.data_structures import (
    LimitOrder,
    OrderBook,
    OrderType,
    PriceLevel,
    Side,
)
from lob.parsers import (
    ITCHAddOrder,
    ITCHMessage,
    ITCHOrderCancel,
    ITCHOrderDelete,
    ITCHOrderExecuted,
    ITCHOrderReplace,
    ITCHParser,
    LOBSTEREventType,
    LOBSTERMessage,
    LOBSTERParser,
)
from lob.state_manager import LOBStateManager

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures
# ==============================================================================


class DataSourceType(Enum):
    """Type of LOB data source."""

    LOBSTER = "lobster"
    ITCH = "itch"
    BINANCE = "binance"
    ALPACA = "alpaca"
    INTERNAL = "internal"
    SYNTHETIC = "synthetic"


@dataclass
class LOBUpdate:
    """
    Single LOB update event.

    Represents any change to the order book:
    - ADD: New order added
    - MODIFY: Existing order modified (cancel + replace)
    - DELETE: Order cancelled
    - EXECUTE: Order (partially) filled
    """

    timestamp_ns: int
    update_type: str  # "ADD", "MODIFY", "DELETE", "EXECUTE"
    order_id: str
    price: float
    qty: float
    side: Side
    remaining_qty: Optional[float] = None
    fill_qty: Optional[float] = None
    source: DataSourceType = DataSourceType.INTERNAL


@dataclass
class DepthLevel:
    """
    Aggregated depth at a single price level (L2 data).

    Used for Binance/Alpaca which provide aggregated data.
    """

    price: float
    qty: float
    order_count: int = 1


@dataclass
class DepthSnapshot:
    """
    Full depth snapshot (L2 data).

    Contains aggregated bids and asks at each price level.
    """

    timestamp_ns: int
    symbol: str
    bids: List[DepthLevel] = field(default_factory=list)
    asks: List[DepthLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return self.best_bid or self.best_ask

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        if self.best_bid and self.best_ask and self.best_bid > 0:
            return (self.best_ask - self.best_bid) / self.best_bid * 10000
        return None


@dataclass
class AdapterStats:
    """Statistics for data adapter operations."""

    messages_processed: int = 0
    messages_failed: int = 0
    orders_added: int = 0
    orders_cancelled: int = 0
    orders_executed: int = 0
    total_volume: float = 0.0
    start_time_ns: int = 0
    end_time_ns: int = 0

    @property
    def duration_sec(self) -> float:
        """Duration of data in seconds."""
        if self.end_time_ns > self.start_time_ns:
            return (self.end_time_ns - self.start_time_ns) / 1e9
        return 0.0

    @property
    def error_rate(self) -> float:
        """Error rate."""
        total = self.messages_processed + self.messages_failed
        return self.messages_failed / total if total > 0 else 0.0


# ==============================================================================
# Base Adapter
# ==============================================================================


class BaseLOBAdapter(ABC):
    """
    Base class for LOB data adapters.

    Provides common interface for loading order book data
    from various sources into OrderBook structures.
    """

    def __init__(
        self,
        source_type: DataSourceType,
        symbol: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize adapter.

        Args:
            source_type: Type of data source
            symbol: Trading symbol
            config: Optional configuration
        """
        self._source_type = source_type
        self._symbol = symbol
        self._config = dict(config) if config else {}
        self._stats = AdapterStats()

    @property
    def source_type(self) -> DataSourceType:
        """Get data source type."""
        return self._source_type

    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self._symbol

    @property
    def stats(self) -> AdapterStats:
        """Get adapter statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = AdapterStats()

    @abstractmethod
    def load_snapshot(self, source: Any) -> DepthSnapshot:
        """
        Load order book snapshot from source.

        Args:
            source: Source-specific data (filepath, dict, etc.)

        Returns:
            DepthSnapshot containing order book state
        """
        pass

    @abstractmethod
    def stream_updates(self, source: Any) -> Iterator[LOBUpdate]:
        """
        Stream order book updates from source.

        Args:
            source: Source-specific data

        Yields:
            LOBUpdate objects
        """
        pass


# ==============================================================================
# LOBSTER Adapter
# ==============================================================================


class LOBSTERAdapter(BaseLOBAdapter):
    """
    Adapter for LOBSTER format data.

    LOBSTER is a standard academic format from NASDAQ OMX,
    commonly used in market microstructure research.

    Message file format:
        time, type, order_id, size, price, direction

    Orderbook file format:
        ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...

    Reference:
        https://lobsterdata.com/info/DataStructure.php
    """

    def __init__(
        self,
        symbol: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize LOBSTER adapter.

        Args:
            symbol: Trading symbol
            config: Configuration options:
                - price_multiplier: Price conversion factor (default: 0.0001)
                - delimiter: CSV delimiter (default: ",")
                - skip_header: Skip first line (default: False)
        """
        super().__init__(DataSourceType.LOBSTER, symbol, config)

        self._parser = LOBSTERParser(
            price_multiplier=self._config.get("price_multiplier", 0.0001),
            delimiter=self._config.get("delimiter", ","),
        )
        self._skip_header = self._config.get("skip_header", False)
        self._state_manager = LOBStateManager(symbol)

    def parse_message_file(
        self,
        path: Union[str, Path],
        max_messages: Optional[int] = None,
    ) -> Iterator[LOBSTERMessage]:
        """
        Parse LOBSTER message file.

        Args:
            path: Path to message file
            max_messages: Maximum messages to parse

        Yields:
            LOBSTERMessage objects
        """
        yield from self._parser.parse_file(
            path,
            skip_header=self._skip_header,
            max_messages=max_messages,
        )

    def parse_orderbook_file(
        self,
        path: Union[str, Path],
        n_levels: int = 10,
    ) -> LOBSnapshot:
        """
        Parse LOBSTER orderbook snapshot file.

        Args:
            path: Path to orderbook file
            n_levels: Number of price levels to parse

        Returns:
            LOBSnapshot containing order book state
        """
        asks, bids = self._parser.parse_orderbook_file(path, n_levels)

        # Build OrderBook from snapshot
        book = OrderBook()
        timestamp_ns = 0

        for i, (ask_price, ask_size) in enumerate(asks):
            order = LimitOrder(
                order_id=f"ask_{i}",
                price=ask_price,
                qty=float(ask_size),
                remaining_qty=float(ask_size),
                timestamp_ns=timestamp_ns,
                side=Side.SELL,
            )
            book.add_limit_order(order)

        for i, (bid_price, bid_size) in enumerate(bids):
            order = LimitOrder(
                order_id=f"bid_{i}",
                price=bid_price,
                qty=float(bid_size),
                remaining_qty=float(bid_size),
                timestamp_ns=timestamp_ns,
                side=Side.BUY,
            )
            book.add_limit_order(order)

        return DepthSnapshot(
            timestamp_ns=timestamp_ns,
            symbol=self._symbol,
            bids=[
                DepthLevel(price=b[0], qty=float(b[1]), order_count=1)
                for b in bids
            ],
            asks=[
                DepthLevel(price=a[0], qty=float(a[1]), order_count=1)
                for a in asks
            ],
        )

    def load_snapshot(
        self,
        source: Union[str, Path],
    ) -> DepthSnapshot:
        """
        Load order book snapshot from file.

        Args:
            source: Path to orderbook file

        Returns:
            DepthSnapshot
        """
        return self.parse_orderbook_file(source)

    def stream_updates(
        self,
        source: Union[str, Path],
        max_messages: Optional[int] = None,
    ) -> Iterator[LOBUpdate]:
        """
        Stream order book updates from message file.

        Args:
            source: Path to message file
            max_messages: Maximum messages to stream

        Yields:
            LOBUpdate objects
        """
        for msg in self.parse_message_file(source, max_messages):
            self._stats.messages_processed += 1

            update = self._message_to_update(msg)
            if update:
                if self._stats.start_time_ns == 0:
                    self._stats.start_time_ns = update.timestamp_ns
                self._stats.end_time_ns = update.timestamp_ns

                if update.update_type == "ADD":
                    self._stats.orders_added += 1
                elif update.update_type == "DELETE":
                    self._stats.orders_cancelled += 1
                elif update.update_type == "EXECUTE":
                    self._stats.orders_executed += 1
                    self._stats.total_volume += update.fill_qty or 0.0

                yield update

    def _message_to_update(self, msg: LOBSTERMessage) -> Optional[LOBUpdate]:
        """Convert LOBSTER message to LOBUpdate."""
        update_type_map = {
            LOBSTEREventType.ADD: "ADD",
            LOBSTEREventType.MODIFY: "MODIFY",
            LOBSTEREventType.DELETE: "DELETE",
            LOBSTEREventType.EXECUTE: "EXECUTE",
            LOBSTEREventType.HIDDEN_EXECUTE: "EXECUTE",
        }

        update_type = update_type_map.get(msg.event_type)
        if update_type is None:
            return None

        return LOBUpdate(
            timestamp_ns=msg.timestamp_ns,
            update_type=update_type,
            order_id=str(msg.order_id),
            price=msg.price,
            qty=float(msg.size),
            side=msg.side,
            fill_qty=float(msg.size) if msg.is_execute else None,
            source=DataSourceType.LOBSTER,
        )

    def build_orderbook_from_messages(
        self,
        message_path: Union[str, Path],
        orderbook_path: Optional[Union[str, Path]] = None,
        max_messages: Optional[int] = None,
    ) -> OrderBook:
        """
        Build OrderBook by replaying messages.

        Args:
            message_path: Path to message file
            orderbook_path: Optional initial orderbook file
            max_messages: Maximum messages to process

        Returns:
            OrderBook after processing all messages
        """
        book = OrderBook()

        # Load initial state if provided
        if orderbook_path:
            snapshot = self.parse_orderbook_file(orderbook_path)
            # Reconstruct book from snapshot
            for level in snapshot.bids:
                order = LimitOrder(
                    order_id=f"init_bid_{level.price}",
                    price=level.price,
                    qty=level.qty,
                    remaining_qty=level.qty,
                    timestamp_ns=snapshot.timestamp_ns,
                    side=Side.BUY,
                )
                book.add_limit_order(order)
            for level in snapshot.asks:
                order = LimitOrder(
                    order_id=f"init_ask_{level.price}",
                    price=level.price,
                    qty=level.qty,
                    remaining_qty=level.qty,
                    timestamp_ns=snapshot.timestamp_ns,
                    side=Side.SELL,
                )
                book.add_limit_order(order)

        # Process messages
        for msg in self.parse_message_file(message_path, max_messages):
            if msg.is_add:
                order = msg.to_limit_order()
                book.add_limit_order(order)
            elif msg.is_delete:
                book.cancel_order(str(msg.order_id))
            elif msg.is_execute:
                # Execute against order - get order, fill it, cancel if fully filled
                order = book.get_order(str(msg.order_id))
                if order:
                    order.fill(float(msg.size))
                    if order.is_filled:
                        book.cancel_order(str(msg.order_id))
            elif msg.is_modify:
                # Modify is cancel + replace
                book.cancel_order(str(msg.order_id))
                order = msg.to_limit_order()
                book.add_limit_order(order)

        return book


# ==============================================================================
# ITCH Adapter
# ==============================================================================


class ITCHAdapter(BaseLOBAdapter):
    """
    Adapter for NASDAQ ITCH format.

    ITCH is the primary market data format for NASDAQ,
    providing order-by-order updates at nanosecond resolution.

    Reference:
        https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf
    """

    def __init__(
        self,
        symbol: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize ITCH adapter.

        Args:
            symbol: Trading symbol to filter (empty = all)
            config: Configuration options:
                - price_divisor: Price conversion divisor (default: 10000)
        """
        super().__init__(DataSourceType.ITCH, symbol, config)

        self._parser = ITCHParser(
            price_divisor=self._config.get("price_divisor", 10000.0),
        )
        self._order_map: Dict[int, LimitOrder] = {}

    def parse_file(
        self,
        path: Union[str, Path],
        max_messages: Optional[int] = None,
    ) -> Iterator[ITCHMessage]:
        """
        Parse ITCH binary file.

        Args:
            path: Path to ITCH file
            max_messages: Maximum messages to parse

        Yields:
            ITCHMessage objects
        """
        yield from self._parser.parse_file(path, max_messages)

    def load_snapshot(self, source: Any) -> DepthSnapshot:
        """
        Load snapshot from ITCH data.

        Note: ITCH doesn't have snapshot format, must replay messages.

        Args:
            source: Path to ITCH file

        Returns:
            DepthSnapshot (empty, use stream_updates + build_orderbook)
        """
        return DepthSnapshot(
            timestamp_ns=0,
            symbol=self._symbol,
            bids=[],
            asks=[],
        )

    def stream_updates(
        self,
        source: Union[str, Path],
        max_messages: Optional[int] = None,
    ) -> Iterator[LOBUpdate]:
        """
        Stream order book updates from ITCH file.

        Args:
            source: Path to ITCH file
            max_messages: Maximum messages

        Yields:
            LOBUpdate objects
        """
        for msg in self.parse_file(source, max_messages):
            self._stats.messages_processed += 1

            update = self._message_to_update(msg)
            if update:
                # Filter by symbol if specified
                if self._symbol and isinstance(msg, ITCHAddOrder):
                    if msg.stock.strip() != self._symbol:
                        continue

                if self._stats.start_time_ns == 0:
                    self._stats.start_time_ns = update.timestamp_ns
                self._stats.end_time_ns = update.timestamp_ns

                yield update

    def _message_to_update(self, msg: ITCHMessage) -> Optional[LOBUpdate]:
        """Convert ITCH message to LOBUpdate."""
        if isinstance(msg, ITCHAddOrder):
            side = Side.BUY if msg.side == "B" else Side.SELL
            self._stats.orders_added += 1
            return LOBUpdate(
                timestamp_ns=msg.timestamp_ns,
                update_type="ADD",
                order_id=str(msg.order_ref),
                price=msg.price,
                qty=float(msg.shares),
                side=side,
                source=DataSourceType.ITCH,
            )

        elif isinstance(msg, ITCHOrderDelete):
            self._stats.orders_cancelled += 1
            return LOBUpdate(
                timestamp_ns=msg.timestamp_ns,
                update_type="DELETE",
                order_id=str(msg.order_ref),
                price=0.0,  # Unknown from delete message
                qty=0.0,
                side=Side.BUY,  # Unknown, will be looked up
                source=DataSourceType.ITCH,
            )

        elif isinstance(msg, ITCHOrderExecuted):
            self._stats.orders_executed += 1
            self._stats.total_volume += float(msg.shares)
            return LOBUpdate(
                timestamp_ns=msg.timestamp_ns,
                update_type="EXECUTE",
                order_id=str(msg.order_ref),
                price=msg.price or 0.0,
                qty=float(msg.shares),
                side=Side.BUY,  # Will be looked up
                fill_qty=float(msg.shares),
                source=DataSourceType.ITCH,
            )

        elif isinstance(msg, ITCHOrderCancel):
            return LOBUpdate(
                timestamp_ns=msg.timestamp_ns,
                update_type="MODIFY",  # Partial cancel
                order_id=str(msg.order_ref),
                price=0.0,
                qty=float(msg.cancelled_shares),
                side=Side.BUY,
                source=DataSourceType.ITCH,
            )

        elif isinstance(msg, ITCHOrderReplace):
            return LOBUpdate(
                timestamp_ns=msg.timestamp_ns,
                update_type="MODIFY",
                order_id=str(msg.new_ref),
                price=msg.price,
                qty=float(msg.shares),
                side=Side.BUY,
                source=DataSourceType.ITCH,
            )

        return None


# ==============================================================================
# Binance L2 Adapter
# ==============================================================================


class BinanceL2Adapter(BaseLOBAdapter):
    """
    Adapter for Binance depth stream.

    Binance provides L2-level data (aggregated by price level)
    via REST API and WebSocket streams.

    Data format:
        {
            "lastUpdateId": 123456,
            "bids": [["10000.00", "1.5"], ...],
            "asks": [["10001.00", "2.0"], ...]
        }

    Reference:
        https://binance-docs.github.io/apidocs/spot/en/#order-book
    """

    def __init__(
        self,
        symbol: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize Binance adapter.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            config: Configuration options:
                - base_url: API base URL
                - depth_limit: Number of depth levels (5, 10, 20, 50, 100, 500, 1000)
        """
        super().__init__(DataSourceType.BINANCE, symbol, config)

        self._depth_limit = self._config.get("depth_limit", 20)
        self._last_update_id = 0

    def load_snapshot(
        self,
        source: Union[Dict[str, Any], str],
    ) -> DepthSnapshot:
        """
        Load depth snapshot from Binance data.

        Args:
            source: Dict from Binance API or path to JSON file

        Returns:
            DepthSnapshot
        """
        import time

        if isinstance(source, str):
            import json
            with open(source, "r") as f:
                data = json.load(f)
        else:
            data = source

        timestamp_ns = int(time.time() * 1e9)
        self._last_update_id = data.get("lastUpdateId", 0)

        bids = [
            DepthLevel(price=float(b[0]), qty=float(b[1]), order_count=1)
            for b in data.get("bids", [])
        ]
        asks = [
            DepthLevel(price=float(a[0]), qty=float(a[1]), order_count=1)
            for a in data.get("asks", [])
        ]

        return DepthSnapshot(
            timestamp_ns=timestamp_ns,
            symbol=self._symbol,
            bids=bids,
            asks=asks,
        )

    def stream_updates(
        self,
        source: Any,
    ) -> Iterator[LOBUpdate]:
        """
        Stream depth updates from Binance.

        Args:
            source: Iterator of Binance depth update dicts

        Yields:
            LOBUpdate objects
        """
        import time

        for update_data in source:
            self._stats.messages_processed += 1

            timestamp_ns = int(update_data.get("E", time.time() * 1000) * 1e6)

            if self._stats.start_time_ns == 0:
                self._stats.start_time_ns = timestamp_ns
            self._stats.end_time_ns = timestamp_ns

            # Process bid updates
            for bid in update_data.get("b", []):
                price = float(bid[0])
                qty = float(bid[1])

                update_type = "DELETE" if qty == 0 else "ADD"
                yield LOBUpdate(
                    timestamp_ns=timestamp_ns,
                    update_type=update_type,
                    order_id=f"bid_{price}",
                    price=price,
                    qty=qty,
                    side=Side.BUY,
                    source=DataSourceType.BINANCE,
                )

            # Process ask updates
            for ask in update_data.get("a", []):
                price = float(ask[0])
                qty = float(ask[1])

                update_type = "DELETE" if qty == 0 else "ADD"
                yield LOBUpdate(
                    timestamp_ns=timestamp_ns,
                    update_type=update_type,
                    order_id=f"ask_{price}",
                    price=price,
                    qty=qty,
                    side=Side.SELL,
                    source=DataSourceType.BINANCE,
                )

    def depth_to_orderbook(
        self,
        depth_data: Dict[str, Any],
        timestamp_ns: Optional[int] = None,
    ) -> OrderBook:
        """
        Convert Binance depth snapshot to OrderBook.

        Note: This creates synthetic orders since Binance only
        provides aggregated L2 data, not individual orders.

        Args:
            depth_data: Binance depth API response
            timestamp_ns: Optional timestamp

        Returns:
            OrderBook with synthetic orders
        """
        import time

        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)

        book = OrderBook()

        # Add bid orders
        for i, (price_str, qty_str) in enumerate(depth_data.get("bids", [])):
            price = float(price_str)
            qty = float(qty_str)
            order = LimitOrder(
                order_id=f"bid_{i}_{price}",
                price=price,
                qty=qty,
                remaining_qty=qty,
                timestamp_ns=timestamp_ns,
                side=Side.BUY,
            )
            book.add_limit_order(order)

        # Add ask orders
        for i, (price_str, qty_str) in enumerate(depth_data.get("asks", [])):
            price = float(price_str)
            qty = float(qty_str)
            order = LimitOrder(
                order_id=f"ask_{i}_{price}",
                price=price,
                qty=qty,
                remaining_qty=qty,
                timestamp_ns=timestamp_ns,
                side=Side.SELL,
            )
            book.add_limit_order(order)

        return book


# ==============================================================================
# Alpaca L2 Adapter
# ==============================================================================


class AlpacaL2Adapter(BaseLOBAdapter):
    """
    Adapter for Alpaca market data.

    Alpaca provides:
    - Quotes (BBO) via REST and WebSocket
    - Trades via REST and WebSocket
    - NBBO data from IEX or SIP

    Reference:
        https://alpaca.markets/docs/api-references/market-data-api/
    """

    def __init__(
        self,
        symbol: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize Alpaca adapter.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            config: Configuration options:
                - api_key: Alpaca API key
                - api_secret: Alpaca API secret
                - feed: "iex" or "sip"
        """
        super().__init__(DataSourceType.ALPACA, symbol, config)

        self._api_key = self._config.get("api_key", "")
        self._api_secret = self._config.get("api_secret", "")
        self._feed = self._config.get("feed", "iex")

    def load_snapshot(
        self,
        source: Union[Dict[str, Any], str],
    ) -> DepthSnapshot:
        """
        Load snapshot from Alpaca quote data.

        Args:
            source: Quote dict or path to JSON file

        Returns:
            DepthSnapshot (only top-of-book for Alpaca)
        """
        import time

        if isinstance(source, str):
            import json
            with open(source, "r") as f:
                data = json.load(f)
        else:
            data = source

        timestamp_ns = int(time.time() * 1e9)

        # Alpaca provides only top-of-book
        # Support both full keys (bid_price) and short keys (bp)
        bids = []
        asks = []

        bid_price = data.get("bid_price") or data.get("bp")
        bid_size = data.get("bid_size") or data.get("bs", 100)
        ask_price = data.get("ask_price") or data.get("ap")
        ask_size = data.get("ask_size") or data.get("as", 100)

        if bid_price:
            bids.append(DepthLevel(
                price=float(bid_price),
                qty=float(bid_size),
                order_count=1,
            ))

        if ask_price:
            asks.append(DepthLevel(
                price=float(ask_price),
                qty=float(ask_size),
                order_count=1,
            ))

        return DepthSnapshot(
            timestamp_ns=timestamp_ns,
            symbol=self._symbol,
            bids=bids,
            asks=asks,
        )

    def stream_updates(
        self,
        source: Any,
    ) -> Iterator[LOBUpdate]:
        """
        Stream quote updates from Alpaca.

        Args:
            source: Iterator of Alpaca quote dicts

        Yields:
            LOBUpdate objects (top-of-book changes)
        """
        last_bid = (0.0, 0.0)
        last_ask = (0.0, 0.0)

        for quote_data in source:
            self._stats.messages_processed += 1

            # Parse timestamp
            ts = quote_data.get("t", "")
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamp_ns = int(dt.timestamp() * 1e9)
                except Exception:
                    import time
                    timestamp_ns = int(time.time() * 1e9)
            else:
                timestamp_ns = int(ts * 1e9) if ts else 0

            if self._stats.start_time_ns == 0:
                self._stats.start_time_ns = timestamp_ns
            self._stats.end_time_ns = timestamp_ns

            # Check for bid changes
            bid_price = float(quote_data.get("bp", 0) or 0)
            bid_size = float(quote_data.get("bs", 0) or 0)

            if (bid_price, bid_size) != last_bid:
                if last_bid[0] > 0 and last_bid != (bid_price, bid_size):
                    # Remove old bid
                    yield LOBUpdate(
                        timestamp_ns=timestamp_ns,
                        update_type="DELETE",
                        order_id=f"bid_{last_bid[0]}",
                        price=last_bid[0],
                        qty=0.0,
                        side=Side.BUY,
                        source=DataSourceType.ALPACA,
                    )

                if bid_price > 0 and bid_size > 0:
                    # Add new bid
                    yield LOBUpdate(
                        timestamp_ns=timestamp_ns,
                        update_type="ADD",
                        order_id=f"bid_{bid_price}",
                        price=bid_price,
                        qty=bid_size,
                        side=Side.BUY,
                        source=DataSourceType.ALPACA,
                    )

                last_bid = (bid_price, bid_size)

            # Check for ask changes
            ask_price = float(quote_data.get("ap", 0) or 0)
            ask_size = float(quote_data.get("as", 0) or 0)

            if (ask_price, ask_size) != last_ask:
                if last_ask[0] > 0 and last_ask != (ask_price, ask_size):
                    yield LOBUpdate(
                        timestamp_ns=timestamp_ns,
                        update_type="DELETE",
                        order_id=f"ask_{last_ask[0]}",
                        price=last_ask[0],
                        qty=0.0,
                        side=Side.SELL,
                        source=DataSourceType.ALPACA,
                    )

                if ask_price > 0 and ask_size > 0:
                    yield LOBUpdate(
                        timestamp_ns=timestamp_ns,
                        update_type="ADD",
                        order_id=f"ask_{ask_price}",
                        price=ask_price,
                        qty=ask_size,
                        side=Side.SELL,
                        source=DataSourceType.ALPACA,
                    )

                last_ask = (ask_price, ask_size)

    def quotes_to_depth_snapshot(
        self,
        quotes: List[Dict[str, Any]],
        timestamp_ns: Optional[int] = None,
    ) -> DepthSnapshot:
        """
        Convert Alpaca quotes to DepthSnapshot.

        Note: Alpaca only provides top-of-book, so depth is limited.

        Args:
            quotes: List of Alpaca quote dicts
            timestamp_ns: Optional timestamp

        Returns:
            DepthSnapshot with available depth
        """
        import time

        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)

        bids: List[DepthLevel] = []
        asks: List[DepthLevel] = []

        for quote in quotes:
            bid_price = float(quote.get("bp", 0) or 0)
            bid_size = float(quote.get("bs", 0) or 0)
            ask_price = float(quote.get("ap", 0) or 0)
            ask_size = float(quote.get("as", 0) or 0)

            if bid_price > 0 and bid_size > 0:
                bids.append(DepthLevel(price=bid_price, qty=bid_size))

            if ask_price > 0 and ask_size > 0:
                asks.append(DepthLevel(price=ask_price, qty=ask_size))

        # Sort and deduplicate
        bids = sorted(bids, key=lambda x: -x.price)
        asks = sorted(asks, key=lambda x: x.price)

        return DepthSnapshot(
            timestamp_ns=timestamp_ns,
            symbol=self._symbol,
            bids=bids,
            asks=asks,
        )


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_lob_adapter(
    source_type: Union[str, DataSourceType],
    symbol: str = "",
    config: Optional[Mapping[str, Any]] = None,
) -> BaseLOBAdapter:
    """
    Factory function to create LOB adapter.

    Args:
        source_type: Type of data source
        symbol: Trading symbol
        config: Optional configuration

    Returns:
        BaseLOBAdapter instance
    """
    if isinstance(source_type, str):
        source_type = DataSourceType(source_type.lower())

    adapter_map = {
        DataSourceType.LOBSTER: LOBSTERAdapter,
        DataSourceType.ITCH: ITCHAdapter,
        DataSourceType.BINANCE: BinanceL2Adapter,
        DataSourceType.ALPACA: AlpacaL2Adapter,
    }

    adapter_class = adapter_map.get(source_type)
    if adapter_class is None:
        raise ValueError(f"Unknown data source type: {source_type}")

    return adapter_class(symbol=symbol, config=config)


def load_orderbook_from_file(
    path: Union[str, Path],
    source_type: Union[str, DataSourceType] = DataSourceType.LOBSTER,
    symbol: str = "",
) -> OrderBook:
    """
    Load OrderBook from file.

    Args:
        path: Path to data file
        source_type: Type of data source
        symbol: Trading symbol

    Returns:
        OrderBook
    """
    adapter = create_lob_adapter(source_type, symbol)

    if isinstance(adapter, LOBSTERAdapter):
        return adapter.build_orderbook_from_messages(path)
    else:
        snapshot = adapter.load_snapshot(path)
        # Convert snapshot to OrderBook
        book = OrderBook()
        for level in snapshot.bids:
            order = LimitOrder(
                order_id=f"bid_{level.price}",
                price=level.price,
                qty=level.qty,
                remaining_qty=level.qty,
                timestamp_ns=snapshot.timestamp_ns,
                side=Side.BUY,
            )
            book.add_limit_order(order)
        for level in snapshot.asks:
            order = LimitOrder(
                order_id=f"ask_{level.price}",
                price=level.price,
                qty=level.qty,
                remaining_qty=level.qty,
                timestamp_ns=snapshot.timestamp_ns,
                side=Side.SELL,
            )
            book.add_limit_order(order)
        return book


def get_supported_formats() -> List[str]:
    """
    Get list of supported data format names.

    Returns:
        List of supported format names: ["lobster", "itch", "binance", "alpaca"]
    """
    return ["lobster", "itch", "binance", "alpaca"]
