"""
LOBSTER and ITCH message format parsers.

Supports:
- LOBSTER: Standard academic format from NASDAQ OMX
- ITCH: NASDAQ TotalView-ITCH 5.0 (binary)

LOBSTER Message Format:
    time, type, order_id, size, price, direction
    - time: Seconds since midnight (float with nanosecond precision)
    - type: 1=Add, 2=Modify, 3=Delete, 4=Execute, 5=Hidden Execute, 6=Cross, 7=Trade Halt
    - order_id: Unique order identifier
    - size: Number of shares
    - price: Price in dollars (float)
    - direction: 1=Buy, -1=Sell

Reference:
    https://lobsterdata.com/info/DataStructure.php
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import (
    BinaryIO,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
    Union,
)

from lob.data_structures import Side, LimitOrder, OrderType


# ==============================================================================
# LOBSTER Format
# ==============================================================================


class LOBSTEREventType(IntEnum):
    """LOBSTER message event types."""

    ADD = 1  # Limit order added
    MODIFY = 2  # Order modified (cancel + replace)
    DELETE = 3  # Order cancelled
    EXECUTE = 4  # Visible execution
    HIDDEN_EXECUTE = 5  # Hidden execution
    CROSS = 6  # Cross trade
    TRADE_HALT = 7  # Trading halt indicator


@dataclass
class LOBSTERMessage:
    """
    Parsed LOBSTER message.

    Attributes:
        timestamp_sec: Seconds since midnight (with nanoseconds as decimal)
        event_type: Type of event (ADD, MODIFY, DELETE, EXECUTE, etc.)
        order_id: Unique order identifier
        size: Number of shares
        price: Price in dollars
        direction: 1 for buy, -1 for sell
        timestamp_ns: Timestamp in nanoseconds from midnight
    """

    timestamp_sec: float
    event_type: LOBSTEREventType
    order_id: int
    size: int
    price: float
    direction: int

    @property
    def timestamp_ns(self) -> int:
        """Convert timestamp to nanoseconds."""
        return int(self.timestamp_sec * 1_000_000_000)

    @property
    def side(self) -> Side:
        """Convert direction to Side enum."""
        return Side.BUY if self.direction == 1 else Side.SELL

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.direction == 1

    @property
    def is_add(self) -> bool:
        """Check if this is an ADD message."""
        return self.event_type == LOBSTEREventType.ADD

    @property
    def is_delete(self) -> bool:
        """Check if this is a DELETE message."""
        return self.event_type == LOBSTEREventType.DELETE

    @property
    def is_execute(self) -> bool:
        """Check if this is an EXECUTE message."""
        return self.event_type in (
            LOBSTEREventType.EXECUTE,
            LOBSTEREventType.HIDDEN_EXECUTE,
        )

    @property
    def is_modify(self) -> bool:
        """Check if this is a MODIFY message."""
        return self.event_type == LOBSTEREventType.MODIFY

    def to_limit_order(self, order_id_prefix: str = "") -> LimitOrder:
        """
        Convert to LimitOrder for book operations.

        Args:
            order_id_prefix: Optional prefix for order ID

        Returns:
            LimitOrder instance
        """
        return LimitOrder(
            order_id=f"{order_id_prefix}{self.order_id}",
            price=self.price,
            qty=float(self.size),
            remaining_qty=float(self.size),
            timestamp_ns=self.timestamp_ns,
            side=self.side,
            order_type=OrderType.LIMIT,
        )


class LOBSTERParser:
    """
    Parser for LOBSTER message format.

    LOBSTER is a standard format for limit order book data,
    commonly used in academic research.

    Usage:
        parser = LOBSTERParser()
        for msg in parser.parse_file("AAPL_2019-01-02_message.csv"):
            if msg.is_add:
                book.add_limit_order(msg.to_limit_order())
            elif msg.is_delete:
                book.cancel_order(str(msg.order_id))
    """

    def __init__(
        self,
        price_multiplier: float = 1.0,
        delimiter: str = ",",
    ) -> None:
        """
        Initialize parser.

        Args:
            price_multiplier: Multiply price by this factor
                             (LOBSTER prices are in dollars × 10000)
            delimiter: CSV delimiter character
        """
        self.price_multiplier = price_multiplier
        self.delimiter = delimiter
        self._line_count = 0
        self._error_count = 0

    def parse_line(self, line: str) -> Optional[LOBSTERMessage]:
        """
        Parse single LOBSTER message line.

        Args:
            line: CSV line to parse

        Returns:
            LOBSTERMessage or None if parsing fails
        """
        self._line_count += 1

        try:
            parts = line.strip().split(self.delimiter)
            if len(parts) < 6:
                self._error_count += 1
                return None

            timestamp_sec = float(parts[0])
            event_type = LOBSTEREventType(int(parts[1]))
            order_id = int(parts[2])
            size = int(parts[3])
            price = float(parts[4]) * self.price_multiplier
            direction = int(parts[5])

            return LOBSTERMessage(
                timestamp_sec=timestamp_sec,
                event_type=event_type,
                order_id=order_id,
                size=size,
                price=price,
                direction=direction,
            )

        except (ValueError, IndexError) as e:
            self._error_count += 1
            return None

    def parse_file(
        self,
        filepath: Union[str, Path],
        skip_header: bool = False,
        max_messages: Optional[int] = None,
    ) -> Generator[LOBSTERMessage, None, None]:
        """
        Parse LOBSTER message file.

        Args:
            filepath: Path to message file
            skip_header: Skip first line as header
            max_messages: Maximum messages to parse (None = all)

        Yields:
            LOBSTERMessage objects
        """
        self._line_count = 0
        self._error_count = 0
        count = 0

        with open(filepath, "r") as f:
            if skip_header:
                next(f, None)

            for line in f:
                if max_messages and count >= max_messages:
                    break

                msg = self.parse_line(line)
                if msg is not None:
                    yield msg
                    count += 1

    def parse_stream(
        self,
        stream: TextIO,
        max_messages: Optional[int] = None,
    ) -> Generator[LOBSTERMessage, None, None]:
        """
        Parse LOBSTER messages from stream.

        Args:
            stream: Text stream to read from
            max_messages: Maximum messages to parse

        Yields:
            LOBSTERMessage objects
        """
        count = 0
        for line in stream:
            if max_messages and count >= max_messages:
                break

            msg = self.parse_line(line)
            if msg is not None:
                yield msg
                count += 1

    def parse_orderbook_file(
        self,
        filepath: Union[str, Path],
        n_levels: int = 10,
    ) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """
        Parse LOBSTER orderbook snapshot file.

        LOBSTER orderbook files have format:
        ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...

        Args:
            filepath: Path to orderbook file
            n_levels: Number of levels to parse

        Returns:
            Tuple of (asks, bids) where each is list of (price, size)
        """
        asks: List[Tuple[float, int]] = []
        bids: List[Tuple[float, int]] = []

        with open(filepath, "r") as f:
            line = f.readline().strip()
            parts = line.split(self.delimiter)

            for i in range(n_levels):
                idx = i * 4
                if idx + 3 >= len(parts):
                    break

                ask_price = float(parts[idx]) * self.price_multiplier
                ask_size = int(parts[idx + 1])
                bid_price = float(parts[idx + 2]) * self.price_multiplier
                bid_size = int(parts[idx + 3])

                asks.append((ask_price, ask_size))
                bids.append((bid_price, bid_size))

        return asks, bids

    @property
    def line_count(self) -> int:
        """Number of lines processed."""
        return self._line_count

    @property
    def error_count(self) -> int:
        """Number of parsing errors."""
        return self._error_count

    @property
    def error_rate(self) -> float:
        """Parsing error rate."""
        if self._line_count == 0:
            return 0.0
        return self._error_count / self._line_count


# ==============================================================================
# ITCH Format (NASDAQ TotalView-ITCH 5.0)
# ==============================================================================


class ITCHMessageType:
    """ITCH 5.0 message type codes."""

    SYSTEM_EVENT = b"S"
    STOCK_DIRECTORY = b"R"
    STOCK_TRADING_ACTION = b"H"
    REG_SHO = b"Y"
    MARKET_PARTICIPANT = b"L"
    MWCB_DECLINE = b"V"
    MWCB_STATUS = b"W"
    IPO_QUOTING = b"K"
    LULD_AUCTION_COLLAR = b"J"
    OPERATIONAL_HALT = b"h"
    ADD_ORDER = b"A"
    ADD_ORDER_MPID = b"F"
    ORDER_EXECUTED = b"E"
    ORDER_EXECUTED_PRICE = b"C"
    ORDER_CANCEL = b"X"
    ORDER_DELETE = b"D"
    ORDER_REPLACE = b"U"
    TRADE = b"P"
    CROSS_TRADE = b"Q"
    BROKEN_TRADE = b"B"
    NOII = b"I"
    RPII = b"N"


@dataclass
class ITCHMessage:
    """
    Parsed ITCH message.

    Base class for all ITCH message types.
    """

    msg_type: bytes
    timestamp_ns: int
    stock_locate: int


@dataclass
class ITCHAddOrder(ITCHMessage):
    """ITCH Add Order message (A or F)."""

    order_ref: int
    side: str  # 'B' or 'S'
    shares: int
    stock: str
    price: float
    mpid: Optional[str] = None  # Only for type F


@dataclass
class ITCHOrderExecuted(ITCHMessage):
    """ITCH Order Executed message (E or C)."""

    order_ref: int
    shares: int
    match_number: int
    price: Optional[float] = None  # Only for type C


@dataclass
class ITCHOrderCancel(ITCHMessage):
    """ITCH Order Cancel message (X)."""

    order_ref: int
    cancelled_shares: int


@dataclass
class ITCHOrderDelete(ITCHMessage):
    """ITCH Order Delete message (D)."""

    order_ref: int


@dataclass
class ITCHOrderReplace(ITCHMessage):
    """ITCH Order Replace message (U)."""

    original_ref: int
    new_ref: int
    shares: int
    price: float


class ITCHParser:
    """
    Parser for NASDAQ ITCH 5.0 binary format.

    ITCH is the primary market data feed format for NASDAQ.
    Messages are variable-length binary records.

    Reference:
        https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf
    """

    # Message lengths by type (excluding length prefix)
    MSG_LENGTHS = {
        b"S": 12,  # System Event
        b"R": 39,  # Stock Directory
        b"H": 25,  # Stock Trading Action
        b"Y": 20,  # Reg SHO
        b"L": 26,  # Market Participant
        b"V": 35,  # MWCB Decline
        b"W": 12,  # MWCB Status
        b"K": 28,  # IPO Quoting
        b"J": 35,  # LULD Auction Collar
        b"h": 21,  # Operational Halt
        b"A": 36,  # Add Order
        b"F": 40,  # Add Order MPID
        b"E": 31,  # Order Executed
        b"C": 36,  # Order Executed with Price
        b"X": 23,  # Order Cancel
        b"D": 19,  # Order Delete
        b"U": 35,  # Order Replace
        b"P": 44,  # Trade
        b"Q": 40,  # Cross Trade
        b"B": 19,  # Broken Trade
        b"I": 50,  # NOII
        b"N": 20,  # RPII
    }

    def __init__(self, price_divisor: float = 10000.0) -> None:
        """
        Initialize parser.

        Args:
            price_divisor: Divisor for price conversion (ITCH uses fixed point)
        """
        self.price_divisor = price_divisor
        self._msg_count = 0
        self._error_count = 0

    def _parse_timestamp(self, data: bytes) -> int:
        """Parse 6-byte ITCH timestamp to nanoseconds."""
        # Timestamp is 6 bytes, big-endian
        ts_bytes = b"\x00\x00" + data[:6]
        return struct.unpack(">Q", ts_bytes)[0]

    def _parse_stock(self, data: bytes) -> str:
        """Parse 8-byte stock symbol."""
        return data[:8].decode("ascii").strip()

    def _parse_price(self, data: bytes) -> float:
        """Parse 4-byte price to float."""
        return struct.unpack(">I", data)[0] / self.price_divisor

    def parse_add_order(self, data: bytes) -> ITCHAddOrder:
        """Parse Add Order message (type A or F)."""
        msg_type = data[0:1]
        stock_locate = struct.unpack(">H", data[1:3])[0]
        # tracking_number = struct.unpack(">H", data[3:5])[0]  # unused
        timestamp_ns = self._parse_timestamp(data[5:11])
        order_ref = struct.unpack(">Q", data[11:19])[0]
        side = chr(data[19])
        shares = struct.unpack(">I", data[20:24])[0]
        stock = self._parse_stock(data[24:32])
        price = self._parse_price(data[32:36])

        mpid = None
        if msg_type == b"F":
            mpid = data[36:40].decode("ascii").strip()

        return ITCHAddOrder(
            msg_type=msg_type,
            timestamp_ns=timestamp_ns,
            stock_locate=stock_locate,
            order_ref=order_ref,
            side=side,
            shares=shares,
            stock=stock,
            price=price,
            mpid=mpid,
        )

    def parse_order_executed(self, data: bytes) -> ITCHOrderExecuted:
        """Parse Order Executed message (type E or C)."""
        msg_type = data[0:1]
        stock_locate = struct.unpack(">H", data[1:3])[0]
        timestamp_ns = self._parse_timestamp(data[5:11])
        order_ref = struct.unpack(">Q", data[11:19])[0]
        shares = struct.unpack(">I", data[19:23])[0]
        match_number = struct.unpack(">Q", data[23:31])[0]

        price = None
        if msg_type == b"C":
            # printable = chr(data[31])  # unused
            price = self._parse_price(data[32:36])

        return ITCHOrderExecuted(
            msg_type=msg_type,
            timestamp_ns=timestamp_ns,
            stock_locate=stock_locate,
            order_ref=order_ref,
            shares=shares,
            match_number=match_number,
            price=price,
        )

    def parse_order_cancel(self, data: bytes) -> ITCHOrderCancel:
        """Parse Order Cancel message (type X)."""
        stock_locate = struct.unpack(">H", data[1:3])[0]
        timestamp_ns = self._parse_timestamp(data[5:11])
        order_ref = struct.unpack(">Q", data[11:19])[0]
        cancelled_shares = struct.unpack(">I", data[19:23])[0]

        return ITCHOrderCancel(
            msg_type=b"X",
            timestamp_ns=timestamp_ns,
            stock_locate=stock_locate,
            order_ref=order_ref,
            cancelled_shares=cancelled_shares,
        )

    def parse_order_delete(self, data: bytes) -> ITCHOrderDelete:
        """Parse Order Delete message (type D)."""
        stock_locate = struct.unpack(">H", data[1:3])[0]
        timestamp_ns = self._parse_timestamp(data[5:11])
        order_ref = struct.unpack(">Q", data[11:19])[0]

        return ITCHOrderDelete(
            msg_type=b"D",
            timestamp_ns=timestamp_ns,
            stock_locate=stock_locate,
            order_ref=order_ref,
        )

    def parse_order_replace(self, data: bytes) -> ITCHOrderReplace:
        """Parse Order Replace message (type U)."""
        stock_locate = struct.unpack(">H", data[1:3])[0]
        timestamp_ns = self._parse_timestamp(data[5:11])
        original_ref = struct.unpack(">Q", data[11:19])[0]
        new_ref = struct.unpack(">Q", data[19:27])[0]
        shares = struct.unpack(">I", data[27:31])[0]
        price = self._parse_price(data[31:35])

        return ITCHOrderReplace(
            msg_type=b"U",
            timestamp_ns=timestamp_ns,
            stock_locate=stock_locate,
            original_ref=original_ref,
            new_ref=new_ref,
            shares=shares,
            price=price,
        )

    def parse_message(self, data: bytes) -> Optional[ITCHMessage]:
        """
        Parse single ITCH message.

        Args:
            data: Raw message bytes (including type byte)

        Returns:
            Parsed message or None if not supported
        """
        self._msg_count += 1

        if len(data) < 1:
            self._error_count += 1
            return None

        msg_type = data[0:1]

        try:
            if msg_type in (b"A", b"F"):
                return self.parse_add_order(data)
            elif msg_type in (b"E", b"C"):
                return self.parse_order_executed(data)
            elif msg_type == b"X":
                return self.parse_order_cancel(data)
            elif msg_type == b"D":
                return self.parse_order_delete(data)
            elif msg_type == b"U":
                return self.parse_order_replace(data)
            else:
                # Skip unsupported message types
                return None
        except (struct.error, IndexError) as e:
            self._error_count += 1
            return None

    def parse_stream(
        self,
        stream: BinaryIO,
        max_messages: Optional[int] = None,
    ) -> Generator[ITCHMessage, None, None]:
        """
        Parse ITCH messages from binary stream.

        Args:
            stream: Binary stream to read from
            max_messages: Maximum messages to parse

        Yields:
            Parsed ITCH messages
        """
        count = 0

        while True:
            if max_messages and count >= max_messages:
                break

            # Read 2-byte length prefix
            length_bytes = stream.read(2)
            if len(length_bytes) < 2:
                break

            msg_length = struct.unpack(">H", length_bytes)[0]
            if msg_length == 0:
                continue

            # Read message body
            msg_data = stream.read(msg_length)
            if len(msg_data) < msg_length:
                break

            msg = self.parse_message(msg_data)
            if msg is not None:
                yield msg
                count += 1

    def parse_file(
        self,
        filepath: Union[str, Path],
        max_messages: Optional[int] = None,
    ) -> Generator[ITCHMessage, None, None]:
        """
        Parse ITCH messages from file.

        Args:
            filepath: Path to ITCH file (binary)
            max_messages: Maximum messages to parse

        Yields:
            Parsed ITCH messages
        """
        self._msg_count = 0
        self._error_count = 0

        with open(filepath, "rb") as f:
            yield from self.parse_stream(f, max_messages)

    @property
    def message_count(self) -> int:
        """Number of messages processed."""
        return self._msg_count

    @property
    def error_count(self) -> int:
        """Number of parsing errors."""
        return self._error_count
