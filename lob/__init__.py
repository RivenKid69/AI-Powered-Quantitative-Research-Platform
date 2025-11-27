"""
L3 Limit Order Book (LOB) module for equity market simulation.

This module provides high-fidelity order book simulation for US equities,
supporting full market microstructure modeling including:
- Price-time priority (FIFO) queue tracking
- Iceberg/hidden order simulation
- Market-by-Order (MBO) and Market-by-Price (MBP) views
- LOBSTER message format parsing

Architecture:
    This module is SEPARATE from the Cython LOB implementations (fast_lob.pyx,
    execlob_book.pyx) which are optimized for crypto environments. The Python
    implementation here provides better flexibility and testability for equity
    market simulation while integrating with execution_providers.py.

Usage:
    from lob import OrderBook, LimitOrder, PriceLevel
    from lob.parsers import LOBSTERParser
    from lob.state_manager import LOBStateManager

Note:
    This module does NOT affect crypto execution paths. Crypto continues
    to use the existing Cython LOB implementations.
"""

from lob.data_structures import (
    Side,
    OrderType,
    LimitOrder,
    PriceLevel,
    OrderBook,
    Fill,
    Trade,
)

from lob.state_manager import (
    LOBStateManager,
    LOBSnapshot,
    LOBMessage,
    MessageType,
)

from lob.parsers import (
    LOBSTERParser,
    LOBSTERMessage,
)

__all__ = [
    # Core data structures
    "Side",
    "OrderType",
    "LimitOrder",
    "PriceLevel",
    "OrderBook",
    "Fill",
    "Trade",
    # State management
    "LOBStateManager",
    "LOBSnapshot",
    "LOBMessage",
    "MessageType",
    # Parsers
    "LOBSTERParser",
    "LOBSTERMessage",
]

__version__ = "1.0.0"
