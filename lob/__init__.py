"""
L3 Limit Order Book (LOB) module for equity market simulation.

This module provides high-fidelity order book simulation for US equities,
supporting full market microstructure modeling including:
- Price-time priority (FIFO) queue tracking
- FIFO matching engine with self-trade prevention
- Queue position tracking (MBO/MBP estimation)
- Order lifecycle management
- Iceberg/hidden order simulation
- Market-by-Order (MBO) and Market-by-Price (MBP) views
- LOBSTER message format parsing

Architecture:
    This module is SEPARATE from the Cython LOB implementations (fast_lob.pyx,
    execlob_book.pyx) which are optimized for crypto environments. The Python
    implementation here provides better flexibility and testability for equity
    market simulation while integrating with execution_providers.py.

Stage 1 (v1.0): Data structures, parsers, state manager
Stage 2 (v2.0): Matching engine, queue tracker, order manager

Usage:
    from lob import OrderBook, LimitOrder, PriceLevel
    from lob.parsers import LOBSTERParser
    from lob.state_manager import LOBStateManager
    from lob.matching_engine import MatchingEngine
    from lob.queue_tracker import QueuePositionTracker
    from lob.order_manager import OrderManager

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

from lob.matching_engine import (
    MatchingEngine,
    MatchResult,
    MatchType,
    STPAction,
    ProRataMatchingEngine,
    create_matching_engine,
)

from lob.queue_tracker import (
    QueuePositionTracker,
    QueueState,
    FillProbability,
    PositionEstimationMethod,
    create_queue_tracker,
)

from lob.order_manager import (
    OrderManager,
    ManagedOrder,
    OrderEvent,
    OrderEventType,
    OrderLifecycleState,
    TimeInForce,
    create_order_manager,
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
    # Matching engine (Stage 2)
    "MatchingEngine",
    "MatchResult",
    "MatchType",
    "STPAction",
    "ProRataMatchingEngine",
    "create_matching_engine",
    # Queue tracking (Stage 2)
    "QueuePositionTracker",
    "QueueState",
    "FillProbability",
    "PositionEstimationMethod",
    "create_queue_tracker",
    # Order management (Stage 2)
    "OrderManager",
    "ManagedOrder",
    "OrderEvent",
    "OrderEventType",
    "OrderLifecycleState",
    "TimeInForce",
    "create_order_manager",
]

__version__ = "2.0.0"
