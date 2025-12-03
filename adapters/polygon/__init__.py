# -*- coding: utf-8 -*-
"""
adapters/polygon/__init__.py
Polygon.io adapter implementations.

Polygon.io is a financial data provider offering:
- Real-time and historical stock data (US markets)
- Options, forex, and crypto data
- News and reference data

This module provides adapter implementations for:
- MarketDataAdapter: Historical bars, real-time streaming
- FeeAdapter: No trading fees (data-only provider)
- TradingHoursAdapter: US market schedule
- ExchangeInfoAdapter: Symbol metadata and filters
- PolygonOptionsAdapter: Historical options data (2018+)

Requirements:
    pip install polygon-api-client

Environment:
    POLYGON_API_KEY: Your Polygon.io API key

Usage:
    from adapters.polygon import PolygonMarketDataAdapter, PolygonOptionsAdapter

    # Stock data
    adapter = PolygonMarketDataAdapter(config={"api_key": "..."})
    bars = adapter.get_bars("AAPL", "1h", limit=100)

    # Options historical data
    options = PolygonOptionsAdapter(config={"api_key": "..."})
    chain = options.get_historical_chain("AAPL", date(2024, 1, 15))

Phase 2: US Exchange Adapters
"""

from .market_data import PolygonMarketDataAdapter
from .trading_hours import PolygonTradingHoursAdapter
from .exchange_info import PolygonExchangeInfoAdapter
from .options import (
    PolygonOptionsAdapter,
    PolygonOptionsContract,
    PolygonOptionsQuote,
    PolygonOptionsSnapshot,
    create_polygon_options_adapter,
    polygon_ticker_to_occ,
    occ_to_polygon_ticker,
    parse_polygon_ticker,
)

__all__ = [
    # Market Data
    "PolygonMarketDataAdapter",
    "PolygonTradingHoursAdapter",
    "PolygonExchangeInfoAdapter",
    # Options Historical Data
    "PolygonOptionsAdapter",
    "PolygonOptionsContract",
    "PolygonOptionsQuote",
    "PolygonOptionsSnapshot",
    "create_polygon_options_adapter",
    "polygon_ticker_to_occ",
    "occ_to_polygon_ticker",
    "parse_polygon_ticker",
]

# Register adapters in global registry
try:
    from ..registry import register, AdapterType
    from ..models import ExchangeVendor

    register(
        vendor=ExchangeVendor.POLYGON,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=PolygonMarketDataAdapter,
        description="Polygon.io market data adapter",
    )

    register(
        vendor=ExchangeVendor.POLYGON,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=PolygonTradingHoursAdapter,
        description="Polygon.io trading hours adapter",
    )

    register(
        vendor=ExchangeVendor.POLYGON,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=PolygonExchangeInfoAdapter,
        description="Polygon.io exchange info adapter",
    )

except ImportError:
    pass  # Registry not available
