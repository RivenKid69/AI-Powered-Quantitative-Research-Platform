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

Requirements:
    pip install polygon-api-client

Environment:
    POLYGON_API_KEY: Your Polygon.io API key

Usage:
    from adapters.polygon import PolygonMarketDataAdapter

    adapter = PolygonMarketDataAdapter(config={"api_key": "..."})
    bars = adapter.get_bars("AAPL", "1h", limit=100)
"""

from .market_data import PolygonMarketDataAdapter
from .trading_hours import PolygonTradingHoursAdapter
from .exchange_info import PolygonExchangeInfoAdapter

__all__ = [
    "PolygonMarketDataAdapter",
    "PolygonTradingHoursAdapter",
    "PolygonExchangeInfoAdapter",
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
