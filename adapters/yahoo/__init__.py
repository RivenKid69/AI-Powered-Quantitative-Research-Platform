# -*- coding: utf-8 -*-
"""
adapters/yahoo/__init__.py
Yahoo Finance adapter for market data, corporate actions, and earnings.

This adapter uses yfinance library to fetch:
- Market data for indices (VIX, DXY, Treasury yields)
- Dividend history
- Stock split history
- Earnings calendar and estimates

Usage:
    from adapters.registry import create_market_data_adapter, create_corporate_actions_adapter

    # Market data (VIX, indices)
    md_adapter = create_market_data_adapter("yahoo")
    vix_bars = md_adapter.get_bars("^VIX", "1d", limit=365)

    # Corporate actions
    ca_adapter = create_corporate_actions_adapter("yahoo")
    dividends = ca_adapter.get_dividends("AAPL", start_date="2023-01-01")
    splits = ca_adapter.get_splits("AAPL", start_date="2020-01-01")

    # Earnings
    earnings_adapter = create_earnings_adapter("yahoo")
    history = earnings_adapter.get_earnings_history("AAPL")
    upcoming = earnings_adapter.get_upcoming_earnings(["AAPL", "MSFT"])

Dependencies:
    pip install yfinance>=0.2.0

Note:
    Yahoo Finance data is free but has rate limits. For production use,
    implement caching via the CorporateActionsService.
"""

from adapters.registry import register, AdapterType
from adapters.models import ExchangeVendor

# Import adapter implementations
from .corporate_actions import YahooCorporateActionsAdapter
from .earnings import YahooEarningsAdapter
from .market_data import YahooMarketDataAdapter

# Register adapters
register(
    vendor=ExchangeVendor.YAHOO,
    adapter_type=AdapterType.MARKET_DATA,
    adapter_class=YahooMarketDataAdapter,
    description="Yahoo Finance market data adapter (VIX, indices, macro)",
)

register(
    vendor=ExchangeVendor.YAHOO,
    adapter_type=AdapterType.CORPORATE_ACTIONS,
    adapter_class=YahooCorporateActionsAdapter,
    description="Yahoo Finance corporate actions adapter (dividends, splits)",
)

register(
    vendor=ExchangeVendor.YAHOO,
    adapter_type=AdapterType.EARNINGS,
    adapter_class=YahooEarningsAdapter,
    description="Yahoo Finance earnings calendar adapter",
)

__all__ = [
    "YahooMarketDataAdapter",
    "YahooCorporateActionsAdapter",
    "YahooEarningsAdapter",
]
