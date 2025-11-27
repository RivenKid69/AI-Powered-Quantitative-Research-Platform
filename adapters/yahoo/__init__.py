# -*- coding: utf-8 -*-
"""
adapters/yahoo/__init__.py
Yahoo Finance adapter for corporate actions and earnings data.

This adapter uses yfinance library to fetch:
- Dividend history
- Stock split history
- Earnings calendar and estimates

Usage:
    from adapters.registry import create_corporate_actions_adapter, create_earnings_adapter

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

# Register adapters
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
    "YahooCorporateActionsAdapter",
    "YahooEarningsAdapter",
]
