# -*- coding: utf-8 -*-
"""
OANDA Forex Adapter Package

Phase 0: Stub module for forex integration foundation.

This package will provide:
- OANDA v20 REST API integration
- Streaming prices via WebSocket
- Historical candles (M1 to M)
- Order execution with last-look handling
- Swap rate queries

API Rate Limits:
- 120 requests per second (streaming is separate)
- Max 5000 candles per request
- Burst: 200 requests allowed

Environment Variables:
- OANDA_API_KEY: API access token
- OANDA_ACCOUNT_ID: Trading account ID
- OANDA_PRACTICE: "true" for demo, "false" for live

Status: Stub (Phase 0 - Foundation)
Planned Implementation: Phase 2

References:
    - OANDA v20 API: https://developer.oanda.com/rest-live-v20/introduction/
    - BIS Triennial Survey 2022: https://www.bis.org/statistics/rpfx22.htm
"""

# Phase 0: Empty stub
# Actual implementations will be added in Phase 2

__all__: list = []

# Placeholder for future imports:
# from .market_data import OandaMarketDataAdapter
# from .fees import OandaFeeAdapter
# from .trading_hours import OandaTradingHoursAdapter
# from .exchange_info import OandaExchangeInfoAdapter
# from .order_execution import OandaOrderExecutionAdapter
