# -*- coding: utf-8 -*-
"""
adapters/binance/__init__.py
Binance adapter implementations.

This package provides adapter implementations for Binance exchange,
wrapping existing functionality from binance_public.py, impl_fees.py, etc.

Usage:
    from adapters.binance import (
        BinanceMarketDataAdapter,
        BinanceFeeAdapter,
        BinanceTradingHoursAdapter,
        BinanceExchangeInfoAdapter,
    )

    # Or use the registry
    from adapters.registry import create_market_data_adapter
    adapter = create_market_data_adapter("binance", config)
"""

from __future__ import annotations

import logging

from adapters.models import ExchangeVendor
from adapters.registry import AdapterType, register

logger = logging.getLogger(__name__)

# Import adapter implementations
from .market_data import BinanceMarketDataAdapter
from .fees import BinanceFeeAdapter
from .trading_hours import BinanceTradingHoursAdapter
from .exchange_info import BinanceExchangeInfoAdapter


# Register adapters with the global registry
def _register_adapters() -> None:
    """Register Binance adapters with the global registry."""

    # Market Data
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=BinanceMarketDataAdapter,
        description="Binance market data adapter (REST/WebSocket)",
    )

    # Fees
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.FEE,
        adapter_class=BinanceFeeAdapter,
        description="Binance fee computation adapter",
    )

    # Trading Hours
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=BinanceTradingHoursAdapter,
        description="Binance trading hours (24/7 crypto)",
    )

    # Exchange Info
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=BinanceExchangeInfoAdapter,
        description="Binance exchange info adapter",
    )

    # Also register for Binance US (same implementations)
    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=BinanceMarketDataAdapter,
        default_config={"base_url": "https://api.binance.us"},
        description="Binance US market data adapter",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.FEE,
        adapter_class=BinanceFeeAdapter,
        description="Binance US fee computation adapter",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=BinanceTradingHoursAdapter,
        description="Binance US trading hours (24/7 crypto)",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=BinanceExchangeInfoAdapter,
        default_config={"base_url": "https://api.binance.us"},
        description="Binance US exchange info adapter",
    )

    logger.debug("Registered Binance adapters")


# Auto-register on import
_register_adapters()


__all__ = [
    "BinanceMarketDataAdapter",
    "BinanceFeeAdapter",
    "BinanceTradingHoursAdapter",
    "BinanceExchangeInfoAdapter",
]
