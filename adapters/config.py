# -*- coding: utf-8 -*-
"""
adapters/config.py
Configuration schema for multi-exchange support.

This module provides Pydantic models for configuring exchange adapters.
It integrates with the existing config system in core_config.py.

Usage in YAML config:
    exchange:
      vendor: "binance"  # or "alpaca"
      market_type: "CRYPTO_SPOT"  # or "EQUITY"

      # Binance-specific settings
      binance:
        base_url: "https://api.binance.com"
        use_futures: false
        timeout: 20

      # Alpaca-specific settings
      alpaca:
        api_key: "${ALPACA_API_KEY}"
        api_secret: "${ALPACA_API_SECRET}"
        paper: true
        feed: "iex"
        extended_hours: false

      # Common settings
      fees:
        include_regulatory: true

      trading_hours:
        allow_extended: true

Usage in Python:
    from adapters.config import ExchangeConfig
    config = ExchangeConfig.from_yaml("configs/exchange.yaml")
    adapter = config.create_market_data_adapter()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    # Fallback for older pydantic
    from pydantic import BaseModel, Field, validator as field_validator, root_validator as model_validator

from .models import ExchangeVendor, MarketType


class BinanceConfig(BaseModel):
    """Binance-specific configuration."""

    base_url: str = Field(
        default="https://api.binance.com",
        description="Binance API base URL",
    )
    futures_url: str = Field(
        default="https://fapi.binance.com",
        description="Binance Futures API URL",
    )
    use_futures: bool = Field(
        default=False,
        description="Use futures endpoints instead of spot",
    )
    timeout: int = Field(
        default=20,
        description="Request timeout in seconds",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Binance API key (for private endpoints)",
    )
    api_secret: Optional[str] = Field(
        default=None,
        description="Binance API secret (for private endpoints)",
    )

    # Fee settings
    maker_bps: float = Field(default=10.0, description="Maker fee in basis points")
    taker_bps: float = Field(default=10.0, description="Taker fee in basis points")
    use_bnb_discount: bool = Field(default=False, description="Apply BNB fee discount")
    vip_tier: int = Field(default=0, description="VIP tier level")

    # Cache paths
    filters_cache_path: Optional[str] = Field(
        default="data/binance_filters.json",
        description="Path to exchange filters cache",
    )
    fees_cache_path: Optional[str] = Field(
        default="data/fees/fees_by_symbol.json",
        description="Path to fees cache",
    )

    def to_adapter_config(self) -> Dict[str, Any]:
        """Convert to adapter configuration dict."""
        config = {
            "base_url": self.base_url,
            "futures_url": self.futures_url,
            "use_futures": self.use_futures,
            "timeout": self.timeout,
            "maker_bps": self.maker_bps,
            "taker_bps": self.taker_bps,
            "use_bnb_discount": self.use_bnb_discount,
            "vip_tier": self.vip_tier,
        }

        if self.api_key:
            config["api_key"] = self._resolve_env(self.api_key)
        if self.api_secret:
            config["api_secret"] = self._resolve_env(self.api_secret)
        if self.filters_cache_path:
            config["filters_cache_path"] = self.filters_cache_path
        if self.fees_cache_path:
            config["fee_cache_path"] = self.fees_cache_path

        return config

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve environment variable if value starts with $."""
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        elif value.startswith("$"):
            env_var = value[1:]
            return os.environ.get(env_var, "")
        return value


class AlpacaConfig(BaseModel):
    """Alpaca-specific configuration."""

    api_key: str = Field(
        default="",
        description="Alpaca API key (required)",
    )
    api_secret: str = Field(
        default="",
        description="Alpaca API secret (required)",
    )
    paper: bool = Field(
        default=True,
        description="Use paper trading endpoint",
    )
    feed: str = Field(
        default="iex",
        description="Data feed: 'iex' (free) or 'sip' (paid)",
    )
    extended_hours: bool = Field(
        default=False,
        description="Allow extended hours trading",
    )
    allow_extended_hours: bool = Field(
        default=True,
        description="Consider extended hours as 'market open'",
    )

    # Regulatory fees
    include_regulatory_fees: bool = Field(
        default=True,
        description="Include SEC/TAF fees in calculations",
    )
    options_per_contract_fee: float = Field(
        default=0.65,
        description="Fee per options contract",
    )

    def to_adapter_config(self) -> Dict[str, Any]:
        """Convert to adapter configuration dict."""
        return {
            "api_key": self._resolve_env(self.api_key),
            "api_secret": self._resolve_env(self.api_secret),
            "paper": self.paper,
            "feed": self.feed,
            "extended_hours": self.extended_hours,
            "allow_extended_hours": self.allow_extended_hours,
            "include_regulatory_fees": self.include_regulatory_fees,
            "options_per_contract_fee": self.options_per_contract_fee,
        }

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve environment variable."""
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        elif value.startswith("$"):
            env_var = value[1:]
            return os.environ.get(env_var, "")
        return value


class FeeConfig(BaseModel):
    """Common fee configuration."""

    include_regulatory: bool = Field(
        default=True,
        description="Include regulatory fees (SEC/TAF for US equities)",
    )
    custom_maker_bps: Optional[float] = Field(
        default=None,
        description="Override maker fee rate",
    )
    custom_taker_bps: Optional[float] = Field(
        default=None,
        description="Override taker fee rate",
    )


class TradingHoursConfig(BaseModel):
    """Common trading hours configuration."""

    allow_extended: bool = Field(
        default=True,
        description="Allow trading during extended hours (pre/post market)",
    )
    use_exchange_calendar: bool = Field(
        default=True,
        description="Fetch calendar from exchange API",
    )


class ExchangeConfig(BaseModel):
    """
    Main exchange configuration.

    Example YAML:
        exchange:
          vendor: "binance"
          market_type: "CRYPTO_SPOT"
          binance:
            timeout: 30
            use_bnb_discount: true
    """

    vendor: str = Field(
        default="binance",
        description="Exchange vendor: 'binance', 'alpaca', etc.",
    )
    market_type: str = Field(
        default="CRYPTO_SPOT",
        description="Market type: CRYPTO_SPOT, EQUITY, etc.",
    )

    # Vendor-specific configs
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)

    # Common configs
    fees: FeeConfig = Field(default_factory=FeeConfig)
    trading_hours: TradingHoursConfig = Field(default_factory=TradingHoursConfig)

    @property
    def exchange_vendor(self) -> ExchangeVendor:
        """Get ExchangeVendor enum."""
        try:
            return ExchangeVendor(self.vendor.lower())
        except ValueError:
            return ExchangeVendor.UNKNOWN

    @property
    def exchange_market_type(self) -> MarketType:
        """Get MarketType enum."""
        try:
            return MarketType(self.market_type)
        except ValueError:
            return MarketType.CRYPTO_SPOT

    def get_adapter_config(self) -> Dict[str, Any]:
        """Get adapter configuration for current vendor."""
        vendor = self.exchange_vendor

        if vendor in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
            config = self.binance.to_adapter_config()
        elif vendor == ExchangeVendor.ALPACA:
            config = self.alpaca.to_adapter_config()
        else:
            config = {}

        # Merge common configs
        if self.fees.custom_maker_bps is not None:
            config["maker_bps"] = self.fees.custom_maker_bps
        if self.fees.custom_taker_bps is not None:
            config["taker_bps"] = self.fees.custom_taker_bps

        config["market_type"] = self.market_type

        return config

    def create_market_data_adapter(self):
        """Create market data adapter for configured vendor."""
        from .registry import create_market_data_adapter
        return create_market_data_adapter(self.vendor, self.get_adapter_config())

    def create_fee_adapter(self):
        """Create fee adapter for configured vendor."""
        from .registry import create_fee_adapter
        return create_fee_adapter(self.vendor, self.get_adapter_config())

    def create_trading_hours_adapter(self):
        """Create trading hours adapter for configured vendor."""
        from .registry import create_trading_hours_adapter
        config = self.get_adapter_config()
        config["allow_extended_hours"] = self.trading_hours.allow_extended
        config["use_alpaca_calendar"] = self.trading_hours.use_exchange_calendar
        return create_trading_hours_adapter(self.vendor, config)

    def create_exchange_info_adapter(self):
        """Create exchange info adapter for configured vendor."""
        from .registry import create_exchange_info_adapter
        return create_exchange_info_adapter(self.vendor, self.get_adapter_config())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExchangeConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExchangeConfig":
        """Load from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Support both root-level and nested under 'exchange' key
        if "exchange" in data:
            data = data["exchange"]

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


# =========================
# Default Configuration
# =========================

DEFAULT_BINANCE_CONFIG = BinanceConfig()
DEFAULT_ALPACA_CONFIG = AlpacaConfig()
DEFAULT_EXCHANGE_CONFIG = ExchangeConfig()


def load_exchange_config(
    path: Optional[Union[str, Path]] = None,
    vendor: Optional[str] = None,
) -> ExchangeConfig:
    """
    Load exchange configuration.

    Priority:
    1. Explicit path
    2. Environment variable EXCHANGE_CONFIG_PATH
    3. Default config with optional vendor override

    Args:
        path: Path to YAML config file
        vendor: Override vendor (if not loading from file)

    Returns:
        ExchangeConfig instance
    """
    if path is not None:
        return ExchangeConfig.from_yaml(path)

    env_path = os.environ.get("EXCHANGE_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return ExchangeConfig.from_yaml(env_path)

    config = ExchangeConfig()
    if vendor:
        config.vendor = vendor

    return config
