# -*- coding: utf-8 -*-
"""
adapters/registry.py
Adapter registry and factory for multi-exchange support.

This module provides:
- Global registry for adapter implementations
- Factory functions for creating adapters
- Configuration-driven adapter instantiation
- Lazy loading of adapter modules

Design:
    Registry uses a two-level structure:
    1. Vendor-level: Maps ExchangeVendor to adapter module
    2. Interface-level: Maps adapter type to implementation class

Usage:
    # Register adapters (usually done in adapter module __init__)
    registry.register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=BinanceMarketDataAdapter,
    )

    # Create adapter via factory
    adapter = registry.create_adapter(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.MARKET_DATA,
        config={"api_key": "..."},
    )

    # Or use convenience functions
    adapter = create_market_data_adapter("binance", config)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
)

from .models import ExchangeVendor, MarketType
from .base import (
    BaseAdapter,
    ExchangeAdapter,
    ExchangeInfoAdapter,
    FeeAdapter,
    MarketDataAdapter,
    OrderExecutionAdapter,
    TradingHoursAdapter,
)

logger = logging.getLogger(__name__)

# Type variable for adapter classes
A = TypeVar("A", bound=BaseAdapter)


# =========================
# Adapter Types
# =========================

class AdapterType(str, Enum):
    """Types of adapters that can be registered."""
    MARKET_DATA = "market_data"
    FEE = "fee"
    TRADING_HOURS = "trading_hours"
    ORDER_EXECUTION = "order_execution"
    EXCHANGE_INFO = "exchange_info"
    COMBINED = "combined"  # ExchangeAdapter implementing all interfaces


# Mapping from AdapterType to base class
ADAPTER_BASE_CLASSES: Dict[AdapterType, Type[BaseAdapter]] = {
    AdapterType.MARKET_DATA: MarketDataAdapter,
    AdapterType.FEE: FeeAdapter,
    AdapterType.TRADING_HOURS: TradingHoursAdapter,
    AdapterType.ORDER_EXECUTION: OrderExecutionAdapter,
    AdapterType.EXCHANGE_INFO: ExchangeInfoAdapter,
    AdapterType.COMBINED: ExchangeAdapter,
}


# =========================
# Registration Entry
# =========================

@dataclass
class AdapterRegistration:
    """Registration entry for an adapter implementation."""
    vendor: ExchangeVendor
    adapter_type: AdapterType
    adapter_class: Type[BaseAdapter]
    module_path: Optional[str] = None  # For lazy loading
    factory_func: Optional[Callable[..., BaseAdapter]] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def create(self, config: Optional[Mapping[str, Any]] = None) -> BaseAdapter:
        """
        Create adapter instance.

        Args:
            config: Configuration to merge with defaults

        Returns:
            Adapter instance
        """
        merged_config = dict(self.default_config)
        if config:
            merged_config.update(config)

        if self.factory_func is not None:
            return self.factory_func(vendor=self.vendor, config=merged_config)

        return self.adapter_class(vendor=self.vendor, config=merged_config)


# =========================
# Adapter Registry
# =========================

class AdapterRegistry:
    """
    Global registry for adapter implementations.

    Thread-safe singleton registry that manages adapter registrations
    and provides factory methods for creating adapter instances.
    """

    _instance: Optional["AdapterRegistry"] = None

    def __new__(cls) -> "AdapterRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        # Registry: vendor -> adapter_type -> registration
        self._registrations: Dict[ExchangeVendor, Dict[AdapterType, AdapterRegistration]] = {}

        # Lazy loading modules: vendor -> module path
        self._lazy_modules: Dict[ExchangeVendor, str] = {
            ExchangeVendor.BINANCE: "adapters.binance",
            ExchangeVendor.BINANCE_US: "adapters.binance",
            ExchangeVendor.ALPACA: "adapters.alpaca",
        }

        # Track which vendors have been loaded
        self._loaded_vendors: set[ExchangeVendor] = set()

        self._initialized = True

    def register(
        self,
        vendor: ExchangeVendor,
        adapter_type: AdapterType,
        adapter_class: Type[BaseAdapter],
        *,
        factory_func: Optional[Callable[..., BaseAdapter]] = None,
        default_config: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        """
        Register an adapter implementation.

        Args:
            vendor: Exchange vendor
            adapter_type: Type of adapter
            adapter_class: Adapter implementation class
            factory_func: Optional factory function for custom instantiation
            default_config: Default configuration
            description: Human-readable description
        """
        if vendor not in self._registrations:
            self._registrations[vendor] = {}

        registration = AdapterRegistration(
            vendor=vendor,
            adapter_type=adapter_type,
            adapter_class=adapter_class,
            factory_func=factory_func,
            default_config=default_config or {},
            description=description,
        )

        self._registrations[vendor][adapter_type] = registration

        logger.debug(
            f"Registered adapter: {vendor.value}/{adapter_type.value} -> {adapter_class.__name__}"
        )

    def unregister(
        self,
        vendor: ExchangeVendor,
        adapter_type: Optional[AdapterType] = None,
    ) -> None:
        """
        Unregister adapter(s).

        Args:
            vendor: Exchange vendor
            adapter_type: Specific adapter type, or None to remove all for vendor
        """
        if vendor not in self._registrations:
            return

        if adapter_type is None:
            del self._registrations[vendor]
        elif adapter_type in self._registrations[vendor]:
            del self._registrations[vendor][adapter_type]

    def get_registration(
        self,
        vendor: ExchangeVendor,
        adapter_type: AdapterType,
    ) -> Optional[AdapterRegistration]:
        """
        Get registration for vendor/type.

        Args:
            vendor: Exchange vendor
            adapter_type: Adapter type

        Returns:
            AdapterRegistration or None
        """
        self._ensure_loaded(vendor)
        vendor_registrations = self._registrations.get(vendor, {})
        return vendor_registrations.get(adapter_type)

    def create_adapter(
        self,
        vendor: Union[ExchangeVendor, str],
        adapter_type: Union[AdapterType, str],
        config: Optional[Mapping[str, Any]] = None,
    ) -> BaseAdapter:
        """
        Create adapter instance from registry.

        Args:
            vendor: Exchange vendor (enum or string)
            adapter_type: Adapter type (enum or string)
            config: Configuration for adapter

        Returns:
            Adapter instance

        Raises:
            ValueError: If vendor/type combination not registered
        """
        # Normalize inputs
        if isinstance(vendor, str):
            try:
                vendor = ExchangeVendor(vendor.lower())
            except ValueError:
                raise ValueError(f"Unknown vendor: {vendor}")

        if isinstance(adapter_type, str):
            try:
                adapter_type = AdapterType(adapter_type.lower())
            except ValueError:
                raise ValueError(f"Unknown adapter type: {adapter_type}")

        registration = self.get_registration(vendor, adapter_type)

        if registration is None:
            # Try combined adapter as fallback
            combined = self.get_registration(vendor, AdapterType.COMBINED)
            if combined is not None:
                return combined.create(config)

            raise ValueError(
                f"No adapter registered for {vendor.value}/{adapter_type.value}"
            )

        return registration.create(config)

    def list_registrations(
        self,
        vendor: Optional[ExchangeVendor] = None,
    ) -> List[AdapterRegistration]:
        """
        List all registrations.

        Args:
            vendor: Optional filter by vendor

        Returns:
            List of registrations
        """
        result: List[AdapterRegistration] = []

        vendors = [vendor] if vendor else list(ExchangeVendor)

        for v in vendors:
            self._ensure_loaded(v)
            vendor_regs = self._registrations.get(v, {})
            result.extend(vendor_regs.values())

        return result

    def get_supported_vendors(self) -> List[ExchangeVendor]:
        """
        Get list of vendors with registered adapters.

        Returns:
            List of supported vendors
        """
        # Load all lazy modules first
        for vendor in self._lazy_modules:
            self._ensure_loaded(vendor)

        return list(self._registrations.keys())

    def get_supported_types(
        self,
        vendor: ExchangeVendor,
    ) -> List[AdapterType]:
        """
        Get adapter types supported by vendor.

        Args:
            vendor: Exchange vendor

        Returns:
            List of supported adapter types
        """
        self._ensure_loaded(vendor)
        return list(self._registrations.get(vendor, {}).keys())

    def _ensure_loaded(self, vendor: ExchangeVendor) -> None:
        """Ensure vendor module is loaded (lazy loading)."""
        if vendor in self._loaded_vendors:
            return

        module_path = self._lazy_modules.get(vendor)
        if module_path:
            try:
                importlib.import_module(module_path)
                logger.debug(f"Lazy-loaded adapter module: {module_path}")
            except ImportError as e:
                logger.warning(f"Could not load adapter module {module_path}: {e}")

        self._loaded_vendors.add(vendor)


# =========================
# Global Registry Instance
# =========================

# Singleton registry instance
_registry: Optional[AdapterRegistry] = None


def get_registry() -> AdapterRegistry:
    """Get global adapter registry instance."""
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def register(
    vendor: ExchangeVendor,
    adapter_type: AdapterType,
    adapter_class: Type[BaseAdapter],
    **kwargs: Any,
) -> None:
    """Register adapter in global registry."""
    get_registry().register(vendor, adapter_type, adapter_class, **kwargs)


# =========================
# Convenience Factory Functions
# =========================

def create_market_data_adapter(
    vendor: Union[ExchangeVendor, str],
    config: Optional[Mapping[str, Any]] = None,
) -> MarketDataAdapter:
    """
    Create market data adapter.

    Args:
        vendor: Exchange vendor
        config: Adapter configuration

    Returns:
        MarketDataAdapter instance
    """
    adapter = get_registry().create_adapter(vendor, AdapterType.MARKET_DATA, config)
    if not isinstance(adapter, MarketDataAdapter):
        raise TypeError(f"Expected MarketDataAdapter, got {type(adapter)}")
    return adapter


def create_fee_adapter(
    vendor: Union[ExchangeVendor, str],
    config: Optional[Mapping[str, Any]] = None,
) -> FeeAdapter:
    """
    Create fee adapter.

    Args:
        vendor: Exchange vendor
        config: Adapter configuration

    Returns:
        FeeAdapter instance
    """
    adapter = get_registry().create_adapter(vendor, AdapterType.FEE, config)
    if not isinstance(adapter, FeeAdapter):
        raise TypeError(f"Expected FeeAdapter, got {type(adapter)}")
    return adapter


def create_trading_hours_adapter(
    vendor: Union[ExchangeVendor, str],
    config: Optional[Mapping[str, Any]] = None,
) -> TradingHoursAdapter:
    """
    Create trading hours adapter.

    Args:
        vendor: Exchange vendor
        config: Adapter configuration

    Returns:
        TradingHoursAdapter instance
    """
    adapter = get_registry().create_adapter(vendor, AdapterType.TRADING_HOURS, config)
    if not isinstance(adapter, TradingHoursAdapter):
        raise TypeError(f"Expected TradingHoursAdapter, got {type(adapter)}")
    return adapter


def create_order_execution_adapter(
    vendor: Union[ExchangeVendor, str],
    config: Optional[Mapping[str, Any]] = None,
) -> OrderExecutionAdapter:
    """
    Create order execution adapter.

    Args:
        vendor: Exchange vendor
        config: Adapter configuration

    Returns:
        OrderExecutionAdapter instance
    """
    adapter = get_registry().create_adapter(vendor, AdapterType.ORDER_EXECUTION, config)
    if not isinstance(adapter, OrderExecutionAdapter):
        raise TypeError(f"Expected OrderExecutionAdapter, got {type(adapter)}")
    return adapter


def create_exchange_info_adapter(
    vendor: Union[ExchangeVendor, str],
    config: Optional[Mapping[str, Any]] = None,
) -> ExchangeInfoAdapter:
    """
    Create exchange info adapter.

    Args:
        vendor: Exchange vendor
        config: Adapter configuration

    Returns:
        ExchangeInfoAdapter instance
    """
    adapter = get_registry().create_adapter(vendor, AdapterType.EXCHANGE_INFO, config)
    if not isinstance(adapter, ExchangeInfoAdapter):
        raise TypeError(f"Expected ExchangeInfoAdapter, got {type(adapter)}")
    return adapter


def create_exchange_adapter(
    vendor: Union[ExchangeVendor, str],
    config: Optional[Mapping[str, Any]] = None,
) -> ExchangeAdapter:
    """
    Create combined exchange adapter.

    Args:
        vendor: Exchange vendor
        config: Adapter configuration

    Returns:
        ExchangeAdapter instance (or equivalent combined adapter)
    """
    adapter = get_registry().create_adapter(vendor, AdapterType.COMBINED, config)
    return adapter  # type: ignore


# =========================
# Configuration-Driven Creation
# =========================

@dataclass
class AdapterConfig:
    """Configuration for adapter creation."""
    vendor: str
    adapter_type: str = "combined"
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AdapterConfig":
        return cls(
            vendor=str(d.get("vendor", "binance")),
            adapter_type=str(d.get("adapter_type", "combined")),
            config=dict(d.get("config", {})),
        )


def create_from_config(adapter_config: Union[AdapterConfig, Mapping[str, Any]]) -> BaseAdapter:
    """
    Create adapter from configuration.

    Args:
        adapter_config: AdapterConfig or dict with vendor/adapter_type/config

    Returns:
        Adapter instance
    """
    if isinstance(adapter_config, Mapping):
        adapter_config = AdapterConfig.from_dict(adapter_config)

    return get_registry().create_adapter(
        vendor=adapter_config.vendor,
        adapter_type=adapter_config.adapter_type,
        config=adapter_config.config,
    )


# =========================
# Decorator for Registration
# =========================

def register_adapter(
    vendor: ExchangeVendor,
    adapter_type: AdapterType,
    **kwargs: Any,
) -> Callable[[Type[A]], Type[A]]:
    """
    Decorator for registering adapter class.

    Usage:
        @register_adapter(ExchangeVendor.BINANCE, AdapterType.MARKET_DATA)
        class BinanceMarketDataAdapter(MarketDataAdapter):
            ...
    """
    def decorator(cls: Type[A]) -> Type[A]:
        register(vendor, adapter_type, cls, **kwargs)
        return cls

    return decorator
