# -*- coding: utf-8 -*-
"""
services/forex_config.py
Forex Configuration Loader and Validator

Phase 8: Configuration System (2025-11-30)

This module provides:
1. ForexConfig - Pydantic model for forex configuration
2. ForexConfigLoader - Load and validate configuration from YAML files
3. ForexConfigValidator - Validate configuration against rules
4. Factory functions for creating forex-related objects from config

Usage:
    from services.forex_config import (
        load_forex_config,
        create_forex_slippage_provider,
        create_forex_dealer_simulator,
    )

    # Load from YAML
    config = load_forex_config("configs/forex_defaults.yaml")

    # Create providers
    slippage_provider = create_forex_slippage_provider(config)
    dealer_sim = create_forex_dealer_simulator(config)

References:
    - Pydantic v2 docs: https://docs.pydantic.dev/
    - YAML config patterns in the codebase
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default configuration paths
DEFAULT_FOREX_CONFIG_PATH = "configs/forex_defaults.yaml"
DEFAULT_ASSET_CLASS_DEFAULTS_PATH = "configs/asset_class_defaults.yaml"
DEFAULT_EXCHANGE_CONFIG_PATH = "configs/exchange.yaml"
DEFAULT_SLIPPAGE_CONFIG_PATH = "configs/slippage.yaml"

# Environment variable prefix for overrides
ENV_PREFIX = "FOREX_"

# Supported vendors
SUPPORTED_VENDORS = frozenset({"oanda", "ig", "dukascopy"})

# Valid leverage ranges
MIN_LEVERAGE = 1.0
MAX_LEVERAGE = 500.0

# Session names
VALID_SESSIONS = frozenset({
    "sydney", "tokyo", "london", "new_york",
    "london_ny_overlap", "tokyo_london_overlap",
    "off_hours", "weekend"
})

# Pair categories
VALID_PAIR_CATEGORIES = frozenset({"major", "minor", "cross", "exotic"})


# =============================================================================
# Enumerations
# =============================================================================


class ForexVendor(str, Enum):
    """Supported forex data vendors."""
    OANDA = "oanda"
    IG = "ig"
    DUKASCOPY = "dukascopy"


class ForexMarketType(str, Enum):
    """Forex market types."""
    SPOT = "spot"
    FORWARD = "forward"
    SWAP = "swap"


class ForexFeeStructure(str, Enum):
    """Fee structure types."""
    SPREAD_ONLY = "spread_only"
    RAW_SPREAD_COMMISSION = "raw_spread_commission"
    ECN = "ecn"


class ForexSlippageLevel(str, Enum):
    """Slippage model levels."""
    L2 = "L2"
    L2_PLUS = "L2+"


class ForexJurisdiction(str, Enum):
    """Regulatory jurisdictions."""
    RETAIL_US = "retail_us"
    RETAIL_EU = "retail_eu"
    RETAIL_UK = "retail_uk"
    RETAIL_AU = "retail_au"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"


class ConfigValidationSeverity(str, Enum):
    """Validation message severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# =============================================================================
# Data Classes - Session Configuration
# =============================================================================


@dataclass
class SessionConfig:
    """Individual session configuration."""
    start_utc: int
    end_utc: int
    liquidity_factor: float = 1.0
    spread_multiplier: float = 1.0

    def __post_init__(self):
        """Validate session times."""
        if not 0 <= self.start_utc <= 23:
            raise ValueError(f"start_utc must be 0-23, got {self.start_utc}")
        if not 0 <= self.end_utc <= 23:
            raise ValueError(f"end_utc must be 0-23, got {self.end_utc}")
        if self.liquidity_factor < 0:
            raise ValueError(f"liquidity_factor must be >= 0, got {self.liquidity_factor}")
        if self.spread_multiplier <= 0:
            raise ValueError(f"spread_multiplier must be > 0, got {self.spread_multiplier}")


@dataclass
class TradingSessionConfig:
    """Complete trading session configuration."""
    calendar: str = "forex_24x5"
    weekend_filter: bool = True
    rollover_time_et: int = 17
    rollover_keepout_minutes: int = 30
    dst_aware: bool = True
    sessions: Dict[str, SessionConfig] = field(default_factory=dict)
    overlaps: Dict[str, SessionConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSessionConfig":
        """Create from dictionary."""
        sessions = {}
        if "sessions" in data:
            for name, sess_data in data["sessions"].items():
                sessions[name] = SessionConfig(**sess_data)

        overlaps = {}
        if "overlaps" in data:
            for name, overlap_data in data["overlaps"].items():
                overlaps[name] = SessionConfig(**overlap_data)

        return cls(
            calendar=data.get("calendar", "forex_24x5"),
            weekend_filter=data.get("weekend_filter", True),
            rollover_time_et=data.get("rollover_time_et", 17),
            rollover_keepout_minutes=data.get("rollover_keepout_minutes", 30),
            dst_aware=data.get("dst_aware", True),
            sessions=sessions,
            overlaps=overlaps,
        )


# =============================================================================
# Data Classes - Fee Configuration
# =============================================================================


@dataclass
class SpreadProfile:
    """Spread profile for a pair category."""
    major: float = 0.3
    minor: float = 0.8
    cross: float = 1.5
    exotic: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SpreadProfile":
        """Create from dictionary."""
        return cls(
            major=data.get("major", 0.3),
            minor=data.get("minor", 0.8),
            cross=data.get("cross", 1.5),
            exotic=data.get("exotic", 10.0),
        )


@dataclass
class FeeConfig:
    """Fee configuration."""
    structure: ForexFeeStructure = ForexFeeStructure.SPREAD_ONLY
    maker_bps: float = 0.0
    taker_bps: float = 0.0
    commission_per_lot: float = 0.0
    swap_enabled: bool = True
    swap_data_source: str = "oanda"
    swap_cache_path: str = "data/forex/swap_rates.json"
    swap_cache_ttl_hours: int = 24
    wednesday_triple_swap: bool = True
    spread_profiles: Dict[str, SpreadProfile] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeeConfig":
        """Create from dictionary."""
        structure = ForexFeeStructure(data.get("structure", "spread_only"))

        spread_profiles = {}
        if "spread_profiles" in data:
            for profile_name, profile_data in data["spread_profiles"].items():
                spread_profiles[profile_name] = SpreadProfile.from_dict(profile_data)

        return cls(
            structure=structure,
            maker_bps=data.get("maker_bps", 0.0),
            taker_bps=data.get("taker_bps", 0.0),
            commission_per_lot=data.get("commission_per_lot", 0.0),
            swap_enabled=data.get("swap_enabled", True),
            swap_data_source=data.get("swap_data_source", "oanda"),
            swap_cache_path=data.get("swap_cache_path", "data/forex/swap_rates.json"),
            swap_cache_ttl_hours=data.get("swap_cache_ttl_hours", 24),
            wednesday_triple_swap=data.get("wednesday_triple_swap", True),
            spread_profiles=spread_profiles,
        )


# =============================================================================
# Data Classes - Slippage Configuration
# =============================================================================


@dataclass
class SlippageConfig:
    """Slippage model configuration."""
    level: ForexSlippageLevel = ForexSlippageLevel.L2_PLUS
    provider: str = "ForexParametricSlippageProvider"
    profile: str = "retail"
    impact_coef_base: float = 0.03
    impact_coef_range: Tuple[float, float] = (0.02, 0.05)
    spread_pips: float = 1.2
    session_adjustment: bool = True
    volatility_adjustment: bool = True
    carry_stress_adjustment: bool = True
    dxy_correlation_adjustment: bool = True
    news_event_adjustment: bool = True
    asymmetric_impact: bool = True
    min_slippage_pips: float = 0.1
    max_slippage_pips: float = 50.0
    whale_threshold: float = 0.005
    whale_twap_adjustment: float = 0.75

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlippageConfig":
        """Create from dictionary."""
        impact_range = data.get("impact_coef_range", [0.02, 0.05])
        if isinstance(impact_range, list):
            impact_range = tuple(impact_range)

        level_str = data.get("level", "L2+")
        level = ForexSlippageLevel.L2_PLUS if "+" in level_str else ForexSlippageLevel.L2

        return cls(
            level=level,
            provider=data.get("provider", "ForexParametricSlippageProvider"),
            profile=data.get("profile", "retail"),
            impact_coef_base=data.get("impact_coef_base", 0.03),
            impact_coef_range=impact_range,
            spread_pips=data.get("spread_pips", 1.2),
            session_adjustment=data.get("session_adjustment", True),
            volatility_adjustment=data.get("volatility_adjustment", True),
            carry_stress_adjustment=data.get("carry_stress_adjustment", True),
            dxy_correlation_adjustment=data.get("dxy_correlation_adjustment", True),
            news_event_adjustment=data.get("news_event_adjustment", True),
            asymmetric_impact=data.get("asymmetric_impact", True),
            min_slippage_pips=data.get("min_slippage_pips", 0.1),
            max_slippage_pips=data.get("max_slippage_pips", 50.0),
            whale_threshold=data.get("whale_threshold", 0.005),
            whale_twap_adjustment=data.get("whale_twap_adjustment", 0.75),
        )


# =============================================================================
# Data Classes - Dealer Simulation Configuration
# =============================================================================


@dataclass
class DealerTierConfig:
    """Configuration for a dealer tier."""
    count: int = 2
    spread_factor: float = 1.0
    max_size_usd: float = 5_000_000.0
    base_reject_prob: float = 0.05
    last_look_window_ms: int = 200


@dataclass
class DealerSimulationConfig:
    """OTC dealer simulation configuration."""
    enabled: bool = True
    provider: str = "ForexDealerSimulator"
    num_dealers: int = 5
    dealer_profiles: Dict[str, DealerTierConfig] = field(default_factory=dict)
    last_look_enabled: bool = True
    adverse_selection_threshold_pips: float = 0.3
    latency_arbitrage_detection: bool = True
    quote_flickering_enabled: bool = True
    quote_validity_ms: int = 200
    rfq_threshold_usd: float = 5_000_000.0
    track_execution_stats: bool = True
    stats_window_trades: int = 1000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DealerSimulationConfig":
        """Create from dictionary."""
        profiles = {}
        if "dealer_profiles" in data:
            for tier_name, tier_data in data["dealer_profiles"].items():
                profiles[tier_name] = DealerTierConfig(**tier_data)

        return cls(
            enabled=data.get("enabled", True),
            provider=data.get("provider", "ForexDealerSimulator"),
            num_dealers=data.get("num_dealers", 5),
            dealer_profiles=profiles,
            last_look_enabled=data.get("last_look_enabled", True),
            adverse_selection_threshold_pips=data.get("adverse_selection_threshold_pips", 0.3),
            latency_arbitrage_detection=data.get("latency_arbitrage_detection", True),
            quote_flickering_enabled=data.get("quote_flickering_enabled", True),
            quote_validity_ms=data.get("quote_validity_ms", 200),
            rfq_threshold_usd=data.get("rfq_threshold_usd", 5_000_000.0),
            track_execution_stats=data.get("track_execution_stats", True),
            stats_window_trades=data.get("stats_window_trades", 1000),
        )


# =============================================================================
# Data Classes - Leverage Configuration
# =============================================================================


@dataclass
class LeverageConfig:
    """Leverage configuration."""
    max_leverage: float = 50.0
    default_leverage: float = 30.0
    margin_warning: float = 1.20
    margin_call: float = 1.00
    stop_out: float = 0.50
    by_category: Dict[str, int] = field(default_factory=dict)
    jurisdictions: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate leverage values."""
        if not MIN_LEVERAGE <= self.max_leverage <= MAX_LEVERAGE:
            raise ValueError(
                f"max_leverage must be {MIN_LEVERAGE}-{MAX_LEVERAGE}, got {self.max_leverage}"
            )
        if self.default_leverage > self.max_leverage:
            raise ValueError(
                f"default_leverage ({self.default_leverage}) cannot exceed max_leverage ({self.max_leverage})"
            )
        if not 0 < self.stop_out < self.margin_call < self.margin_warning:
            raise ValueError(
                f"Must have 0 < stop_out < margin_call < margin_warning"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeverageConfig":
        """Create from dictionary."""
        return cls(
            max_leverage=data.get("max_leverage", 50.0),
            default_leverage=data.get("default_leverage", 30.0),
            margin_warning=data.get("margin_warning", 1.20),
            margin_call=data.get("margin_call", 1.00),
            stop_out=data.get("stop_out", 0.50),
            by_category=data.get("by_category", {}),
            jurisdictions=data.get("jurisdictions", {}),
        )


# =============================================================================
# Data Classes - Position Sync Configuration
# =============================================================================


@dataclass
class PositionSyncConfig:
    """Position synchronization configuration."""
    enabled: bool = True
    interval_sec: float = 30.0
    position_tolerance_pct: float = 0.01
    price_tolerance_pct: float = 0.01
    auto_reconcile: bool = False
    max_reconcile_units: float = 100_000.0
    alert_on_discrepancy: bool = True
    track_financing: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionSyncConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            interval_sec=data.get("interval_sec", 30.0),
            position_tolerance_pct=data.get("position_tolerance_pct", 0.01),
            price_tolerance_pct=data.get("price_tolerance_pct", 0.01),
            auto_reconcile=data.get("auto_reconcile", False),
            max_reconcile_units=data.get("max_reconcile_units", 100_000.0),
            alert_on_discrepancy=data.get("alert_on_discrepancy", True),
            track_financing=data.get("track_financing", True),
        )


# =============================================================================
# Data Classes - Data Sources Configuration
# =============================================================================


@dataclass
class DataSourcesConfig:
    """Data sources configuration."""
    price_data: str = "oanda"
    interest_rates: str = "fred"
    economic_calendar: str = "forexfactory"
    swap_rates: str = "oanda"
    dxy: str = "yahoo"
    cot_data: str = "cftc"
    fallbacks: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSourcesConfig":
        """Create from dictionary."""
        return cls(
            price_data=data.get("price_data", "oanda"),
            interest_rates=data.get("interest_rates", "fred"),
            economic_calendar=data.get("economic_calendar", "forexfactory"),
            swap_rates=data.get("swap_rates", "oanda"),
            dxy=data.get("dxy", "yahoo"),
            cot_data=data.get("cot_data", "cftc"),
            fallbacks=data.get("fallbacks", {}),
        )


# =============================================================================
# Data Classes - Rate Limits Configuration
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a vendor."""
    requests_per_second: int = 120
    burst: int = 200
    streaming_connections: int = 20

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateLimitConfig":
        """Create from dictionary."""
        return cls(
            requests_per_second=data.get("requests_per_second", 120),
            burst=data.get("burst", 200),
            streaming_connections=data.get("streaming_connections", 20),
        )


# =============================================================================
# Main Configuration Class
# =============================================================================


@dataclass
class ForexConfig:
    """
    Complete forex configuration.

    Attributes:
        asset_class: Asset class identifier (always "forex")
        data_vendor: Primary data vendor
        market: Market type (spot, forward, swap)
        session: Trading session configuration
        fees: Fee configuration
        slippage: Slippage model configuration
        dealer_simulation: OTC dealer simulation configuration
        leverage: Leverage configuration
        position_sync: Position synchronization configuration
        data_sources: Data sources configuration
        pairs: Currency pair lists by category
        pip_sizes: Pip sizes by pair type
        rate_limits: Rate limits by vendor
    """
    asset_class: str = "forex"
    data_vendor: ForexVendor = ForexVendor.OANDA
    market: ForexMarketType = ForexMarketType.SPOT
    session: TradingSessionConfig = field(default_factory=TradingSessionConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    dealer_simulation: DealerSimulationConfig = field(default_factory=DealerSimulationConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    position_sync: PositionSyncConfig = field(default_factory=PositionSyncConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    pairs: Dict[str, List[str]] = field(default_factory=dict)
    pip_sizes: Dict[str, float] = field(default_factory=dict)
    rate_limits: Dict[str, RateLimitConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForexConfig":
        """
        Create ForexConfig from a dictionary.

        Args:
            data: Configuration dictionary (typically from YAML)

        Returns:
            ForexConfig instance
        """
        forex_data = data.get("forex", data)

        # Parse vendor
        vendor_str = forex_data.get("data_vendor", "oanda")
        vendor = ForexVendor(vendor_str.lower())

        # Parse market type
        market_str = forex_data.get("market", "spot")
        market = ForexMarketType(market_str.lower())

        # Parse sub-configurations
        session = TradingSessionConfig.from_dict(forex_data.get("session", {}))
        fees = FeeConfig.from_dict(forex_data.get("fees", {}))
        slippage = SlippageConfig.from_dict(forex_data.get("slippage", {}))
        dealer_sim = DealerSimulationConfig.from_dict(forex_data.get("dealer_simulation", {}))
        leverage = LeverageConfig.from_dict(forex_data.get("leverage", {}))
        position_sync = PositionSyncConfig.from_dict(forex_data.get("position_sync", {}))
        data_sources = DataSourcesConfig.from_dict(forex_data.get("data_sources", {}))

        # Parse pairs
        pairs = forex_data.get("pairs", {})

        # Parse pip sizes
        pip_sizes = forex_data.get("pip_sizes", {"standard": 0.0001, "jpy": 0.01})

        # Parse rate limits
        rate_limits_data = data.get("rate_limits", {})
        rate_limits = {}
        for vendor_name, limits in rate_limits_data.items():
            rate_limits[vendor_name] = RateLimitConfig.from_dict(limits)

        return cls(
            asset_class="forex",
            data_vendor=vendor,
            market=market,
            session=session,
            fees=fees,
            slippage=slippage,
            dealer_simulation=dealer_sim,
            leverage=leverage,
            position_sync=position_sync,
            data_sources=data_sources,
            pairs=pairs,
            pip_sizes=pip_sizes,
            rate_limits=rate_limits,
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ForexConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            ForexConfig instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file is invalid YAML
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "forex": {
                "asset_class": self.asset_class,
                "data_vendor": self.data_vendor.value,
                "market": self.market.value,
                "session": {
                    "calendar": self.session.calendar,
                    "weekend_filter": self.session.weekend_filter,
                    "rollover_time_et": self.session.rollover_time_et,
                    "rollover_keepout_minutes": self.session.rollover_keepout_minutes,
                    "dst_aware": self.session.dst_aware,
                },
                "fees": {
                    "structure": self.fees.structure.value,
                    "maker_bps": self.fees.maker_bps,
                    "taker_bps": self.fees.taker_bps,
                    "swap_enabled": self.fees.swap_enabled,
                },
                "slippage": {
                    "level": self.slippage.level.value,
                    "profile": self.slippage.profile,
                    "impact_coef_base": self.slippage.impact_coef_base,
                    "spread_pips": self.slippage.spread_pips,
                },
                "leverage": {
                    "max_leverage": self.leverage.max_leverage,
                    "default_leverage": self.leverage.default_leverage,
                    "margin_warning": self.leverage.margin_warning,
                    "margin_call": self.leverage.margin_call,
                    "stop_out": self.leverage.stop_out,
                },
            },
            "rate_limits": {
                name: {
                    "requests_per_second": rl.requests_per_second,
                    "burst": rl.burst,
                    "streaming_connections": rl.streaming_connections,
                }
                for name, rl in self.rate_limits.items()
            },
        }


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationMessage:
    """Validation message."""
    severity: ConfigValidationSeverity
    field: str
    message: str
    suggestion: Optional[str] = None


class ForexConfigValidator:
    """Validates forex configuration."""

    def __init__(self):
        """Initialize validator."""
        self.messages: List[ValidationMessage] = []

    def validate(self, config: ForexConfig) -> List[ValidationMessage]:
        """
        Validate a forex configuration.

        Args:
            config: ForexConfig to validate

        Returns:
            List of validation messages (empty if valid)
        """
        self.messages = []

        self._validate_vendor(config)
        self._validate_leverage(config)
        self._validate_session(config)
        self._validate_slippage(config)
        self._validate_dealer_simulation(config)
        self._validate_fees(config)

        return self.messages

    def _validate_vendor(self, config: ForexConfig) -> None:
        """Validate vendor configuration."""
        if config.data_vendor.value not in SUPPORTED_VENDORS:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.ERROR,
                field="data_vendor",
                message=f"Unsupported vendor: {config.data_vendor.value}",
                suggestion=f"Use one of: {', '.join(SUPPORTED_VENDORS)}",
            ))

    def _validate_leverage(self, config: ForexConfig) -> None:
        """Validate leverage configuration."""
        if config.leverage.max_leverage > 50:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.WARNING,
                field="leverage.max_leverage",
                message=f"Leverage {config.leverage.max_leverage} exceeds CFTC 50:1 limit for US retail",
                suggestion="Set max_leverage <= 50 for US retail compliance",
            ))

        if config.leverage.max_leverage > 500:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.ERROR,
                field="leverage.max_leverage",
                message=f"Leverage {config.leverage.max_leverage} exceeds maximum allowed (500:1)",
                suggestion="Set max_leverage <= 500",
            ))

    def _validate_session(self, config: ForexConfig) -> None:
        """Validate session configuration."""
        if not config.session.weekend_filter:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.WARNING,
                field="session.weekend_filter",
                message="Weekend filter disabled - forex markets are closed on weekends",
                suggestion="Enable weekend_filter to avoid weekend gap risk",
            ))

        if config.session.rollover_time_et != 17:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.INFO,
                field="session.rollover_time_et",
                message=f"Non-standard rollover time: {config.session.rollover_time_et} ET (standard is 5pm/17)",
            ))

    def _validate_slippage(self, config: ForexConfig) -> None:
        """Validate slippage configuration."""
        if config.slippage.impact_coef_base > 0.10:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.WARNING,
                field="slippage.impact_coef_base",
                message=f"High impact coefficient: {config.slippage.impact_coef_base} (typical forex is 0.02-0.05)",
                suggestion="Consider lowering impact_coef_base for forex",
            ))

        if config.slippage.max_slippage_pips > 100:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.WARNING,
                field="slippage.max_slippage_pips",
                message=f"Very high max slippage: {config.slippage.max_slippage_pips} pips",
                suggestion="Consider setting max_slippage_pips <= 50",
            ))

    def _validate_dealer_simulation(self, config: ForexConfig) -> None:
        """Validate dealer simulation configuration."""
        if not config.dealer_simulation.enabled:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.WARNING,
                field="dealer_simulation.enabled",
                message="Dealer simulation disabled - OTC market realism may be reduced",
                suggestion="Enable dealer_simulation for realistic forex execution",
            ))

        if config.dealer_simulation.num_dealers < 1:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.ERROR,
                field="dealer_simulation.num_dealers",
                message=f"Invalid num_dealers: {config.dealer_simulation.num_dealers}",
                suggestion="Set num_dealers >= 1",
            ))

    def _validate_fees(self, config: ForexConfig) -> None:
        """Validate fee configuration."""
        if config.fees.structure != ForexFeeStructure.SPREAD_ONLY:
            if config.fees.maker_bps == 0 and config.fees.taker_bps == 0:
                self.messages.append(ValidationMessage(
                    severity=ConfigValidationSeverity.INFO,
                    field="fees",
                    message=f"ECN fee structure but no commission set",
                    suggestion="Set commission_per_lot for ECN mode",
                ))

        if not config.fees.swap_enabled:
            self.messages.append(ValidationMessage(
                severity=ConfigValidationSeverity.WARNING,
                field="fees.swap_enabled",
                message="Swap costs disabled - overnight positions won't account for financing",
                suggestion="Enable swap_enabled for realistic cost modeling",
            ))

    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors)."""
        return not any(m.severity == ConfigValidationSeverity.ERROR for m in self.messages)

    def get_errors(self) -> List[ValidationMessage]:
        """Get error messages only."""
        return [m for m in self.messages if m.severity == ConfigValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationMessage]:
        """Get warning messages only."""
        return [m for m in self.messages if m.severity == ConfigValidationSeverity.WARNING]


# =============================================================================
# Configuration Loader
# =============================================================================


class ForexConfigLoader:
    """
    Loads and merges forex configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables (FOREX_*)
    2. User config file (if provided)
    3. Default config file (configs/forex_defaults.yaml)
    """

    def __init__(
        self,
        default_path: str = DEFAULT_FOREX_CONFIG_PATH,
        validate: bool = True,
    ):
        """
        Initialize loader.

        Args:
            default_path: Path to default configuration file
            validate: Whether to validate configuration after loading
        """
        self.default_path = Path(default_path)
        self.validate = validate
        self.validator = ForexConfigValidator()

    def load(
        self,
        user_config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ForexConfig:
        """
        Load configuration with optional overrides.

        Args:
            user_config_path: Optional user configuration file path
            overrides: Optional dictionary of override values

        Returns:
            ForexConfig instance

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If validation fails with errors
        """
        # Start with defaults
        config_data = {}
        if self.default_path.exists():
            with open(self.default_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        # Merge user config
        if user_config_path:
            user_path = Path(user_config_path)
            if user_path.exists():
                with open(user_path, "r", encoding="utf-8") as f:
                    user_data = yaml.safe_load(f) or {}
                config_data = self._deep_merge(config_data, user_data)

        # Apply overrides
        if overrides:
            config_data = self._deep_merge(config_data, overrides)

        # Apply environment variables
        config_data = self._apply_env_overrides(config_data)

        # Create config object
        config = ForexConfig.from_dict(config_data)

        # Validate
        if self.validate:
            messages = self.validator.validate(config)
            errors = self.validator.get_errors()
            if errors:
                error_msgs = "; ".join(f"{e.field}: {e.message}" for e in errors)
                raise ValueError(f"Configuration validation failed: {error_msgs}")

            for msg in self.validator.get_warnings():
                logger.warning(f"Config warning [{msg.field}]: {msg.message}")

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Check for common overrides
        env_mappings = {
            f"{ENV_PREFIX}VENDOR": ("forex", "data_vendor"),
            f"{ENV_PREFIX}MAX_LEVERAGE": ("forex", "leverage", "max_leverage"),
            f"{ENV_PREFIX}DEALER_ENABLED": ("forex", "dealer_simulation", "enabled"),
            f"{ENV_PREFIX}SWAP_ENABLED": ("forex", "fees", "swap_enabled"),
        }

        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config, path, self._parse_env_value(value))

        return config

    def _set_nested(self, d: Dict, path: Tuple[str, ...], value: Any) -> None:
        """Set a nested dictionary value by path."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        lower = value.lower()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
        try:
            return float(value)
        except ValueError:
            return value


# =============================================================================
# Factory Functions
# =============================================================================


def load_forex_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> ForexConfig:
    """
    Load forex configuration.

    Args:
        path: Optional path to configuration file (uses default if not provided)
        overrides: Optional dictionary of override values
        validate: Whether to validate configuration

    Returns:
        ForexConfig instance
    """
    loader = ForexConfigLoader(validate=validate)
    return loader.load(user_config_path=path, overrides=overrides)


def create_forex_slippage_provider(
    config: ForexConfig,
) -> Any:
    """
    Create a forex slippage provider from configuration.

    Args:
        config: ForexConfig instance

    Returns:
        ForexParametricSlippageProvider instance
    """
    from execution_providers import ForexParametricSlippageProvider, ForexParametricConfig

    provider_config = ForexParametricConfig(
        impact_coef_base=config.slippage.impact_coef_base,
        impact_coef_range=config.slippage.impact_coef_range,
        spread_pips=config.slippage.spread_pips,
        session_adjustment=config.slippage.session_adjustment,
        volatility_adjustment=config.slippage.volatility_adjustment,
        carry_stress_adjustment=config.slippage.carry_stress_adjustment,
        dxy_correlation_adjustment=config.slippage.dxy_correlation_adjustment,
        news_event_adjustment=config.slippage.news_event_adjustment,
        asymmetric_impact=config.slippage.asymmetric_impact,
        min_slippage_pips=config.slippage.min_slippage_pips,
        max_slippage_pips=config.slippage.max_slippage_pips,
        whale_threshold=config.slippage.whale_threshold,
        whale_twap_adjustment=config.slippage.whale_twap_adjustment,
    )

    return ForexParametricSlippageProvider(config=provider_config)


def create_forex_dealer_simulator(
    config: ForexConfig,
    seed: Optional[int] = None,
) -> Any:
    """
    Create a forex dealer simulator from configuration.

    Args:
        config: ForexConfig instance
        seed: Optional random seed

    Returns:
        ForexDealerSimulator instance
    """
    from services.forex_dealer import ForexDealerSimulator, ForexDealerConfig

    dealer_config = ForexDealerConfig(
        num_dealers=config.dealer_simulation.num_dealers,
        last_look_enabled=config.dealer_simulation.last_look_enabled,
        max_slippage_pips=config.slippage.max_slippage_pips,
        quote_refresh_interval_ms=config.dealer_simulation.quote_validity_ms,
        size_impact_threshold_usd=config.dealer_simulation.rfq_threshold_usd,
        max_history_size=config.dealer_simulation.stats_window_trades,
    )

    return ForexDealerSimulator(config=dealer_config, seed=seed)


def create_forex_fee_provider(config: ForexConfig) -> Any:
    """
    Create a forex fee provider from configuration.

    Args:
        config: ForexConfig instance

    Returns:
        ForexFeeProvider instance
    """
    from execution_providers import ForexFeeProvider

    return ForexFeeProvider(
        swap_enabled=config.fees.swap_enabled,
        swap_data_source=config.fees.swap_data_source,
    )


def get_pip_size(pair: str, config: Optional[ForexConfig] = None) -> float:
    """
    Get pip size for a currency pair.

    Args:
        pair: Currency pair (e.g., "EUR_USD", "USD_JPY")
        config: Optional ForexConfig instance

    Returns:
        Pip size (0.0001 for standard, 0.01 for JPY pairs)
    """
    # Check for JPY pairs
    pair_upper = pair.upper().replace("/", "_")
    if "JPY" in pair_upper:
        return 0.01

    # Check config overrides
    if config and config.pip_sizes:
        overrides = config.pip_sizes.get("overrides", {})
        if pair_upper in overrides:
            return overrides[pair_upper]

    # Default standard pip size
    return 0.0001


def pips_to_price(pips: float, pair: str, config: Optional[ForexConfig] = None) -> float:
    """
    Convert pips to price movement.

    Args:
        pips: Number of pips
        pair: Currency pair
        config: Optional ForexConfig instance

    Returns:
        Price movement
    """
    return pips * get_pip_size(pair, config)


def price_to_pips(price: float, pair: str, config: Optional[ForexConfig] = None) -> float:
    """
    Convert price movement to pips.

    Args:
        price: Price movement
        pair: Currency pair
        config: Optional ForexConfig instance

    Returns:
        Number of pips
    """
    pip_size = get_pip_size(pair, config)
    if pip_size == 0:
        return 0.0
    return price / pip_size
