# -*- coding: utf-8 -*-
"""
adapters/models.py
Exchange-agnostic data models for multi-exchange support.

These models provide a unified interface for different asset classes (crypto, stocks)
and exchanges (Binance, Alpaca, etc.) while maintaining compatibility with existing
core_models.py entities.

Design Principles:
- Immutable dataclasses for thread safety and clarity
- Decimal for financial precision (prices, quantities)
- int milliseconds UTC for timestamps
- Backward compatible with existing core_models.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Mapping, Tuple
import json


# =========================
# Enumerations
# =========================

class MarketType(str, Enum):
    """Type of market/asset class."""
    CRYPTO_SPOT = "CRYPTO_SPOT"
    CRYPTO_FUTURES = "CRYPTO_FUTURES"
    CRYPTO_PERP = "CRYPTO_PERP"
    EQUITY = "EQUITY"
    EQUITY_OPTIONS = "EQUITY_OPTIONS"
    FOREX = "FOREX"

    @property
    def is_crypto(self) -> bool:
        return self in (MarketType.CRYPTO_SPOT, MarketType.CRYPTO_FUTURES, MarketType.CRYPTO_PERP)

    @property
    def is_equity(self) -> bool:
        return self in (MarketType.EQUITY, MarketType.EQUITY_OPTIONS)

    @property
    def has_trading_hours(self) -> bool:
        """Returns True if this market type has trading hours (not 24/7)."""
        return self.is_equity or self == MarketType.FOREX


class ExchangeVendor(str, Enum):
    """Supported exchange vendors."""
    BINANCE = "binance"
    BINANCE_US = "binance_us"
    ALPACA = "alpaca"
    POLYGON = "polygon"  # Data provider
    UNKNOWN = "unknown"

    @property
    def market_type(self) -> MarketType:
        """Default market type for vendor."""
        if self in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
            return MarketType.CRYPTO_SPOT
        elif self == ExchangeVendor.ALPACA:
            return MarketType.EQUITY
        return MarketType.CRYPTO_SPOT


class FeeStructure(str, Enum):
    """Fee calculation structure type."""
    PERCENTAGE = "percentage"      # Fee as % of notional (e.g., 0.1% = 10 bps)
    PER_SHARE = "per_share"        # Fee per share/unit (e.g., $0.005/share)
    FLAT = "flat"                  # Flat fee per trade (e.g., $1.00)
    TIERED = "tiered"              # Tiered based on volume
    MIXED = "mixed"                # Combination of above


class SessionType(str, Enum):
    """Trading session type."""
    REGULAR = "regular"            # Regular market hours (9:30-16:00 ET for US)
    PRE_MARKET = "pre_market"      # Pre-market (4:00-9:30 ET)
    AFTER_HOURS = "after_hours"    # After hours (16:00-20:00 ET)
    EXTENDED = "extended"          # All extended hours
    CONTINUOUS = "continuous"      # 24/7 (crypto)


# =========================
# Exchange Rules
# =========================

@dataclass(frozen=True)
class ExchangeRule:
    """
    Trading rules and constraints for a symbol on an exchange.

    Generalizes Binance's filter concept to support multiple exchanges.
    Designed to be exchange-agnostic while capturing key trading constraints.

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
        tick_size: Minimum price increment (e.g., 0.01)
        step_size: Minimum quantity increment (e.g., 0.001)
        min_notional: Minimum order value in quote currency
        min_qty: Minimum order quantity
        max_qty: Maximum order quantity
        max_notional: Maximum order value (optional)
        price_precision: Decimal places for price
        qty_precision: Decimal places for quantity
        market_type: Type of market (crypto, equity)
        base_asset: Base asset symbol (e.g., "BTC", "AAPL")
        quote_asset: Quote asset symbol (e.g., "USDT", "USD")
        lot_size: Standard lot size (1 for most, 100 for some options)
        is_tradable: Whether symbol is currently tradable
        is_marginable: Whether margin trading is allowed
        is_shortable: Whether short selling is allowed
        raw_filters: Original exchange-specific filter data
    """
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_notional: Decimal
    min_qty: Decimal = Decimal("0")
    max_qty: Optional[Decimal] = None
    max_notional: Optional[Decimal] = None
    price_precision: int = 8
    qty_precision: int = 8
    market_type: MarketType = MarketType.CRYPTO_SPOT
    base_asset: str = ""
    quote_asset: str = ""
    lot_size: int = 1
    is_tradable: bool = True
    is_marginable: bool = False
    is_shortable: bool = False
    raw_filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate immutable constraints (done via object.__setattr__ for frozen)
        if self.tick_size <= Decimal("0"):
            object.__setattr__(self, 'tick_size', Decimal("0.00000001"))
        if self.step_size <= Decimal("0"):
            object.__setattr__(self, 'step_size', Decimal("0.00000001"))

    def quantize_price(self, price: Decimal) -> Decimal:
        """Round price to valid tick size."""
        if self.tick_size <= 0:
            return price
        return (price // self.tick_size) * self.tick_size

    def quantize_qty(self, qty: Decimal) -> Decimal:
        """Round quantity to valid step size."""
        if self.step_size <= 0:
            return qty
        return (qty // self.step_size) * self.step_size

    def validate_order(self, price: Decimal, qty: Decimal) -> Tuple[bool, Optional[str]]:
        """
        Validate order against exchange rules.

        Returns:
            (is_valid, error_message)
        """
        notional = price * qty

        if qty < self.min_qty:
            return False, f"Quantity {qty} below minimum {self.min_qty}"

        if self.max_qty is not None and qty > self.max_qty:
            return False, f"Quantity {qty} exceeds maximum {self.max_qty}"

        if notional < self.min_notional:
            return False, f"Notional {notional} below minimum {self.min_notional}"

        if self.max_notional is not None and notional > self.max_notional:
            return False, f"Notional {notional} exceeds maximum {self.max_notional}"

        if not self.is_tradable:
            return False, f"Symbol {self.symbol} is not tradable"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "tick_size": str(self.tick_size),
            "step_size": str(self.step_size),
            "min_notional": str(self.min_notional),
            "min_qty": str(self.min_qty),
            "max_qty": str(self.max_qty) if self.max_qty else None,
            "max_notional": str(self.max_notional) if self.max_notional else None,
            "price_precision": self.price_precision,
            "qty_precision": self.qty_precision,
            "market_type": self.market_type.value,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "lot_size": self.lot_size,
            "is_tradable": self.is_tradable,
            "is_marginable": self.is_marginable,
            "is_shortable": self.is_shortable,
            "raw_filters": self.raw_filters,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExchangeRule":
        def _to_decimal(v: Any, default: str = "0") -> Decimal:
            if v is None:
                return Decimal(default)
            try:
                return Decimal(str(v))
            except Exception:
                return Decimal(default)

        def _to_decimal_opt(v: Any) -> Optional[Decimal]:
            if v is None:
                return None
            try:
                return Decimal(str(v))
            except Exception:
                return None

        market_type_val = d.get("market_type", "CRYPTO_SPOT")
        if isinstance(market_type_val, MarketType):
            market_type = market_type_val
        else:
            try:
                market_type = MarketType(str(market_type_val))
            except ValueError:
                market_type = MarketType.CRYPTO_SPOT

        return cls(
            symbol=str(d.get("symbol", "")),
            tick_size=_to_decimal(d.get("tick_size"), "0.00000001"),
            step_size=_to_decimal(d.get("step_size"), "0.00000001"),
            min_notional=_to_decimal(d.get("min_notional"), "0"),
            min_qty=_to_decimal(d.get("min_qty"), "0"),
            max_qty=_to_decimal_opt(d.get("max_qty")),
            max_notional=_to_decimal_opt(d.get("max_notional")),
            price_precision=int(d.get("price_precision", 8)),
            qty_precision=int(d.get("qty_precision", 8)),
            market_type=market_type,
            base_asset=str(d.get("base_asset", "")),
            quote_asset=str(d.get("quote_asset", "")),
            lot_size=int(d.get("lot_size", 1)),
            is_tradable=bool(d.get("is_tradable", True)),
            is_marginable=bool(d.get("is_marginable", False)),
            is_shortable=bool(d.get("is_shortable", False)),
            raw_filters=dict(d.get("raw_filters", {})),
        )


# =========================
# Trading Sessions
# =========================

@dataclass(frozen=True)
class TradingSession:
    """
    Represents a trading session with start/end times.

    Times are stored as minutes from midnight UTC.
    For timezone-aware sessions (like US market), conversion should happen
    at the adapter level.

    Attributes:
        session_type: Type of session (regular, pre-market, etc.)
        start_minutes: Start time as minutes from midnight (0-1439)
        end_minutes: End time as minutes from midnight (0-1439)
        timezone: Timezone string (e.g., "America/New_York")
        days_of_week: Days when session is active (0=Monday, 6=Sunday)
        is_active: Whether session is currently active
    """
    session_type: SessionType
    start_minutes: int  # Minutes from midnight (0-1439)
    end_minutes: int    # Minutes from midnight (0-1439)
    timezone: str = "UTC"
    days_of_week: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon-Fri by default
    is_active: bool = True

    @property
    def start_time_str(self) -> str:
        """Returns start time as HH:MM string."""
        h, m = divmod(self.start_minutes, 60)
        return f"{h:02d}:{m:02d}"

    @property
    def end_time_str(self) -> str:
        """Returns end time as HH:MM string."""
        h, m = divmod(self.end_minutes, 60)
        return f"{h:02d}:{m:02d}"

    @property
    def duration_minutes(self) -> int:
        """Duration of session in minutes."""
        if self.end_minutes > self.start_minutes:
            return self.end_minutes - self.start_minutes
        # Overnight session
        return (1440 - self.start_minutes) + self.end_minutes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_type": self.session_type.value,
            "start_minutes": self.start_minutes,
            "end_minutes": self.end_minutes,
            "timezone": self.timezone,
            "days_of_week": list(self.days_of_week),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TradingSession":
        session_type_val = d.get("session_type", "regular")
        try:
            session_type = SessionType(str(session_type_val))
        except ValueError:
            session_type = SessionType.REGULAR

        days = d.get("days_of_week", [0, 1, 2, 3, 4])
        if isinstance(days, (list, tuple)):
            days_tuple = tuple(int(x) for x in days)
        else:
            days_tuple = (0, 1, 2, 3, 4)

        return cls(
            session_type=session_type,
            start_minutes=int(d.get("start_minutes", 0)),
            end_minutes=int(d.get("end_minutes", 0)),
            timezone=str(d.get("timezone", "UTC")),
            days_of_week=days_tuple,
            is_active=bool(d.get("is_active", True)),
        )


@dataclass(frozen=True)
class MarketCalendar:
    """
    Market calendar with trading sessions and holidays.

    Attributes:
        vendor: Exchange vendor
        market_type: Type of market
        sessions: List of trading sessions
        holidays: List of holiday dates as (year, month, day) tuples
        half_days: List of half-day dates with early close
        timezone: Primary timezone for this market
    """
    vendor: ExchangeVendor
    market_type: MarketType
    sessions: List[TradingSession] = field(default_factory=list)
    holidays: List[Tuple[int, int, int]] = field(default_factory=list)  # (year, month, day)
    half_days: List[Tuple[int, int, int]] = field(default_factory=list)
    timezone: str = "UTC"

    @property
    def is_24_7(self) -> bool:
        """Returns True if market trades 24/7."""
        return self.market_type.is_crypto

    def get_regular_session(self) -> Optional[TradingSession]:
        """Returns the regular trading session if available."""
        for session in self.sessions:
            if session.session_type == SessionType.REGULAR:
                return session
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor.value,
            "market_type": self.market_type.value,
            "sessions": [s.to_dict() for s in self.sessions],
            "holidays": list(self.holidays),
            "half_days": list(self.half_days),
            "timezone": self.timezone,
        }


# =========================
# Fee Structures
# =========================

@dataclass(frozen=True)
class FeeSchedule:
    """
    Fee schedule for a symbol or exchange.

    Supports multiple fee structures:
    - Percentage: fee = notional * rate (e.g., crypto exchanges)
    - Per-share: fee = shares * rate (e.g., some stock brokers)
    - Flat: fixed fee per trade
    - Tiered: different rates based on volume

    Attributes:
        structure: Type of fee calculation
        maker_rate: Maker fee rate (bps for percentage, $ for per-share)
        taker_rate: Taker fee rate
        flat_fee: Flat fee per trade (if applicable)
        min_fee: Minimum fee per trade
        max_fee: Maximum fee per trade (cap)
        currency: Fee currency (usually quote currency)
        volume_tiers: Volume-based tier discounts {volume_threshold: rate_multiplier}
        rebate_enabled: Whether maker rebates are possible
    """
    structure: FeeStructure = FeeStructure.PERCENTAGE
    maker_rate: float = 10.0  # bps for percentage, $ for per-share
    taker_rate: float = 10.0
    flat_fee: float = 0.0
    min_fee: float = 0.0
    max_fee: Optional[float] = None
    currency: str = "USD"
    volume_tiers: Dict[float, float] = field(default_factory=dict)  # {volume: rate_multiplier}
    rebate_enabled: bool = False

    def compute_fee(
        self,
        notional: float,
        qty: float,
        is_maker: bool,
        monthly_volume: float = 0.0,
    ) -> float:
        """
        Compute fee for a trade.

        Args:
            notional: Trade value (price * qty)
            qty: Number of shares/units
            is_maker: True for maker orders
            monthly_volume: Monthly trading volume for tier calculation

        Returns:
            Fee amount in fee currency
        """
        base_rate = self.maker_rate if is_maker else self.taker_rate

        # Apply volume tier discount if applicable
        tier_mult = 1.0
        for threshold, mult in sorted(self.volume_tiers.items(), reverse=True):
            if monthly_volume >= threshold:
                tier_mult = mult
                break

        effective_rate = base_rate * tier_mult

        if self.structure == FeeStructure.PERCENTAGE:
            # Rate is in bps (basis points)
            fee = notional * (effective_rate / 10_000)
        elif self.structure == FeeStructure.PER_SHARE:
            # Rate is $ per share
            fee = abs(qty) * effective_rate
        elif self.structure == FeeStructure.FLAT:
            fee = self.flat_fee
        else:
            # Default to percentage
            fee = notional * (effective_rate / 10_000)

        # Add flat fee if mixed structure
        if self.structure == FeeStructure.MIXED:
            fee += self.flat_fee

        # Apply min/max constraints
        if self.min_fee > 0:
            fee = max(fee, self.min_fee)
        if self.max_fee is not None and self.max_fee > 0:
            fee = min(fee, self.max_fee)

        return fee

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure": self.structure.value,
            "maker_rate": self.maker_rate,
            "taker_rate": self.taker_rate,
            "flat_fee": self.flat_fee,
            "min_fee": self.min_fee,
            "max_fee": self.max_fee,
            "currency": self.currency,
            "volume_tiers": self.volume_tiers,
            "rebate_enabled": self.rebate_enabled,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeeSchedule":
        structure_val = d.get("structure", "percentage")
        try:
            structure = FeeStructure(str(structure_val))
        except ValueError:
            structure = FeeStructure.PERCENTAGE

        return cls(
            structure=structure,
            maker_rate=float(d.get("maker_rate", 10.0)),
            taker_rate=float(d.get("taker_rate", 10.0)),
            flat_fee=float(d.get("flat_fee", 0.0)),
            min_fee=float(d.get("min_fee", 0.0)),
            max_fee=float(d.get("max_fee")) if d.get("max_fee") is not None else None,
            currency=str(d.get("currency", "USD")),
            volume_tiers={float(k): float(v) for k, v in d.get("volume_tiers", {}).items()},
            rebate_enabled=bool(d.get("rebate_enabled", False)),
        )


# =========================
# Account Info
# =========================

@dataclass(frozen=True)
class AccountInfo:
    """
    Exchange account information.

    Attributes:
        vendor: Exchange vendor
        account_id: Account identifier
        account_type: Account type (margin, cash, etc.)
        vip_tier: VIP/tier level for fee discounts
        maker_fee_rate: Effective maker fee rate
        taker_fee_rate: Effective taker fee rate
        buying_power: Available buying power
        cash_balance: Cash/USDT balance
        margin_enabled: Whether margin is enabled
        pattern_day_trader: PDT status (for US equities)
        raw_data: Original account data from exchange
    """
    vendor: ExchangeVendor
    account_id: str = ""
    account_type: str = "cash"
    vip_tier: int = 0
    maker_fee_rate: Optional[float] = None  # bps
    taker_fee_rate: Optional[float] = None  # bps
    buying_power: Optional[Decimal] = None
    cash_balance: Optional[Decimal] = None
    margin_enabled: bool = False
    pattern_day_trader: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor.value,
            "account_id": self.account_id,
            "account_type": self.account_type,
            "vip_tier": self.vip_tier,
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "buying_power": str(self.buying_power) if self.buying_power else None,
            "cash_balance": str(self.cash_balance) if self.cash_balance else None,
            "margin_enabled": self.margin_enabled,
            "pattern_day_trader": self.pattern_day_trader,
            "raw_data": self.raw_data,
        }


# =========================
# Symbol Info
# =========================

@dataclass(frozen=True)
class SymbolInfo:
    """
    Comprehensive symbol information combining exchange rules and metadata.

    This is the primary model for symbol information across exchanges.
    It combines trading rules, asset details, and exchange-specific metadata.

    Attributes:
        symbol: Trading symbol
        vendor: Exchange vendor
        market_type: Type of market
        exchange_rule: Trading rules and constraints
        fee_schedule: Fee structure for this symbol
        name: Full name/description
        sector: Sector (for equities)
        industry: Industry (for equities)
        is_etf: Whether symbol is an ETF
        is_fractionable: Whether fractional shares are allowed
        status: Trading status
        listed_date: Listing date (ISO format)
        delisted_date: Delisting date if applicable
        raw_data: Original exchange data
    """
    symbol: str
    vendor: ExchangeVendor
    market_type: MarketType
    exchange_rule: ExchangeRule
    fee_schedule: Optional[FeeSchedule] = None
    name: str = ""
    sector: str = ""
    industry: str = ""
    is_etf: bool = False
    is_fractionable: bool = False
    status: str = "active"
    listed_date: Optional[str] = None
    delisted_date: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradable(self) -> bool:
        return self.exchange_rule.is_tradable and self.status == "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "vendor": self.vendor.value,
            "market_type": self.market_type.value,
            "exchange_rule": self.exchange_rule.to_dict(),
            "fee_schedule": self.fee_schedule.to_dict() if self.fee_schedule else None,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "is_etf": self.is_etf,
            "is_fractionable": self.is_fractionable,
            "status": self.status,
            "listed_date": self.listed_date,
            "delisted_date": self.delisted_date,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SymbolInfo":
        vendor_val = d.get("vendor", "unknown")
        try:
            vendor = ExchangeVendor(str(vendor_val))
        except ValueError:
            vendor = ExchangeVendor.UNKNOWN

        market_type_val = d.get("market_type", "CRYPTO_SPOT")
        try:
            market_type = MarketType(str(market_type_val))
        except ValueError:
            market_type = MarketType.CRYPTO_SPOT

        rule_data = d.get("exchange_rule", {})
        if isinstance(rule_data, ExchangeRule):
            exchange_rule = rule_data
        else:
            exchange_rule = ExchangeRule.from_dict(rule_data)

        fee_data = d.get("fee_schedule")
        fee_schedule = FeeSchedule.from_dict(fee_data) if fee_data else None

        return cls(
            symbol=str(d.get("symbol", "")),
            vendor=vendor,
            market_type=market_type,
            exchange_rule=exchange_rule,
            fee_schedule=fee_schedule,
            name=str(d.get("name", "")),
            sector=str(d.get("sector", "")),
            industry=str(d.get("industry", "")),
            is_etf=bool(d.get("is_etf", False)),
            is_fractionable=bool(d.get("is_fractionable", False)),
            status=str(d.get("status", "active")),
            listed_date=d.get("listed_date"),
            delisted_date=d.get("delisted_date"),
            raw_data=dict(d.get("raw_data", {})),
        )


# =========================
# Predefined Sessions
# =========================

# US Equity Market Sessions (Eastern Time)
US_EQUITY_SESSIONS = [
    TradingSession(
        session_type=SessionType.PRE_MARKET,
        start_minutes=4 * 60,   # 04:00 ET
        end_minutes=9 * 60 + 30,  # 09:30 ET
        timezone="America/New_York",
        days_of_week=(0, 1, 2, 3, 4),
    ),
    TradingSession(
        session_type=SessionType.REGULAR,
        start_minutes=9 * 60 + 30,  # 09:30 ET
        end_minutes=16 * 60,  # 16:00 ET
        timezone="America/New_York",
        days_of_week=(0, 1, 2, 3, 4),
    ),
    TradingSession(
        session_type=SessionType.AFTER_HOURS,
        start_minutes=16 * 60,  # 16:00 ET
        end_minutes=20 * 60,    # 20:00 ET
        timezone="America/New_York",
        days_of_week=(0, 1, 2, 3, 4),
    ),
]

# Crypto 24/7 Session
CRYPTO_CONTINUOUS_SESSION = TradingSession(
    session_type=SessionType.CONTINUOUS,
    start_minutes=0,
    end_minutes=0,  # 0-0 means 24h
    timezone="UTC",
    days_of_week=(0, 1, 2, 3, 4, 5, 6),
)


# =========================
# Default Calendars
# =========================

def create_us_equity_calendar(vendor: ExchangeVendor = ExchangeVendor.ALPACA) -> MarketCalendar:
    """Create US equity market calendar."""
    return MarketCalendar(
        vendor=vendor,
        market_type=MarketType.EQUITY,
        sessions=list(US_EQUITY_SESSIONS),
        timezone="America/New_York",
    )


def create_crypto_calendar(vendor: ExchangeVendor = ExchangeVendor.BINANCE) -> MarketCalendar:
    """Create 24/7 crypto market calendar."""
    return MarketCalendar(
        vendor=vendor,
        market_type=MarketType.CRYPTO_SPOT,
        sessions=[CRYPTO_CONTINUOUS_SESSION],
        timezone="UTC",
    )
