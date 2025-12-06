# MiFID II Compliance Roadmap

**Версия**: 1.0
**Дата**: 2025-12-06
**Статус**: ПЛАН (требуется имплементация)

---

## Оглавление

1. [Executive Summary](#1-executive-summary)
2. [Регуляторная база](#2-регуляторная-база)
3. [GAP Analysis](#3-gap-analysis)
4. [Фаза 1: Foundational Compliance](#4-фаза-1-foundational-compliance)
5. [Фаза 2: Transaction Reporting](#5-фаза-2-transaction-reporting)
6. [Фаза 3: Algorithmic Trading Controls](#6-фаза-3-algorithmic-trading-controls)
7. [Фаза 4: Record Keeping & Audit Trail](#7-фаза-4-record-keeping--audit-trail)
8. [Фаза 5: Best Execution](#8-фаза-5-best-execution)
9. [Фаза 6: Governance & Documentation](#9-фаза-6-governance--documentation)
10. [Фаза 7: Testing & Certification](#10-фаза-7-testing--certification)
11. [Архитектура решения](#11-архитектура-решения)
12. [Референсы](#12-референсы)

---

## 1. Executive Summary

### Цель документа

Данный документ определяет пошаговый план приведения AI-Powered Quantitative Research Platform в соответствие с требованиями **MiFID II** (Directive 2014/65/EU) и связанных регуляторных технических стандартов (RTS).

### Scope

| Область | Применимость | Основной RTS |
|---------|--------------|--------------|
| Алгоритмическая торговля | **Полная** | RTS 6 (Article 17) |
| Transaction Reporting | **Полная** | RTS 22, RTS 25 |
| Record Keeping | **Полная** | RTS 24, Article 25 MiFIR |
| Best Execution | **Частичная** | RTS 27, RTS 28* |
| Клиентская классификация | Не применимо** | - |

*RTS 28 reporting отменён с 2024 года, но best execution policy остаётся обязательной
**Система торгует на собственный счёт (proprietary trading), не для клиентов

### Текущий статус

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiFID II COMPLIANCE SCORE                    │
├─────────────────────────────────────────────────────────────────┤
│ Kill Switch (Art. 17)         [████████░░] 75%                  │
│ Pre-Trade Controls (RTS 6)    [██████░░░░] 60%                  │
│ Transaction Reporting         [░░░░░░░░░░] 0%                   │
│ Record Keeping               [██░░░░░░░░] 20%                   │
│ Best Execution Policy        [░░░░░░░░░░] 0%                    │
│ LEI Integration              [░░░░░░░░░░] 0%                    │
│ Audit Trail (RTS 25)         [░░░░░░░░░░] 0%                    │
│ Algorithm Registration       [░░░░░░░░░░] 0%                    │
├─────────────────────────────────────────────────────────────────┤
│ OVERALL COMPLIANCE:          [██░░░░░░░░] ~20%                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Регуляторная база

### 2.1 Ключевые документы

| Документ | Статья | Описание |
|----------|--------|----------|
| **MiFID II** (Directive 2014/65/EU) | Article 17 | Требования к алгоритмической торговле |
| **MiFIR** (Regulation 600/2014) | Article 25-26 | Transaction reporting, record keeping |
| **RTS 6** (Regulation 2017/589) | Articles 1-18 | Детальные требования к algo trading |
| **RTS 22** (Regulation 2017/590) | - | Transaction reporting format |
| **RTS 24** (Regulation 2017/580) | - | Order book data |
| **RTS 25** (Regulation 2017/574) | - | Clock synchronisation |

### 2.2 Применимость к проекту

Согласно [Article 17 MiFID II](https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifid-ii/article-17-algorithmic-trading):

> "An investment firm that engages in algorithmic trading shall have in place effective systems and risk controls suitable to the business it operates to ensure that its trading systems are resilient and have sufficient capacity..."

**Проект классифицируется как:**
- ✅ Algorithmic trading system (автоматическое принятие решений)
- ✅ Proprietary trading (торговля на собственный счёт)
- ⚠️ Потенциально HFT (зависит от частоты сообщений)

### 2.3 Изменения 2024-2025

Согласно [MiFID II Review Directive (2024/791)](https://www.dlapiper.com/en/insights/publications/2024/10/esma-consults-on-revisions-rts-22-on-transaction-data-reporting-and-rts-24):

- **RTS 27** (quarterly venue reports) — приостановлен
- **RTS 28** (annual execution quality reports) — отменён
- **RTS 22** — консультации по расширению (ожидается 2025)
- **Best Execution Policy** — остаётся **обязательной**

---

## 3. GAP Analysis

### 3.1 Что УЖЕ есть в проекте

| Компонент | Файл | MiFID II Mapping | Статус |
|-----------|------|------------------|--------|
| Kill Switch | `services/ops_kill_switch.py` | Art. 12 RTS 6 | ⚠️ Частично |
| Risk Guards | `risk_guard.py` | Art. 15 RTS 6 | ⚠️ Частично |
| Position Limits | `services/futures_risk_guards.py` | Art. 15(4) RTS 6 | ⚠️ Частично |
| Circuit Breaker Awareness | `impl_circuit_breaker.py` | Art. 18 RTS 6 | ⚠️ CME only |
| TCA Models | `execution_providers.py` | Best Execution | ⚠️ Частично |
| Event Bus | `event_bus.py` | Logging | ⚠️ Не compliance-ready |

### 3.2 Что ОТСУТСТВУЕТ (критические gaps)

| Требование | RTS | Приоритет | Сложность |
|------------|-----|-----------|-----------|
| **Transaction Reporting to ARM** | RTS 22 | 🔴 Critical | High |
| **LEI Integration** | RTS 22 Art. 4 | 🔴 Critical | Medium |
| **5-Year Audit Trail** | Art. 25 MiFIR | 🔴 Critical | High |
| **Clock Synchronisation** | RTS 25 | 🟡 High | Medium |
| **Algorithm Registration** | Art. 17(2) | 🟡 High | Low |
| **Order-to-Trade Ratio Monitoring** | RTS 6 Art. 16 | 🟡 High | Medium |
| **Best Execution Policy** | Art. 27 MiFID II | 🟡 High | Low |
| **Annual Self-Assessment** | RTS 6 Art. 9 | 🟡 High | Medium |
| **Real-Time Monitoring Dashboard** | Art. 17 RTS 6 | 🟠 Medium | Medium |
| **Business Continuity Plan** | Art. 3 RTS 6 | 🟠 Medium | Low |

---

## 4. Фаза 1: Foundational Compliance

**Длительность**: 2-3 недели
**Зависимости**: Нет
**Приоритет**: 🔴 Critical

### 4.1 Этап 1.1: LEI Integration

**Требование**: [GLEIF Guidelines](https://www.gleif.org/en/newsroom/blog/reminder-failure-to-obtain-an-lei-by-the-firm-or-its-client-will-prevent-firms-from-being-able-to-comply-with-the-reporting-requirements-under-mifir-applicable-from-january-2018)

> "No LEI, No Trade" — без LEI невозможно подать transaction report

**Задачи:**

```
1.1.1 Получить LEI для юридического лица
      - Регистрация через LOUs (Local Operating Units)
      - Стоимость: ~€50-100/год
      - Срок: 1-3 рабочих дня

1.1.2 Создать модуль lei_manager.py
      services/compliance/
      ├── lei_manager.py      # LEI validation, caching
      ├── gleif_client.py     # GLEIF API integration
      └── __init__.py

1.1.3 Интегрировать LEI validation в order flow
      - Проверка LEI перед каждым ордером
      - Автообновление expired LEIs
      - Caching для производительности
```

**Структура модуля:**

```python
# services/compliance/lei_manager.py
from dataclasses import dataclass
from datetime import date
from typing import Optional
import re

@dataclass
class LEIRecord:
    lei: str                    # 20-char ISO 17442
    legal_name: str
    country: str
    registration_date: date
    next_renewal_date: date
    status: str                 # ISSUED, LAPSED, RETIRED, etc.

    def is_valid(self) -> bool:
        return self.status in ("ISSUED", "PENDING_TRANSFER", "PENDING_ARCHIVAL")

    def is_expired(self) -> bool:
        return self.next_renewal_date < date.today()

class LEIManager:
    """MiFID II LEI management and validation."""

    LEI_PATTERN = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")

    def validate_format(self, lei: str) -> bool:
        """Validate LEI format per ISO 17442."""
        return bool(self.LEI_PATTERN.match(lei))

    async def verify_with_gleif(self, lei: str) -> Optional[LEIRecord]:
        """Verify LEI against GLEIF database."""
        ...

    def check_before_trade(self, lei: str) -> tuple[bool, str]:
        """Pre-trade LEI check. Returns (allowed, reason)."""
        ...
```

**Тесты:**

```python
# tests/test_lei_manager.py
def test_lei_format_validation():
    manager = LEIManager()
    assert manager.validate_format("5493001KJTIIGC8Y1R12")  # Valid
    assert not manager.validate_format("INVALID")

def test_lei_gleif_verification():
    ...

def test_no_lei_no_trade():
    """Verify order rejection without valid LEI."""
    ...
```

### 4.2 Этап 1.2: Clock Synchronisation (RTS 25)

**Требование**: [ESMA RTS 25](https://www.esma.europa.eu/press-news/esma-news/esma-provides-guidance-transaction-reporting-order-record-keeping-and-clock)

> "All records include a timestamp synchronized with UTC"

**Текущее состояние:**
```python
# clock.py (существующий)
def now_ms() -> int:
    return int(time.time() * 1000)
```

**Требуемые доработки:**

```
1.2.1 Добавить NTP синхронизацию
      - Primary: time.google.com
      - Secondary: pool.ntp.org
      - Max drift: ±100ms для algo trading, ±1ms для HFT

1.2.2 Добавить clock drift monitoring
      - Логировать drift > 50ms
      - Alert при drift > 100ms
      - Kill switch при drift > 1s

1.2.3 Создать compliance_clock.py
```

**Структура модуля:**

```python
# services/compliance/compliance_clock.py
import ntplib
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class ClockSyncStatus:
    offset_ms: float
    stratum: int
    reference_server: str
    last_sync_time: float
    sync_success: bool

    def is_compliant(self, max_offset_ms: float = 100.0) -> bool:
        return self.sync_success and abs(self.offset_ms) <= max_offset_ms

class ComplianceClock:
    """RTS 25 compliant clock with NTP synchronisation."""

    NTP_SERVERS = [
        "time.google.com",
        "pool.ntp.org",
        "time.windows.com",
    ]

    def __init__(self, max_offset_ms: float = 100.0):
        self.max_offset_ms = max_offset_ms
        self._offset_ms: float = 0.0
        self._last_sync: Optional[ClockSyncStatus] = None

    def sync(self) -> ClockSyncStatus:
        """Synchronise with NTP servers."""
        client = ntplib.NTPClient()
        for server in self.NTP_SERVERS:
            try:
                response = client.request(server, version=3)
                self._offset_ms = response.offset * 1000
                self._last_sync = ClockSyncStatus(
                    offset_ms=self._offset_ms,
                    stratum=response.stratum,
                    reference_server=server,
                    last_sync_time=time.time(),
                    sync_success=True,
                )
                return self._last_sync
            except Exception:
                continue
        return ClockSyncStatus(0, 0, "", time.time(), False)

    def now_utc_ns(self) -> int:
        """Current UTC timestamp in nanoseconds (RTS 25 compliant)."""
        return int((time.time() + self._offset_ms / 1000) * 1e9)

    def now_utc_ms(self) -> int:
        """Current UTC timestamp in milliseconds."""
        return self.now_utc_ns() // 1_000_000
```

### 4.3 Этап 1.3: Algorithm Registration

**Требование**: [Article 17(2) MiFID II](https://www.kroll.com/en/publications/financial-compliance-regulation/algorithmic-trading-under-mifid-ii)

> "Investment firms shall notify the competent authority of its home Member State that it engages in algorithmic trading."

**Задачи:**

```
1.3.1 Создать реестр алгоритмов
      - Уникальный ID для каждого алгоритма
      - Версионирование
      - Описание стратегии
      - Ответственное лицо

1.3.2 Создать algorithm_registry.py
```

**Структура:**

```python
# services/compliance/algorithm_registry.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum
import uuid

class AlgorithmType(Enum):
    EXECUTION = "execution"       # TWAP, VWAP, POV
    DECISION = "decision"         # Signal generation
    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"

@dataclass
class AlgorithmRecord:
    algorithm_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    type: AlgorithmType = AlgorithmType.DECISION
    description: str = ""
    responsible_person: str = ""
    deployment_date: datetime = field(default_factory=datetime.utcnow)
    asset_classes: List[str] = field(default_factory=list)
    risk_controls: List[str] = field(default_factory=list)
    last_modification: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

class AlgorithmRegistry:
    """MiFID II Algorithm Registry for regulatory reporting."""

    def register(self, algo: AlgorithmRecord) -> str:
        """Register algorithm and return ID."""
        ...

    def get_for_reporting(self) -> List[dict]:
        """Get algorithms in NCA reporting format."""
        ...

    def generate_annual_report(self) -> dict:
        """Generate annual self-assessment data."""
        ...
```

---

## 5. Фаза 2: Transaction Reporting

**Длительность**: 4-6 недель
**Зависимости**: Фаза 1 (LEI)
**Приоритет**: 🔴 Critical

### 5.1 Архитектура Transaction Reporting

Согласно [RTS 22](https://www.esma.europa.eu/sites/default/files/library/esma65-8-2356_mifir_transaction_reporting_technical_reporting_instructions.pdf):

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING PLATFORM                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Order Exec  │──│ Trade Log   │──│ TX Builder  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└────────────────────────────┬────────────────────────────────────┘
                             │ Transaction Report
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REPORTING LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Validation  │──│ Enrichment  │──│ ARM Client  │              │
│  │ (65 fields) │  │ (LEI, ISIN) │  │ (API/SFTP)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└────────────────────────────┬────────────────────────────────────┘
                             │ XML/ISO 20022
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 APPROVED REPORTING MECHANISM (ARM)               │
│  Examples: Bloomberg BTRL, TRAX, Tradeweb, UnaVista             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               NATIONAL COMPETENT AUTHORITY (NCA)                 │
│  Examples: FCA (UK), BaFin (DE), AMF (FR), AFM (NL)             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Этап 2.1: Transaction Report Data Model

**Требование**: RTS 22 определяет 65 полей для transaction report

```python
# services/compliance/transaction_report.py
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from enum import Enum

class BuySellIndicator(Enum):
    BUY = "BUYI"
    SELL = "SELL"

class TradingCapacity(Enum):
    DEAL = "DEAL"  # Dealing on own account
    MTCH = "MTCH"  # Matched principal
    AOTC = "AOTC"  # Any other trading capacity

@dataclass
class TransactionReport:
    """MiFIR Article 26 Transaction Report (RTS 22 format)."""

    # === Identification Fields (1-10) ===
    transaction_reference_number: str = ""
    trading_venue_transaction_id: str = ""
    executing_entity_id_code: str = ""  # LEI
    executing_entity_id_type: str = "LEI"
    investment_firm_covered: bool = True

    # === Buyer/Seller (11-30) ===
    buyer_id_code: str = ""  # LEI or National ID
    buyer_id_type: str = "LEI"
    buyer_country: str = ""
    buyer_first_name: str = ""
    buyer_surname: str = ""
    buyer_dob: Optional[datetime] = None

    seller_id_code: str = ""
    seller_id_type: str = "LEI"
    seller_country: str = ""

    # === Trading Decision (31-35) ===
    transmission_indicator: bool = False
    transmitting_firm_id: str = ""
    trading_decision_maker_id: str = ""
    trading_decision_maker_id_type: str = "ALGO"  # ALGO = algorithm

    # === Order Details (36-45) ===
    trading_capacity: TradingCapacity = TradingCapacity.DEAL
    quantity: Decimal = Decimal("0")
    quantity_currency: str = ""
    derivative_notional_increase: Optional[Decimal] = None
    price: Decimal = Decimal("0")
    price_currency: str = ""
    net_amount: Decimal = Decimal("0")

    # === Venue & Timing (46-55) ===
    venue: str = ""  # MIC code
    country_of_branch: str = ""
    upfront_payment: Optional[Decimal] = None
    upfront_payment_currency: str = ""
    trading_datetime: datetime = field(default_factory=datetime.utcnow)

    # === Instrument (56-65) ===
    instrument_id_code: str = ""  # ISIN
    instrument_id_type: str = "ISIN"
    instrument_full_name: str = ""
    instrument_classification: str = ""  # CFI code
    notional_currency_1: str = ""
    notional_currency_2: str = ""
    price_multiplier: Decimal = Decimal("1")
    underlying_instrument_code: str = ""
    underlying_index_name: str = ""
    term_of_contract: str = ""

    # === Additional ===
    buy_sell_indicator: BuySellIndicator = BuySellIndicator.BUY

    def validate(self) -> List[str]:
        """Validate all required fields. Returns list of errors."""
        errors = []

        if not self.executing_entity_id_code:
            errors.append("Missing executing entity LEI")
        if not self.instrument_id_code:
            errors.append("Missing instrument ISIN")
        if self.quantity <= 0:
            errors.append("Invalid quantity")
        if self.price <= 0:
            errors.append("Invalid price")

        return errors

    def to_xml(self) -> str:
        """Convert to ISO 20022 XML format for ARM submission."""
        ...

    def to_json(self) -> dict:
        """Convert to JSON for API submission."""
        ...
```

### 5.3 Этап 2.2: ARM Integration

**Задачи:**

```
2.2.1 Выбор ARM провайдера
      Опции:
      - Bloomberg BTRL (~€500-1000/month)
      - TRAX (CME Group)
      - UnaVista (LSEG)
      - Tradeweb

2.2.2 Реализация ARM клиента

2.2.3 Тестирование с ARM test environment
```

**ARM Client:**

```python
# services/compliance/arm_client.py
from abc import ABC, abstractmethod
from typing import List
import httpx

class ARMClient(ABC):
    """Abstract ARM (Approved Reporting Mechanism) client."""

    @abstractmethod
    async def submit_report(self, report: TransactionReport) -> str:
        """Submit single transaction report. Returns confirmation ID."""
        pass

    @abstractmethod
    async def submit_batch(self, reports: List[TransactionReport]) -> List[str]:
        """Submit batch of reports."""
        pass

    @abstractmethod
    async def query_status(self, confirmation_id: str) -> dict:
        """Query report status."""
        pass

    @abstractmethod
    async def cancel_report(self, original_id: str) -> bool:
        """Cancel previously submitted report."""
        pass

class BloombergBTRLClient(ARMClient):
    """Bloomberg Transaction Reporting (BTRL) client."""

    def __init__(self, api_key: str, environment: str = "test"):
        self.api_key = api_key
        self.base_url = (
            "https://btrl-api.bloomberg.com" if environment == "prod"
            else "https://btrl-api-uat.bloomberg.com"
        )

    async def submit_report(self, report: TransactionReport) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/transactions",
                json=report.to_json(),
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()["confirmationId"]
```

### 5.4 Этап 2.3: Reporting Pipeline

```python
# services/compliance/reporting_pipeline.py
from typing import Optional
import asyncio
from datetime import datetime, timedelta

class TransactionReportingPipeline:
    """End-to-end transaction reporting pipeline."""

    def __init__(
        self,
        arm_client: ARMClient,
        lei_manager: LEIManager,
        clock: ComplianceClock,
    ):
        self.arm = arm_client
        self.lei = lei_manager
        self.clock = clock
        self._pending_reports: list = []

    async def on_trade_executed(self, trade: dict) -> None:
        """Hook called after each trade execution."""
        report = self._build_report(trade)
        errors = report.validate()

        if errors:
            raise ValueError(f"Report validation failed: {errors}")

        self._pending_reports.append(report)

    async def flush_reports(self) -> List[str]:
        """Submit all pending reports to ARM."""
        if not self._pending_reports:
            return []

        confirmations = await self.arm.submit_batch(self._pending_reports)
        self._pending_reports.clear()
        return confirmations

    def _build_report(self, trade: dict) -> TransactionReport:
        """Build transaction report from trade data."""
        return TransactionReport(
            transaction_reference_number=trade["order_id"],
            executing_entity_id_code=self.lei.get_own_lei(),
            trading_datetime=datetime.fromtimestamp(
                self.clock.now_utc_ms() / 1000
            ),
            instrument_id_code=trade.get("isin", ""),
            quantity=trade["quantity"],
            price=trade["price"],
            buy_sell_indicator=(
                BuySellIndicator.BUY if trade["side"] == "BUY"
                else BuySellIndicator.SELL
            ),
            trading_capacity=TradingCapacity.DEAL,
            venue=trade.get("mic", "XOFF"),
        )
```

---

## 6. Фаза 3: Algorithmic Trading Controls

**Длительность**: 3-4 недели
**Зависимости**: Фаза 1
**Приоритет**: 🔴 Critical

### 6.1 Этап 3.1: Kill Switch Enhancement (RTS 6 Article 12)

**Текущее состояние**: `services/ops_kill_switch.py` — базовая реализация

**Требования [Article 12 RTS 6](https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj/chapter-ii/section-3/):**

> "The investment firm shall be able to cancel immediately, as an emergency measure, any or all of its unexecuted orders submitted to any or all trading venues"

**Необходимые доработки:**

```python
# services/compliance/enhanced_kill_switch.py
from dataclasses import dataclass
from typing import List, Optional, Callable
from enum import Enum
import threading
import logging

class KillSwitchScope(Enum):
    ALL = "all"                 # All orders across all venues
    VENUE = "venue"             # Specific venue
    ALGORITHM = "algorithm"     # Specific algorithm
    TRADER = "trader"           # Specific trader/desk
    INSTRUMENT = "instrument"   # Specific instrument

@dataclass
class KillSwitchEvent:
    timestamp_ns: int
    scope: KillSwitchScope
    scope_id: str
    reason: str
    triggered_by: str           # Person or system
    orders_cancelled: int
    confirmation_id: str

class EnhancedKillSwitch:
    """MiFID II RTS 6 Article 12 compliant kill switch."""

    def __init__(
        self,
        order_cancellation_callback: Callable[[KillSwitchScope, str], int],
        alert_callback: Optional[Callable[[KillSwitchEvent], None]] = None,
    ):
        self.cancel_orders = order_cancellation_callback
        self.alert = alert_callback
        self._lock = threading.Lock()
        self._events: List[KillSwitchEvent] = []
        self._armed = True

    def trigger(
        self,
        scope: KillSwitchScope,
        scope_id: str = "",
        reason: str = "",
        triggered_by: str = "system",
    ) -> KillSwitchEvent:
        """
        Trigger kill switch. Immediately cancels orders per scope.

        Per RTS 6 Article 12: "cancel immediately, as an emergency measure"
        """
        if not self._armed:
            raise RuntimeError("Kill switch is disarmed")

        with self._lock:
            orders_cancelled = self.cancel_orders(scope, scope_id)

            event = KillSwitchEvent(
                timestamp_ns=time.time_ns(),
                scope=scope,
                scope_id=scope_id,
                reason=reason,
                triggered_by=triggered_by,
                orders_cancelled=orders_cancelled,
                confirmation_id=str(uuid.uuid4()),
            )

            self._events.append(event)

            if self.alert:
                self.alert(event)

            logging.critical(
                f"KILL SWITCH TRIGGERED: {scope.value} "
                f"cancelled {orders_cancelled} orders. "
                f"Reason: {reason}"
            )

            return event

    def trigger_all(self, reason: str = "Emergency") -> KillSwitchEvent:
        """Trigger kill switch for ALL orders on ALL venues."""
        return self.trigger(KillSwitchScope.ALL, "", reason)

    def get_contact_info(self) -> dict:
        """
        Per RTS 6: "compliance staff must maintain contact with the
        individual at the firm who is able to cancel immediately"
        """
        return {
            "primary_contact": "trading_desk@firm.com",
            "emergency_phone": "+1-XXX-XXX-XXXX",
            "out_of_hours": "+1-XXX-XXX-XXXX",
        }
```

### 6.2 Этап 3.2: Pre-Trade Controls (RTS 6 Article 15)

**Требования [Article 15 RTS 6](https://www.eventus.com/cat-article/enforcement-action-from-esma-on-rts-6/):**

```
(1) Price collars
(2) Maximum order values
(3) Maximum order volumes
(4) Maximum message limits
(5) Automatic blocking of orders from unauthorized traders
```

**Реализация:**

```python
# services/compliance/pre_trade_controls.py
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple
from enum import Enum

class RejectionReason(Enum):
    PRICE_COLLAR = "price_collar_breach"
    MAX_ORDER_VALUE = "max_order_value_exceeded"
    MAX_ORDER_VOLUME = "max_order_volume_exceeded"
    MESSAGE_RATE = "message_rate_exceeded"
    UNAUTHORIZED_TRADER = "unauthorized_trader"
    UNAUTHORIZED_INSTRUMENT = "unauthorized_instrument"
    RISK_LIMIT = "risk_limit_breach"

@dataclass
class PreTradeControlsConfig:
    # Price collars (% from reference)
    price_collar_pct: float = 5.0

    # Maximum order values (EUR equivalent)
    max_order_value_eur: Decimal = Decimal("1000000")

    # Maximum order volumes (units)
    max_order_volume: Decimal = Decimal("10000")

    # Message rate limits (per second)
    max_messages_per_second: int = 100

    # Fat finger protection
    fat_finger_price_deviation_pct: float = 10.0
    fat_finger_volume_multiplier: float = 10.0

class PreTradeControls:
    """MiFID II RTS 6 Article 15 pre-trade risk controls."""

    def __init__(self, config: PreTradeControlsConfig):
        self.config = config
        self._message_timestamps: list = []
        self._authorized_traders: set = set()
        self._authorized_instruments: set = set()

    def check_order(
        self,
        order: dict,
        reference_price: Decimal,
        trader_id: str,
    ) -> Tuple[bool, Optional[RejectionReason], str]:
        """
        Pre-trade validation per RTS 6 Article 15.

        Returns: (allowed, rejection_reason, message)
        """
        # (1) Price collar check
        if not self._check_price_collar(order["price"], reference_price):
            return (
                False,
                RejectionReason.PRICE_COLLAR,
                f"Price {order['price']} exceeds collar "
                f"({self.config.price_collar_pct}% from {reference_price})"
            )

        # (2) Max order value
        order_value = order["price"] * order["quantity"]
        if order_value > self.config.max_order_value_eur:
            return (
                False,
                RejectionReason.MAX_ORDER_VALUE,
                f"Order value {order_value} exceeds max {self.config.max_order_value_eur}"
            )

        # (3) Max order volume
        if order["quantity"] > self.config.max_order_volume:
            return (
                False,
                RejectionReason.MAX_ORDER_VOLUME,
                f"Quantity {order['quantity']} exceeds max {self.config.max_order_volume}"
            )

        # (4) Message rate
        if not self._check_message_rate():
            return (
                False,
                RejectionReason.MESSAGE_RATE,
                f"Message rate exceeded {self.config.max_messages_per_second}/sec"
            )

        # (5) Trader authorization
        if trader_id not in self._authorized_traders:
            return (
                False,
                RejectionReason.UNAUTHORIZED_TRADER,
                f"Trader {trader_id} not authorized"
            )

        return (True, None, "OK")

    def _check_price_collar(
        self,
        order_price: Decimal,
        reference_price: Decimal,
    ) -> bool:
        if reference_price <= 0:
            return False
        deviation_pct = abs(order_price - reference_price) / reference_price * 100
        return deviation_pct <= self.config.price_collar_pct

    def _check_message_rate(self) -> bool:
        now = time.time()
        # Remove old timestamps
        self._message_timestamps = [
            ts for ts in self._message_timestamps
            if now - ts < 1.0
        ]
        # Check limit
        if len(self._message_timestamps) >= self.config.max_messages_per_second:
            return False
        self._message_timestamps.append(now)
        return True
```

### 6.3 Этап 3.3: Real-Time Monitoring (RTS 6 Article 17)

**Требование:**

> "Real-time alerts shall be generated within five seconds after the relevant event"

```python
# services/compliance/realtime_monitor.py
from dataclasses import dataclass
from typing import List, Callable, Optional
from enum import Enum
import asyncio

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ComplianceAlert:
    timestamp_ns: int
    severity: AlertSeverity
    category: str
    message: str
    data: dict
    acknowledged: bool = False

class RealTimeMonitor:
    """
    MiFID II RTS 6 Article 17 real-time monitoring.

    Generates alerts within 5 seconds per regulatory requirement.
    """

    ALERT_DEADLINE_SEC = 5.0  # RTS 6 Art. 17 requirement

    def __init__(
        self,
        alert_callback: Callable[[ComplianceAlert], None],
        escalation_callback: Optional[Callable[[ComplianceAlert], None]] = None,
    ):
        self.alert = alert_callback
        self.escalate = escalation_callback
        self._alerts: List[ComplianceAlert] = []
        self._thresholds: dict = {}

    async def monitor_loop(self):
        """Main monitoring loop."""
        while True:
            await self._check_all_metrics()
            await asyncio.sleep(1.0)  # Check every second

    async def _check_all_metrics(self):
        """Check all monitored metrics."""
        # Order-to-trade ratio
        await self._check_order_to_trade_ratio()

        # Position limits
        await self._check_position_limits()

        # P&L thresholds
        await self._check_pnl_thresholds()

        # System health
        await self._check_system_health()

    async def _check_order_to_trade_ratio(self):
        """Monitor OTR per RTS 6."""
        # Implementation
        pass

    def _generate_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        data: dict,
    ):
        """Generate alert within 5-second deadline."""
        alert = ComplianceAlert(
            timestamp_ns=time.time_ns(),
            severity=severity,
            category=category,
            message=message,
            data=data,
        )

        self._alerts.append(alert)
        self.alert(alert)

        if severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
            if self.escalate:
                self.escalate(alert)
```

### 6.4 Этап 3.4: Order-to-Trade Ratio Monitoring

**Требование RTS 6:**

```python
# services/compliance/otr_monitor.py
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict
import time

@dataclass
class OTRMetrics:
    orders_submitted: int
    orders_cancelled: int
    trades_executed: int
    otr_ratio: float
    window_start: float
    window_end: float

class OrderToTradeRatioMonitor:
    """
    Monitor Order-to-Trade Ratio per MiFID II requirements.

    High OTR may indicate:
    - Quote stuffing
    - Layering/spoofing
    - System malfunction
    """

    def __init__(
        self,
        warning_threshold: float = 50.0,   # 50:1 OTR warning
        critical_threshold: float = 100.0,  # 100:1 OTR critical
        window_seconds: float = 60.0,       # Rolling 1-minute window
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_seconds = window_seconds

        self._orders: Deque[float] = deque()
        self._cancels: Deque[float] = deque()
        self._trades: Deque[float] = deque()

    def record_order(self):
        """Record order submission."""
        self._orders.append(time.time())
        self._cleanup()

    def record_cancel(self):
        """Record order cancellation."""
        self._cancels.append(time.time())
        self._cleanup()

    def record_trade(self):
        """Record trade execution."""
        self._trades.append(time.time())
        self._cleanup()

    def get_metrics(self) -> OTRMetrics:
        """Calculate current OTR metrics."""
        self._cleanup()

        orders = len(self._orders)
        trades = max(len(self._trades), 1)  # Avoid div by zero

        return OTRMetrics(
            orders_submitted=orders,
            orders_cancelled=len(self._cancels),
            trades_executed=len(self._trades),
            otr_ratio=orders / trades,
            window_start=time.time() - self.window_seconds,
            window_end=time.time(),
        )

    def check_compliance(self) -> tuple[bool, str]:
        """Check if OTR is within acceptable limits."""
        metrics = self.get_metrics()

        if metrics.otr_ratio >= self.critical_threshold:
            return (False, f"CRITICAL: OTR {metrics.otr_ratio:.1f} exceeds {self.critical_threshold}")

        if metrics.otr_ratio >= self.warning_threshold:
            return (True, f"WARNING: OTR {metrics.otr_ratio:.1f} approaching limit")

        return (True, "OK")

    def _cleanup(self):
        """Remove old entries outside window."""
        cutoff = time.time() - self.window_seconds

        while self._orders and self._orders[0] < cutoff:
            self._orders.popleft()
        while self._cancels and self._cancels[0] < cutoff:
            self._cancels.popleft()
        while self._trades and self._trades[0] < cutoff:
            self._trades.popleft()
```

---

## 7. Фаза 4: Record Keeping & Audit Trail

**Длительность**: 4-5 недель
**Зависимости**: Фаза 1, 3
**Приоритет**: 🔴 Critical

### 7.1 Требования Article 25 MiFIR

Согласно [ESMA Guidelines](https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifir/article-25-obligation-maintain-records):

> "Keep at the disposal of the competent authority, for **five years**, the relevant data relating to all orders and all transactions"

**Требуемые записи:**
- Все ордера (submitted, modified, cancelled)
- Все транзакции
- Параметры алгоритмов
- Решения risk controls
- Timestamps (microsecond precision)

### 7.2 Этап 4.1: Audit Trail Database Schema

```python
# services/compliance/audit_models.py
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum
import json

class AuditEventType(Enum):
    # Order lifecycle
    ORDER_SUBMITTED = "order_submitted"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_EXPIRED = "order_expired"

    # Risk events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"

    # Algorithm events
    ALGO_STARTED = "algo_started"
    ALGO_STOPPED = "algo_stopped"
    ALGO_PARAMETER_CHANGED = "algo_parameter_changed"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"

@dataclass
class AuditRecord:
    """
    Immutable audit record per MiFIR Article 25.

    Stored for 5 years minimum, 7 years if requested by NCA.
    """

    # Identification
    record_id: str
    event_type: AuditEventType

    # Timestamps (RTS 25 compliant)
    event_timestamp_ns: int          # Nanosecond precision
    record_timestamp_ns: int         # When record was created

    # Entity identification
    firm_lei: str
    algorithm_id: Optional[str]
    trader_id: Optional[str]

    # Order details (if applicable)
    order_id: Optional[str]
    instrument_isin: Optional[str]
    venue_mic: Optional[str]
    side: Optional[str]
    quantity: Optional[Decimal]
    price: Optional[Decimal]

    # Event details
    details: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    previous_record_hash: Optional[str] = None
    record_hash: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps(self.__dict__, default=str)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for integrity verification."""
        import hashlib
        data = self.to_json().encode()
        return hashlib.sha256(data).hexdigest()

class AuditTrailWriter:
    """
    Write-once audit trail with integrity verification.

    Per MiFIR: "Records must be tamper-proof and cannot be altered"
    """

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self._last_hash: Optional[str] = None

    def write(self, record: AuditRecord) -> str:
        """
        Write audit record. Returns record hash.

        Records are chained via hashes for integrity verification.
        """
        record.previous_record_hash = self._last_hash
        record.record_timestamp_ns = time.time_ns()
        record.record_hash = record.compute_hash()

        self.storage.append(record)
        self._last_hash = record.record_hash

        return record.record_hash

    def verify_chain(self) -> bool:
        """Verify integrity of audit trail chain."""
        records = self.storage.read_all()

        for i, record in enumerate(records):
            # Verify hash
            computed = record.compute_hash()
            if computed != record.record_hash:
                return False

            # Verify chain
            if i > 0:
                if record.previous_record_hash != records[i-1].record_hash:
                    return False

        return True
```

### 7.3 Этап 4.2: Storage Backend

**Требования:**
- 5-7 лет retention
- Tamper-proof
- High availability
- Fast retrieval для regulators

```python
# services/compliance/audit_storage.py
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime, timedelta
import sqlite3
import json

class AuditStorageBackend(ABC):
    """Abstract storage backend for audit records."""

    @abstractmethod
    def append(self, record: AuditRecord) -> None:
        """Append record (write-once)."""
        pass

    @abstractmethod
    def read_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[AuditRecord]:
        """Read records in time range."""
        pass

    @abstractmethod
    def read_by_order_id(self, order_id: str) -> List[AuditRecord]:
        """Read all records for an order."""
        pass

class SQLiteAuditStorage(AuditStorageBackend):
    """
    SQLite storage for development/testing.

    For production, use:
    - PostgreSQL with partitioning
    - TimescaleDB
    - AWS Timestream
    - Azure Data Explorer
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                event_timestamp_ns INTEGER NOT NULL,
                record_timestamp_ns INTEGER NOT NULL,
                firm_lei TEXT NOT NULL,
                algorithm_id TEXT,
                trader_id TEXT,
                order_id TEXT,
                instrument_isin TEXT,
                venue_mic TEXT,
                side TEXT,
                quantity TEXT,
                price TEXT,
                details TEXT,
                previous_record_hash TEXT,
                record_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for fast retrieval
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_timestamp "
            "ON audit_trail(event_timestamp_ns)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_id "
            "ON audit_trail(order_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_instrument "
            "ON audit_trail(instrument_isin)"
        )

        conn.commit()
        conn.close()

    def append(self, record: AuditRecord) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO audit_trail (
                    record_id, event_type, event_timestamp_ns,
                    record_timestamp_ns, firm_lei, algorithm_id,
                    trader_id, order_id, instrument_isin, venue_mic,
                    side, quantity, price, details,
                    previous_record_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.event_type.value,
                    record.event_timestamp_ns,
                    record.record_timestamp_ns,
                    record.firm_lei,
                    record.algorithm_id,
                    record.trader_id,
                    record.order_id,
                    record.instrument_isin,
                    record.venue_mic,
                    record.side,
                    str(record.quantity) if record.quantity else None,
                    str(record.price) if record.price else None,
                    json.dumps(record.details),
                    record.previous_record_hash,
                    record.record_hash,
                )
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Duplicate record_id: {record.record_id}")
        finally:
            conn.close()
```

### 7.4 Этап 4.3: Retention Policy

```python
# services/compliance/retention_policy.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class RetentionPolicyConfig:
    """
    MiFIR Article 25 retention requirements.

    - Default: 5 years
    - Extended (NCA request): 7 years
    """

    default_retention_years: int = 5
    extended_retention_years: int = 7
    archive_after_years: int = 1  # Move to cold storage after 1 year

class RetentionManager:
    """Manage audit record retention per MiFIR."""

    def __init__(
        self,
        config: RetentionPolicyConfig,
        hot_storage: AuditStorageBackend,
        cold_storage: Optional[AuditStorageBackend] = None,
    ):
        self.config = config
        self.hot = hot_storage
        self.cold = cold_storage

    def archive_old_records(self) -> int:
        """
        Move records older than archive_after_years to cold storage.

        Returns number of archived records.
        """
        if not self.cold:
            return 0

        cutoff = datetime.utcnow() - timedelta(
            days=self.config.archive_after_years * 365
        )

        # Implementation: move records from hot to cold storage
        pass

    def delete_expired_records(self, extended: bool = False) -> int:
        """
        Delete records past retention period.

        Per MiFIR: Only after 5 years (or 7 if NCA requested).
        Returns number of deleted records.
        """
        years = (
            self.config.extended_retention_years if extended
            else self.config.default_retention_years
        )

        cutoff = datetime.utcnow() - timedelta(days=years * 365)

        # Implementation: delete from both hot and cold storage
        pass

    def prepare_for_nca_request(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: dict,
    ) -> str:
        """
        Prepare audit data export for NCA request.

        Returns path to export file.
        """
        # Implementation: export filtered records
        pass
```

---

## 8. Фаза 5: Best Execution

**Длительность**: 3-4 недели
**Зависимости**: Фаза 4
**Приоритет**: 🟡 High

### 8.1 Требования Article 27 MiFID II

Согласно [ESMA Best Execution Guidelines](https://www.esma.europa.eu/sites/default/files/library/esma35-43-3088_final_report_review_of_mifid_ii_framework_on_best_execution_reports.pdf):

> "Investment firms must take all sufficient steps to obtain the best possible result for their clients"

**Факторы Best Execution:**
1. Price
2. Costs
3. Speed
4. Likelihood of execution
5. Settlement likelihood
6. Size
7. Nature of order

### 8.2 Этап 5.1: Best Execution Policy

```python
# services/compliance/best_execution.py
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict
from enum import Enum

class ExecutionFactor(Enum):
    PRICE = "price"
    COST = "cost"
    SPEED = "speed"
    LIKELIHOOD = "likelihood"
    SETTLEMENT = "settlement"
    SIZE = "size"
    NATURE = "nature"

@dataclass
class ExecutionVenue:
    mic: str
    name: str
    ranking: int
    avg_spread_bps: Decimal
    avg_latency_ms: float
    fill_rate_pct: float
    cost_bps: Decimal

@dataclass
class BestExecutionPolicy:
    """
    Article 27 MiFID II Best Execution Policy.

    Must be:
    - Documented
    - Reviewed annually
    - Disclosed to clients (if applicable)
    """

    version: str
    effective_date: str
    review_date: str
    approved_by: str

    # Factor weights (must sum to 1.0)
    factor_weights: Dict[ExecutionFactor, float]

    # Venue rankings by instrument class
    venue_rankings: Dict[str, List[ExecutionVenue]]

    # Policies
    order_routing_policy: str
    conflict_of_interest_policy: str
    monitoring_policy: str

    def validate(self) -> List[str]:
        errors = []

        total_weight = sum(self.factor_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            errors.append(f"Factor weights sum to {total_weight}, must be 1.0")

        return errors

class BestExecutionAnalyzer:
    """Analyze execution quality for best execution monitoring."""

    def __init__(self, policy: BestExecutionPolicy):
        self.policy = policy

    def analyze_execution(
        self,
        order: dict,
        fill: dict,
        market_data: dict,
    ) -> dict:
        """
        Analyze single execution against best execution criteria.

        Returns analysis report with scores per factor.
        """
        analysis = {}

        # Price analysis
        mid_price = (market_data["bid"] + market_data["ask"]) / 2
        price_improvement = (mid_price - fill["price"]) / mid_price * 10000
        analysis["price_improvement_bps"] = price_improvement

        # Cost analysis
        total_cost = fill.get("commission", 0) + fill.get("fees", 0)
        analysis["total_cost_bps"] = total_cost / fill["notional"] * 10000

        # Speed analysis
        latency_ms = fill["fill_time_ms"] - order["submit_time_ms"]
        analysis["latency_ms"] = latency_ms

        # Overall score
        analysis["overall_score"] = self._compute_score(analysis)

        return analysis

    def generate_monthly_report(self, executions: List[dict]) -> dict:
        """Generate monthly best execution monitoring report."""
        pass

    def _compute_score(self, analysis: dict) -> float:
        """Compute weighted overall score."""
        pass
```

### 8.3 Этап 5.2: TCA Integration

**Текущее состояние:** Проект имеет L2+ Parametric TCA models в `execution_providers.py`

**Доработки для compliance:**

```python
# services/compliance/tca_compliance.py
from execution_providers import (
    CryptoParametricSlippageProvider,
    EquityParametricSlippageProvider,
)

class ComplianceTCAWrapper:
    """
    Wrap existing TCA models for best execution compliance.

    Adds:
    - Audit logging
    - Pre/post trade analysis
    - Regulatory reporting
    """

    def __init__(
        self,
        tca_provider,
        audit_writer: AuditTrailWriter,
    ):
        self.tca = tca_provider
        self.audit = audit_writer

    def pre_trade_estimate(self, order: dict) -> dict:
        """
        Pre-trade cost estimation for best execution.

        Per Article 27: "sufficient steps to obtain best result"
        """
        estimate = self.tca.estimate_impact_cost(
            notional=order["notional"],
            adv=order["adv"],
            side=order["side"],
            hour_utc=order.get("hour_utc"),
        )

        # Log for audit trail
        self.audit.write(AuditRecord(
            record_id=str(uuid.uuid4()),
            event_type=AuditEventType.RISK_CHECK_PASSED,
            event_timestamp_ns=time.time_ns(),
            firm_lei=self.firm_lei,
            order_id=order.get("order_id"),
            details={
                "type": "pre_trade_tca",
                "estimated_impact_bps": estimate["impact_bps"],
                "recommendation": estimate["recommendation"],
            }
        ))

        return estimate

    def post_trade_analysis(self, order: dict, fill: dict) -> dict:
        """
        Post-trade analysis for best execution monitoring.
        """
        expected_slippage = self.pre_trade_estimate(order)["impact_bps"]
        actual_slippage = (
            (fill["price"] - order["expected_price"])
            / order["expected_price"] * 10000
        )

        return {
            "expected_slippage_bps": expected_slippage,
            "actual_slippage_bps": actual_slippage,
            "slippage_vs_estimate": actual_slippage - expected_slippage,
        }
```

---

## 9. Фаза 6: Governance & Documentation

**Длительность**: 2-3 недели
**Зависимости**: Все предыдущие фазы
**Приоритет**: 🟡 High

### 9.1 Этап 6.1: Annual Self-Assessment (RTS 6 Article 9)

**Требование [Deloitte RTS 6 Guide](https://www.deloitte.com/uk/en/services/audit-assurance/blogs/mifid-ii-rts-6-requirements-annual-self-assessment.html):**

> "MiFID II investment firms engaged in algorithmic trading activities must perform an annual self-assessment"

```python
# services/compliance/self_assessment.py
from dataclasses import dataclass
from datetime import date
from typing import List, Dict

@dataclass
class SelfAssessmentQuestion:
    id: str
    category: str
    question: str
    rts_reference: str
    response: str
    evidence: List[str]
    compliant: bool
    remediation_plan: str

@dataclass
class AnnualSelfAssessment:
    """RTS 6 Article 9 Annual Self-Assessment."""

    assessment_date: date
    assessment_period_start: date
    assessment_period_end: date
    assessor: str
    reviewer: str

    # Categories
    governance: List[SelfAssessmentQuestion]
    risk_controls: List[SelfAssessmentQuestion]
    testing: List[SelfAssessmentQuestion]
    business_continuity: List[SelfAssessmentQuestion]
    record_keeping: List[SelfAssessmentQuestion]

    def overall_compliance_score(self) -> float:
        all_questions = (
            self.governance + self.risk_controls +
            self.testing + self.business_continuity +
            self.record_keeping
        )
        compliant = sum(1 for q in all_questions if q.compliant)
        return compliant / len(all_questions) * 100

    def get_remediation_items(self) -> List[SelfAssessmentQuestion]:
        """Get all items requiring remediation."""
        all_questions = (
            self.governance + self.risk_controls +
            self.testing + self.business_continuity +
            self.record_keeping
        )
        return [q for q in all_questions if not q.compliant]

    def generate_report(self) -> str:
        """Generate formal assessment report for NCA."""
        pass

# Template questions based on RTS 6
SELF_ASSESSMENT_TEMPLATE = {
    "governance": [
        SelfAssessmentQuestion(
            id="GOV-001",
            category="Governance",
            question="Does the firm have clear lines of accountability for algorithmic trading?",
            rts_reference="RTS 6 Article 1",
            response="",
            evidence=[],
            compliant=False,
            remediation_plan="",
        ),
        # ... more questions
    ],
    "risk_controls": [
        SelfAssessmentQuestion(
            id="RISK-001",
            category="Risk Controls",
            question="Are pre-trade controls in place per RTS 6 Article 15?",
            rts_reference="RTS 6 Article 15",
            response="",
            evidence=[],
            compliant=False,
            remediation_plan="",
        ),
        # ... more questions
    ],
}
```

### 9.2 Этап 6.2: Business Continuity Plan (RTS 6 Article 3)

```python
# services/compliance/bcp.py
from dataclasses import dataclass
from typing import List

@dataclass
class BCPScenario:
    id: str
    name: str
    description: str
    impact: str  # HIGH, MEDIUM, LOW
    response_procedure: str
    responsible_person: str
    recovery_time_objective: str
    last_tested: str

@dataclass
class BusinessContinuityPlan:
    """
    RTS 6 Article 3 Business Continuity Plan.

    Must cover:
    - System failures
    - Network outages
    - Data center failures
    - Market disruptions
    """

    version: str
    effective_date: str
    approved_by: str

    scenarios: List[BCPScenario]

    emergency_contacts: dict
    escalation_matrix: dict

    def generate_document(self) -> str:
        """Generate formal BCP document."""
        pass

# Standard BCP scenarios for algo trading
BCP_SCENARIOS = [
    BCPScenario(
        id="BCP-001",
        name="Primary System Failure",
        description="Complete failure of primary trading system",
        impact="HIGH",
        response_procedure="1. Activate kill switch\n2. Switch to backup system\n3. Notify regulators",
        responsible_person="Head of Trading Technology",
        recovery_time_objective="15 minutes",
        last_tested="",
    ),
    BCPScenario(
        id="BCP-002",
        name="Market Data Feed Failure",
        description="Loss of market data from primary vendor",
        impact="HIGH",
        response_procedure="1. Switch to backup feed\n2. Reduce trading limits\n3. Monitor closely",
        responsible_person="Market Data Manager",
        recovery_time_objective="5 minutes",
        last_tested="",
    ),
    BCPScenario(
        id="BCP-003",
        name="Kill Switch Activation",
        description="Emergency cancellation of all orders",
        impact="HIGH",
        response_procedure="1. Execute kill switch\n2. Document reason\n3. Review before restart",
        responsible_person="Compliance Officer",
        recovery_time_objective="Immediate",
        last_tested="",
    ),
]
```

### 9.3 Этап 6.3: Policy Documents

**Требуемые документы:**

| Документ | RTS | Статус |
|----------|-----|--------|
| Algorithmic Trading Policy | RTS 6 Art. 1 | 📋 Требуется |
| Best Execution Policy | Art. 27 MiFID II | 📋 Требуется |
| Business Continuity Plan | RTS 6 Art. 3 | 📋 Требуется |
| Risk Management Policy | RTS 6 Art. 14-17 | 📋 Требуется |
| Order Handling Policy | Art. 28 MiFID II | 📋 Требуется |
| Conflicts of Interest Policy | Art. 23 MiFID II | 📋 Требуется |

---

## 10. Фаза 7: Testing & Certification

**Длительность**: 3-4 недели
**Зависимости**: Все предыдущие фазы
**Приоритет**: 🟡 High

### 10.1 Этап 7.1: Conformance Testing (RTS 6 Article 5)

**Требование:**

> "Investment firms shall test the trading algorithm and trading system prior to deployment or substantial update"

```python
# services/compliance/conformance_testing.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum

class TestResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"

@dataclass
class ConformanceTest:
    test_id: str
    name: str
    category: str
    description: str
    rts_reference: str
    result: TestResult
    details: str
    timestamp: datetime

@dataclass
class ConformanceTestSuite:
    """RTS 6 Article 5 Conformance Testing Suite."""

    algorithm_id: str
    algorithm_version: str
    test_date: datetime
    tester: str
    environment: str  # "sandbox", "uat", "production"

    tests: List[ConformanceTest]

    def overall_result(self) -> TestResult:
        if any(t.result == TestResult.FAIL for t in self.tests):
            return TestResult.FAIL
        if any(t.result == TestResult.ERROR for t in self.tests):
            return TestResult.ERROR
        return TestResult.PASS

    def generate_certificate(self) -> Optional[str]:
        """Generate conformance certificate if all tests pass."""
        if self.overall_result() != TestResult.PASS:
            return None

        return f"""
        CONFORMANCE CERTIFICATE
        =======================
        Algorithm ID: {self.algorithm_id}
        Version: {self.algorithm_version}
        Test Date: {self.test_date}
        Environment: {self.environment}
        Result: PASS

        This algorithm has been tested in accordance with
        MiFID II RTS 6 Article 5 requirements.

        Tests Passed: {sum(1 for t in self.tests if t.result == TestResult.PASS)}
        Total Tests: {len(self.tests)}
        """

# Standard conformance tests
CONFORMANCE_TESTS = [
    ConformanceTest(
        test_id="CT-001",
        name="Kill Switch Functionality",
        category="Risk Controls",
        description="Verify kill switch cancels all orders immediately",
        rts_reference="RTS 6 Article 12",
        result=TestResult.SKIP,
        details="",
        timestamp=datetime.now(),
    ),
    ConformanceTest(
        test_id="CT-002",
        name="Pre-Trade Price Collar",
        category="Risk Controls",
        description="Verify orders outside price collar are rejected",
        rts_reference="RTS 6 Article 15",
        result=TestResult.SKIP,
        details="",
        timestamp=datetime.now(),
    ),
    ConformanceTest(
        test_id="CT-003",
        name="Maximum Order Value",
        category="Risk Controls",
        description="Verify orders exceeding max value are rejected",
        rts_reference="RTS 6 Article 15",
        result=TestResult.SKIP,
        details="",
        timestamp=datetime.now(),
    ),
    ConformanceTest(
        test_id="CT-004",
        name="Clock Synchronisation",
        category="Technical",
        description="Verify clock is synchronized within RTS 25 tolerance",
        rts_reference="RTS 25",
        result=TestResult.SKIP,
        details="",
        timestamp=datetime.now(),
    ),
    ConformanceTest(
        test_id="CT-005",
        name="Audit Trail Integrity",
        category="Record Keeping",
        description="Verify audit trail is tamper-proof",
        rts_reference="MiFIR Article 25",
        result=TestResult.SKIP,
        details="",
        timestamp=datetime.now(),
    ),
]
```

### 10.2 Этап 7.2: External Audit

**Рекомендации:**

1. **Big 4 Firms** (Deloitte, PwC, EY, KPMG)
   - Comprehensive MiFID II compliance review
   - Стоимость: €50,000-150,000

2. **Specialized RegTech Auditors**
   - Capco, Accenture, Eventus
   - Focus on algo trading controls
   - Стоимость: €20,000-50,000

3. **NCA Pre-Notification Review**
   - Voluntary consultation with regulator
   - Перед запуском в production

---

## 11. Архитектура решения

### 11.1 Итоговая структура модулей

```
services/compliance/
├── __init__.py
├── config.py                    # Compliance configuration
│
├── # Phase 1: Foundation
├── lei_manager.py               # LEI management
├── gleif_client.py              # GLEIF API client
├── compliance_clock.py          # RTS 25 clock sync
├── algorithm_registry.py        # Algorithm registration
│
├── # Phase 2: Transaction Reporting
├── transaction_report.py        # RTS 22 data model
├── arm_client.py                # ARM integration
├── reporting_pipeline.py        # End-to-end reporting
│
├── # Phase 3: Algo Controls
├── enhanced_kill_switch.py      # RTS 6 Art. 12
├── pre_trade_controls.py        # RTS 6 Art. 15
├── realtime_monitor.py          # RTS 6 Art. 17
├── otr_monitor.py               # Order-to-trade ratio
│
├── # Phase 4: Record Keeping
├── audit_models.py              # Audit trail models
├── audit_storage.py             # Storage backends
├── retention_policy.py          # 5-7 year retention
│
├── # Phase 5: Best Execution
├── best_execution.py            # Article 27 policy
├── tca_compliance.py            # TCA wrapper
│
├── # Phase 6: Governance
├── self_assessment.py           # Annual self-assessment
├── bcp.py                       # Business continuity
│
├── # Phase 7: Testing
├── conformance_testing.py       # Conformance tests
└── test_scenarios.py            # Test scenarios

configs/compliance/
├── compliance.yaml              # Main compliance config
├── lei.yaml                     # LEI settings
├── arm.yaml                     # ARM connection
├── pre_trade_controls.yaml      # Pre-trade limits
├── audit.yaml                   # Audit settings
└── bcp.yaml                     # BCP scenarios

tests/compliance/
├── test_lei_manager.py
├── test_compliance_clock.py
├── test_transaction_report.py
├── test_pre_trade_controls.py
├── test_audit_trail.py
├── test_best_execution.py
└── test_conformance.py
```

### 11.2 Конфигурация

```yaml
# configs/compliance/compliance.yaml
compliance:
  enabled: true
  mode: "production"  # or "testing"

  # LEI Configuration
  lei:
    own_lei: "5493001KJTIIGC8Y1R12"  # Your firm's LEI
    gleif_api_url: "https://api.gleif.org/api/v1"
    cache_ttl_hours: 24

  # Clock Synchronisation (RTS 25)
  clock:
    ntp_servers:
      - "time.google.com"
      - "pool.ntp.org"
    max_offset_ms: 100  # For algo trading
    sync_interval_seconds: 60

  # Transaction Reporting
  reporting:
    arm_provider: "bloomberg_btrl"  # or "trax", "unavista"
    arm_environment: "uat"  # or "production"
    batch_size: 100
    retry_attempts: 3

  # Pre-Trade Controls (RTS 6 Art. 15)
  pre_trade:
    price_collar_pct: 5.0
    max_order_value_eur: 1000000
    max_order_volume: 10000
    max_messages_per_second: 100

  # Audit Trail
  audit:
    storage_backend: "postgresql"
    retention_years: 5
    archive_after_years: 1
    integrity_check_interval_hours: 24

  # Best Execution
  best_execution:
    factor_weights:
      price: 0.35
      cost: 0.25
      speed: 0.15
      likelihood: 0.15
      settlement: 0.05
      size: 0.03
      nature: 0.02
```

---

## 12. Референсы

### Официальные источники

1. **ESMA** (European Securities and Markets Authority)
   - [MiFID II Interactive Single Rulebook](https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifid-ii)
   - [Transaction Reporting Guidelines (ESMA/2016/1452)](https://www.esma.europa.eu/sites/default/files/library/2016-1452_guidelines_mifid_ii_transaction_reporting.pdf)
   - [Algorithmic Trading Review Report (ESMA70-156-4572)](https://www.esma.europa.eu/sites/default/files/library/esma70-156-4572_mifid_ii_final_report_on_algorithmic_trading.pdf)

2. **EUR-Lex** (Official EU Law)
   - [MiFID II Directive 2014/65/EU](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32014L0065)
   - [MiFIR Regulation 600/2014](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32014R0600)
   - [RTS 6 (Regulation 2017/589)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32017R0589)

3. **GLEIF** (Global LEI Foundation)
   - [LEI Registration Guide](https://www.gleif.org/en/about-lei/get-an-lei-find-lei-issuing-organizations)
   - [LEI Lookup API](https://www.gleif.org/en/lei-data/gleif-lei-look-up-api)

### Industry Guides

4. **Consulting Firms**
   - [Kroll: Algorithmic Trading Under MiFID II](https://www.kroll.com/en/publications/financial-compliance-regulation/algorithmic-trading-under-mifid-ii)
   - [Deloitte: RTS 6 Annual Self-Assessment](https://www.deloitte.com/uk/en/services/audit-assurance/blogs/mifid-ii-rts-6-requirements-annual-self-assessment.html)
   - [KPMG: MiFID II RTS 6 – 5 Years On](https://kpmg.com/uk/en/home/insights/2023/08/mifid-ii-rts-6.html)

5. **Law Firms**
   - [Norton Rose Fulbright: MiFID II RTS](https://www.nortonrosefulbright.com/en-gb/knowledge/publications/a1a5be12/10-things-you-should-know-the-mifid-ii-mifir-rts)
   - [DLA Piper: ESMA RTS 22 Consultation](https://www.dlapiper.com/en/insights/publications/2024/10/esma-consults-on-revisions-rts-22-on-transaction-data-reporting-and-rts-24)

### Technology Vendors

6. **ARM Providers**
   - [Bloomberg BTRL](https://www.bloomberg.com/professional/solution/regulatory-reporting/)
   - [CME TRAX](https://www.cmegroup.com/market-data/trax.html)
   - [LSEG UnaVista](https://www.lseg.com/en/post-trade/unavista)

7. **RegTech Platforms**
   - [Eventus: RTS 6 Compliance](https://www.eventus.com/cat-article/enforcement-action-from-esma-on-rts-6/)
   - [Trading Technologies: MiFID II](https://tradingtechnologies.com/resources/mifid-ii-compliance/)

### Case Studies

8. **Implementation Examples**
   - [Synetec: Argentex MiFID II Case Study](https://www.synetec.co.uk/case-study/speedy-response-to-mifid-ii-compliance)
   - [S&P Global: MiFID II Solutions](https://www.spglobal.com/marketintelligence/en/mi/solutions/mifidii.html)

---

## Приложения

### A. Checklist для внедрения

```
Phase 1: Foundation
[ ] Получен LEI
[ ] LEI Manager реализован
[ ] Clock sync работает
[ ] Algorithm Registry создан

Phase 2: Transaction Reporting
[ ] ARM provider выбран
[ ] ARM client реализован
[ ] Transaction Report model готов
[ ] Reporting pipeline работает
[ ] UAT тестирование пройдено

Phase 3: Algo Controls
[ ] Enhanced Kill Switch готов
[ ] Pre-trade controls работают
[ ] Real-time monitoring активен
[ ] OTR monitoring включён

Phase 4: Record Keeping
[ ] Audit trail schema создана
[ ] Storage backend работает
[ ] 5-year retention настроена
[ ] Integrity verification работает

Phase 5: Best Execution
[ ] Policy документирована
[ ] TCA wrapper интегрирован
[ ] Monthly reporting работает

Phase 6: Governance
[ ] Self-assessment template готов
[ ] BCP документирован
[ ] Все policy documents готовы

Phase 7: Testing
[ ] Conformance tests пройдены
[ ] External audit проведён (опционально)
[ ] NCA notification подготовлена
```

### B. Оценка ресурсов

| Фаза | Срок | FTE | Стоимость* |
|------|------|-----|------------|
| Phase 1 | 2-3 нед | 1 | €5,000 |
| Phase 2 | 4-6 нед | 2 | €15,000 |
| Phase 3 | 3-4 нед | 1.5 | €10,000 |
| Phase 4 | 4-5 нед | 2 | €15,000 |
| Phase 5 | 3-4 нед | 1 | €8,000 |
| Phase 6 | 2-3 нед | 0.5 | €3,000 |
| Phase 7 | 3-4 нед | 1 | €5,000 |
| **TOTAL** | **21-29 нед** | - | **~€60,000** |

*Без учёта: ARM subscription (~€500-1000/мес), external audit (~€50,000+), LEI renewal (~€100/год)

---

**Документ подготовлен**: 2025-12-06
**Следующий review**: После завершения Phase 1
