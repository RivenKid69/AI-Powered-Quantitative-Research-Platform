# OPTIONS_INTEGRATION_PLAN.md

## AI-Powered Quantitative Research Platform — Options Integration

**Version**: 2.0
**Status**: PLANNED
**Target Completion**: Q3 2026
**Estimated Tests**: 1,800+
**Realism Target**: 95%+

---

## Executive Summary

Полная интеграция опционов всех типов на уровне L3 LOB simulation с поддержкой:
- **US Listed Options** (CBOE, NYSE Arca, NASDAQ PHLX)
- **Index Options** (SPX, VIX, NDX — cash-settled, European)
- **Futures Options** (ES, CL, GC options — via CME/CBOT)
- **Crypto Options** (Deribit BTC/ETH — European, cash-settled)

**Ключевые метрики качества:**
| Metric | Target |
|--------|--------|
| Greeks Accuracy | < 0.1% vs analytical |
| IV Surface Error | < 0.5 vol points |
| Fill Rate (L3) | > 90% |
| Slippage Error | < 3 bps |
| Exercise Timing | < 1 bar |
| Test Coverage | 100% per phase |

---

## Critical Design Principles

### Integration with Existing Codebase

**ВАЖНО**: План интегрируется с существующей архитектурой проекта:

1. **AssetClass.OPTIONS** уже определён в `execution_providers.py:54` — использовать его
2. **LOB модуль** (`lob/`) уже имеет 8-stage архитектуру v8.0.0 — расширять, не дублировать
3. **IB адаптер** (`adapters/ib/`) уже существует — расширять `market_data.py`, `order_execution.py`
4. **Risk guards** следуют паттерну из `services/futures_risk_guards.py` — dataclasses + enums + multi-level alerts
5. **Env wrappers** следуют паттерну из `wrappers/futures_env.py` — gymnasium.Wrapper
6. **Factory functions** следуют паттерну из `execution_providers.py` — `create_*_provider()`

### Academic Foundation

Все алгоритмы основаны на peer-reviewed исследованиях:

| Component | Primary Reference | Year |
|-----------|-------------------|------|
| IV Surface | Gatheral & Jacquier: "Arbitrage-free SVI volatility surfaces" | 2014 |
| American IV | Brenner & Subrahmanyam: "A simple approach to option valuation" | 1994 |
| MM Behavior | Cho & Engle: "Market Maker Quotes in Options Markets" | 2022 |
| Early Exercise | Broadie & Detemple: "American Option Valuation" | 1996 |
| Local Vol | Dupire: "Pricing with a Smile" + Tikhonov regularization | 1994 |
| Pin Risk | Avellaneda & Lipkin: "A market-induced mechanism for stock pinning" | 2003 |
| Jump Diffusion | Merton: "Option pricing when underlying stock returns are discontinuous" | 1976 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIONS INTEGRATION ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 1: Core Models         │  Phase 2: Exchange Adapters         │
│  ├─ core_options.py           │  ├─ adapters/ib/options.py          │
│  ├─ impl_greeks.py            │  ├─ adapters/theta_data/options.py  │
│  ├─ impl_pricing.py           │  ├─ adapters/deribit/options.py     │
│  └─ impl_iv_calculation.py    │  └─ adapters/polygon/options.py     │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: IV Surface          │  Phase 4: L2 Execution              │
│  ├─ impl_iv_surface.py        │  ├─ execution_providers_options.py  │
│  ├─ impl_ssvi.py (SSVI model) │  ├─ impl_options_slippage.py        │
│  ├─ impl_heston.py            │  └─ impl_options_fees.py            │
│  └─ service_iv_calibration.py │                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 5: L3 LOB              │  Phase 6: Risk Management           │
│  ├─ lob/options_matching.py   │  ├─ services/options_risk_guards.py │
│  ├─ lob/options_mm.py         │  ├─ impl_occ_margin.py              │
│  └─ lob/pin_risk.py           │  └─ impl_exercise_assignment.py     │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 7: Complex Orders      │  Phase 8: Training Integration      │
│  ├─ impl_multi_leg.py         │  ├─ wrappers/options_env.py         │
│  ├─ impl_spread_execution.py  │  ├─ options_features.py             │
│  └─ strategies/vol_arb.py     │  └─ impl_options_reward.py          │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 9: Live Trading        │  Phase 10: Validation               │
│  ├─ services/options_live.py  │  ├─ tests/test_options_*.py         │
│  ├─ services/greeks_monitor.py│  ├─ benchmarks/bench_options.py     │
│  └─ services/exercise_mgr.py  │  └─ docs/options/*.md               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Models & Data Structures (4 weeks)

### 1.1 Objectives
- Define options contract specifications
- Implement analytical Greeks (all 12, not just 8)
- Build Black-Scholes, Leisen-Reimer Binomial, Monte Carlo pricing
- Create robust IV solver (Newton-Raphson + Brent's hybrid)
- Add Merton dividend model and jump diffusion

### 1.2 Components

#### 1.2.1 Core Options Models (`core_options.py`)

**ВАЖНО**: Интегрируется с существующим `AssetClass.OPTIONS` из `execution_providers.py:54`

```python
from execution_providers import AssetClass  # Use existing enum

@dataclass
class OptionsContractSpec:
    symbol: str                    # "AAPL240315C00175000" (OCC symbology)
    underlying: str                # "AAPL"
    option_type: OptionType        # CALL, PUT
    strike: Decimal
    expiration: date
    exercise_style: ExerciseStyle  # AMERICAN, EUROPEAN
    settlement: SettlementType     # PHYSICAL, CASH
    multiplier: int                # 100 for US equities
    tick_size: Decimal
    exchange: str                  # "CBOE", "PHLX"
    asset_class: AssetClass = AssetClass.OPTIONS  # Integrate with existing

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(Enum):
    AMERICAN = "american"
    EUROPEAN = "european"

class SettlementType(Enum):
    PHYSICAL = "physical"  # Stock delivery
    CASH = "cash"          # SPX, VIX
```

#### 1.2.2 Greeks Implementation (`impl_greeks.py`)

**Полный набор Greeks (12, не 8)**:

| Greek | Formula | Order | Use Case |
|-------|---------|-------|----------|
| **Delta** | ∂V/∂S | 1st | Directional exposure |
| **Gamma** | ∂²V/∂S² | 2nd | Convexity risk |
| **Theta** | ∂V/∂t | 1st | Time decay |
| **Vega** | ∂V/∂σ | 1st | Vol exposure |
| **Rho** | ∂V/∂r | 1st | Rate sensitivity |
| **Vanna** | ∂²V/∂S∂σ | 2nd | Skew risk |
| **Volga** (Vomma) | ∂²V/∂σ² | 2nd | Vol-of-vol |
| **Charm** | ∂Δ/∂t = ∂²V/∂S∂t | 2nd | Delta decay |
| **Speed** | ∂Γ/∂S = ∂³V/∂S³ | **3rd** | Gamma convexity |
| **Color** | ∂Γ/∂t = ∂³V/∂S²∂t | **3rd** | Gamma decay |
| **Zomma** | ∂Γ/∂σ = ∂³V/∂S²∂σ | **3rd** | Gamma vol sensitivity |
| **Ultima** | ∂Volga/∂σ = ∂³V/∂σ³ | **3rd** | Vol-of-vol-of-vol |

```python
@dataclass
class GreeksResult:
    # First-order
    delta: float
    gamma: float
    theta: float      # Per day (NOT per year)
    vega: float       # Per 1% vol (0.01 absolute)
    rho: float        # Per 1% rate

    # Second-order
    vanna: float      # ∂Δ/∂σ
    volga: float      # ∂ν/∂σ (Vomma)
    charm: float      # ∂Δ/∂t

    # Third-order (critical for MM risk)
    speed: float      # ∂Γ/∂S
    color: float      # ∂Γ/∂t
    zomma: float      # ∂Γ/∂σ
    ultima: float     # ∂Volga/∂σ

    timestamp_ns: int

    def to_dollar_terms(self, multiplier: int = 100) -> "GreeksResult":
        """Convert to dollar terms (Delta dollars, Gamma dollars, etc.)."""
```

#### 1.2.3 Pricing Models (`impl_pricing.py`)

**Black-Scholes-Merton (European with dividends)**:

Merton (1973) extension for continuous dividends:
```
C = S·e^(-qT)·N(d₁) - K·e^(-rT)·N(d₂)
P = K·e^(-rT)·N(-d₂) - S·e^(-qT)·N(-d₁)

d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

где q = continuous dividend yield
```

**Leisen-Reimer Binomial Tree (American)** — superior to CRR:

Reference: Leisen & Reimer (1996) "Binomial models for option valuation - examining and improving convergence"

```python
def leisen_reimer_tree(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: OptionType,
    steps: int = 501,  # ODD number required, min 501 for production
) -> Tuple[float, GreeksResult]:
    """
    Leisen-Reimer binomial tree with O(1/n²) convergence.

    Advantages over CRR:
    - O(1/n²) convergence vs O(1/n) for CRR
    - No oscillation between even/odd steps
    - Better Greeks stability

    Parameters:
        steps: Must be ODD, minimum 501 for production accuracy (<0.01% error)
    """
```

**Merton Jump-Diffusion** (for earnings, M&A events):

Reference: Merton (1976) "Option pricing when underlying stock returns are discontinuous"

```
dS/S = (μ - λk)dt + σdW + (J-1)dN

где:
- λ = jump intensity (jumps per year)
- k = E[J-1] = expected relative jump size
- J = jump multiplier (lognormal)
- N = Poisson process
```

```python
def merton_jump_diffusion_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    jump_intensity: float,      # λ, typically 1-5 per year
    jump_mean: float,           # μ_J, typically -0.05 to 0
    jump_vol: float,            # σ_J, typically 0.1-0.3
    option_type: OptionType,
    n_terms: int = 50,          # Series truncation
) -> float:
    """
    Merton (1976) jump-diffusion pricing via series expansion.

    Critical for:
    - Earnings announcements
    - M&A situations
    - Flash crash scenarios
    """
```

**Variance Swap Replication** (for vol trading):

Reference: Carr & Madan (1998) "Towards a theory of volatility trading"

```python
def variance_swap_strike(
    option_chain: List[OptionsQuote],
    forward: float,
    time_to_expiry: float,
    rate: float,
) -> float:
    """
    Fair variance strike via replication:

    K_var² = (2/T) × [∫₀^F (1/K²)P(K)dK + ∫_F^∞ (1/K²)C(K)dK]

    Discretized using listed strikes.
    """
```

#### 1.2.4 IV Solver (`impl_iv_calculation.py`)

**CRITICAL FIX**: Newton-Raphson fails for deep OTM options (vega → 0). Use hybrid approach:

Reference: Jäckel (2015) "Let's Be Rational"

```python
def compute_implied_volatility_european(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    method: str = "hybrid",  # "newton", "brent", "hybrid", "jaeckel"
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """
    Robust IV solver with multiple fallback methods.

    Methods:
    - "newton": Newton-Raphson with Brenner-Subrahmanyam seed (fast, but fails deep OTM)
    - "brent": Brent's method (robust, but slower)
    - "hybrid": Newton-Raphson with Brent fallback (recommended)
    - "jaeckel": Jäckel (2015) rational approximation (fastest, most robust)

    Brenner-Subrahmanyam (1994) initial guess:
        σ₀ = √(2π/T) × C/S  (for ATM)

    For deep OTM (|Δ| < 0.1):
        Use Brent's method on [0.01, 5.0] interval
    """
    if method == "hybrid":
        try:
            return _newton_raphson_iv(
                option_price, spot, strike, time_to_expiry, rate,
                dividend_yield, option_type, max_iterations // 2, tolerance
            )
        except IVConvergenceError:
            # Fallback to Brent's method for difficult cases
            return _brent_iv(
                option_price, spot, strike, time_to_expiry, rate,
                dividend_yield, option_type, tolerance
            )


def compute_implied_volatility_american(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    tree_steps: int = 201,
    tolerance: float = 1e-6,
) -> float:
    """
    American IV solver using Leisen-Reimer tree inversion.

    CANNOT use Black-Scholes for American options!
    Must use binomial tree with early exercise.

    Uses bisection on σ ∈ [0.01, 3.0] with tree pricing.
    """
```

### 1.3 Test Matrix (Phase 1)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsContractSpec | 15 | Creation, validation, OCC symbology |
| Greeks (all 12, BS analytical) | 35 | First/second/third order, edge cases |
| Black-Scholes-Merton pricing | 25 | Calls/puts, ATM/OTM/ITM, dividends |
| Leisen-Reimer (American) | 25 | Early exercise, convergence vs BS |
| Merton Jump-Diffusion | 15 | Jump parameters, series convergence |
| Variance Swap | 10 | Replication, strike calculation |
| IV solver (European) | 20 | N-R, Brent, hybrid, deep OTM |
| IV solver (American) | 15 | Tree inversion, dividends |
| **Total** | **160** | **100%** |

### 1.4 Deliverables
- [ ] `core_options.py` — Contract specs, enums (integrates with AssetClass.OPTIONS)
- [ ] `impl_greeks.py` — All 12 Greeks with numerical validation
- [ ] `impl_pricing.py` — BS-Merton, Leisen-Reimer, Merton JD, Variance Swap
- [ ] `impl_iv_calculation.py` — Hybrid IV solver (European + American)
- [ ] `tests/test_options_core.py` — 160 tests
- [ ] Documentation: `docs/options/core_models.md`

### 1.5 Regression Check
```bash
pytest tests/ -x --ignore=tests/test_options_*.py  # All existing tests pass
pytest tests/test_options_core.py -v               # All new tests pass
```

---

## Phase 2: Exchange Adapters (5 weeks)

### 2.1 Objectives
- Extend existing IB TWS adapter for US options
- Add Theta Data as primary options data source (cost-effective)
- Deribit crypto options
- Polygon.io for historical options data

### 2.2 Critical Notes on Data Sources

**OPRA Feed**: $2,500/month minimum — NOT recommended for retail/small funds.
Use IB or Theta Data instead.

**CBOE Direct API**: No direct retail access. Use broker APIs (IB, Schwab).

**Recommended Stack**:
| Data Need | Primary Source | Backup |
|-----------|---------------|--------|
| Real-time US options | IB TWS API | Theta Data |
| Historical US options | Theta Data | Polygon.io |
| Index options (SPX, VIX) | IB TWS API | — |
| Crypto options | Deribit API | — |

### 2.3 Supported Exchanges

| Exchange | Asset Class | Data | Execution | Protocol | Cost |
|----------|-------------|------|-----------|----------|------|
| **IB TWS** | US Equity Options | ✅ | ✅ | TWS API | Subscription |
| **Theta Data** | US Options (historical+RT) | ✅ | ❌ | REST/WS | $100/mo |
| **Deribit** | Crypto Options | ✅ | ✅ | REST/WS | Free |
| **Polygon.io** | US Options (historical) | ✅ | ❌ | REST | $200/mo |

### 2.4 Components

#### 2.4.1 IB Options Extension (`adapters/ib/options.py`)

**ВАЖНО**: Расширяет существующий `adapters/ib/market_data.py` и `adapters/ib/order_execution.py`

```python
from adapters.ib.market_data import IBMarketDataAdapter
from adapters.ib.order_execution import IBOrderExecutionAdapter

class IBOptionsMarketDataAdapter(IBMarketDataAdapter):
    """Extension for options-specific market data."""

    def get_option_chain(
        self, underlying: str, expiration: Optional[date] = None
    ) -> List[OptionsContractSpec]:
        """Fetch full option chain for underlying."""

    def get_option_quote(
        self, contract: OptionsContractSpec
    ) -> OptionsQuote:
        """Real-time bid/ask/last with Greeks from IB."""

    async def stream_option_quotes_async(
        self, contracts: List[OptionsContractSpec]
    ) -> AsyncIterator[OptionsQuote]:
        """Real-time streaming quotes."""


class IBOptionsOrderExecutionAdapter(IBOrderExecutionAdapter):
    """Extension for options order execution."""

    def submit_option_order(
        self, order: OptionsOrder
    ) -> OrderResult:
        """Submit single-leg option order."""

    def submit_combo_order(
        self, legs: List[ComboLeg], order_type: str
    ) -> OrderResult:
        """Submit multi-leg spread order."""

    def get_option_margin_requirement(
        self, order: OptionsOrder
    ) -> MarginRequirement:
        """Query margin requirement before order."""


@dataclass
class OptionsQuote:
    bid: Decimal
    ask: Decimal
    last: Decimal
    bid_size: int
    ask_size: int
    underlying_price: Decimal
    iv: float                    # IB-calculated IV
    greeks: GreeksResult         # IB-provided Greeks
    open_interest: int
    volume: int
    timestamp_ns: int
```

**IB Rate Limits for Options** (from existing IBRateLimiter):
| Limit Type | IB Limit | Safety Margin |
|------------|----------|---------------|
| Option chains | 10/min | 8/min |
| Quote requests | 100/sec | 80/sec |
| Order submissions | 50/sec | 40/sec |
| Concurrent market data | 100 lines | 100 lines |

#### 2.4.2 Theta Data Adapter (`adapters/theta_data/options.py`)

Reference: https://www.thetadata.io/

Cost-effective alternative to OPRA ($100/mo vs $2,500/mo):

```python
class ThetaDataOptionsAdapter:
    """
    Theta Data options adapter.

    Advantages:
    - $100/month (vs OPRA $2,500/month)
    - Full US options universe
    - Historical data back to 2013
    - Real-time with 15-min delay (free) or real-time ($)
    """

    def get_option_chain(
        self, underlying: str, expiration: Optional[date] = None
    ) -> List[OptionsContractSpec]:
        """Full option chain with OCC symbology."""

    def get_historical_quotes(
        self, contract: OptionsContractSpec,
        start_date: date,
        end_date: date,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """Historical OHLCV + Greeks + IV."""

    def get_historical_trades(
        self, contract: OptionsContractSpec,
        date: date,
    ) -> pd.DataFrame:
        """Tick-level trade data for backtesting."""

    def get_eod_chain(
        self, underlying: str,
        date: date,
    ) -> pd.DataFrame:
        """End-of-day snapshot of full chain."""
```

#### 2.4.3 Deribit Crypto Options (`adapters/deribit/options.py`)

```python
class DeribitOptionsAdapter:
    """
    Deribit BTC/ETH options (European, cash-settled).

    Specifics:
    - Settlement in underlying crypto (not USD)
    - Inverse contracts: 1 BTC = 1 contract
    - Expiration: 08:00 UTC on expiry date
    - IV index: DVOL (Deribit Volatility Index)
    - 24/7 trading
    """

    def get_btc_options(self) -> List[OptionsContractSpec]:
        """BTC options (European, cash-settled in BTC)."""

    def get_eth_options(self) -> List[OptionsContractSpec]:
        """ETH options (European, cash-settled in ETH)."""

    def get_orderbook(
        self, instrument: str, depth: int = 20
    ) -> OptionsOrderBook:
        """L2 order book for options."""

    async def stream_quotes_async(
        self, instruments: List[str]
    ) -> AsyncIterator[OptionsQuote]:
        """Real-time quote stream via WebSocket."""

    def get_dvol(self) -> float:
        """Current DVOL (Deribit Volatility Index)."""
```

#### 2.4.4 Polygon Options Historical (`adapters/polygon/options.py`)

Extends existing `adapters/polygon/market_data.py`:

```python
from adapters.polygon.market_data import PolygonMarketDataAdapter

class PolygonOptionsAdapter(PolygonMarketDataAdapter):
    """
    Polygon.io options historical data.

    Use for:
    - Historical backtesting (2018+)
    - EOD chain snapshots
    - Corporate actions (splits, dividends)
    """

    def get_historical_chain(
        self, underlying: str, date: date
    ) -> pd.DataFrame:
        """Historical EOD chain snapshot."""

    def get_historical_quotes(
        self, contract_symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Historical NBBO quotes."""
```

### 2.5 Test Matrix (Phase 2)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| IB Options Extension | 40 | Chain fetch, quotes, orders, combos |
| IB Rate Limiting | 10 | Throttling, backoff |
| Theta Data Adapter | 30 | Chains, historical, EOD |
| Deribit Adapter | 30 | BTC/ETH, orderbook, streaming |
| Polygon Options | 20 | Historical chains, quotes |
| Registry Integration | 10 | Factory functions |
| **Total** | **140** | **100%** |

### 2.6 Deliverables
- [ ] `adapters/ib/options.py` — IB options extension
- [ ] `adapters/theta_data/options.py` — Theta Data adapter
- [ ] `adapters/deribit/options.py` — Crypto options
- [ ] `adapters/polygon/options.py` — Historical options
- [ ] Registry updates in `adapters/registry.py`
- [ ] `tests/test_options_adapters.py` — 140 tests
- [ ] Documentation: `docs/options/exchange_adapters.md`

---

## Phase 3: IV Surface & Volatility Models (5 weeks)

### 3.1 Objectives
- Construct arbitrage-free IV surface using SSVI
- Implement Gatheral & Jacquier (2014) arbitrage conditions
- Add Heston for equity (NOT SABR — SABR is for rates/FX only)
- Build Dupire local vol with Tikhonov regularization
- Robust calibration service

### 3.2 Components

#### 3.2.1 IV Surface (`impl_iv_surface.py`)

```python
class IVSurface:
    def __init__(
        self,
        underlying: str,
        spot: float,
        rate: float,
        dividend_yield: float = 0.0,
    ):
        self._strikes: np.ndarray = None
        self._expiries: np.ndarray = None  # In years
        self._iv_grid: np.ndarray = None   # 2D: [expiry, strike]
        self._ssvi_params: Optional[SSVIParams] = None

    def fit_from_quotes(
        self, quotes: List[OptionsQuote], method: str = "ssvi"
    ) -> None:
        """
        Fit surface from market quotes.

        Methods:
        - "ssvi": Surface SVI (Gatheral & Jacquier 2014) — RECOMMENDED
        - "svi": Raw SVI per slice (may have calendar arbitrage!)
        - "cubic": Cubic spline (fast, but not arbitrage-free)
        """

    def get_iv(self, strike: float, expiry: float) -> float:
        """Interpolate IV at given strike/expiry."""

    def get_local_vol(self, strike: float, expiry: float) -> float:
        """Dupire local volatility with regularization."""

    def get_forward_iv(
        self, strike: float, t1: float, t2: float
    ) -> float:
        """Forward-starting implied vol."""

    def check_arbitrage_free(self) -> ArbitrageReport:
        """
        Check all arbitrage conditions:
        - Butterfly: d²w/dk² > 0 (convexity in total variance)
        - Calendar: ∂w/∂T > 0 (total variance increasing in time)
        - Call spread: -1 < ∂C/∂K < 0
        """
```

#### 3.2.2 SSVI (Surface SVI) — Gatheral & Jacquier (2014)

Reference: Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces"

**CRITICAL**: Use SSVI, not raw SVI! Raw SVI per slice does NOT guarantee calendar arbitrage-free.

```python
@dataclass
class SSVIParams:
    """
    Surface SVI parameterization (Gatheral & Jacquier 2014).

    Total variance: w(k, θ) = (θ/2) × {1 + ρφ(θ)k + √[(φ(θ)k + ρ)² + (1-ρ²)]}

    where:
    - θ = ATM total variance = σ_ATM² × T
    - k = log-moneyness = ln(K/F)
    - φ(θ) = η / θ^γ  (power-law ATM skew)
    - ρ = correlation (typically -0.4 to -0.9 for equity)
    - η = skew magnitude
    - γ = skew decay (typically 0.4-0.6)
    """
    rho: float        # Correlation, |ρ| < 1
    eta: float        # Skew magnitude, η > 0
    gamma: float      # Skew decay, 0 < γ < 1

    def phi(self, theta: float) -> float:
        """ATM skew function φ(θ) = η/θ^γ"""
        return self.eta / (theta ** self.gamma)


class SSVICalibrator:
    """
    SSVI calibration with Gatheral-Jacquier arbitrage-free conditions.
    """

    def calibrate(
        self, slices: Dict[float, List[OptionsQuote]]
    ) -> SSVIParams:
        """
        Calibrate SSVI to multiple expiry slices.

        Arbitrage-free conditions (Gatheral & Jacquier 2014, Theorem 4.1):

        1. Butterfly condition (no vertical spread arbitrage):
           g(k) ≥ 0 for all k, where
           g(k) = (1 - kw'/(2w))² - (w'/4)(1/w + 1/4) + w''/2

        2. Calendar spread condition (no horizontal arbitrage):
           ∂θ/∂T ≥ 0 (ATM variance increasing in time)

        3. Large-strike asymptotics:
           w(k)/|k| → 2 as |k| → ∞ (Roger Lee moment formula)

        Uses constrained optimization (SLSQP) to enforce all conditions.
        """
```

#### 3.2.3 Heston Model (`impl_heston.py`)

**ВАЖНО**: Use Heston for equity options, NOT SABR!
SABR is designed for rates/FX (no mean reversion), Heston has mean-reverting variance.

Reference: Heston (1993) "A Closed-Form Solution for Options with Stochastic Volatility"

```
dS = μS dt + √V S dW₁
dV = κ(θ - V) dt + ξ√V dW₂
dW₁·dW₂ = ρ dt

Parameters:
- κ: Mean reversion speed (typically 1-5)
- θ: Long-term variance (typically 0.04-0.16, i.e., 20-40% vol)
- ξ: Vol-of-vol (typically 0.3-1.0)
- ρ: Correlation (typically -0.5 to -0.9 for equity)
- V₀: Initial variance

Feller condition for positive variance:
    2κθ > ξ² (ensures V > 0 almost surely)
```

```python
@dataclass
class HestonParams:
    kappa: float      # Mean reversion speed
    theta: float      # Long-term variance (NOT volatility!)
    xi: float         # Vol-of-vol
    rho: float        # Correlation
    v0: float         # Initial variance

    def check_feller(self) -> bool:
        """Check Feller condition: 2κθ > ξ²"""
        return 2 * self.kappa * self.theta > self.xi ** 2


class HestonPricer:
    """Heston model pricing via characteristic function."""

    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        params: HestonParams,
        option_type: OptionType,
    ) -> float:
        """
        Heston (1993) semi-analytical pricing.

        Uses Gauss-Laguerre quadrature for characteristic function inversion.
        Lewis (2000) formulation for numerical stability.
        """
```

#### 3.2.4 Dupire Local Volatility with Regularization (`impl_local_vol.py`)

Reference: Dupire (1994) "Pricing with a Smile"

**CRITICAL**: Dupire formula is numerically unstable! Must use Tikhonov regularization.

```python
def dupire_local_vol(
    iv_surface: IVSurface,
    strike: float,
    expiry: float,
    regularization: str = "tikhonov",
    lambda_reg: float = 0.01,
) -> float:
    """
    Dupire (1994) local volatility:

    σ_loc²(K, T) = [∂w/∂T] / [1 - (k/w)∂w/∂k + (1/4)(-1/4 - 1/w + k²/w²)(∂w/∂k)² + (1/2)∂²w/∂k²]

    where w = σ²T (total variance)

    PROBLEM: Denominator can be very small or negative → unstable!

    SOLUTION: Tikhonov regularization
        min ||σ_loc - σ_target||² + λ||∇²σ_loc||²

    This smooths the local vol surface and prevents spikes.
    """
```

#### 3.2.5 Calibration Service (`service_iv_calibration.py`)

```python
class IVCalibrationService:
    """
    Production IV surface calibration service.

    Features:
    - Real-time recalibration on quote updates
    - Arbitrage-free enforcement (SSVI)
    - Stale quote filtering
    - Bid-ask midpoint with spread weighting
    """

    def calibrate_ssvi(
        self,
        quotes: List[OptionsQuote],
        min_volume: int = 10,
        max_spread_pct: float = 0.5,
    ) -> IVSurface:
        """
        Calibrate arbitrage-free SSVI surface.

        Filters:
        - Remove quotes with volume < min_volume
        - Remove quotes with spread > max_spread_pct
        - Use volume-weighted midpoint
        """

    def calibrate_heston(
        self,
        quotes: List[OptionsQuote],
        method: str = "differential_evolution",
    ) -> HestonParams:
        """
        Calibrate Heston to surface.

        Methods:
        - "differential_evolution": Global optimization (recommended)
        - "levenberg_marquardt": Local optimization (faster, needs good seed)
        """

    def check_arbitrage(self, surface: IVSurface) -> ArbitrageReport:
        """
        Full arbitrage check:
        - Butterfly (strike convexity)
        - Calendar (time monotonicity)
        - Put-call parity violations
        """
```

### 3.3 Test Matrix (Phase 3)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| IVSurface construction | 25 | From quotes, interpolation |
| SSVI fitting | 35 | Gatheral-Jacquier conditions, calibration |
| Heston pricing | 25 | Characteristic function, Greeks |
| Heston calibration | 20 | DE, LM, parameter recovery |
| Dupire local vol | 20 | Regularization, stability |
| Arbitrage detection | 20 | Butterfly, calendar, PCP |
| Forward vol | 10 | Term structure |
| **Total** | **155** | **100%** |

### 3.4 Deliverables
- [ ] `impl_iv_surface.py` — IV surface with SSVI
- [ ] `impl_ssvi.py` — SSVI model with Gatheral-Jacquier conditions
- [ ] `impl_heston.py` — Heston model (NOT SABR for equity!)
- [ ] `impl_local_vol.py` — Dupire with Tikhonov regularization
- [ ] `service_iv_calibration.py` — Production calibration service
- [ ] `tests/test_iv_surface.py` — 155 tests
- [ ] Documentation: `docs/options/volatility_surface.md`

---

## Phase 4: L2 Execution Provider (4 weeks)

### 4.1 Objectives
- Options-specific slippage model with moneyness/DTE factors
- Greeks-aware execution
- Bid-ask spread modeling (including PFOF effects)
- Options fee structures

### 4.2 Components

#### 4.2.1 Options Slippage Model (`impl_options_slippage.py`)

**CRITICAL**: Options slippage fundamentally differs from equity!

Key differences:
- Spread is NOT constant — varies with moneyness, DTE, underlying vol
- Gamma exposure creates asymmetric slippage
- Retail flow (PFOF) gets better execution than displayed

Reference: Muravyev & Pearson (2020) "Options Trading Costs Are Lower Than You Think"

```python
@dataclass
class OptionsSlippageConfig:
    # Base factors
    base_spread_pct: float = 0.05          # 5% of mid as base

    # Moneyness adjustments
    atm_spread_mult: float = 1.0           # ATM is base
    otm_10_spread_mult: float = 1.5        # 10% OTM: 1.5x spread
    otm_20_spread_mult: float = 2.5        # 20% OTM: 2.5x spread
    deep_otm_spread_mult: float = 4.0      # 30%+ OTM: 4x spread

    # DTE adjustments
    dte_30_spread_mult: float = 1.0        # 30+ DTE: base
    dte_7_spread_mult: float = 1.5         # 7-30 DTE: 1.5x
    dte_1_spread_mult: float = 3.0         # 0-7 DTE: 3x (expiration gamma)

    # PFOF retail advantage
    pfof_improvement_bps: float = 5.0      # 5 bps price improvement for retail


class OptionsSlippageProvider:
    """
    Options slippage model with:
    - Moneyness dependency (ATM tighter than OTM)
    - DTE dependency (near-expiry wider)
    - Greeks impact (gamma, vega exposure)
    - PFOF retail flow improvement
    - Underlying correlation (delta hedge cost)

    Reference: Muravyev & Pearson (2020)
    """

    def compute_slippage_bps(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        greeks: GreeksResult,
        is_retail: bool = False,
    ) -> float:
        """
        Total slippage in basis points.

        Components:
        1. spread_component = spread/2 × moneyness_mult × dte_mult
        2. gamma_component = γ_impact × |gamma| × spot² × vol × √participation
        3. vega_component = ν_impact × |vega| × expected_iv_move
        4. theta_component = θ_impact × |theta| × execution_time
        5. delta_hedge_component = |delta| × underlying_slippage
        6. pfof_improvement = -pfof_bps if is_retail else 0
        """
```

#### 4.2.2 Options Fee Provider (`impl_options_fees.py`)

**ВАЖНО**: Следует паттерну из `execution_providers.py:FeeProvider`

| Exchange | Per Contract | Index Premium | Regulatory | Notes |
|----------|--------------|---------------|------------|-------|
| CBOE | $0.44 | $0.66 (SPX) | OCC $0.055 | Maker/taker varies |
| NYSE Arca | $0.47 | — | OCC $0.055 | |
| Deribit | 0.03% notional | — | None | Capped at 12.5% of premium |
| IB (Tiered) | $0.15-0.65 | — | Varies | Volume-based |
| IB (Fixed) | $0.65 | — | Varies | Simple |

```python
class OptionsFeeProvider:
    """Options fee provider following FeeProvider protocol."""

    def compute_fee(
        self,
        contract: OptionsContractSpec,
        qty: int,
        price: Decimal,
        exchange: str,
        is_maker: bool = False,
        is_index: bool = False,
    ) -> Decimal:
        """
        Total fee = exchange_fee + occ_fee + regulatory_fee

        OCC fee: $0.055/contract (all US options)
        Regulatory: SEC fee ($0.0000278/$ on sells), TAF, etc.
        """
```

#### 4.2.3 Options Execution Provider (`execution_providers_options.py`)

**ВАЖНО**: Integrates with existing `AssetClass.OPTIONS` and follows `L2ExecutionProvider` pattern

```python
from execution_providers import (
    SlippageProvider,
    FillProvider,
    FeeProvider,
    AssetClass,
    create_execution_provider,
)

class OptionsL2ExecutionProvider:
    """
    L2 execution provider for options.

    Follows existing pattern from execution_providers.py.
    Integrates with AssetClass.OPTIONS.
    """

    def __init__(
        self,
        slippage_provider: OptionsSlippageProvider,
        fee_provider: OptionsFeeProvider,
        greeks_provider: GreeksProvider,
    ):
        self.asset_class = AssetClass.OPTIONS

    def execute(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        bar: BarData,
        underlying_bar: BarData,
        is_retail: bool = False,
    ) -> OptionsFill:
        """Execute single-leg option order."""

    def estimate_cost(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
    ) -> CostEstimate:
        """Pre-trade cost estimation with Greeks impact."""


# Factory function following existing pattern
def create_options_execution_provider(
    level: str = "L2",
    profile: str = "default",
) -> OptionsL2ExecutionProvider:
    """
    Factory function for options execution provider.

    Follows pattern from execution_providers.create_execution_provider().
    """
```

### 4.3 Test Matrix (Phase 4)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Options Slippage (moneyness) | 20 | ATM, OTM, ITM |
| Options Slippage (DTE) | 15 | 30+, 7-30, 0-7 DTE |
| Options Slippage (Greeks) | 20 | Gamma, vega, theta impact |
| Options Slippage (PFOF) | 10 | Retail vs institutional |
| Fee Calculation | 25 | All exchanges, index premium |
| L2 Execution | 35 | Market/limit, fills |
| Cost Estimation | 15 | Pre-trade analysis |
| Factory Integration | 10 | create_options_execution_provider |
| **Total** | **150** | **100%** |

### 4.4 Deliverables
- [ ] `impl_options_slippage.py` — Moneyness/DTE/Greeks-aware slippage
- [ ] `impl_options_fees.py` — Fee structures (all exchanges)
- [ ] `execution_providers_options.py` — L2 provider (integrates with AssetClass.OPTIONS)
- [ ] `tests/test_options_execution_l2.py` — 150 tests
- [ ] Documentation: `docs/options/execution_l2.md`

---

## Phase 5: L3 LOB Simulation (6 weeks)

### 5.1 Objectives
- Options-specific order book (fundamentally different from equity!)
- Market maker behavior simulation (Cho & Engle 2022)
- Quote dynamics with regime detection
- Pin risk simulation (Avellaneda & Lipkin 2003)
- Cross-strike arbitrage detection

### 5.2 Critical Design Note

**Options LOB is FUNDAMENTALLY DIFFERENT from Equity LOB!**

| Aspect | Equity LOB | Options LOB |
|--------|-----------|-------------|
| Spread | 1 cent typical | $0.05-$1.00 typical |
| Depth | 1000s of shares | 10-100 contracts |
| Liquidity | Continuous | Episodic |
| Quote updates | Tick-by-tick | On underlying move |
| MM behavior | Inventory-based | Greeks-based |
| Expiration | N/A | Liquidity collapse |

**IMPORTANT**: Do NOT simply extend `lob/matching_engine.py`. Create `lob/options_matching.py` with options-specific logic, but reuse primitives from `lob/data_structures.py`.

### 5.3 Components

#### 5.3.1 Options Matching Engine (`lob/options_matching.py`)

```python
from lob.data_structures import LimitOrder, Side, OrderType, Fill

class OptionsMatchingEngine:
    """
    Options-specific matching engine.

    Key differences from equity:
    1. Priority: Price-Time-Pro-Rata hybrid (CBOE uses pro-rata for MM)
    2. Quote updates: Entire book shifts on underlying move
    3. Order types: Complex orders (spreads) have priority
    4. Minimum size: Often 1 contract minimum displayed
    """

    def __init__(
        self,
        contract: OptionsContractSpec,
        mm_pro_rata_allocation: float = 0.4,  # 40% pro-rata to MMs
    ):
        self.contract = contract
        self._bids: SortedDict[Decimal, PriceLevel] = SortedDict()
        self._asks: SortedDict[Decimal, PriceLevel] = SortedDict()

    def match_with_pro_rata(
        self, order: LimitOrder
    ) -> List[Fill]:
        """
        CBOE-style pro-rata matching:
        1. First, fill any price-improving orders
        2. Then, allocate 40% pro-rata to MMs at NBBO
        3. Finally, FIFO for remainder
        """

    def shift_book_on_underlying_move(
        self,
        underlying_move_pct: float,
        delta: float,
        gamma: float,
    ) -> None:
        """
        Shift all quotes based on delta/gamma.

        New price = old_price + delta × ΔS + 0.5 × gamma × ΔS²
        """
```

#### 5.3.2 Market Maker Simulator (`lob/options_mm.py`)

Reference: Cho & Engle (2022) "Market Maker Quotes in Options Markets"

**Key insight**: MM quoting is regime-dependent!

```python
@dataclass
class MMQuotingRegime(Enum):
    """
    Market maker quoting regimes (Cho & Engle 2022).

    Regime affects spread, size, and requote frequency.
    """
    NORMAL = "normal"           # Standard quoting
    HIGH_VOL = "high_vol"       # Wider spreads, smaller size
    EARNINGS = "earnings"       # Much wider, minimal size
    EXPIRATION = "expiration"   # Gamma-driven, very wide
    LOW_LIQUIDITY = "low_liq"   # Off-hours, wide spreads


class OptionsMMSimulator:
    """
    Regime-dependent market maker simulator.

    Based on Cho & Engle (2022) empirical findings:
    - Spread = f(|gamma|, vega/theta ratio, inventory, regime)
    - Size = f(max_position, concentration, regime)
    - Requote trigger = f(delta_change, iv_change, time)
    """

    def __init__(
        self,
        inventory_limit: int = 500,        # Lower than equity!
        base_spread_pct: float = 0.03,     # 3% base
        gamma_factor: float = 0.8,         # Cho & Engle coefficient
        vega_theta_factor: float = 0.4,    # Vega/theta impact
        regime_spreads: Dict[MMQuotingRegime, float] = None,
    ):
        self.regime_spreads = regime_spreads or {
            MMQuotingRegime.NORMAL: 1.0,
            MMQuotingRegime.HIGH_VOL: 2.5,
            MMQuotingRegime.EARNINGS: 5.0,
            MMQuotingRegime.EXPIRATION: 3.0,
            MMQuotingRegime.LOW_LIQUIDITY: 4.0,
        }

    def detect_regime(
        self,
        underlying_rv: float,
        iv_level: float,
        dte: float,
        has_earnings: bool,
        hour_utc: int,
    ) -> MMQuotingRegime:
        """Detect current quoting regime."""

    def generate_quotes(
        self,
        contract: OptionsContractSpec,
        underlying: float,
        iv: float,
        greeks: GreeksResult,
        current_inventory: int,
        regime: MMQuotingRegime,
    ) -> Tuple[Quote, Quote]:
        """
        Generate bid/ask quotes.

        Spread formula (Cho & Engle 2022):
        spread = base × regime_mult × (1 + γ_factor × |gamma/gamma_max|
                                         + ν_factor × |vega/theta|)

        Size formula:
        size = max_pos × (1 - |inventory/limit|) × (1 - concentration_penalty)
        """

    def should_requote(
        self,
        delta_change: float,
        iv_change: float,
        time_elapsed_sec: float,
        regime: MMQuotingRegime,
    ) -> bool:
        """
        Check if requote needed.

        Thresholds vary by regime:
        - NORMAL: Δdelta > 0.01 OR Δiv > 0.5%
        - HIGH_VOL: Δdelta > 0.005 OR Δiv > 0.25%
        - EXPIRATION: Δdelta > 0.001 (very sensitive)
        """
```

#### 5.3.3 Pin Risk Simulation (`lob/pin_risk.py`)

Reference: Avellaneda & Lipkin (2003) "A market-induced mechanism for stock pinning"

```python
class PinRiskSimulator:
    """
    Stock pinning simulation near expiration.

    Mechanism (Avellaneda & Lipkin 2003):
    1. Near expiry, delta hedging by MMs creates feedback
    2. If spot near strike with high OI, hedging pushes spot toward strike
    3. Probability of pinning ∝ OI × gamma × sqrt(T)
    """

    def compute_pin_probability(
        self,
        strike: float,
        spot: float,
        time_to_expiry: float,
        iv: float,
        open_interest: int,
        delta_hedger_volume: float,
    ) -> float:
        """
        Pin probability using Avellaneda-Lipkin model:

        P_pin ∝ OI × Γ × √T × (hedger_volume / total_volume)

        where Γ = gamma at strike
        """

    def simulate_expiry_dynamics(
        self,
        strikes: List[float],
        spot: float,
        oi_by_strike: Dict[float, int],
        hedger_fraction: float,
        n_paths: int = 10000,
    ) -> PinningAnalysis:
        """
        Monte Carlo simulation of spot dynamics near expiry.

        Returns:
        - Pin probabilities by strike
        - Expected spot distribution at expiry
        - Gamma exposure concentration
        """
```

#### 5.3.4 Cross-Strike Arbitrage (`lob/options_arbitrage.py`)

```python
class OptionsArbitrageDetector:
    """
    Real-time arbitrage detection in options markets.

    Types:
    1. Butterfly: C(K-) - 2C(K) + C(K+) should be positive
    2. Box spread: Should price at risk-free rate
    3. Conversion/Reversal: Put-call parity violations
    4. Calendar: Far expiry should be worth more
    """

    def detect_butterfly_arb(
        self,
        quotes: Dict[float, OptionsQuote],
        transaction_cost: float,
    ) -> List[ButterflyArbitrage]:
        """
        Find butterfly arbitrage opportunities.

        Condition: C(K-) + C(K+) > 2C(K) + 2 × transaction_cost
        """

    def detect_box_spread_arb(
        self,
        call_quotes: Dict[float, OptionsQuote],
        put_quotes: Dict[float, OptionsQuote],
        risk_free_rate: float,
        time_to_expiry: float,
    ) -> Optional[BoxArbitrage]:
        """
        Find box spread mispricing.

        Box value should be: (K_high - K_low) × e^(-rT)
        """
```

### 5.4 Test Matrix (Phase 5)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsMatchingEngine | 35 | FIFO, pro-rata, book shift |
| MM Simulator (regimes) | 40 | All 5 regimes, transitions |
| MM Simulator (quoting) | 30 | Spread formula, size, requote |
| Pin Risk | 25 | Probability, dynamics |
| Arbitrage Detection | 25 | Butterfly, box, calendar |
| Integration | 25 | Full L3 simulation |
| **Total** | **180** | **100%** |

### 5.5 Deliverables
- [ ] `lob/options_matching.py` — Options matching engine
- [ ] `lob/options_mm.py` — Cho & Engle MM simulator
- [ ] `lob/pin_risk.py` — Avellaneda-Lipkin pin simulation
- [ ] `lob/options_arbitrage.py` — Real-time arbitrage detection
- [ ] `tests/test_options_l3_lob.py` — 180 tests
- [ ] Documentation: `docs/options/l3_lob.md`

---

## Phase 6: Risk Management (5 weeks)

### 6.1 Objectives
- Options-specific risk guards (following `services/futures_risk_guards.py` pattern)
- OCC Clearing Margin (NOT TIMS — TIMS is outdated!)
- Reg T margin with correct formulas
- Exercise/assignment simulation with gamma convexity

### 6.2 Components

#### 6.2.1 Options Risk Guards (`services/options_risk_guards.py`)

**ВАЖНО**: Follows pattern from `services/futures_risk_guards.py` — dataclasses + enums + multi-level alerts

```python
from enum import Enum
from dataclasses import dataclass

class OptionsRiskLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class GreeksLimitConfig:
    max_delta: float = 100.0      # Net delta in underlying terms
    max_gamma: float = 50.0       # Per 1% move
    max_vega: float = 10000.0     # Per 1 vol point
    max_theta: float = -500.0     # Daily theta limit (negative OK for theta-positive)
    max_speed: float = 100.0      # Third-order gamma risk


@dataclass
class GreeksRiskResult:
    level: OptionsRiskLevel
    delta_utilization: float
    gamma_utilization: float
    vega_utilization: float
    theta_utilization: float
    speed_utilization: float      # Third-order risk
    breach_details: List[str]


class OptionsGreeksGuard:
    """
    Monitor portfolio Greeks limits.

    Follows pattern from FuturesLeverageGuard in services/futures_risk_guards.py.
    """

    def __init__(self, config: GreeksLimitConfig):
        self.config = config

    def check_position(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> GreeksRiskResult:
        """Check if aggregate Greeks within limits."""

    def check_new_trade(
        self,
        proposed_trade: OptionsOrder,
        current_positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Pre-trade Greeks check."""


class ExerciseRiskGuard:
    """
    Monitor early exercise risk for American options.

    Based on Broadie & Detemple (1996) optimal exercise boundary.
    """

    def check_early_exercise_risk(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry: float,
        dividend_schedule: List[Dividend],
        rate: float,
    ) -> ExerciseRiskResult:
        """
        Identify positions at risk of early exercise.

        Decision criteria (Broadie & Detemple 1996):
        1. For calls: Exercise if dividend > time_value + gamma_convexity_value
        2. For puts: Exercise if rate income > time_value + gamma_convexity_value

        Gamma convexity value is often ignored but can be significant!
        """


class AssignmentRiskGuard:
    """Monitor assignment risk for short options."""

    def check_assignment_probability(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry: float,
        dividend: Optional[Dividend],
    ) -> AssignmentProbability:
        """
        Assess probability of assignment.

        High risk triggers:
        - ITM calls before ex-dividend
        - Deep ITM puts
        - Day before expiration
        """
```

#### 6.2.2 OCC Clearing Margin (`impl_occ_margin.py`)

**CRITICAL**: Use OCC Clearing Margin methodology, NOT TIMS!
TIMS (Theoretical Intermarket Margin System) was replaced by OCC's STANS in 2006.

Reference: OCC "Margin Methodology" (current as of 2024)

```python
class OCCMarginCalculator:
    """
    OCC Clearing Margin calculation.

    Based on STANS (System for Theoretical Analysis and Numerical Simulations):
    - Monte Carlo VaR-based
    - 2-day 99% Expected Shortfall
    - Correlation offsets for hedged positions

    For retail (Reg T), uses simpler rules below.
    """

    def calculate_portfolio_margin(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
        correlations: Optional[np.ndarray] = None,
    ) -> PortfolioMarginResult:
        """
        OCC STANS methodology:

        1. Generate 10,000 scenarios (±15% spot, ±40% vol, etc.)
        2. Compute portfolio P&L for each scenario
        3. Margin = 2-day 99% Expected Shortfall
        4. Apply correlation credits for hedged positions
        """


class RegTMarginCalculator:
    """
    Reg T margin for retail accounts.

    Correct formulas (not simplified):
    """

    REG_T_RULES = {
        "long_option": {
            "initial": "100% of premium",
            "maintenance": "100% of premium",
        },
        "short_naked_call": {
            # max(20% underlying - OTM amount, 10% underlying) + premium
            "initial": "max(0.20 × S - max(K - S, 0), 0.10 × S) + premium",
            "maintenance": "max(0.15 × S - max(K - S, 0), 0.10 × S) + premium",
        },
        "short_naked_put": {
            # max(20% underlying - OTM amount, 10% strike) + premium
            "initial": "max(0.20 × S - max(S - K, 0), 0.10 × K) + premium",
            "maintenance": "max(0.15 × S - max(S - K, 0), 0.10 × K) + premium",
        },
        "covered_call": {
            "initial": "Underlying margin only (50% Reg T)",
            "maintenance": "25% of underlying",
        },
        "vertical_spread": {
            "initial": "max loss (K_diff for credit, net debit for debit)",
            "maintenance": "max loss",
        },
        "iron_condor": {
            "initial": "max(put_spread_margin, call_spread_margin)",
            "maintenance": "same",
        },
    }

    def calculate_reg_t_margin(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> RegTMarginResult:
        """
        Calculate Reg T margin for retail account.

        Rules vary by strategy:
        - Naked short: Highest margin
        - Covered: Only underlying margin
        - Spreads: Limited to max loss
        """
```

#### 6.2.3 Exercise/Assignment Engine (`impl_exercise_assignment.py`)

Reference: Broadie & Detemple (1996) "American Option Valuation: New Bounds, Approximations, and a Comparison of Existing Methods"

```python
class ExerciseAssignmentEngine:
    """
    Optimal exercise decision and assignment simulation.

    Key insight (Broadie & Detemple 1996):
    Early exercise sacrifices "gamma convexity value" — the option's
    ability to benefit from future volatility. This is often ignored
    but can be 1-5% of option value for ATM options.
    """

    def should_exercise_early(
        self,
        position: OptionsPosition,
        underlying_price: float,
        rate: float,
        iv: float,
        dividend: Optional[Dividend] = None,
    ) -> ExerciseDecision:
        """
        Optimal early exercise decision.

        Full decision rule:
        Exercise if: intrinsic_value > continuation_value

        where continuation_value includes:
        1. Time value (theta erosion)
        2. Gamma convexity value (Broadie & Detemple 1996)
        3. Future dividend optionality (for calls)

        For calls before dividend:
        Exercise if: dividend > time_value + carrying_cost + gamma_convexity

        For puts (rate benefit):
        Exercise if: K × r × dt > time_value + gamma_convexity
        """

    def compute_gamma_convexity_value(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        rate: float,
    ) -> float:
        """
        Gamma convexity value (Broadie & Detemple 1996).

        Approximation:
        GCV ≈ 0.5 × gamma × spot² × iv² × time_to_expiry

        This is the value of keeping the option alive to
        benefit from future spot movements.
        """

    def simulate_assignment(
        self,
        short_position: OptionsPosition,
        underlying_price: float,
        assignment_probability: float,
    ) -> AssignmentResult:
        """
        Simulate random assignment for short options.

        Uses binomial distribution for partial assignment.
        """
```

### 6.3 Test Matrix (Phase 6)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Greeks Guards | 30 | All limits, scaling, multi-level |
| Exercise Risk | 25 | American, dividends, gamma convexity |
| Assignment Risk | 20 | Short calls/puts, probability |
| Reg T Margin | 35 | All position types, spreads |
| OCC Portfolio Margin | 35 | STANS, correlations |
| Exercise/Assignment Engine | 30 | Decision logic, simulation |
| Pattern Compliance | 10 | futures_risk_guards pattern |
| **Total** | **185** | **100%** |

### 6.4 Deliverables
- [ ] `services/options_risk_guards.py` — All risk guards (following futures pattern)
- [ ] `impl_occ_margin.py` — OCC STANS + Reg T
- [ ] `impl_exercise_assignment.py` — Broadie-Detemple exercise engine
- [ ] `tests/test_options_risk.py` — 185 tests
- [ ] Documentation: `docs/options/risk_management.md`

---

## Phase 7: Complex Orders & Strategies (5 weeks)

### 7.1 Objectives
- Multi-leg order execution with proper Greeks netting
- Spread trading simulation with leg risk
- Delta hedging automation
- Vol trading strategies (variance swaps, vol arb)

### 7.2 Components

#### 7.2.1 Multi-Leg Orders (`impl_multi_leg.py`)

**Supported Spread Types**:

| Spread | Legs | Max Loss | Max Gain | Greeks Profile |
|--------|------|----------|----------|----------------|
| Vertical | 2 | Limited | Limited | Delta-directional |
| Calendar | 2 | Premium | Unlimited | Vega-long, Theta-short |
| Diagonal | 2 | Limited | Varies | Mixed |
| Straddle | 2 | Premium | Unlimited | Gamma-long, Theta-short |
| Strangle | 2 | Premium | Unlimited | Vega-long |
| Iron Condor | 4 | Limited | Limited | Theta-positive |
| Butterfly | 3-4 | Limited | Limited | Gamma-short at center |
| Box | 4 | None | Risk-free rate | Arb strategy |
| Ratio Spread | 2 | Unlimited | Limited | Directional + vol view |
| Jade Lizard | 3 | Limited (up) | Limited | Premium collection |

```python
@dataclass
class ComboOrder:
    legs: List[ComboLeg]
    order_type: ComboOrderType  # NET_DEBIT, NET_CREDIT, EVEN
    limit_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    execution_style: str = "atomic"  # "atomic" or "legging"

@dataclass
class ComboLeg:
    contract: OptionsContractSpec
    side: Side
    ratio: int = 1

@dataclass
class ComboGreeks:
    """Net Greeks for entire combo."""
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    leg_deltas: Dict[str, float]  # Per-leg attribution


class MultiLegExecutor:
    def execute_combo(
        self,
        order: ComboOrder,
        market_state: Dict[str, OptionsMarketState],
    ) -> ComboFill:
        """
        Execute multi-leg order.

        For atomic: All legs fill at once or none
        For legging: Sequential with leg risk monitoring
        """

    def compute_combo_greeks(
        self,
        legs: List[ComboLeg],
        greeks_by_contract: Dict[str, GreeksResult],
    ) -> ComboGreeks:
        """Net Greeks for spread with proper signing."""

    def simulate_legging_risk(
        self,
        order: ComboOrder,
        execution_style: str = "sequential",
        underlying_vol: float = 0.01,
    ) -> LeggingRiskAnalysis:
        """
        Simulate leg execution risk.

        Risk = exposure while partially filled.
        For 2-leg spread with 1 leg filled:
        - Delta exposure until second leg
        - Vega exposure until completed
        """
```

#### 7.2.2 Delta Hedging (`impl_delta_hedge.py`)

```python
class DeltaHedger:
    """
    Automated delta hedging for options portfolios.

    Strategies:
    - Continuous: Hedge every tick (expensive)
    - Discrete: Hedge at fixed intervals
    - Threshold: Hedge when delta exceeds threshold
    - Gamma-scaled: More frequent near gamma peaks
    """

    def __init__(
        self,
        hedge_threshold: float = 0.1,  # Rehedge when |delta| > threshold
        hedge_frequency: str = "threshold",  # "continuous", "daily", "threshold", "gamma_scaled"
        hedge_instrument: str = "stock",  # "stock", "futures", "mini_futures"
        gamma_threshold: float = 5.0,  # For gamma-scaled
    ):
        pass

    def compute_hedge_order(
        self,
        portfolio: OptionsPortfolio,
        target_delta: float = 0.0,
    ) -> Optional[Order]:
        """Compute hedge order to neutralize delta."""

    def simulate_hedge_pnl(
        self,
        option_position: OptionsPosition,
        underlying_path: np.ndarray,
        hedge_frequency: str,
        transaction_cost_bps: float = 5.0,
    ) -> HedgePnLResult:
        """
        Simulate hedging P&L vs BS theoretical.

        Returns:
        - Realized vol (from hedge P&L)
        - Hedge slippage (vs continuous)
        - Transaction cost impact
        - Gamma P&L attribution
        """

    def compute_realized_volatility(
        self,
        underlying_path: np.ndarray,
        hedge_times: np.ndarray,
    ) -> float:
        """
        Realized vol from discrete hedging:

        σ_realized² = (1/T) × Σ (ln(S_t/S_{t-1}))²

        Annualized appropriately.
        """
```

#### 7.2.3 Vol Trading Strategies (`strategies/vol_trading.py`)

```python
class VolatilityTrader:
    """
    Strategies based on IV vs RV mismatch.

    Strategies:
    1. Straddle/strangle: Direct vol bet
    2. Calendar: Term structure trades
    3. Variance swap: Pure vol exposure
    4. Dispersion: Index vs components
    """

    def identify_vol_opportunity(
        self,
        iv_surface: IVSurface,
        realized_vol: float,
        forecast_vol: float,
    ) -> Optional[VolTrade]:
        """
        Find IV mispricing opportunities.

        Signal = IV - RV_forecast
        Long vol if IV << forecast
        Short vol if IV >> forecast
        """

    def construct_variance_swap(
        self,
        chain: OptionsChain,
        target_vega: float,
        use_replication: bool = True,
    ) -> VarianceSwapPosition:
        """
        Construct synthetic variance swap from options.

        Uses Carr-Madan (1998) replication:
        - OTM puts below forward
        - OTM calls above forward
        - Weighted by 1/K²
        """

    def construct_vega_neutral_trade(
        self,
        chain: OptionsChain,
        vol_view: str,  # "long_gamma" or "short_gamma"
    ) -> ComboOrder:
        """
        Build vega-neutral gamma trade.

        For long gamma: Long ATM straddle, short OTM strangle
        For short gamma: Short ATM straddle, long wings
        """
```

### 7.3 Test Matrix (Phase 7)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| ComboOrder creation | 25 | All spread types |
| Multi-leg execution | 35 | Atomic, legging, Greeks netting |
| Legging risk | 20 | Simulation, exposure |
| Delta hedging | 30 | All frequencies, instruments |
| Hedge P&L simulation | 25 | Realized vol, costs |
| Variance swap | 20 | Replication, pricing |
| Vol strategies | 25 | Identification, construction |
| **Total** | **180** | **100%** |

### 7.4 Deliverables
- [ ] `impl_multi_leg.py` — Multi-leg execution with Greeks netting
- [ ] `impl_delta_hedge.py` — Delta hedging strategies
- [ ] `strategies/vol_trading.py` — Vol strategies (variance swaps, gamma trades)
- [ ] `tests/test_options_complex.py` — 180 tests
- [ ] Documentation: `docs/options/complex_orders.md`

---

## Phase 8: Training Integration (5 weeks)

### 8.1 Objectives
- Options trading environment wrapper (following `wrappers/futures_env.py` pattern)
- Options-specific features
- Reward shaping for options
- Greeks-aware policy

### 8.2 Components

#### 8.2.1 Options Environment (`wrappers/options_env.py`)

**ВАЖНО**: Follows pattern from `wrappers/futures_env.py`

```python
import gymnasium as gym
from wrappers.futures_env import FuturesEnvWrapper  # Pattern reference

class OptionsEnvWrapper(gym.Wrapper):
    """
    Options trading environment wrapper.

    Follows pattern from FuturesEnvWrapper in wrappers/futures_env.py.

    Key additions:
    - Greeks tracking in state
    - Margin requirement checks
    - Expiration handling
    - Exercise/assignment events
    """

    def __init__(
        self,
        env: gym.Env,
        options_config: OptionsEnvConfig,
    ):
        super().__init__(env)
        self.greeks_provider = GreeksProvider()
        self.margin_calculator = OCCMarginCalculator()
        self.iv_surface = None
        self._portfolio_greeks = GreeksResult()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step with options-specific logic:

        1. Decode action (strike, expiry, side, qty, or close)
        2. Check margin requirement (Reg T or PM)
        3. Check Greeks limits
        4. Execute option trade
        5. Update portfolio Greeks
        6. Handle any expiration/assignment events
        7. Compute reward with Greeks penalty
        """
        # Decode action
        trade = self._decode_action(action)

        # Pre-trade checks
        margin_ok, margin_msg = self._check_margin(trade)
        if not margin_ok:
            return self._blocked_step(margin_msg)

        greeks_ok, greeks_msg = self._check_greeks(trade)
        if not greeks_ok:
            return self._blocked_step(greeks_msg)

        # Execute
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Post-trade updates
        self._update_portfolio_greeks()
        self._handle_expirations()

        # Adjust reward
        options_reward = self._compute_options_reward(base_reward)

        # Add options info
        info["portfolio_greeks"] = self._portfolio_greeks
        info["margin_used"] = self._current_margin

        return obs, options_reward, terminated, truncated, info

    def _update_portfolio_greeks(self) -> None:
        """Recompute portfolio Greeks after trade."""

    def _handle_expirations(self) -> None:
        """Handle expiring options (exercise/expire worthless)."""

    def _compute_options_reward(
        self,
        base_reward: float,
    ) -> float:
        """
        Adjust reward based on Greeks exposure.

        Penalty for exceeding soft limits.
        Bonus for theta-positive strategies (if configured).
        """


@dataclass
class OptionsEnvConfig:
    max_delta: float = 100.0
    max_gamma: float = 50.0
    max_vega: float = 10000.0
    max_theta: float = -500.0
    gamma_penalty: float = 0.1
    theta_bonus: float = 0.05
    margin_type: str = "reg_t"  # or "portfolio"
```

#### 8.2.2 Options Features (`options_features.py`)

| Feature | Description | Normalization | Source |
|---------|-------------|---------------|--------|
| `iv_rank` | IV percentile (52w) | [0, 1] | Historical IV |
| `iv_percentile` | IV percentile (1y) | [0, 1] | Historical IV |
| `iv_skew_25d` | 25Δ put - 25Δ call | Z-score | IV surface |
| `term_slope` | Front/back IV ratio | Z-score | Term structure |
| `rv_iv_spread` | IV - RV (20d) | Z-score | Historical |
| `gamma_exposure` | Market GEX (normalized) | Z-score | OI × gamma |
| `put_call_ratio` | Put/call volume | Log transform | Volume |
| `put_call_oi_ratio` | Put/call OI | Log transform | Open interest |
| `max_pain` | Max pain strike (normalized) | [0, 1] | OI analysis |
| `vanna_exposure` | Market vanna | Z-score | Dealer positioning |
| `charm_exposure` | Market charm | Z-score | Time decay of delta |

```python
class OptionsFeatureExtractor:
    """
    Extract options-specific features for RL.

    Features designed for:
    - Vol regime detection
    - Dealer positioning signals
    - Term structure opportunities
    - Skew trading signals
    """

    def extract_features(
        self,
        chain: OptionsChain,
        underlying_history: pd.DataFrame,
        iv_surface: IVSurface,
    ) -> np.ndarray:
        """Extract normalized feature vector."""

    def compute_gex(
        self,
        chain: OptionsChain,
        dealer_gamma_assumption: str = "short_gamma",
    ) -> float:
        """
        Gamma Exposure (GEX).

        GEX = Σ (OI × gamma × contract_multiplier × spot)

        Sign assumption: Dealers typically short gamma (sold options).
        """

    def compute_vanna_exposure(
        self,
        chain: OptionsChain,
    ) -> float:
        """
        Vanna exposure = Σ (OI × vanna × multiplier)

        Predicts delta change as IV moves.
        """
```

#### 8.2.3 Options Reward (`impl_options_reward.py`)

```python
def compute_options_reward(
    pnl: float,
    greeks: GreeksResult,
    config: OptionsRewardConfig,
) -> float:
    """
    Options-aware reward shaping.

    Reward = PnL
        - γ_penalty × max(0, |gamma| - max_gamma)
        - speed_penalty × max(0, |speed| - max_speed)  # Third-order risk!
        + θ_bonus × theta (if theta > 0)  # Reward theta-positive
        - vega_penalty × |vega| (if large)  # Discourage naked vol)

    Penalty terms encourage:
    - Risk management (not just P&L maximization)
    - Defined-risk strategies
    - Theta-positive positions
    """


@dataclass
class OptionsRewardConfig:
    gamma_penalty: float = 0.1
    speed_penalty: float = 0.05   # Third-order penalty
    theta_bonus: float = 0.05
    vega_penalty: float = 0.02
    max_gamma: float = 50.0
    max_speed: float = 100.0
    max_vega: float = 10000.0
```

### 8.3 Test Matrix (Phase 8)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsEnvWrapper | 40 | Lifecycle, actions, Greeks tracking |
| Margin integration | 20 | Reg T, PM, blocking |
| Expiration handling | 15 | Exercise, expire worthless |
| Feature extraction | 30 | All 11 features |
| GEX/vanna calculation | 15 | Dealer positioning |
| Reward shaping | 25 | All penalties/bonuses |
| Training loop | 25 | Convergence, stability |
| Pattern compliance | 10 | futures_env.py pattern |
| **Total** | **180** | **100%** |

### 8.4 Deliverables
- [ ] `wrappers/options_env.py` — Training environment (follows futures_env pattern)
- [ ] `options_features.py` — Feature extraction (11 features)
- [ ] `impl_options_reward.py` — Greeks-aware reward shaping
- [ ] `configs/config_train_options.yaml` — Training config
- [ ] `tests/test_options_training.py` — 180 tests
- [ ] Documentation: `docs/options/training.md`

---

## Phase 9: Live Trading (5 weeks)

### 9.1 Objectives
- Options position sync (with Greeks)
- Real-time Greeks monitoring with alerts
- Exercise/expiration management
- Roll management

### 9.2 Components

#### 9.2.1 Options Live Runner (`services/options_live.py`)

```python
class OptionsLiveRunner:
    """
    Live trading coordinator for options.

    Features:
    - Greeks-aware position monitoring
    - Automatic expiration handling
    - Roll management (calendar rolls)
    - Margin monitoring integration
    """

    def __init__(
        self,
        config: OptionsLiveConfig,
        adapter: IBOptionsAdapter,
        greeks_monitor: GreeksMonitor,
        exercise_manager: ExerciseManager,
        margin_monitor: MarginMonitor,
    ):
        pass

    async def run(self) -> None:
        """Main live trading loop."""

    async def _process_expiration(self, date: date) -> None:
        """
        Handle expiring positions:
        1. Identify all positions expiring today
        2. Check exercise eligibility (ITM > threshold)
        3. Execute exercise or let expire
        4. Update positions
        """

    async def _manage_rolls(self) -> None:
        """
        Roll positions before expiry.

        Strategy:
        - Roll 5-7 DTE for monthlies
        - Roll 1-2 DTE for weeklies
        - Maintain similar Greeks exposure
        """

    async def _monitor_margin(self) -> None:
        """Check margin status and alert/reduce if needed."""
```

#### 9.2.2 Greeks Monitor (`services/greeks_monitor.py`)

```python
class GreeksMonitor:
    """
    Real-time portfolio Greeks monitoring.

    Features:
    - Per-second Greeks updates
    - Multi-level alerts
    - Greeks attribution by position
    - Historical Greeks tracking
    """

    def __init__(
        self,
        check_interval_sec: float = 1.0,
        alert_thresholds: GreeksThresholds,
        alert_callback: Optional[Callable] = None,
    ):
        pass

    async def start(self) -> None:
        """Start continuous Greeks monitoring."""

    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current aggregated portfolio Greeks."""

    def get_greeks_attribution(self) -> Dict[str, GreeksResult]:
        """Greeks breakdown by position."""

    def get_greeks_history(
        self,
        lookback_minutes: int = 60,
    ) -> pd.DataFrame:
        """Historical Greeks for analysis."""

    async def _check_alerts(self) -> List[GreeksAlert]:
        """
        Check for Greeks breaches.

        Alert levels:
        - WARNING: 80% of limit
        - DANGER: 100% of limit
        - CRITICAL: 120% of limit (immediate action)
        """
```

#### 9.2.3 Exercise Manager (`services/exercise_mgr.py`)

```python
class ExerciseManager:
    """
    Exercise/assignment management for live trading.

    Handles:
    - Optimal exercise decisions (Broadie-Detemple)
    - Assignment notifications
    - Expiration processing
    """

    def __init__(
        self,
        adapter: IBOptionsAdapter,
        exercise_policy: ExercisePolicy,
    ):
        pass

    async def check_exercise_opportunities(self) -> List[ExerciseRecommendation]:
        """
        Identify positions to exercise.

        Uses Broadie-Detemple (1996) optimal exercise criteria
        including gamma convexity value.
        """

    async def execute_exercise(
        self,
        position: OptionsPosition,
    ) -> ExerciseResult:
        """Submit exercise instruction to broker."""

    async def handle_assignment(
        self,
        assignment: Assignment,
    ) -> AssignmentResult:
        """
        Process incoming assignment.

        Actions:
        1. Update positions (receive/deliver underlying)
        2. Recompute Greeks
        3. Check margin impact
        4. Alert if necessary
        """

    async def manage_expiration(
        self,
        expiring_positions: List[OptionsPosition],
    ) -> ExpirationResult:
        """
        Handle expiring positions.

        For each position:
        - If ITM > threshold: Exercise
        - If ITM < threshold: Let expire
        - If OTM: Expires worthless
        """
```

### 9.3 Test Matrix (Phase 9)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsLiveRunner | 35 | Lifecycle, events |
| Greeks Monitor | 30 | Real-time, alerts, attribution |
| Exercise Manager | 35 | Exercise, assignment, expiration |
| Margin Monitor | 20 | Real-time margin, alerts |
| Roll Management | 25 | Calendar rolls, Greeks preservation |
| Integration | 20 | Full flow simulation |
| **Total** | **165** | **100%** |

### 9.4 Deliverables
- [ ] `services/options_live.py` — Live runner
- [ ] `services/greeks_monitor.py` — Greeks monitoring
- [ ] `services/exercise_mgr.py` — Exercise/assignment management
- [ ] `services/options_margin_monitor.py` — Margin monitoring
- [ ] `configs/config_live_options.yaml` — Live config
- [ ] `tests/test_options_live.py` — 165 tests
- [ ] Documentation: `docs/options/live_trading.md`

---

## Phase 10: Validation & Documentation (4 weeks)

### 10.1 Objectives
- Comprehensive validation testing
- Backward compatibility verification
- Performance benchmarks
- Complete documentation

### 10.2 Validation Tests

#### 10.2.1 Greeks Accuracy Validation

| Test | Target | Method |
|------|--------|--------|
| BS Greeks (all 12) | < 0.01% | vs analytical |
| Leisen-Reimer Greeks | < 0.1% | vs BS for European |
| MC Greeks | < 1% | vs BS (pathwise) |
| Numerical Greeks | < 0.5% | vs analytical |
| Third-order Greeks | < 1% | vs finite difference |

#### 10.2.2 IV Surface Validation

| Metric | Target |
|--------|--------|
| SSVI interpolation error | < 0.3 vol points |
| Butterfly arbitrage-free | 100% |
| Calendar arbitrage-free | 100% |
| SSVI fit R² | > 0.995 |
| Heston calibration error | < 1% |

#### 10.2.3 Execution Validation

| Metric | Target |
|--------|--------|
| Fill rate (L2) | > 95% |
| Fill rate (L3) | > 90% |
| Slippage error | < 3 bps |
| Fee accuracy | 100% |
| MM behavior realism | Qualitative |

### 10.3 Backward Compatibility

```bash
# Verify no regressions in existing functionality
pytest tests/test_unit_*.py -v               # Core unit tests
pytest tests/test_execution_*.py -v          # Execution
pytest tests/test_futures_*.py -v            # Futures
pytest tests/test_forex_*.py -v              # Forex
pytest tests/test_stock_*.py -v              # Stocks
pytest tests/test_lob*.py -v                 # LOB modules

# All new options tests
pytest tests/test_options_*.py -v
```

### 10.4 Benchmarks

| Operation | Target | Method |
|-----------|--------|--------|
| BS pricing | < 1 μs | Vectorized NumPy |
| Greeks (all 12) | < 10 μs | Analytical |
| Leisen-Reimer (501 steps) | < 500 μs | NumPy |
| IV solve (European) | < 50 μs | Jäckel |
| IV solve (American) | < 5 ms | Tree bisection |
| SSVI interpolation | < 10 μs | Cubic spline |
| L3 matching | < 100 μs | Options engine |
| Heston pricing | < 1 ms | Characteristic function |

### 10.5 Documentation Suite

| Document | Description |
|----------|-------------|
| `docs/options/overview.md` | Architecture overview |
| `docs/options/core_models.md` | Greeks (all 12), pricing, IV |
| `docs/options/volatility_surface.md` | SSVI, Heston, Dupire |
| `docs/options/exchange_adapters.md` | IB, Theta Data, Deribit |
| `docs/options/execution_l2.md` | L2 slippage, fees, PFOF |
| `docs/options/l3_lob.md` | L3 simulation, MM behavior |
| `docs/options/risk_management.md` | Guards, OCC margin |
| `docs/options/complex_orders.md` | Multi-leg, hedging, vol trading |
| `docs/options/training.md` | RL environment, features |
| `docs/options/live_trading.md` | Live runner, exercise mgmt |
| `docs/options/api_reference.md` | Full API docs |
| `docs/options/migration_guide.md` | From L2 to L3 |

### 10.6 Test Matrix (Phase 10)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Greeks validation | 60 | All 12 Greeks, all methods |
| IV surface validation | 50 | SSVI, arbitrage-free |
| Execution validation | 40 | Fill rates, slippage |
| Backward compatibility | 100 | All asset classes |
| Performance benchmarks | 30 | Timing targets |
| **Total** | **280** | **100%** |

### 10.7 Deliverables
- [ ] `tests/test_options_validation.py` — 180 validation tests
- [ ] `tests/test_options_backward_compat.py` — 100 compat tests
- [ ] `benchmarks/bench_options.py` — Performance suite
- [ ] `OPTIONS_INTEGRATION_REPORT.md` — Completion report
- [ ] `docs/options/*.md` — 12 documentation files

---

## Summary

### Test Count by Phase

| Phase | Tests | Cumulative |
|-------|-------|------------|
| 1: Core Models | 160 | 160 |
| 2: Exchange Adapters | 140 | 300 |
| 3: IV Surface | 155 | 455 |
| 4: L2 Execution | 150 | 605 |
| 5: L3 LOB | 180 | 785 |
| 6: Risk Management | 185 | 970 |
| 7: Complex Orders | 180 | 1,150 |
| 8: Training | 180 | 1,330 |
| 9: Live Trading | 165 | 1,495 |
| 10: Validation | 280 | **1,775** |

**Buffer for edge cases**: +225 tests
**Total**: **~2,000 tests**

### Timeline (Revised)

| Phase | Duration | Dependencies | Notes |
|-------|----------|--------------|-------|
| 1 | 4 weeks | None | Extended for 12 Greeks, jump diffusion |
| 2 | 5 weeks | Phase 1 | Extended for Theta Data integration |
| 3 | 5 weeks | Phase 1 | Extended for SSVI, Heston |
| 4 | 4 weeks | Phases 1-3 | Extended for PFOF, moneyness factors |
| 5 | 6 weeks | Phases 1-4 | Extended for Cho & Engle MM model |
| 6 | 5 weeks | Phases 1-5 | Extended for OCC margin, gamma convexity |
| 7 | 5 weeks | Phases 1-6 | Extended for variance swaps |
| 8 | 5 weeks | Phases 1-7 | Extended for 11 features |
| 9 | 5 weeks | Phases 1-8 | Extended for roll management |
| 10 | 4 weeks | All | Extended for 12 Greeks validation |

**Total**: ~51 weeks (~12 months)

**Buffer**: 10% contingency = 5 weeks

**Final Estimate**: ~56 weeks (~13 months)

### Key References

**Academic**:
- Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
- Merton (1973): "Theory of Rational Option Pricing" (dividends)
- Merton (1976): "Option pricing when underlying stock returns are discontinuous" (jumps)
- Leisen & Reimer (1996): "Binomial models for option valuation - examining and improving convergence"
- Brenner & Subrahmanyam (1994): "A simple approach to option valuation and hedging" (IV seed)
- Jäckel (2015): "Let's Be Rational" (robust IV solver)
- Gatheral & Jacquier (2014): "Arbitrage-free SVI volatility surfaces" (SSVI)
- Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
- Dupire (1994): "Pricing with a Smile" (local vol)
- Broadie & Detemple (1996): "American Option Valuation" (gamma convexity)
- Carr & Madan (1998): "Towards a theory of volatility trading" (variance swaps)
- Avellaneda & Lipkin (2003): "A market-induced mechanism for stock pinning"
- Cho & Engle (2022): "Market Maker Quotes in Options Markets" (regime-dependent MM)
- Muravyev & Pearson (2020): "Options Trading Costs Are Lower Than You Think" (PFOF)
- Hull (2017): "Options, Futures, and Other Derivatives" (textbook reference)

**Industry**:
- OCC: Options Clearing Corporation — STANS margin methodology
- CBOE: Index options methodology, VIX calculation
- CME: Futures options specifications
- Deribit: Crypto options documentation, DVOL
- Theta Data: API documentation

### Quick Reference Table (for CLAUDE.md)

| Task | Location | Test Command |
|------|----------|--------------|
| Options pricing (BS, LR, JD) | `impl_pricing.py` | `pytest tests/test_options_core.py` |
| Greeks calculation (12) | `impl_greeks.py` | `pytest tests/test_options_core.py::TestGreeks` |
| IV solver (European) | `impl_iv_calculation.py` | `pytest tests/test_options_core.py::TestIV` |
| IV solver (American) | `impl_iv_calculation.py` | `pytest tests/test_options_core.py::TestAmericanIV` |
| SSVI surface | `impl_ssvi.py` | `pytest tests/test_iv_surface.py::TestSSVI` |
| Heston calibration | `impl_heston.py` | `pytest tests/test_iv_surface.py::TestHeston` |
| IB options adapter | `adapters/ib/options.py` | `pytest tests/test_options_adapters.py::TestIB` |
| Theta Data adapter | `adapters/theta_data/options.py` | `pytest tests/test_options_adapters.py::TestThetaData` |
| Deribit adapter | `adapters/deribit/options.py` | `pytest tests/test_options_adapters.py::TestDeribit` |
| Options L2 execution | `execution_providers_options.py` | `pytest tests/test_options_execution_l2.py` |
| Options L3 LOB | `lob/options_matching.py` | `pytest tests/test_options_l3_lob.py` |
| Options MM simulator | `lob/options_mm.py` | `pytest tests/test_options_l3_lob.py::TestMM` |
| Pin risk | `lob/pin_risk.py` | `pytest tests/test_options_l3_lob.py::TestPinRisk` |
| Options risk guards | `services/options_risk_guards.py` | `pytest tests/test_options_risk.py` |
| OCC margin | `impl_occ_margin.py` | `pytest tests/test_options_risk.py::TestMargin` |
| Exercise engine | `impl_exercise_assignment.py` | `pytest tests/test_options_risk.py::TestExercise` |
| Multi-leg orders | `impl_multi_leg.py` | `pytest tests/test_options_complex.py` |
| Delta hedging | `impl_delta_hedge.py` | `pytest tests/test_options_complex.py::TestHedge` |
| Vol strategies | `strategies/vol_trading.py` | `pytest tests/test_options_complex.py::TestVol` |
| Options env wrapper | `wrappers/options_env.py` | `pytest tests/test_options_training.py` |
| Options features | `options_features.py` | `pytest tests/test_options_training.py::TestFeatures` |
| Options live runner | `services/options_live.py` | `pytest tests/test_options_live.py` |
| Greeks monitor | `services/greeks_monitor.py` | `pytest tests/test_options_live.py::TestMonitor` |
| Exercise manager | `services/exercise_mgr.py` | `pytest tests/test_options_live.py::TestExercise` |

---

**Document Version**: 2.0
**Created**: 2025-12-03
**Last Updated**: 2025-12-03
**Author**: Claude Code

### Changelog v2.0 (2025-12-03)

**Phase 1 Fixes**:
- Added American IV solver using binomial tree inversion
- Increased binomial steps: 100 → 501 (Leisen-Reimer)
- Replaced CRR with Leisen-Reimer (O(1/n²) convergence)
- Added hybrid Newton-Raphson + Brent's IV solver for deep OTM
- Added 4 missing Greeks: Speed, Color, Zomma, Ultima
- Added Merton dividend model
- Added Merton jump-diffusion for earnings/M&A
- Added variance swap replication (Carr-Madan 1998)

**Phase 2 Fixes**:
- Removed OPRA direct feed ($2,500/month not practical)
- Removed CBOE direct API (no retail access)
- Extended existing IB adapter instead of creating new
- Added Theta Data as cost-effective alternative ($100/mo)
- Added Polygon for historical options data

**Phase 3 Fixes**:
- Replaced raw SVI with SSVI (Gatheral & Jacquier 2014)
- Added full arbitrage-free conditions (butterfly, calendar, Lee)
- Clarified SABR is for rates/FX only, use Heston for equity
- Added Tikhonov regularization for Dupire local vol
- Added complete calendar arbitrage check (∂w/∂T > 0)

**Phase 4 Fixes**:
- Added moneyness-dependent slippage (ATM vs OTM)
- Added DTE-dependent slippage (near-expiry wider)
- Added PFOF retail flow model (Muravyev & Pearson 2020)
- Integrated with existing AssetClass.OPTIONS

**Phase 5 Fixes**:
- Redesigned options LOB (fundamentally different from equity)
- Added Cho & Engle (2022) regime-dependent MM model
- Added 5 MM quoting regimes (NORMAL, HIGH_VOL, EARNINGS, EXPIRATION, LOW_LIQ)
- Added Avellaneda-Lipkin pin risk model
- Added cross-strike arbitrage detection (butterfly, box)

**Phase 6 Fixes**:
- Fixed Reg T formulas (added correct naked put/call calculations)
- Replaced TIMS with OCC STANS margin methodology
- Added Broadie & Detemple (1996) gamma convexity value for exercise
- Added pattern compliance with futures_risk_guards.py

**Phase 7-10 Fixes**:
- Added variance swap construction (Carr-Madan)
- Added 11 options features (including GEX, vanna exposure)
- Extended env wrapper pattern from futures_env.py
- Extended all component durations by 1-2 weeks

**Architectural Fixes**:
- Integration with existing AssetClass.OPTIONS (execution_providers.py:54)
- Extension of existing lob/ module (not new module)
- Extension of existing IB adapter (adapters/ib/)
- Factory functions following existing pattern
- Risk guards following futures_risk_guards.py pattern
- Env wrapper following futures_env.py pattern

**Timeline Revision**:
- Original: 38 weeks
- Revised: 51 weeks (+13 weeks for added complexity)
- With 10% buffer: 56 weeks (~13 months)

**Test Count Revision**:
- Original: 1,265 tests
- Revised: 1,775 tests (+510 for added features)
- With buffer: ~2,000 tests

**Added References**:
- Leisen & Reimer (1996)
- Jäckel (2015)
- Gatheral & Jacquier (2014)
- Broadie & Detemple (1996)
- Carr & Madan (1998)
- Avellaneda & Lipkin (2003)
- Cho & Engle (2022)
- Muravyev & Pearson (2020)
