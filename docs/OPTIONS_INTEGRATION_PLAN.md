# OPTIONS_INTEGRATION_PLAN.md

## AI-Powered Quantitative Research Platform — Options Integration

**Version**: 1.0
**Status**: PLANNED
**Target Completion**: Q2 2026
**Estimated Tests**: 1,500+
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

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIONS INTEGRATION ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 1: Core Models         │  Phase 2: Exchange Adapters         │
│  ├─ core_options.py           │  ├─ adapters/ib/options.py          │
│  ├─ impl_greeks.py            │  ├─ adapters/cboe/market_data.py    │
│  ├─ impl_pricing.py           │  ├─ adapters/deribit/options.py     │
│  └─ impl_iv_calculation.py    │  └─ adapters/opra/feed.py           │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: IV Surface          │  Phase 4: L2 Execution              │
│  ├─ impl_iv_surface.py        │  ├─ execution_providers_options.py  │
│  ├─ impl_vol_models.py        │  ├─ impl_options_slippage.py        │
│  └─ service_iv_calibration.py │  └─ impl_options_fees.py            │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 5: L3 LOB              │  Phase 6: Risk Management           │
│  ├─ lob/options_book.py       │  ├─ services/options_risk_guards.py │
│  ├─ lob/mm_simulator.py       │  ├─ impl_options_margin.py          │
│  └─ lob/quote_dynamics.py     │  └─ impl_exercise_assignment.py     │
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

## Phase 1: Core Models & Data Structures (3 weeks)

### 1.1 Objectives
- Define options contract specifications
- Implement analytical Greeks
- Build Black-Scholes and Binomial pricing
- Create IV solver

### 1.2 Components

#### 1.2.1 Core Options Models (`core_options.py`)

```python
@dataclass
class OptionsContractSpec:
    symbol: str                    # "AAPL240315C00175000"
    underlying: str                # "AAPL"
    option_type: OptionType        # CALL, PUT
    strike: Decimal
    expiration: date
    exercise_style: ExerciseStyle  # AMERICAN, EUROPEAN
    settlement: SettlementType     # PHYSICAL, CASH
    multiplier: int                # 100 for US equities
    tick_size: Decimal
    exchange: str                  # "CBOE", "PHLX"

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

| Greek | Formula | Use Case |
|-------|---------|----------|
| **Delta** | ∂V/∂S | Directional exposure |
| **Gamma** | ∂²V/∂S² | Convexity risk |
| **Theta** | ∂V/∂t | Time decay |
| **Vega** | ∂V/∂σ | Vol exposure |
| **Rho** | ∂V/∂r | Rate sensitivity |
| **Vanna** | ∂²V/∂S∂σ | Skew risk |
| **Volga** | ∂²V/∂σ² | Vol-of-vol |
| **Charm** | ∂Δ/∂t | Delta decay |

```python
@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float      # Per day
    vega: float       # Per 1% vol
    rho: float        # Per 1% rate
    vanna: float
    volga: float
    charm: float
    timestamp_ns: int
```

#### 1.2.3 Pricing Models (`impl_pricing.py`)

**Black-Scholes (European)**:
```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Binomial Tree (American)**:
- CRR (Cox-Ross-Rubinstein) with early exercise check
- Min 100 steps for production accuracy
- Richardson extrapolation for convergence

**Monte Carlo (Exotic)**:
- Antithetic variates
- Control variates (BS as control)
- Quasi-random sequences (Sobol)

#### 1.2.4 IV Solver (`impl_iv_calculation.py`)

**Newton-Raphson with Brenner-Subrahmanyam seed**:
```python
def compute_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> float:
    # Brenner-Subrahmanyam initial guess
    sigma_0 = sqrt(2 * pi / time_to_expiry) * option_price / spot

    for _ in range(max_iterations):
        price = black_scholes_price(spot, strike, time_to_expiry, rate, sigma, option_type)
        vega = black_scholes_vega(spot, strike, time_to_expiry, rate, sigma)

        diff = price - option_price
        if abs(diff) < tolerance:
            return sigma

        sigma -= diff / (vega + 1e-10)

    raise IVConvergenceError(f"IV failed to converge")
```

### 1.3 Test Matrix (Phase 1)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsContractSpec | 15 | Creation, validation, OCC symbology |
| Greeks (BS analytical) | 25 | All 8 Greeks, edge cases |
| Black-Scholes pricing | 20 | Calls/puts, ATM/OTM/ITM |
| Binomial (American) | 20 | Early exercise, dividends |
| IV solver | 15 | Convergence, edge cases |
| **Total** | **95** | **100%** |

### 1.4 Deliverables
- [ ] `core_options.py` — Contract specs, enums
- [ ] `impl_greeks.py` — All Greeks with numerical validation
- [ ] `impl_pricing.py` — BS, Binomial, MC
- [ ] `impl_iv_calculation.py` — Newton-Raphson IV solver
- [ ] `tests/test_options_core.py` — 95 tests
- [ ] Documentation: `docs/options/core_models.md`

### 1.5 Regression Check
```bash
pytest tests/ -x --ignore=tests/test_options_*.py  # All existing tests pass
pytest tests/test_options_core.py -v               # All new tests pass
```

---

## Phase 2: Exchange Adapters (4 weeks)

### 2.1 Objectives
- IB TWS adapter for US options
- CBOE index options data
- Deribit crypto options
- OPRA market data feed

### 2.2 Supported Exchanges

| Exchange | Asset Class | Data | Execution | Protocol |
|----------|-------------|------|-----------|----------|
| **IB TWS** | US Equity Options | ✅ | ✅ | TWS API |
| **CBOE** | Index Options (SPX, VIX) | ✅ | Via broker | REST/WS |
| **Deribit** | Crypto Options | ✅ | ✅ | REST/WS |
| **OPRA** | Consolidated US | ✅ | N/A | SIP feed |

### 2.3 Components

#### 2.3.1 IB Options Adapter (`adapters/ib/options.py`)

```python
class IBOptionsAdapter:
    def get_option_chain(
        self, underlying: str, expiration: Optional[date] = None
    ) -> List[OptionsContractSpec]:
        """Fetch full option chain for underlying."""

    def get_option_quote(
        self, contract: OptionsContractSpec
    ) -> OptionsQuote:
        """Real-time bid/ask/last with Greeks."""

    def submit_option_order(
        self, order: OptionsOrder
    ) -> OrderResult:
        """Submit single-leg option order."""

    def submit_combo_order(
        self, legs: List[ComboLeg], order_type: str
    ) -> OrderResult:
        """Submit multi-leg spread order."""

@dataclass
class OptionsQuote:
    bid: Decimal
    ask: Decimal
    last: Decimal
    bid_size: int
    ask_size: int
    underlying_price: Decimal
    iv: float
    greeks: GreeksResult
    timestamp_ns: int
```

**IB Rate Limits for Options**:
| Limit Type | Value | Safety Margin |
|------------|-------|---------------|
| Option chains | 10/min | 8/min |
| Quote requests | 100/sec | 80/sec |
| Order submissions | 50/sec | 40/sec |

#### 2.3.2 CBOE Index Options (`adapters/cboe/market_data.py`)

```python
class CBOEMarketDataAdapter:
    def get_spx_chain(self, expirations: int = 4) -> List[OptionsContractSpec]:
        """Get SPX option chain (AM/PM settled)."""

    def get_vix_chain(self) -> List[OptionsContractSpec]:
        """Get VIX options (cash-settled, European)."""

    def get_term_structure(self, underlying: str) -> TermStructure:
        """IV term structure from listed expirations."""
```

**SPX Settlement Types**:
- **AM-settled (standard)**: Settlement at open on expiration
- **PM-settled (SPXW)**: Settlement at close on expiration

#### 2.3.3 Deribit Crypto Options (`adapters/deribit/options.py`)

```python
class DeribitOptionsAdapter:
    def get_btc_options(self) -> List[OptionsContractSpec]:
        """BTC options (European, cash-settled in BTC)."""

    def get_eth_options(self) -> List[OptionsContractSpec]:
        """ETH options (European, cash-settled in ETH)."""

    def get_orderbook(
        self, instrument: str, depth: int = 10
    ) -> OptionsOrderBook:
        """L2 order book for options."""

    async def stream_quotes_async(
        self, instruments: List[str]
    ) -> AsyncIterator[OptionsQuote]:
        """Real-time quote stream."""
```

**Deribit Specifics**:
- Settlement in underlying crypto (not USD)
- Inverse contracts: 1 BTC = 1 contract
- Expiration: 08:00 UTC on expiry date
- IV index: DVOL (Deribit Volatility Index)

### 2.4 Test Matrix (Phase 2)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| IB Options Adapter | 35 | Chain fetch, quotes, orders |
| IB Combo Orders | 15 | Spreads, straddles |
| CBOE Adapter | 20 | SPX, VIX, term structure |
| Deribit Adapter | 25 | BTC/ETH, orderbook, streaming |
| Rate Limiting | 10 | Throttling, backoff |
| **Total** | **105** | **100%** |

### 2.5 Deliverables
- [ ] `adapters/ib/options.py` — IB options adapter
- [ ] `adapters/cboe/market_data.py` — CBOE index options
- [ ] `adapters/deribit/options.py` — Crypto options
- [ ] `adapters/opra/feed.py` — OPRA consolidated feed
- [ ] `tests/test_options_adapters.py` — 105 tests
- [ ] Documentation: `docs/options/exchange_adapters.md`

---

## Phase 3: IV Surface & Volatility Models (4 weeks)

### 3.1 Objectives
- Construct IV surface from market quotes
- Implement smile/skew interpolation
- Add stochastic volatility models (Heston)
- Build calibration service

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

    def fit_from_quotes(
        self, quotes: List[OptionsQuote], method: str = "svi"
    ) -> None:
        """Fit surface from market quotes."""

    def get_iv(self, strike: float, expiry: float) -> float:
        """Interpolate IV at given strike/expiry."""

    def get_local_vol(self, strike: float, expiry: float) -> float:
        """Dupire local volatility."""

    def get_forward_iv(
        self, strike: float, t1: float, t2: float
    ) -> float:
        """Forward-starting implied vol."""
```

#### 3.2.2 SVI Parameterization (Gatheral)

**Raw SVI**:
```
w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

где:
- w = σ²·T (total variance)
- k = ln(K/F) (log-moneyness)
- a, b, ρ, m, σ — 5 параметров
```

**Constraints for arbitrage-free surface**:
1. `b ≥ 0` (butterfly arbitrage)
2. `|ρ| < 1` (correlation bound)
3. `a + b·σ·√(1-ρ²) ≥ 0` (variance positive)

#### 3.2.3 Stochastic Volatility Models (`impl_vol_models.py`)

**Heston Model**:
```
dS = μS dt + √V S dW₁
dV = κ(θ - V) dt + ξ√V dW₂
dW₁·dW₂ = ρ dt

Parameters:
- κ: Mean reversion speed
- θ: Long-term variance
- ξ: Vol-of-vol
- ρ: Correlation (typically -0.7 for equities)
- V₀: Initial variance
```

**SABR Model** (для rates/FX):
```
dF = σF^β dW₁
dσ = α·σ dW₂
dW₁·dW₂ = ρ dt
```

#### 3.2.4 Calibration Service (`service_iv_calibration.py`)

```python
class IVCalibrationService:
    def calibrate_svi(
        self,
        quotes: List[OptionsQuote],
        expiry: float,
        method: str = "quasi_explicit",
    ) -> SVIParams:
        """Calibrate SVI slice for single expiry."""

    def calibrate_heston(
        self,
        quotes: List[OptionsQuote],
        method: str = "differential_evolution",
    ) -> HestonParams:
        """Calibrate Heston to full surface."""

    def check_arbitrage(self, surface: IVSurface) -> ArbitrageReport:
        """Check for butterfly/calendar arbitrage."""
```

### 3.3 Test Matrix (Phase 3)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| IVSurface construction | 20 | From quotes, interpolation |
| SVI fitting | 25 | All 5 params, constraints |
| Heston calibration | 20 | Parameter recovery |
| Local vol (Dupire) | 15 | Numerical stability |
| Arbitrage detection | 15 | Butterfly, calendar |
| Forward vol | 10 | Term structure |
| **Total** | **105** | **100%** |

### 3.4 Deliverables
- [ ] `impl_iv_surface.py` — IV surface with interpolation
- [ ] `impl_vol_models.py` — Heston, SABR
- [ ] `service_iv_calibration.py` — Calibration service
- [ ] `tests/test_iv_surface.py` — 105 tests
- [ ] Documentation: `docs/options/volatility_surface.md`

---

## Phase 4: L2 Execution Provider (3 weeks)

### 4.1 Objectives
- Options-specific slippage model
- Greeks-aware execution
- Bid-ask spread modeling
- Options fee structures

### 4.2 Components

#### 4.2.1 Options Slippage Model (`impl_options_slippage.py`)

**Slippage Factors**:

| Factor | Formula | Impact |
|--------|---------|--------|
| **Bid-Ask Spread** | (ask-bid)/mid | Base cost |
| **Gamma Exposure** | Γ × S² × σ × √(Q/ADV) | Convexity cost |
| **Vega Exposure** | ν × ΔIV_expected | Vol uncertainty |
| **Time Decay** | θ × execution_time | Intraday theta |
| **Delta Hedge** | Δ × underlying_slippage | Hedging cost |

**Total Slippage**:
```
slippage = spread/2
    + γ_impact × gamma × spot² × vol × √participation
    + ν_impact × vega × expected_iv_move
    + theta_adjustment
    + delta × underlying_slippage
```

#### 4.2.2 Options Fee Provider (`impl_options_fees.py`)

| Exchange | Per Contract | Regulatory | Notes |
|----------|--------------|------------|-------|
| CBOE | $0.44 | OCC $0.055 | Index options higher |
| NYSE Arca | $0.47 | OCC $0.055 | |
| Deribit | 0.03% | None | Capped at 12.5% |
| IB | $0.65 | Varies | Tiered pricing |

#### 4.2.3 Options Execution Provider (`execution_providers_options.py`)

```python
class OptionsL2ExecutionProvider:
    def __init__(
        self,
        slippage_provider: OptionsSlippageProvider,
        fee_provider: OptionsFeeProvider,
        greeks_provider: GreeksProvider,
    ):
        pass

    def execute(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
        bar: BarData,
        underlying_bar: BarData,
    ) -> OptionsFill:
        """Execute single-leg option order."""

    def estimate_cost(
        self,
        order: OptionsOrder,
        market: OptionsMarketState,
    ) -> CostEstimate:
        """Pre-trade cost estimation with Greeks impact."""
```

### 4.3 Test Matrix (Phase 4)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Options Slippage | 25 | All factors, edge cases |
| Fee Calculation | 20 | All exchanges, tiers |
| L2 Execution | 30 | Market/limit, fills |
| Cost Estimation | 15 | Greeks impact |
| Integration | 10 | Full flow |
| **Total** | **100** | **100%** |

### 4.4 Deliverables
- [ ] `impl_options_slippage.py` — Greeks-aware slippage
- [ ] `impl_options_fees.py` — Fee structures
- [ ] `execution_providers_options.py` — L2 provider
- [ ] `tests/test_options_execution_l2.py` — 100 tests
- [ ] Documentation: `docs/options/execution_l2.md`

---

## Phase 5: L3 LOB Simulation (5 weeks)

### 5.1 Objectives
- Options-specific order book
- Market maker behavior simulation
- Quote dynamics modeling
- Pin risk simulation

### 5.2 Components

#### 5.2.1 Options Order Book (`lob/options_book.py`)

**Key Differences from Equity LOB**:
1. **Wider spreads** — 5-50 ticks typical
2. **Lower liquidity** — 10-100 contracts per level
3. **Quote updates** — Continuous rehedging by MMs
4. **Expiration dynamics** — Liquidity collapse near expiry

```python
class OptionsOrderBook:
    def __init__(
        self,
        contract: OptionsContractSpec,
        underlying_price: float,
        iv: float,
    ):
        self._bids: SortedDict[Decimal, PriceLevel] = SortedDict()
        self._asks: SortedDict[Decimal, PriceLevel] = SortedDict()
        self._greeks: GreeksResult = None

    def update_underlying(self, new_price: float, new_iv: float) -> None:
        """Shift quotes based on underlying move."""

    def simulate_mm_requote(
        self,
        underlying_move: float,
        iv_move: float,
        time_elapsed: float,
    ) -> None:
        """Simulate market maker requoting."""
```

#### 5.2.2 Market Maker Simulator (`lob/mm_simulator.py`)

**MM Behavior Model**:
```
quoted_spread = base_spread × (1 + γ_factor × |gamma| + ν_factor × |vega/theta|)
quote_size = max_position / (1 + concentration_penalty)
requote_threshold = delta_change > 0.01 OR iv_change > 0.5%
```

```python
class OptionsMMSimulator:
    def __init__(
        self,
        inventory_limit: int = 1000,
        base_spread_pct: float = 0.02,
        gamma_factor: float = 0.5,
        vega_factor: float = 0.3,
    ):
        pass

    def generate_quotes(
        self,
        contract: OptionsContractSpec,
        underlying: float,
        iv: float,
        current_inventory: int,
    ) -> Tuple[Quote, Quote]:
        """Generate bid/ask quotes."""

    def should_requote(
        self,
        underlying_move: float,
        iv_move: float,
        time_elapsed_sec: float,
    ) -> bool:
        """Check if requote needed."""
```

#### 5.2.3 Quote Dynamics (`lob/quote_dynamics.py`)

**Factors affecting options quotes**:

| Factor | Effect on Spread | Effect on Size |
|--------|------------------|----------------|
| Gamma (high) | Widen 20-50% | Reduce 50% |
| Near expiry (<7d) | Widen 100%+ | Reduce 70% |
| Vol spike | Widen 30-100% | Reduce 40% |
| Earnings | Widen 50-200% | Reduce 60% |
| Low underlying vol | Tighten | Increase |

#### 5.2.4 Pin Risk Simulation (`lob/pin_risk.py`)

```python
class PinRiskSimulator:
    def compute_pin_probability(
        self,
        strike: float,
        spot: float,
        time_to_expiry: float,
        iv: float,
        open_interest: int,
    ) -> float:
        """Probability of pinning to strike at expiry."""

    def simulate_expiry_dynamics(
        self,
        strikes: List[float],
        spot: float,
        oi_by_strike: Dict[float, int],
        delta_hedging_volume: float,
    ) -> SpotPath:
        """Simulate spot dynamics near high-OI strikes."""
```

### 5.3 Test Matrix (Phase 5)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsOrderBook | 30 | Add/remove, matching |
| MM Simulator | 35 | Quoting, inventory |
| Quote Dynamics | 25 | All factors |
| Pin Risk | 20 | Probability, dynamics |
| Integration | 20 | Full LOB simulation |
| **Total** | **130** | **100%** |

### 5.4 Deliverables
- [ ] `lob/options_book.py` — Options order book
- [ ] `lob/mm_simulator.py` — Market maker simulation
- [ ] `lob/quote_dynamics.py` — Quote behavior model
- [ ] `lob/pin_risk.py` — Pin risk simulation
- [ ] `tests/test_options_l3_lob.py` — 130 tests
- [ ] Documentation: `docs/options/l3_lob.md`

---

## Phase 6: Risk Management (4 weeks)

### 6.1 Objectives
- Options-specific risk guards
- SPAN/Portfolio Margin for options
- Exercise/assignment simulation
- Greeks limit monitoring

### 6.2 Components

#### 6.2.1 Options Risk Guards (`services/options_risk_guards.py`)

```python
class OptionsGreeksGuard:
    """Monitor portfolio Greeks limits."""

    def __init__(
        self,
        max_delta: float = 100.0,      # Net delta in underlying terms
        max_gamma: float = 50.0,       # Per 1% move
        max_vega: float = 10000.0,     # Per 1 vol point
        max_theta: float = -500.0,     # Daily theta limit
    ):
        pass

    def check_position(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> GreeksRiskResult:
        """Check if Greeks within limits."""

class ExerciseRiskGuard:
    """Monitor early exercise risk for American options."""

    def check_early_exercise_risk(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry: float,
        dividend_schedule: List[Dividend],
    ) -> ExerciseRiskResult:
        """Identify positions at risk of early exercise."""

class AssignmentRiskGuard:
    """Monitor assignment risk for short options."""

    def check_assignment_risk(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry: float,
    ) -> AssignmentRiskResult:
        """Assess probability of assignment."""
```

#### 6.2.2 Options Margin (`impl_options_margin.py`)

**Reg T Margin (Retail)**:
| Position | Initial | Maintenance |
|----------|---------|-------------|
| Long option | 100% premium | 100% |
| Short naked call | 20% underlying + premium - OTM | 15% + premium |
| Short naked put | 20% underlying + premium - OTM | 15% + premium |
| Covered call | Underlying margin only | Same |
| Spread | Max loss | Max loss |

**Portfolio Margin (Professional)**:
- TIMS (Theoretical Intermarket Margin System)
- Based on 10 price scenarios (±8% spot, ±20% vol)
- Significant reduction for hedged positions

```python
class OptionsMarginCalculator:
    def calculate_reg_t_margin(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> MarginRequirement:
        """Calculate Reg T margin."""

    def calculate_portfolio_margin(
        self,
        positions: List[OptionsPosition],
        underlying_prices: Dict[str, float],
    ) -> PortfolioMarginResult:
        """Calculate PM using TIMS scenarios."""
```

#### 6.2.3 Exercise/Assignment (`impl_exercise_assignment.py`)

**Early Exercise Decision (American)**:
```
Exercise if: intrinsic_value > time_value + transaction_cost

For calls on dividend-paying stocks:
Exercise if: dividend > time_value + carrying_cost
```

```python
class ExerciseAssignmentEngine:
    def should_exercise_early(
        self,
        position: OptionsPosition,
        underlying_price: float,
        rate: float,
        dividend: Optional[Dividend] = None,
    ) -> ExerciseDecision:
        """Determine if early exercise optimal."""

    def simulate_assignment(
        self,
        short_position: OptionsPosition,
        underlying_price: float,
        assignment_probability: float,
    ) -> AssignmentResult:
        """Simulate random assignment for short options."""
```

### 6.3 Test Matrix (Phase 6)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Greeks Guards | 25 | All limits, scaling |
| Exercise Risk | 20 | American, dividends |
| Assignment Risk | 15 | Short calls/puts |
| Reg T Margin | 25 | All position types |
| Portfolio Margin | 30 | TIMS scenarios |
| Exercise/Assignment Engine | 25 | Decision logic |
| **Total** | **140** | **100%** |

### 6.4 Deliverables
- [ ] `services/options_risk_guards.py` — All risk guards
- [ ] `impl_options_margin.py` — Reg T + PM
- [ ] `impl_exercise_assignment.py` — Exercise engine
- [ ] `tests/test_options_risk.py` — 140 tests
- [ ] Documentation: `docs/options/risk_management.md`

---

## Phase 7: Complex Orders & Strategies (4 weeks)

### 7.1 Objectives
- Multi-leg order execution
- Spread trading simulation
- Delta hedging automation
- Vol trading strategies

### 7.2 Components

#### 7.2.1 Multi-Leg Orders (`impl_multi_leg.py`)

**Supported Spread Types**:

| Spread | Legs | Max Loss | Max Gain |
|--------|------|----------|----------|
| Vertical | 2 | Limited | Limited |
| Calendar | 2 | Premium | Unlimited |
| Diagonal | 2 | Limited | Varies |
| Straddle | 2 | Premium | Unlimited |
| Strangle | 2 | Premium | Unlimited |
| Iron Condor | 4 | Limited | Limited |
| Butterfly | 3-4 | Limited | Limited |
| Box | 4 | None | Risk-free rate |

```python
@dataclass
class ComboOrder:
    legs: List[ComboLeg]
    order_type: ComboOrderType  # NET_DEBIT, NET_CREDIT, EVEN
    limit_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY

@dataclass
class ComboLeg:
    contract: OptionsContractSpec
    side: Side
    ratio: int = 1

class MultiLegExecutor:
    def execute_combo(
        self,
        order: ComboOrder,
        market_state: Dict[str, OptionsMarketState],
    ) -> ComboFill:
        """Execute multi-leg order atomically."""

    def simulate_legging_in(
        self,
        order: ComboOrder,
        execution_style: str = "sequential",
    ) -> List[OptionsFill]:
        """Simulate leg-by-leg execution."""
```

#### 7.2.2 Delta Hedging (`impl_delta_hedge.py`)

```python
class DeltaHedger:
    def __init__(
        self,
        hedge_threshold: float = 0.1,  # Rehedge when |delta| > threshold
        hedge_frequency: str = "continuous",  # or "daily", "on_threshold"
        use_futures: bool = False,
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
    ) -> HedgePnLResult:
        """Simulate hedging P&L vs BS theoretical."""
```

#### 7.2.3 Vol Trading Strategies (`strategies/vol_trading.py`)

```python
class VolatilityTrader:
    """Strategies based on IV vs RV mismatch."""

    def identify_vol_opportunity(
        self,
        iv_surface: IVSurface,
        realized_vol: float,
        forecast_vol: float,
    ) -> Optional[VolTrade]:
        """Find IV mispricing opportunities."""

    def construct_vega_neutral_trade(
        self,
        chain: OptionsChain,
        vol_view: str,  # "long_vol" or "short_vol"
    ) -> ComboOrder:
        """Build vega-neutral gamma trade."""
```

### 7.3 Test Matrix (Phase 7)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| ComboOrder creation | 20 | All spread types |
| Multi-leg execution | 30 | Atomic, legging |
| Delta hedging | 25 | Continuous, discrete |
| Hedge P&L | 20 | BS comparison |
| Vol strategies | 25 | Identification, construction |
| **Total** | **120** | **100%** |

### 7.4 Deliverables
- [ ] `impl_multi_leg.py` — Multi-leg execution
- [ ] `impl_delta_hedge.py` — Delta hedging
- [ ] `strategies/vol_trading.py` — Vol strategies
- [ ] `tests/test_options_complex.py` — 120 tests
- [ ] Documentation: `docs/options/complex_orders.md`

---

## Phase 8: Training Integration (4 weeks)

### 8.1 Objectives
- Options trading environment wrapper
- Options-specific features
- Reward shaping for options
- Greeks-aware policy

### 8.2 Components

#### 8.2.1 Options Environment (`wrappers/options_env.py`)

```python
class OptionsEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: TradingEnv,
        options_config: OptionsEnvConfig,
    ):
        self.greeks_provider = GreeksProvider()
        self.margin_calculator = OptionsMarginCalculator()
        self.iv_surface = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Decode action (strike, expiry, side, qty)
        # 2. Check margin requirement
        # 3. Execute option trade
        # 4. Update portfolio Greeks
        # 5. Compute reward with Greeks penalty
        pass

    def _update_greeks(self) -> None:
        """Recompute portfolio Greeks."""

    def _compute_options_reward(
        self,
        base_reward: float,
        greeks: GreeksResult,
    ) -> float:
        """Adjust reward based on Greeks exposure."""
```

#### 8.2.2 Options Features (`options_features.py`)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `iv_rank` | IV percentile (52w) | [0, 1] |
| `iv_skew` | 25Δ put - 25Δ call | Z-score |
| `term_slope` | Front/back IV ratio | Z-score |
| `realized_iv_spread` | IV - RV | Z-score |
| `gamma_exposure` | Market GEX | Z-score |
| `put_call_ratio` | Put/call volume | Log transform |
| `max_pain` | Max pain strike | Normalized |

```python
class OptionsFeatureExtractor:
    def extract_features(
        self,
        chain: OptionsChain,
        underlying_history: pd.DataFrame,
        iv_surface: IVSurface,
    ) -> np.ndarray:
        """Extract options-specific features."""
```

#### 8.2.3 Options Reward (`impl_options_reward.py`)

```python
def compute_options_reward(
    pnl: float,
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    max_delta: float = 100.0,
    max_gamma: float = 50.0,
    gamma_penalty: float = 0.1,
    theta_penalty: float = 0.05,
) -> float:
    """
    Reward = PnL
        - γ_penalty × max(0, |gamma| - max_gamma)
        - θ_penalty × |theta|  (encourage theta-positive)
    """
```

### 8.3 Test Matrix (Phase 8)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsEnvWrapper | 35 | Lifecycle, actions |
| Feature extraction | 25 | All features |
| Reward shaping | 20 | Greeks penalties |
| Training loop | 20 | Convergence |
| **Total** | **100** | **100%** |

### 8.4 Deliverables
- [ ] `wrappers/options_env.py` — Training environment
- [ ] `options_features.py` — Feature extraction
- [ ] `impl_options_reward.py` — Reward shaping
- [ ] `configs/config_train_options.yaml` — Training config
- [ ] `tests/test_options_training.py` — 100 tests
- [ ] Documentation: `docs/options/training.md`

---

## Phase 9: Live Trading (4 weeks)

### 9.1 Objectives
- Options position sync
- Real-time Greeks monitoring
- Exercise/expiration management
- Roll management

### 9.2 Components

#### 9.2.1 Options Live Runner (`services/options_live.py`)

```python
class OptionsLiveRunner:
    def __init__(
        self,
        config: OptionsLiveConfig,
        adapter: OptionsAdapter,
        greeks_monitor: GreeksMonitor,
        exercise_manager: ExerciseManager,
    ):
        pass

    async def run(self) -> None:
        """Main live trading loop."""

    async def _process_expiration(self, date: date) -> None:
        """Handle expiring positions."""

    async def _manage_rolls(self) -> None:
        """Roll positions before expiry."""
```

#### 9.2.2 Greeks Monitor (`services/greeks_monitor.py`)

```python
class GreeksMonitor:
    def __init__(
        self,
        check_interval_sec: float = 1.0,
        alert_thresholds: GreeksThresholds,
    ):
        pass

    async def start(self) -> None:
        """Start continuous Greeks monitoring."""

    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current portfolio Greeks."""

    def get_greeks_attribution(self) -> Dict[str, GreeksResult]:
        """Greeks breakdown by position."""
```

#### 9.2.3 Exercise Manager (`services/exercise_mgr.py`)

```python
class ExerciseManager:
    def __init__(
        self,
        adapter: OptionsAdapter,
        exercise_policy: ExercisePolicy,
    ):
        pass

    async def check_exercise_opportunities(self) -> List[ExerciseRecommendation]:
        """Identify positions to exercise."""

    async def handle_assignment(
        self,
        assignment: Assignment,
    ) -> AssignmentResult:
        """Process incoming assignment."""

    async def manage_expiration(
        self,
        expiring_positions: List[OptionsPosition],
    ) -> ExpirationResult:
        """Handle expiring positions."""
```

### 9.3 Test Matrix (Phase 9)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| OptionsLiveRunner | 30 | Lifecycle, events |
| Greeks Monitor | 25 | Real-time, alerts |
| Exercise Manager | 30 | Exercise, assignment |
| Roll Management | 20 | Auto-roll logic |
| Integration | 15 | Full flow |
| **Total** | **120** | **100%** |

### 9.4 Deliverables
- [ ] `services/options_live.py` — Live runner
- [ ] `services/greeks_monitor.py` — Greeks monitoring
- [ ] `services/exercise_mgr.py` — Exercise management
- [ ] `configs/config_live_options.yaml` — Live config
- [ ] `tests/test_options_live.py` — 120 tests
- [ ] Documentation: `docs/options/live_trading.md`

---

## Phase 10: Validation & Documentation (3 weeks)

### 10.1 Objectives
- Comprehensive validation testing
- Backward compatibility verification
- Performance benchmarks
- Complete documentation

### 10.2 Validation Tests

#### 10.2.1 Greeks Accuracy Validation

| Test | Target | Method |
|------|--------|--------|
| BS Greeks | < 0.01% | vs analytical |
| Binomial Greeks | < 0.1% | vs BS for European |
| MC Greeks | < 1% | vs BS (pathwise) |
| Numerical Greeks | < 0.5% | vs analytical |

#### 10.2.2 IV Surface Validation

| Metric | Target |
|--------|--------|
| Interpolation error | < 0.5 vol points |
| No butterfly arbitrage | 100% |
| No calendar arbitrage | 100% |
| SVI fit R² | > 0.99 |

#### 10.2.3 Execution Validation

| Metric | Target |
|--------|--------|
| Fill rate (L2) | > 95% |
| Fill rate (L3) | > 90% |
| Slippage error | < 3 bps |
| Fee accuracy | 100% |

### 10.3 Backward Compatibility

```bash
# Verify no regressions
pytest tests/test_unit_*.py -v           # Core unit tests
pytest tests/test_execution_*.py -v       # Execution
pytest tests/test_futures_*.py -v         # Futures
pytest tests/test_forex_*.py -v           # Forex
pytest tests/test_stock_*.py -v           # Stocks
pytest tests/test_options_*.py -v         # New options tests
```

### 10.4 Benchmarks

| Operation | Target | Method |
|-----------|--------|--------|
| BS pricing | < 1 μs | Vectorized NumPy |
| Greeks (all 8) | < 5 μs | Analytical |
| IV solve | < 50 μs | Newton-Raphson |
| Surface interpolation | < 10 μs | Cubic spline |
| L3 matching | < 100 μs | FIFO engine |

### 10.5 Documentation Suite

| Document | Description |
|----------|-------------|
| `docs/options/overview.md` | Architecture overview |
| `docs/options/core_models.md` | Greeks, pricing |
| `docs/options/volatility_surface.md` | IV surface, SVI |
| `docs/options/exchange_adapters.md` | IB, CBOE, Deribit |
| `docs/options/execution_l2.md` | L2 slippage, fees |
| `docs/options/l3_lob.md` | L3 simulation |
| `docs/options/risk_management.md` | Guards, margin |
| `docs/options/complex_orders.md` | Multi-leg, hedging |
| `docs/options/training.md` | RL environment |
| `docs/options/live_trading.md` | Live runner |
| `docs/options/api_reference.md` | Full API docs |
| `docs/options/migration_guide.md` | From L2 to L3 |

### 10.6 Test Matrix (Phase 10)

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| Greeks validation | 50 | Analytical comparison |
| IV surface validation | 40 | Arbitrage-free |
| Execution validation | 35 | Fill rates, slippage |
| Backward compatibility | 100 | All asset classes |
| Performance benchmarks | 25 | Timing targets |
| **Total** | **250** | **100%** |

### 10.7 Deliverables
- [ ] `tests/test_options_validation.py` — 150 validation tests
- [ ] `tests/test_options_backward_compat.py` — 100 compat tests
- [ ] `benchmarks/bench_options.py` — Performance suite
- [ ] `OPTIONS_INTEGRATION_REPORT.md` — Completion report
- [ ] `docs/options/*.md` — 12 documentation files

---

## Summary

### Test Count by Phase

| Phase | Tests | Cumulative |
|-------|-------|------------|
| 1: Core Models | 95 | 95 |
| 2: Exchange Adapters | 105 | 200 |
| 3: IV Surface | 105 | 305 |
| 4: L2 Execution | 100 | 405 |
| 5: L3 LOB | 130 | 535 |
| 6: Risk Management | 140 | 675 |
| 7: Complex Orders | 120 | 795 |
| 8: Training | 100 | 895 |
| 9: Live Trading | 120 | 1,015 |
| 10: Validation | 250 | **1,265** |

**Buffer for edge cases**: +235 tests
**Total**: **~1,500 tests**

### Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1 | 3 weeks | None |
| 2 | 4 weeks | Phase 1 |
| 3 | 4 weeks | Phase 1 |
| 4 | 3 weeks | Phases 1-3 |
| 5 | 5 weeks | Phases 1-4 |
| 6 | 4 weeks | Phases 1-5 |
| 7 | 4 weeks | Phases 1-6 |
| 8 | 4 weeks | Phases 1-7 |
| 9 | 4 weeks | Phases 1-8 |
| 10 | 3 weeks | All |

**Total**: ~38 weeks (~9 months)

### Key References

**Academic**:
- Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
- Gatheral (2006): "The Volatility Surface: A Practitioner's Guide"
- Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
- Dupire (1994): "Pricing with a Smile"
- Hull (2017): "Options, Futures, and Other Derivatives"

**Industry**:
- OCC: Options Clearing Corporation specifications
- CBOE: Index options methodology
- CME: Futures options specifications
- Deribit: Crypto options documentation

### Quick Reference Table (for CLAUDE.md)

| Task | Location | Test Command |
|------|----------|--------------|
| Options pricing | `impl_pricing.py` | `pytest tests/test_options_core.py` |
| Greeks calculation | `impl_greeks.py` | `pytest tests/test_options_core.py::TestGreeks` |
| IV surface | `impl_iv_surface.py` | `pytest tests/test_iv_surface.py` |
| Options L2 execution | `execution_providers_options.py` | `pytest tests/test_options_execution_l2.py` |
| Options L3 LOB | `lob/options_book.py` | `pytest tests/test_options_l3_lob.py` |
| Options risk guards | `services/options_risk_guards.py` | `pytest tests/test_options_risk.py` |
| Options margin | `impl_options_margin.py` | `pytest tests/test_options_risk.py::TestMargin` |
| Multi-leg orders | `impl_multi_leg.py` | `pytest tests/test_options_complex.py` |
| Options training | `wrappers/options_env.py` | `pytest tests/test_options_training.py` |
| Options live | `services/options_live.py` | `pytest tests/test_options_live.py` |
| IB options adapter | `adapters/ib/options.py` | `pytest tests/test_options_adapters.py::TestIB` |
| Deribit adapter | `adapters/deribit/options.py` | `pytest tests/test_options_adapters.py::TestDeribit` |

---

**Document Version**: 1.0
**Created**: 2025-12-03
**Author**: Claude Code
