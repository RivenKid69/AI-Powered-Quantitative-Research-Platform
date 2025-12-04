# MVP Focus: Core Use-Case Definition

*What We're Building First — And Why*

---

## The One Problem We Solve

**Proprietary trading firms spend 6-12 months building infrastructure before deploying their first strategy.**

This is our singular focus. Everything in MVP serves this problem.

---

## Target Customer Profile

### Primary Persona: The Quant CTO

| Attribute | Profile |
|-----------|---------|
| **Title** | CTO, Head of Technology, Lead Developer |
| **Company** | Prop trading firm (5-50 traders) |
| **Location** | EU (Netherlands, Germany, Ireland, France) |
| **Background** | Quantitative finance, software engineering |
| **Experience** | Built trading systems at banks or hedge funds |
| **Current State** | Evaluating build vs. buy for new firm/strategy |

### Their Day-to-Day Pain

| Pain Point | Frequency | Intensity |
|------------|-----------|-----------|
| Building execution infrastructure from scratch | Every new firm | High |
| Implementing risk management that satisfies compliance | Every system | High |
| Backtesting with realistic execution simulation | Daily | Medium |
| Managing multiple exchange integrations | Ongoing | Medium |
| Explaining technical architecture to non-technical partners | Weekly | Low |

### Jobs to Be Done

1. **Primary Job**: Get a new quantitative strategy live in production with proper risk controls
2. **Secondary Job**: Validate strategy performance before committing capital
3. **Tertiary Job**: Demonstrate to investors/partners that infrastructure is institutional-grade

---

## MVP Scope Definition

### In Scope (Must Have)

| Feature | Why Essential | Customer Value |
|---------|---------------|----------------|
| **Crypto execution (Binance)** | Fastest market to deploy | Days, not months to go live |
| **Risk-aware position sizing** | Regulatory requirement | MiFID II alignment |
| **CVaR-constrained optimization** | Key differentiator | Better risk-adjusted returns |
| **Basic backtesting** | Strategy validation | Confidence before capital |
| **Real-time monitoring** | Operational necessity | Know what's happening |
| **Configurable risk limits** | Compliance requirement | Max drawdown, position limits |

### Out of Scope (Deferred)

| Feature | Why Deferred | Reintroduce When |
|---------|--------------|------------------|
| US Equities (Alpaca) | Different regulatory regime | EU validation complete |
| CME Futures (IB) | Complex, institutional-only | After 10 paying customers |
| Options pricing | Specialized user base | Customer requests (3+) |
| L3 LOB simulation | Advanced research feature | Power user demand |
| Multi-strategy orchestration | Complexity | Single-strategy validated |
| White-label/API | Enterprise feature | Series A roadmap |

### The MVP Feature Boundary

```
┌─────────────────────────────────────────────────────────────┐
│                      MVP BOUNDARY                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │   Crypto Execution ──► Risk Management ──► Backtest │    │
│  │         │                    │                │     │    │
│  │         ▼                    ▼                ▼     │    │
│  │   Binance Spot         CVaR Limits      Historical  │    │
│  │   Binance Futures      Position Sizing  Simulation  │    │
│  │                        Max Drawdown                 │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ════════════════════ DEFERRED ═══════════════════════════  │
│                                                              │
│  [ Equities ] [ Options ] [ CME ] [ L3 LOB ] [ Multi-Strat ] │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Value Proposition (MVP)

### For European Prop Trading Firms

**Before Our Product:**
- 6-12 months to build execution infrastructure
- €200K-500K in development costs
- Risk management built as afterthought
- Backtesting accuracy questionable

**After Our Product:**
- Days to first live strategy
- €2,000-5,000/month subscription
- Risk management built-in (CVaR, limits)
- Research-grade execution simulation

### The Pitch (30 Seconds)

> "We help prop trading firms go from strategy idea to live trading in days, not months. Our platform handles execution, risk management, and backtesting — so you can focus on alpha generation. We're starting with crypto markets and European firms."

---

## Success Metrics for MVP

### North Star Metric

**Time from signup to first live trade**

Target: < 1 week

### Supporting Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Activation Rate** | % completing onboarding | > 80% |
| **Time to First Backtest** | Hours from signup | < 4 hours |
| **Time to First Live Trade** | Days from signup | < 7 days |
| **Weekly Active Users** | Users with 1+ action/week | > 70% |
| **Strategies Deployed** | Live strategies per user | > 2 |
| **NPS Score** | Net Promoter Score | > 40 |

---

## What Success Looks Like

### Month 1: Validation

- 3-5 pilot firms onboarded
- 80%+ complete setup
- First feedback collected
- Top 3 friction points identified

### Month 3: Product-Market Fit Signals

- 70%+ weekly active rate maintained
- NPS > 40
- 2+ firms express willingness to pay
- Referral from existing pilot

### Month 6: Revenue Validation

- 3+ paying customers (€2K+/month each)
- < 20% monthly churn
- Clear feature roadmap from customer input
- Repeatable sales process documented

---

## Feature Prioritization Framework

### How We Decide What to Build Next

```
                    HIGH CUSTOMER DEMAND
                           │
              ┌────────────┼────────────┐
              │            │            │
              │   BUILD    │   BUILD    │
              │   NEXT     │   NOW      │
              │            │            │
    LOW ──────┼────────────┼────────────┼────── HIGH
    EFFORT    │            │            │       EFFORT
              │   CONSIDER │   DEFER    │
              │   LATER    │            │
              │            │            │
              └────────────┼────────────┘
                           │
                    LOW CUSTOMER DEMAND
```

### Validation Requirements for New Features

| Requirement | Threshold |
|-------------|-----------|
| Customer requests | 3+ independent requests |
| Revenue impact | Affects conversion or retention |
| Competitive necessity | Losing deals without it |
| Strategic alignment | Fits European market focus |

---

## Competitive Positioning (MVP)

### We Are NOT Competing With:

| Competitor | Their Focus | Why We're Different |
|------------|-------------|---------------------|
| QuantConnect | Retail algo traders | We target institutional |
| Alpaca | Broker/API | We're full platform |
| Trading Technologies | Established institutions | We serve emerging firms |
| In-house development | Custom everything | We reduce build time |

### Our MVP Positioning

**"The fastest path from quant strategy to live trading for European prop firms."**

We don't compete on features. We compete on time-to-market.

---

## Risk Mitigation

### What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Customers want equities, not crypto | Medium | High | Pivot to equity MVP if 5+ requests |
| Pricing too high | Medium | Medium | Test €500/month pilot pricing |
| Regulatory concerns (MiFID II) | Low | High | Legal review before paid launch |
| Technical reliability issues | Low | High | Extensive testing already done |
| Competitive response | Low | Medium | Speed of execution, customer focus |

### Pivot Triggers

We will consider pivoting if:

1. **Asset class mismatch**: 70%+ of prospects want equities/futures instead of crypto
2. **Price sensitivity**: Zero conversion at €2K/month after 10 prospects
3. **Feature gap**: Consistent loss to competitors on specific capability
4. **Market timing**: Crypto market conditions make prop trading unviable

---

## The Build vs. Validate Commitment

### What We Will Do

- Weekly customer conversations (minimum 3/week)
- Bi-weekly releases based on feedback
- Monthly business reviews with pilot customers
- Quarterly strategy assessment

### What We Will NOT Do

- Build features without customer validation
- Expand asset classes without proving current ones
- Add complexity before simplicity works
- Chase enterprise deals before SMB validation

---

## Appendix: Customer Interview Guide

### Questions for Discovery Calls

**Problem Exploration:**
1. Walk me through how you built your last trading system. What took the longest?
2. What's the most frustrating part of your current infrastructure?
3. How do you handle risk management today? What's missing?

**Solution Validation:**
4. If you could wave a magic wand, what would your ideal platform do?
5. What would make you switch from your current solution?
6. How much would you pay to save 6 months of development time?

**Competitive Landscape:**
7. What tools/platforms do you use today? What do you like/dislike?
8. Have you evaluated other solutions? Why did/didn't you choose them?
9. What would prevent you from using a third-party platform?

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Owner: Product Team*

