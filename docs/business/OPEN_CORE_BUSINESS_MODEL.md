# Open-Core Business Model Strategy

## QuantBot AI - Strategic Framework

**Document Version:** 1.0
**Date:** December 2025
**Classification:** Internal Strategy Document

---

## 1. Introduction

### 1.1 What is Open-Core?

Open-core is a business model where a company:
- **Releases a core product** under an open-source license
- **Sells proprietary extensions** for enterprise features
- **Monetizes via SaaS/Cloud** hosting of the open-source core

**Key Insight:** The open component drives adoption and community; the proprietary component drives revenue.

### 1.2 Why Open-Core for QuantBot?

| Driver | How Open-Core Helps |
|--------|---------------------|
| **Market Entry** | Lower friction than enterprise sales |
| **Trust Building** | Financial firms can inspect SDK code |
| **Developer Adoption** | Familiar model (pip install) |
| **Competitive Response** | Hard to compete against "free" |
| **Talent Acquisition** | Hire from contributor community |
| **Grant Eligibility** | Open-source contribution valued |

---

## 2. Industry Case Studies

### 2.1 Databricks: The Gold Standard

**Background:**
- Founded 2013 by Apache Spark creators
- Open-sourced Spark, built commercial platform on top
- 2021 valuation: $43B

**What's Open (Apache 2.0):**
- Apache Spark (compute engine)
- Delta Lake (storage layer)
- MLflow (ML lifecycle)

**What's Proprietary:**
- Databricks Runtime optimizations
- Unity Catalog (governance)
- Managed infrastructure
- Enterprise security features

**Lesson for QuantBot:** Open the "substrate" (SDK, connectors), monetize the "intelligence" (RL engine, risk management).

### 2.2 Elastic: License Evolution

**Background:**
- Created Elasticsearch (2010)
- IPO 2018, peaked at $14B market cap
- Changed license in 2021 after AWS competition

**Evolution:**
1. **Phase 1:** Fully open (Apache 2.0)
2. **Phase 2:** Open-core (Apache + proprietary X-Pack)
3. **Phase 3:** SSPL license (anti-cloud copy protection)

**What's Open (SSPL):**
- Elasticsearch core
- Kibana visualization
- Logstash (data pipeline)

**What's Proprietary:**
- Machine learning features
- Advanced security (RBAC)
- Alerting and monitoring

**Lesson for QuantBot:** Start with MIT for maximum adoption, but reserve right to change license for proprietary components if cloud providers try to commoditize.

### 2.3 GitLab: CE vs EE

**Background:**
- Founded 2011 as DevOps platform
- IPO 2021 at $11B valuation
- 30M+ registered users

**What's Open (MIT):**
- GitLab Community Edition (CE)
- Core SCM, CI/CD, project management

**What's Proprietary:**
- GitLab Enterprise Edition (EE)
- Advanced CI/CD (DORA metrics)
- Security scanning
- Compliance features

**Pricing Model:**
| Tier | Price | Key Features |
|------|-------|--------------|
| Free | $0 | Core CE features |
| Premium | $19/user/month | CI/CD, security |
| Ultimate | $99/user/month | Compliance, support |

**Lesson for QuantBot:** Clear tier differentiation with enterprise features gated.

### 2.4 HashiCorp: Multi-Product Open-Core

**Background:**
- Founded 2012
- Multiple open-core products
- IPO 2021 at $14B valuation

**Portfolio Approach:**
| Product | Open | Enterprise |
|---------|------|------------|
| Terraform | Core IaC | Sentinel, governance |
| Vault | Secrets engine | HSM, namespaces |
| Consul | Service mesh | Admin partitions |
| Nomad | Orchestrator | Multi-region |

**What's Open (MPL 2.0):**
- Core functionality of all products
- Sufficient for small/medium deployments

**What's Proprietary:**
- Enterprise governance
- Multi-tenancy
- Premium support

**Lesson for QuantBot:** Consider future multi-product expansion with consistent open-core pattern.

---

## 3. QuantBot Open-Core Architecture

### 3.1 Repository Structure

```
ORGANIZATION: quantbot-ai

├── quantbot-sdk (PUBLIC - MIT License)
│   │
│   ├── quantbot/
│   │   ├── client.py          # API client
│   │   ├── websocket.py       # Real-time streaming
│   │   ├── models/
│   │   │   ├── orders.py      # Order data models
│   │   │   ├── positions.py   # Position data models
│   │   │   └── signals.py     # Signal data models
│   │   └── connectors/
│   │       ├── binance.py     # Public API only
│   │       ├── alpaca.py
│   │       └── oanda.py
│   │
│   ├── examples/
│   │   ├── basic_backtest.ipynb
│   │   ├── signal_streaming.ipynb
│   │   └── portfolio_analytics.ipynb
│   │
│   ├── docs/
│   │   ├── getting_started.md
│   │   ├── api_reference.md
│   │   └── tutorials/
│   │
│   ├── LICENSE (MIT)
│   ├── README.md
│   └── pyproject.toml

└── quantbot-core (PRIVATE - Proprietary)
    │
    ├── core_*/           # Core abstractions
    ├── impl_*/           # Implementation layer
    ├── service_*/        # Business logic
    ├── strategies/       # Trading strategies
    ├── lob/              # L3 LOB simulation
    ├── adapters/         # Exchange adapters (proprietary)
    ├── distributional_ppo.py
    ├── execution_sim.py
    ├── risk_guard.py
    └── [11,000+ proprietary components]
```

### 3.2 Feature Distribution Matrix

| Feature Category | Open SDK | Cloud Pro | Enterprise |
|------------------|----------|-----------|------------|
| **Data Access** | | | |
| Market data API | ✅ | ✅ | ✅ |
| Historical data | Limited | ✅ | ✅ |
| Real-time streaming | ✅ | ✅ | ✅ |
| **Backtesting** | | | |
| Basic backtest | ✅ | ✅ | ✅ |
| L2 execution sim | ❌ | ✅ | ✅ |
| L3 LOB simulation | ❌ | ❌ | ✅ |
| **Strategy** | | | |
| Signal API | ✅ | ✅ | ✅ |
| Custom strategy training | ❌ | ✅ | ✅ |
| Multi-asset strategies | ❌ | ✅ | ✅ |
| **Risk Management** | | | |
| Basic risk metrics | ✅ | ✅ | ✅ |
| CVaR-aware optimization | ❌ | ✅ | ✅ |
| Advanced risk guards | ❌ | ❌ | ✅ |
| **Deployment** | | | |
| Cloud (multi-tenant) | N/A | ✅ | ✅ |
| On-premise | ❌ | ❌ | ✅ |
| Private cloud | ❌ | ❌ | ✅ |
| **Support** | | | |
| Community support | ✅ | ✅ | ✅ |
| Email support | ❌ | ✅ | ✅ |
| Dedicated CSM | ❌ | ❌ | ✅ |
| SLA | ❌ | 99.9% | 99.95% |

### 3.3 Pricing Model

| Tier | Price | Target Customer |
|------|-------|-----------------|
| **SDK (Free)** | $0 | Developers, students, researchers |
| **Cloud Pro** | $500/month | Small quant teams, indie traders |
| **Cloud Team** | $2,000/month | Small funds, trading desks |
| **Enterprise** | Custom ($50K+/year) | Institutional clients |

**Revenue Drivers:**
- Compute usage (backtest hours)
- Data consumption (market data calls)
- Seat licenses (team size)
- Feature upgrades (L3 simulation, custom models)

---

## 4. License Strategy

### 4.1 Open Component License: MIT

```
MIT License

Copyright (c) 2025 QuantBot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

**Why MIT:**
- Maximum permissiveness for adoption
- No copyleft concerns for enterprise
- Compatible with proprietary internal systems
- Industry standard for developer tools

### 4.2 Proprietary License

```
QUANTBOT AI PROPRIETARY LICENSE

Copyright (c) 2025 QuantBot AI. All Rights Reserved.

This software and associated documentation files (the "Software") are
proprietary and confidential.

RESTRICTIONS:
1. No part of the Software may be copied, modified, distributed, sold,
   sublicensed, or transferred without prior written consent.
2. Reverse engineering, decompilation, or disassembly is prohibited.
3. The Software may only be used under a valid license agreement.

ACCESS:
- SaaS access via QuantBot AI cloud platform (terms of service)
- Enterprise access via signed Enterprise License Agreement
- Evaluation access via time-limited trial agreement

For licensing inquiries: enterprise@quantbot.ai
```

### 4.3 Contributor License Agreement (CLA)

For open SDK contributions:

```
CONTRIBUTOR LICENSE AGREEMENT

By submitting a Contribution, you agree that:

1. GRANT OF COPYRIGHT LICENSE
   You grant QuantBot AI a perpetual, worldwide, non-exclusive, royalty-free
   license to use, copy, modify, and distribute your Contribution.

2. GRANT OF PATENT LICENSE
   You grant QuantBot AI a perpetual, worldwide, non-exclusive, royalty-free
   patent license for any patents you own that cover your Contribution.

3. AUTHORITY
   You represent that you have the legal authority to enter this Agreement.

4. ORIGINAL WORK
   You represent that your Contribution is your original creation.

5. NO WARRANTY
   You provide your Contribution "AS IS" without warranty.
```

---

## 5. Community Strategy

### 5.1 Community Building

**Channels:**
- **Discord:** Community discussions, support
- **GitHub Discussions:** Technical Q&A, feature requests
- **Twitter/X:** Announcements, thought leadership
- **Blog:** Tutorials, case studies, releases

**Events:**
- Webinars (monthly): Feature deep-dives
- Meetups (quarterly): In-person community events
- Conference presence: NeurIPS, ICML, QuantCon

### 5.2 Contribution Governance

**Contributor Levels:**
| Level | Criteria | Privileges |
|-------|----------|------------|
| Contributor | 1+ merged PR | Listed in CONTRIBUTORS |
| Committer | 10+ PRs, consistent quality | Direct commit access |
| Maintainer | Proven leadership | Release management, code review |

**Contribution Areas (SDK only):**
- Bug fixes
- Documentation improvements
- New connector integrations
- Example notebooks
- Performance optimizations

**NOT Open for Contribution:**
- Core RL algorithms
- Execution simulation
- Risk management
- Proprietary features

### 5.3 Ecosystem Development

**SDK Integrations (encourage/support):**
- Jupyter/JupyterLab extensions
- VS Code extension
- Streamlit apps
- Grafana dashboards

**Partner Integrations:**
- Cloud providers (AWS, GCP, Azure)
- Data providers (Polygon, Alpaca, Binance)
- Broker integrations

---

## 6. Competitive Response

### 6.1 If Competitor Forks SDK

**Mitigation:**
1. SDK is only the "edge" – limited value without platform
2. Continuous innovation in SDK keeps fork outdated
3. Brand and community are non-forkable
4. Enterprise features remain proprietary

### 6.2 If Cloud Provider Offers Competing Service

**Mitigation:**
1. Trade secrets protect core differentiation
2. Patents (pending) provide legal recourse
3. Specialized domain expertise hard to replicate
4. Existing customer relationships and trust

**Historical Example:** MongoDB vs AWS DocumentDB
- MongoDB's SSPL license response
- Customer preference for "authentic" vendor
- Continued MongoDB growth despite competition

### 6.3 If Startup Copies Approach

**Mitigation:**
1. 18-24 month head start
2. Established customer base
3. Trained models and calibrations
4. 11,063 test cases as quality moat

---

## 7. Success Metrics

### 7.1 Open Source Metrics

| Metric | Target Y1 | Target Y2 |
|--------|-----------|-----------|
| GitHub Stars | 500 | 2,000 |
| Monthly Downloads (PyPI) | 1,000 | 10,000 |
| Contributors | 20 | 50 |
| Discord Members | 500 | 2,000 |

### 7.2 Conversion Metrics

| Metric | Target |
|--------|--------|
| SDK → Cloud Pro | 2% of active SDK users |
| Cloud Pro → Team | 20% upgrade rate |
| Team → Enterprise | 10% escalation rate |

### 7.3 Revenue Metrics

| Metric | Y1 | Y2 | Y3 |
|--------|-------|-------|-------|
| ARR | $300K | $1.5M | $5M |
| Customers | 50 | 200 | 500 |
| Enterprise Deals | 5 | 15 | 40 |

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Q1 2025)
- [ ] Extract SDK from monorepo
- [ ] Create public GitHub repository
- [ ] Implement CLA signing workflow
- [ ] Launch documentation site
- [ ] PyPI package publication

### Phase 2: Community (Q2 2025)
- [ ] Discord community launch
- [ ] First blog posts / tutorials
- [ ] Webinar series kickoff
- [ ] Conference submissions

### Phase 3: Expansion (Q3-Q4 2025)
- [ ] Contributor program launch
- [ ] Partner integrations
- [ ] SDK v2.0 with expanded features
- [ ] Community meetup events

---

## 9. References

### Academic & Industry Research

1. Nagle, F. (2019). "Open Source Software and Firm Productivity." *Management Science*.

2. O'Mahony, S., & Ferraro, F. (2007). "The emergence of governance in an open source community." *Academy of Management Journal*.

3. West, J., & Gallagher, S. (2006). "Challenges of open innovation: the paradox of firm investment in open-source software." *R&D Management*.

### Case Study Sources

4. Databricks S-1 Filing (2021). SEC EDGAR.

5. GitLab IPO Prospectus (2021). SEC EDGAR.

6. Elastic N.V. Annual Report (2022). SEC EDGAR.

7. HashiCorp S-1 Filing (2021). SEC EDGAR.

### Best Practices

8. Open Source Initiative. "Open Source Business Models." https://opensource.org/

9. Linux Foundation. "Guide to Enterprise Open Source." https://www.linuxfoundation.org/

10. Tidelift. "The Open Source Maintainer's Guide to Revenue." (2023).

---

**Document Classification:** INTERNAL STRATEGY
**Owner:** CEO / CTO
**Review Cycle:** Quarterly
