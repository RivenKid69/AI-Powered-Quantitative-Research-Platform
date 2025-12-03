# -*- coding: utf-8 -*-
"""
adapters/theta_data/__init__.py
Theta Data options data adapter package.

Theta Data provides cost-effective US options data:
- Full US options universe
- Historical data back to 2013
- Real-time with 15-min delay (free) or real-time ($)
- $100/month (vs OPRA $2,500/month)

Reference: https://www.thetadata.io/

Phase 2: US Exchange Adapters
"""

from adapters.theta_data.options import (
    ThetaDataOptionsAdapter,
    ThetaDataConfig,
    ThetaDataQuote,
    ThetaDataTrade,
    create_theta_data_adapter,
)

__all__ = [
    "ThetaDataOptionsAdapter",
    "ThetaDataConfig",
    "ThetaDataQuote",
    "ThetaDataTrade",
    "create_theta_data_adapter",
]
