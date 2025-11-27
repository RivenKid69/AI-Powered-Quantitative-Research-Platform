"""Global interpreter tweaks for AI-Powered Quantitative Research Platform."""

import os

# Ensure the no-trade feature mask stays disabled unless a caller
# explicitly overrides the environment variable before interpreter start.
os.environ.setdefault("NO_TRADE_FEATURES_DISABLED", "1")
