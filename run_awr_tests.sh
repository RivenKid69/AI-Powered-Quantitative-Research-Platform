#!/bin/bash
# Quick validation script for AWR weighting tests
# Run this after installing dependencies (torch, pytest)

set -e

echo "=================================="
echo "AWR Weighting Tests"
echo "=================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "✗ PyTorch not found. Install with: pip3 install torch"
    exit 1
}

python3 -c "import pytest; print(f'✓ pytest {pytest.__version__}')" 2>/dev/null || {
    echo "✗ pytest not found. Install with: pip3 install pytest"
    exit 1
}

echo ""
echo "Running AWR weighting tests..."
echo ""

# Run tests with pytest
pytest tests/test_distributional_ppo_awr_weighting.py -v --tb=short

echo ""
echo "=================================="
echo "All tests completed successfully!"
echo "=================================="
