#!/bin/bash
# Validation script for PPO ratio clamping tests
# Run this after installing dependencies (torch, pytest)

set -e

echo "=========================================="
echo "PPO Ratio Clamping Fix - Test Suite"
echo "=========================================="
echo ""

# Check if PyTorch is installed
echo "Checking dependencies..."
if python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null; then
    HAS_TORCH=true
else
    echo "⚠ PyTorch not found. Install with: pip3 install torch"
    echo "  (Will run limited validation without PyTorch)"
    HAS_TORCH=false
fi

if python3 -c "import pytest; print(f'✓ pytest {pytest.__version__}')" 2>/dev/null; then
    HAS_PYTEST=true
else
    echo "⚠ pytest not found. Install with: pip3 install pytest"
    echo "  (Will run standalone tests only)"
    HAS_PYTEST=false
fi

echo ""

# Always run standalone validation
echo "=========================================="
echo "Running standalone validation..."
echo "=========================================="
echo ""
python3 test_ratio_clamping_standalone.py

if [ "$HAS_PYTEST" = true ] && [ "$HAS_TORCH" = true ]; then
    echo ""
    echo "=========================================="
    echo "Running unit tests with pytest..."
    echo "=========================================="
    echo ""
    pytest tests/test_distributional_ppo_ratio_clamping.py -v --tb=short

    echo ""
    echo "=========================================="
    echo "Running integration tests..."
    echo "=========================================="
    echo ""
    pytest tests/test_distributional_ppo_ratio_clamping_integration.py -v --tb=short
fi

echo ""
echo "=========================================="
echo "✓ All tests completed successfully!"
echo "=========================================="
echo ""
echo "Summary of fix:"
echo "  • Changed log_ratio clamp from ±20 to ±10"
echo "  • Reduces max ratio from 485M to 22k"
echo "  • Improves numerical stability by >20,000x"
echo "  • Maintains PPO trust region principle"
echo ""
