#!/bin/bash
# Run advantage std floor fix tests
#
# This script runs the comprehensive test suite for the advantage std floor fix.
# The fix addresses critical numerical stability issues when normalizing advantages
# with very low variance.

set -e

echo "=============================================================================="
echo "ADVANTAGE STD FLOOR FIX - TEST RUNNER"
echo "=============================================================================="
echo ""

# Check if numpy is available
if python3 -c "import numpy; print(f'✓ numpy {numpy.__version__}')" 2>/dev/null; then
    echo "Dependencies OK"
else
    echo "✗ numpy not found. Install with: pip3 install numpy"
    exit 1
fi

echo ""
echo "Running unit tests..."
echo ""

# Run the test suite
python3 tests/test_advantage_std_floor_fix.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✓ ALL TESTS PASSED"
    echo "=============================================================================="
    exit 0
else
    echo ""
    echo "=============================================================================="
    echo "✗ TESTS FAILED"
    echo "=============================================================================="
    exit 1
fi
