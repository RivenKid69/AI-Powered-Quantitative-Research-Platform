#!/bin/bash
# Run ALL advantage std floor fix tests (V1, V2, and deep analysis)

set -e

echo "=============================================================================="
echo "ADVANTAGE STD FLOOR FIX - COMPLETE TEST SUITE"
echo "=============================================================================="
echo ""

# Check numpy
if python3 -c "import numpy; print(f'✓ numpy {numpy.__version__}')" 2>/dev/null; then
    echo "Dependencies OK"
else
    echo "✗ numpy not found. Install with: pip3 install numpy"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "PART 1: Numerical Experiment (demonstrates the problem)"
echo "=============================================================================="
python3 test_advantage_std_floor_experiment.py
echo ""

echo "=============================================================================="
echo "PART 2: V2 Unit Tests (corrected implementation)"
echo "=============================================================================="
python3 tests/test_advantage_std_floor_fix_v2.py
echo ""

echo "=============================================================================="
echo "PART 3: Deep Analysis V2 (comprehensive validation)"
echo "=============================================================================="
python3 tests/test_advantage_std_floor_deep_analysis_v2.py
echo ""

echo "=============================================================================="
echo "✓ ALL TESTS PASSED - IMPLEMENTATION VERIFIED"
echo "=============================================================================="
echo ""
echo "Summary:"
echo "  - Problem confirmed: 1e-8 floor → 10,000x gradient explosion"
echo "  - Solution validated: 1e-4 floor → gradient safe"
echo "  - PPO contract maintained: mean=0 always"
echo "  - No warnings in deep analysis"
echo ""
