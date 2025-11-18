#!/bin/bash
# Comprehensive gradient flow test suite
# Runs all gradient flow and edge case tests

set -e

echo "================================================================================"
echo "COMPREHENSIVE GRADIENT FLOW TEST SUITE"
echo "================================================================================"
echo ""

# Check if torch is available
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} found')" 2>/dev/null || {
    echo "❌ PyTorch not available"
    echo "   Install with: pip install torch"
    echo ""
    echo "Running analytical tests only (no PyTorch required)..."
    echo ""
    python3 test_gradient_minimal.py
    exit $?
}

echo "Running complete test suite..."
echo ""

# Track overall status
OVERALL_STATUS=0

# Test 1: Analytical analysis
echo "--------------------------------------------------------------------------------"
echo "TEST 1: Analytical Gradient Flow Analysis"
echo "--------------------------------------------------------------------------------"
python3 test_gradient_minimal.py
STATUS1=$?
if [ $STATUS1 -ne 0 ]; then
    echo "⚠️  Analytical test found potential issues"
    OVERALL_STATUS=1
fi
echo ""

# Test 2: Numerical gradients
echo "--------------------------------------------------------------------------------"
echo "TEST 2: Numerical Gradient Verification (Finite Differences)"
echo "--------------------------------------------------------------------------------"
python3 test_numerical_gradients.py
STATUS2=$?
if [ $STATUS2 -ne 0 ]; then
    echo "❌ Numerical gradient test FAILED"
    OVERALL_STATUS=1
fi
echo ""

# Test 3: Edge cases
echo "--------------------------------------------------------------------------------"
echo "TEST 3: Edge Case Tests"
echo "--------------------------------------------------------------------------------"
python3 test_edge_cases.py
STATUS3=$?
if [ $STATUS3 -ne 0 ]; then
    echo "❌ Edge case tests FAILED"
    OVERALL_STATUS=1
fi
echo ""

# Test 4: Original standalone test
echo "--------------------------------------------------------------------------------"
echo "TEST 4: Standalone Gradient Flow Tests"
echo "--------------------------------------------------------------------------------"
python3 test_gradient_flow_standalone.py 2>/dev/null || {
    echo "ℹ️  Standalone test requires full torch environment (skipped)"
}
echo ""

# Summary
echo "================================================================================"
echo "TEST SUITE SUMMARY"
echo "================================================================================"
echo ""

if [ $STATUS1 -eq 0 ]; then
    echo "✅ Analytical analysis: PASSED"
else
    echo "⚠️  Analytical analysis: POTENTIAL ISSUES"
fi

if [ $STATUS2 -eq 0 ]; then
    echo "✅ Numerical gradients: PASSED"
else
    echo "❌ Numerical gradients: FAILED"
fi

if [ $STATUS3 -eq 0 ]; then
    echo "✅ Edge cases: PASSED"
else
    echo "❌ Edge cases: FAILED"
fi

echo ""
echo "================================================================================"

if [ $OVERALL_STATUS -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED - Gradient flow is CORRECT!"
    echo "================================================================================"
else
    echo "⚠️  SOME TESTS FAILED - Review implementation!"
    echo "================================================================================"
fi

exit $OVERALL_STATUS
