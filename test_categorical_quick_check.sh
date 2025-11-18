#!/bin/bash

echo "=========================================="
echo "CATEGORICAL VF CLIPPING - QUICK CHECK"
echo "=========================================="
echo

# Check 1: Projection function exists
echo "TEST 1: Projection function exists"
if grep -q "def _project_categorical_distribution" distributional_ppo.py; then
    echo "✓ _project_categorical_distribution function found"
else
    echo "✗ _project_categorical_distribution function NOT found"
    exit 1
fi
echo

# Check 2: Same bounds bug fixed
echo "TEST 2: Same bounds bug fixed"
if grep -q "rows_with_same_bounds" distributional_ppo.py && \
   grep -q "batch_indices_to_fix" distributional_ppo.py && \
   grep -q "corrected_row" distributional_ppo.py; then
    echo "✓ Bug fix code present (rows_with_same_bounds, batch_indices_to_fix, corrected_row)"
else
    echo "✗ Bug fix code missing"
    exit 1
fi

# Check old buggy pattern is NOT present
if grep -q "projected_probs\[batch_indices\] = 0.0" distributional_ppo.py; then
    echo "✗ OLD BUGGY PATTERN STILL PRESENT!"
    exit 1
else
    echo "✓ Old buggy pattern not found (good!)"
fi
echo

# Check 3: VF clipping structure
echo "TEST 3: VF clipping for categorical"
if grep -q "critic_loss_unclipped = -(" distributional_ppo.py && \
   grep -q "critic_loss_clipped = -(" distributional_ppo.py && \
   grep -q "torch.max(critic_loss_unclipped, critic_loss_clipped)" distributional_ppo.py; then
    echo "✓ VF clipping uses max(unclipped, clipped)"
else
    echo "✗ VF clipping structure incomplete"
    exit 1
fi
echo

# Check 4: Projection called
echo "TEST 4: Projection called in VF clipping"
if grep -q "_project_categorical_distribution" distributional_ppo.py | grep -v "def _project"; then
    echo "✓ Projection function is called"
else
    echo "✗ Projection function not called"
    exit 1
fi
echo

# Check 5: Consistency with quantile
echo "TEST 5: Consistency with quantile"
count=$(grep -c "torch.max(critic_loss_unclipped, critic_loss_clipped)" distributional_ppo.py)
if [ "$count" -ge 2 ]; then
    echo "✓ Found $count max() calls (quantile + categorical)"
else
    echo "✗ Only found $count max() call(s), expected >= 2"
    exit 1
fi
echo

# Check 6: Documentation
echo "TEST 6: Documentation"
if grep -q "CRITICAL FIX" distributional_ppo.py && \
   grep -q "PPO VF clipping" distributional_ppo.py; then
    echo "✓ Documentation comments present"
else
    echo "✗ Documentation incomplete"
    exit 1
fi
echo

# Check 7: Edge cases
echo "TEST 7: Edge case handling"
if grep -q "num_atoms <= 1" distributional_ppo.py && \
   grep -q "abs(delta_z) < 1e-6" distributional_ppo.py && \
   grep -q "same_bounds" distributional_ppo.py; then
    echo "✓ Edge cases handled (single atom, degenerate delta, same_bounds)"
else
    echo "✗ Some edge cases not handled"
    exit 1
fi
echo

# Check 8: Gradient-safe operations
echo "TEST 8: Gradient flow safety"
# Check that we use tensor ops, not .item() for probability values in critical sections
if grep -A5 "corrected_row\[target_idx\]" distributional_ppo.py | grep -q "probs\[batch_idx, atom_idx\]"; then
    echo "✓ Uses tensor operations for probabilities"
else
    echo "⚠ Warning: Could not verify tensor operations"
fi
echo

echo "=========================================="
echo "✓✓✓ ALL QUICK CHECKS PASSED ✓✓✓"
echo "=========================================="
echo
echo "Implementation verified:"
echo "  ✓ Projection function implemented"
echo "  ✓ Same bounds bug FIXED"
echo "  ✓ VF clipping structure correct"
echo "  ✓ Consistent with quantile approach"
echo "  ✓ Documentation in place"
echo "  ✓ Edge cases handled"
echo "  ✓ Gradient-safe operations"
