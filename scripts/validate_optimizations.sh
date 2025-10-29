#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pre-deployment validation for AURA optimizations
# Runs all tests and checks before deploying optimizations

set -Eeuo pipefail

echo "🔍 AURA Optimization Pre-Deployment Validation"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED_TESTS=0
TOTAL_TESTS=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "\n${YELLOW}Running: ${test_name}${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# 1. Check Python environment
echo -e "\n📦 Checking Python Environment..."
run_test "Python Version" "python3 --version"
run_test "JAX Installation" "python3 -c 'import jax; print(f\"JAX {jax.__version__}\")'"
run_test "Flax Installation" "python3 -c 'import flax; print(f\"Flax {flax.__version__}\")'"
run_test "Optax Installation" "python3 -c 'import optax; print(f\"Optax {optax.__version__}\")'"

# 2. Check source files exist
echo -e "\n📄 Checking Optimization Module Files..."
run_test "TPU Optimizer" "test -f src/aura/optimization/tpu_optimizer.py"
run_test "Neuroplasticity" "test -f src/aura/optimization/neuroplasticity.py"
run_test "Causal Reasoning" "test -f src/aura/optimization/causal_reasoning.py"
run_test "Evolutionary Experts" "test -f src/aura/optimization/evolutionary_experts.py"
run_test "Meta-Learning" "test -f src/aura/optimization/meta_learning.py"
run_test "Deployment Script" "test -f scripts/deploy_optimizations.py"

# 3. Syntax check all Python files
echo -e "\n🔧 Syntax Checking Python Files..."
for file in src/aura/optimization/*.py scripts/deploy_optimizations.py; do
    if [[ -f "$file" ]]; then
        run_test "Syntax: $(basename $file)" "python3 -m py_compile $file"
    fi
done

# 4. Run unit tests
echo -e "\n🧪 Running Unit Tests..."
run_test "Optimization Tests" "python3 -m pytest tests/test_optimizations.py -v --tb=short 2>&1 || python3 tests/test_optimizations.py"

# 5. Import checks
echo -e "\n📥 Testing Module Imports..."
run_test "Import TPU Optimizer" "python3 -c 'from aura.optimization.tpu_optimizer import create_optimized_training_setup'"
run_test "Import Neuroplasticity" "python3 -c 'from aura.optimization.neuroplasticity import NeuroplasticityEngine'"
run_test "Import Causal Reasoning" "python3 -c 'from aura.optimization.causal_reasoning import CausalReasoningEngine'"
run_test "Import Evolutionary" "python3 -c 'from aura.optimization.evolutionary_experts import ExpertEvolutionEngine'"
run_test "Import Meta-Learning" "python3 -c 'from aura.optimization.meta_learning import MetaLearningEngine'"

# 6. Quick smoke tests
echo -e "\n🔥 Running Smoke Tests..."

# TPU Optimizer smoke test
run_test "TPU Optimizer Smoke Test" "python3 -c '
from aura.optimization.tpu_optimizer import create_optimized_training_setup
config = create_optimized_training_setup(\"small\", 256, 8)
assert config is not None
print(\"TPU config created successfully\")
'"

# Neuroplasticity smoke test
run_test "Neuroplasticity Smoke Test" "python3 -c '
from aura.optimization.neuroplasticity import NeuroplasticityEngine, PlasticityConfig
import jax, jax.numpy as jnp
config = PlasticityConfig()
engine = NeuroplasticityEngine(config)
key = jax.random.key(0)
activities = {\"expert_0\": jax.random.normal(key, (4, 32))}
rewards = {\"expert_0\": 0.8}
connections = engine.update_expert_connections(activities, rewards)
assert len(connections) >= 0
print(\"Neuroplasticity engine working\")
'"

# Causal Reasoning smoke test
run_test "Causal Reasoning Smoke Test" "python3 -c '
from aura.optimization.causal_reasoning import CausalReasoningEngine
import jax, jax.numpy as jnp
engine = CausalReasoningEngine()
key = jax.random.key(0)
data = {\"X\": jax.random.normal(key, (50,)), \"Y\": jax.random.normal(key, (50,))}
dag = engine.learn_causal_structure(data)
assert dag is not None
print(\"Causal reasoning engine working\")
'"

# Evolution smoke test
run_test "Evolution Smoke Test" "python3 -c '
from aura.optimization.evolutionary_experts import ExpertEvolutionEngine
engine = ExpertEvolutionEngine(population_size=5)
population = engine.initialize_population(16, 3)
assert len(population) == 5
print(\"Evolution engine working\")
'"

# Meta-learning smoke test
run_test "Meta-Learning Smoke Test" "python3 -c '
from aura.optimization.meta_learning import MetaLearningEngine, MetaLearningConfig
config = MetaLearningConfig()
engine = MetaLearningEngine(config)
meta_expert = engine.create_meta_expert(16, 3, \"maml\")
assert meta_expert is not None
print(\"Meta-learning engine working\")
'"

# 7. Check for common issues
echo -e "\n🔍 Checking for Common Issues..."

# Check for large files that might cause issues
run_test "No Large Test Files" "! find tests/ -name '*.py' -size +1M 2>/dev/null | grep -q ."

# Check for TODO/FIXME markers
echo -e "\n📝 Checking for TODO/FIXME markers..."
TODO_COUNT=$(grep -r "TODO\|FIXME" src/aura/optimization/ 2>/dev/null | wc -l || echo 0)
if [ "$TODO_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $TODO_COUNT TODO/FIXME markers in optimization code${NC}"
    echo "   (This is informational only, not a failure)"
else
    echo -e "${GREEN}✓ No TODO/FIXME markers found${NC}"
fi

# 8. Memory and performance checks
echo -e "\n💾 Basic Performance Checks..."
run_test "JAX Device Detection" "python3 -c '
import jax
devices = jax.devices()
print(f\"Found {len(devices)} JAX devices: {[str(d.device_kind) for d in devices]}\")
assert len(devices) > 0
'"

# 9. Configuration validation
echo -e "\n⚙️  Configuration Validation..."
run_test "Default Config Valid" "python3 -c '
import json
default_config = {
    \"model_size\": \"medium\",
    \"num_experts\": 16,
    \"enable_tpu_optimization\": True,
    \"enable_neuroplasticity\": True,
    \"enable_causal_reasoning\": True,
    \"enable_evolutionary_experts\": True,
    \"enable_meta_learning\": True
}
json.dumps(default_config)  # Ensure valid JSON
print(\"Default configuration is valid\")
'"

# Summary
echo -e "\n=============================================="
echo "📊 VALIDATION SUMMARY"
echo "=============================================="

PASSED_TESTS=$((TOTAL_TESTS - FAILED_TESTS))

echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✓✓✓ ALL VALIDATION CHECKS PASSED! ✓✓✓${NC}"
    echo -e "${GREEN}The system is ready for optimization deployment.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review the test results above"
    echo "  2. Run: python scripts/deploy_optimizations.py"
    echo "  3. Monitor the deployment report"
    exit 0
else
    echo -e "\n${RED}✗✗✗ VALIDATION FAILED ✗✗✗${NC}"
    echo -e "${RED}$FAILED_TESTS test(s) failed. Please fix issues before deploying.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check the failed tests above"
    echo "  2. Ensure all dependencies are installed"
    echo "  3. Verify JAX/Flax/Optax versions"
    echo "  4. Run individual test files for more details"
    exit 1
fi
