# Gap Analysis: Auto Optimization

> **Feature**: Auto Weight Optimization
> **Date**: 2026-02-18
> **Status**: ⚠️ Significant Gaps Detected (< 90%)
> **Agent**: gap-detector (simulated)

## 1. Summary
The implementation of the Auto Optimization feature (`auto_optimize.py`) has diverged significantly from the initial design (`docs/02-design/features/auto-optimization.md`). While the core functionality exists, the algorithmic approach and several key requirements (Weighted Moving Average, Exploration vs Exploitation) are missing or implemented differently.

## 2. Detailed Gaps

### A. Algorithm Deviation (Critical)
- **Design**: Suggests a simple score-based redistribution (Softmax or Ratio) based on recent performance.
- **Implementation**: Uses a complex **Genetic Algorithm** with a `Vectorized Optimization Cache`.
- **Impact**: The genetic algorithm is more powerful but harder to tune and slower. The design document needs to be updated to reflect this change.

### B. Missing Features (Major)
1.  **Weighted Moving Average**:
    - **Requirement**: The design specifies using a weighted average of recent 5, 10, and 20 weeks to prioritize trends.
    - **Current State**: `OptimizationCache.evaluate_weights` uses a simple arithmetic mean over the entire test window.
    - **Action**: Implement the weighted moving average logic in `OptimizationCache`.

2.  **Exploration vs Exploitation (Epsilon)**:
    - **Requirement**: A minimum weight (Epsilon) should be enforced to prevent engines from being completely discarded.
    - **Current State**: The Genetic Algorithm can drive weights to near-zero, potentially permanently disabling engines.
    - **Action**: Add a `min_weight` constraint in the normalization step.

### C. Redundancy & Inconsistency (Moderate)
- **Code Duplication**: `auto_optimize.py` and `train_1000.py` share significant logic.
- **Weight Application**: Multiple methods exist for applying weights, leading to potential conflicts.
- **Default Parameters**: Design says 100 rounds, code defaults to 200.

### D. Unplanned Features (Positive)
- **Dynamic Boost**: `EnsemblePredictor` implements a runtime dynamic adjustment based on the last 10 rounds. This is a good feature but undocumented.

## 3. Recommendations
1.  **Update Design**: Revise `docs/02-design/features/auto-optimization.md` to document the Genetic Algorithm and Dynamic Boost.
2.  **Refactor Code**:
    - Implement the **Weighted Moving Average** in `OptimizationCache`.
    - Add **Minimum Weight Constraint** (Epsilon) to the Genetic Algorithm.
    - Merge `train_1000.py` logic into `auto_optimize.py` or a shared utility.
3.  **Proceed to Iteration**: Run `/pdca iterate auto-optimization` to implement these fixes.
