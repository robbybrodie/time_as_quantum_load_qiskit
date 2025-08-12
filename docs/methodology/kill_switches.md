# Kill Switch Criteria

This document defines the explicit pass/fail criteria for each experiment. If any kill switch is triggered (test fails), the capacity-time dilation hypothesis requires major theoretical revision.

## Philosophy

**No Post-Hoc Explanations**: Numbers either meet criteria or they don't. No parameter tweaking, no "almost passed", no explanations after seeing results.

**Objective Thresholds**: All criteria are defined before running experiments with specific numerical bounds.

**Binary Outcomes**: Each test is either PASS or FAIL - no ambiguous middle ground.

---

## KS-1: Clock Quadratic Test

### Primary Kill Switch
**FAIL if**: DB²/θ² ratio shows > 20% relative standard deviation for small angles

### Detailed Criteria

#### Pure State Test
- **θ range**: [0.01, 0.2] radians, 20 points
- **Expected ratio**: ~0.25 for pointer state |+⟩ → Rz(θ)|+⟩
- **PASS if**: 
  - Mean ratio ∈ [0.15, 0.4]
  - Relative std dev < 0.2
  - No individual ratio deviates > 50% from mean
- **FAIL if**: Any condition above violated

#### Noisy State Test  
- **Noise model**: T₁ = 50μs, T₂ = 30μs
- **PASS if**: 
  - Maintains quadratic behavior (relative std dev < 0.3)
  - Mean ratio > 0.1 (noise reduces but doesn't eliminate effect)
- **FAIL if**: No clear quadratic trend or ratio < 0.05

### Interpretation
- **PASS**: Quantum clocks have correct small-delta behavior → proceed to KS-2
- **FAIL**: Fundamental mathematical assumptions wrong → revise theory

---

## KS-2: Motion → Demand Test

### Primary Kill Switch
**FAIL if**: Moving patterns do NOT show systematically higher demand than stationary patterns

### Detailed Criteria

#### Demand Comparison
- **Peak demand ratio**: D_max(moving) / D_max(stationary)
  - **PASS if**: > 1.5
  - **FAIL if**: ≤ 1.5
  
- **Center region ratio**: Mean demand in center qubits
  - **PASS if**: > 1.2
  - **FAIL if**: ≤ 1.2

#### Spatial Correlation
- **PASS if**: Demand enhancement correlates with pattern activity regions
- **FAIL if**: No spatial correlation visible

#### Multi-Pattern Validation
- **PASS if**: Moving > Stationary > Uniform in at least 70% of measurements
- **FAIL if**: No clear ordering or inverted relationships

### Interpretation
- **PASS**: Motion creates computational demand → proceed to KS-3
- **FAIL**: No connection between information transport and demand → revise demand model

---

## KS-3: Back-Reaction Test

### Primary Kill Switch
**FAIL if**: Back-reaction creates no measurable difference vs control evolution

### Detailed Criteria

#### Observable Divergence
- **Maximum trajectory divergence**: max|⟨O⟩_feedback - ⟨O⟩_control|
  - **PASS if**: > 0.1
  - **FAIL if**: ≤ 0.1

- **Final state difference**: |⟨O⟩_final_feedback - ⟨O⟩_final_control|
  - **PASS if**: > 0.05
  - **FAIL if**: ≤ 0.05

#### Time Dilation Evidence
- **Minimum time factor**: min(N) across all sites and times
  - **PASS if**: < 0.7 (significant slowdown)
  - **FAIL if**: ≥ 0.9 (minimal effect)

- **Time factor variation**: max(N)/min(N)
  - **PASS if**: > 2.0 (clear spatial variation)
  - **FAIL if**: < 1.5 (too uniform)

#### Multi-Observable Consistency
- **PASS if**: ≥ 1 Pauli observable (X, Y, Z) shows clear effect
- **FAIL if**: No observable shows significant divergence

#### Feedback Scaling
- **PASS if**: Stronger feedback creates larger effects (monotonic relationship)
- **FAIL if**: No scaling or inverted relationship

### Interpretation
- **PASS**: Demand creates time dilation → proceed to KS-4  
- **FAIL**: Back-reaction mechanism doesn't work → revise feedback model

---

## KS-4: Response Kernel Test

### Primary Kill Switch
**FAIL if**: Response kernels show no spatial structure (symmetry + locality)

### Detailed Criteria

#### Detectability
- **Maximum response**: max|G(r)| across all distances
  - **PASS if**: > 0.01
  - **FAIL if**: ≤ 0.01

#### Symmetry Test
- **Mean asymmetry**: Mean of |G(+r) - G(-r)| for symmetric pairs
  - **PASS if**: < 0.1
  - **FAIL if**: ≥ 0.1

- **Maximum asymmetry**: Max of |G(+r) - G(-r)| for symmetric pairs  
  - **PASS if**: < 0.2
  - **FAIL if**: ≥ 0.2

#### Locality Test  
- **Remote response**: |G(r)| at maximum distance
  - **PASS if**: < 0.1 (response decays)
  - **FAIL if**: ≥ 0.1 (no decay)

- **Decay trend**: G_far/G_near ratio
  - **PASS if**: < 0.5
  - **FAIL if**: ≥ 0.5

#### Cross-Validation
- **Perturbation types**: Test gate_depth, rotation, entangling
  - **PASS if**: ≥ 1 type shows structure
  - **FAIL if**: No type shows clear structure

- **Position validation**: Test multiple bump positions
  - **PASS if**: ≥ 50% positions show structure
  - **FAIL if**: < 50% positions work

### Interpretation
- **PASS**: Time dilation has field-like spatial structure → theory validated
- **FAIL**: No spatial correlations → revise response theory

---

## Overall Project Kill Switches

### Sequential Dependencies
- **If KS-1 FAILS**: Stop immediately, revise fundamental assumptions
- **If KS-2 FAILS**: Motion-demand connection wrong, revise before KS-3
- **If KS-3 FAILS**: Core mechanism broken, major theory revision needed
- **If KS-4 FAILS**: Spatial structure assumptions wrong

### Global Success Criteria
**COMPLETE SUCCESS**: All 4 KS tests PASS
**PARTIAL SUCCESS**: KS-1,2 PASS (motion creates demand) but KS-3,4 FAIL (no time dilation)
**FUNDAMENTAL FAILURE**: KS-1 FAILS (wrong mathematical foundation)

### Interpretation Guidelines

#### If ALL Tests Pass
- Strong evidence for capacity-time dilation in quantum systems
- Computational demand affects time evolution rates
- Effects have proper spatial structure
- Framework ready for more complex applications

#### If Some Tests Fail
- **KS-1 only**: Mathematical foundations need work
- **KS-2 only**: Demand estimation methods need revision  
- **KS-3 only**: Feedback mechanism needs different approach
- **KS-4 only**: Spatial assumptions need reconsideration

#### If Most/All Tests Fail
- Core hypothesis likely incorrect
- May need entirely different theoretical approach
- Could indicate fundamental misunderstanding of information-spacetime relationship

---

## Execution Protocol

1. **Pre-execution**: Review all criteria, ensure understanding of thresholds
2. **During execution**: No parameter changes, no threshold adjustments
3. **Post-execution**: Apply criteria mechanically, report PASS/FAIL objectively
4. **If FAIL**: Document exactly which criterion failed and by how much
5. **Documentation**: Record all results in `docs/results_log.md` with timestamp

**The numbers decide. We follow where they lead.**
