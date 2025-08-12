# Experiments Overview

This document provides a technical overview of the four Kill Switch (KS) experiments designed to test the capacity-time dilation hypothesis.

## KS-1: Clock Quadratic Test

**Objective**: Verify the fundamental small-delta law for quantum distinguishability.

**Method**: 
- Create quantum "pointer states" |+⟩ → Rz(θ)|+⟩
- Measure Bures distance squared DB²(θ) for small θ
- Compute ratio DB²/θ² and verify it approaches a constant

**Expected Result**: DB²/θ² ≈ 0.25 (theoretical value for this pointer state)

**Pass Criterion**: 
- Ratio variation < 20% for θ ∈ [0.01, 0.2] rad
- Mean ratio in range [0.15, 0.4]
- Both pure and noisy states show quadratic behavior

**Significance**: Establishes quantum Fisher Information scaling essential for all subsequent experiments.

---

## KS-2: Motion → Demand Test

**Objective**: Demonstrate that quantum information transport creates higher computational demand than static storage.

**Method**:
- Create 6-qubit lattice with three evolution patterns:
  - **Moving**: Localized rotation translates across lattice
  - **Stationary**: Localized rotation remains at center  
  - **Uniform**: Equal rotation on all qubits (control)
- Estimate local demand D using generator variance method
- Compare demand profiles: D_moving vs D_stationary

**Expected Result**: Moving patterns show systematically higher demand in active regions.

**Pass Criterion**:
- Peak demand ratio (moving/stationary) > 1.5
- Center region ratio > 1.2
- Clear spatial correlation with pattern activity

**Significance**: Establishes that motion creates computational demand, prerequisite for time dilation effects.

---

## KS-3: Back-Reaction Toy Model

**Objective**: Demonstrate observable time dilation when demand D feeds back into evolution rate via N = C/D.

**Method**:
- Implement feedback loop: measure D → compute N = C/D → scale gate angles by N → evolve → repeat
- Compare back-reaction evolution vs control (fixed schedule)
- Monitor observable trajectories (e.g., ⟨Z⟩ on probe qubit)
- Measure trajectory divergence between feedback and control runs

**Expected Result**: Back-reaction creates 10-30% slower evolution in high-demand regions.

**Pass Criterion**:
- Maximum trajectory divergence > 0.1
- Final state difference > 0.05
- At least one Pauli observable shows clear effect
- Measurable time dilation (N_min < 0.7)

**Significance**: Core test of the capacity-time dilation mechanism.

---

## KS-4: Response Kernel Mapping

**Objective**: Map spatial structure of time dilation effects via response kernels G(r).

**Method**:
- Create local demand "bump" at site j (extra gate depth, rotation, or entangling gates)
- Measure response Δln(N) at all other sites
- Extract empirical kernel G(r) = Δln(N)(r) / Δbump_strength
- Test spatial properties: symmetry G(+r) ≈ G(-r) and locality G(r) → 0

**Expected Result**: Response kernels show field-like spatial structure with decay and symmetry.

**Pass Criterion**:
- Detectable response |G(r)| > 0.01 at nearby sites
- Symmetry: |G(+r) - G(-r)| < 0.1 for symmetric pairs
- Locality: G(r) decays with distance, remote response < 0.1  
- Consistency across perturbation types and positions

**Significance**: Tests whether capacity-time dilation exhibits spatial correlations similar to physical fields.

---

## Experimental Design Principles

### 1. Emulator-Only Approach
- All experiments run on IBM Qiskit Aer (classical simulation)
- No real quantum hardware required
- Focus on testing theoretical logic, not hardware limitations

### 2. Kill Switch Methodology  
- Each experiment has explicit pass/fail criteria
- No post-hoc parameter adjustment
- Clear numerical thresholds that must be met
- If any KS fails, theory requires major revision

### 3. Progressive Validation
- KS-1 establishes mathematical foundations
- KS-2 connects motion to demand
- KS-3 demonstrates demand → time dilation
- KS-4 maps spatial structure

### 4. Multiple Controls
- Compare moving vs stationary vs uniform patterns
- Test different perturbation types and positions
- Vary feedback strengths and system sizes
- Multiple observables (X, Y, Z) for robustness

### 5. Reproducibility
- Fixed random seeds where applicable
- Documented parameter choices
- Complete code and data availability
- Automated test suite

---

## Integration with Broader Theory

These experiments test specific aspects of a larger theoretical framework:

**Information-Geometric Foundation**: Quantum states have intrinsic computational complexity that affects spacetime geometry.

**Emergent Spacetime**: Time flow emerges from computational limitations rather than being fundamental.

**Holographic Duality**: Bulk spacetime properties arise from boundary computational processes.

**Quantum Gravity Connection**: Information-theoretic effects could explain how quantum matter creates gravitational fields.

The KS experiments isolate testable predictions of this framework while remaining agnostic about its ultimate interpretation.
