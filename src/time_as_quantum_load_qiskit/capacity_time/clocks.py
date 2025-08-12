"""
Quantum Clock States and KS-1 Test Utilities

This module implements simple quantum "clocks" - states that evolve with
a parameter θ and exhibit the quadratic small-delta law. Essential for KS-1.

References:
- Giovannetti, Lloyd, Maccone "Quantum-enhanced measurements: beating the standard quantum limit"
- Holevo "Probabilistic and Statistical Aspects of Quantum Theory"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator, noise
from typing import Union, List, Tuple, Optional
import matplotlib.pyplot as plt
from .qfi import bures_small_delta, bures_distance_squared


def pointer_state(theta: float) -> Statevector:
    """
    Create a simple pointer state |+⟩ → Rz(theta) |+⟩
    
    This is a minimal quantum clock: |ψ(θ)⟩ = Rz(θ)|+⟩
    Expected: DB²(θ)/θ² → 0.25 as θ → 0
    
    Args:
        theta: Rotation angle parameter
        
    Returns:
        Evolved state vector
    """
    # Create circuit: |+⟩ state then Rz rotation
    qc = QuantumCircuit(1)
    qc.h(0)  # |+⟩ = (|0⟩ + |1⟩)/√2
    qc.rz(theta, 0)  # Rz(θ) rotation
    
    # Execute and return state
    simulator = AerSimulator(method='statevector')
    result = simulator.run(qc, shots=1).result()
    return result.get_statevector()


def noisy_pointer_state(theta: float, 
                       T1: Optional[float] = None, 
                       T2: Optional[float] = None,
                       gate_time: float = 1e-6) -> DensityMatrix:
    """
    Create pointer state with realistic noise (amplitude damping + dephasing)
    
    Args:
        theta: Rotation angle parameter
        T1: Amplitude damping time (seconds), None = no amplitude damping
        T2: Dephasing time (seconds), None = no dephasing  
        gate_time: Gate execution time (seconds)
        
    Returns:
        Mixed state density matrix
    """
    # Build base circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(theta, 0)
    
    # Add noise model if specified
    noise_model = noise.NoiseModel()
    
    if T1 is not None:
        # Amplitude damping: γ = 1 - exp(-gate_time/T1)
        gamma_ad = 1.0 - np.exp(-gate_time / T1)
        ad_error = noise.amplitude_damping_error(gamma_ad)
        noise_model.add_all_qubit_quantum_error(ad_error, ['h', 'rz'])
    
    if T2 is not None:
        # Pure dephasing: γ = 1 - exp(-gate_time/T2)  
        gamma_ph = 1.0 - np.exp(-gate_time / T2)
        ph_error = noise.phase_damping_error(gamma_ph)
        noise_model.add_all_qubit_quantum_error(ph_error, ['h', 'rz'])
    
    # Execute with noise
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    result = simulator.run(qc, shots=1).result()
    return result.get_statevector().to_density_matrix()


def two_qubit_clock(theta: float, coupling_strength: float = 0.1) -> Statevector:
    """
    Two-qubit entangled clock state for more complex demand patterns
    
    Creates |ψ(θ)⟩ = Rz(θ) ⊗ I + coupling * (XX + YY)
    
    Args:
        theta: Primary rotation parameter
        coupling_strength: Entanglement strength
        
    Returns:
        Two-qubit state vector
    """
    qc = QuantumCircuit(2)
    
    # Initial |++⟩ state
    qc.h([0, 1])
    
    # Primary evolution
    qc.rz(theta, 0)
    
    # Entangling interaction (approximate XX + YY coupling)
    if coupling_strength > 0:
        qc.ry(coupling_strength, 0)
        qc.cx(0, 1)
        qc.ry(-coupling_strength, 1)
        qc.cx(0, 1)
        qc.ry(coupling_strength, 1)
    
    simulator = AerSimulator(method='statevector')
    result = simulator.run(qc, shots=1).result()
    return result.get_statevector()


def test_ks1_pure_state(theta_max: float = 0.2, 
                       n_points: int = 20, 
                       plot: bool = True) -> Tuple[np.ndarray, float, bool]:
    """
    Run KS-1 test for pure pointer state
    
    Tests: DB²(θ)/θ² ≈ constant for small θ → quadratic behavior
    
    Args:
        theta_max: Maximum angle to test
        n_points: Number of test points
        plot: Whether to generate plots
        
    Returns:
        Tuple of (ratio_values, mean_ratio, pass_test)
        where pass_test indicates if KS-1 criterion is met
    """
    # Generate test angles (skip θ=0 for ratio calculation)
    theta_list = np.linspace(0.01, theta_max, n_points)
    
    # Test small-delta behavior
    theta_arr, DB2_arr, ratio_arr = bures_small_delta(
        theta_list, 
        lambda theta: pointer_state(theta)
    )
    
    # Check consistency of ratio (should be ~constant)
    mean_ratio = np.mean(ratio_arr)
    std_ratio = np.std(ratio_arr)
    relative_std = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')
    
    # Pass criterion: relative standard deviation < 10%
    pass_test = relative_std < 0.1 and mean_ratio > 0.1
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: DB² vs θ²
        ax1.plot(theta_arr**2, DB2_arr, 'bo-', markersize=4)
        ax1.set_xlabel('θ² (radians²)')
        ax1.set_ylabel('D_B² (Bures distance²)')
        ax1.set_title('KS-1: Bures Distance² vs θ²')
        ax1.grid(True, alpha=0.3)
        
        # Fit line to check linearity
        if len(theta_arr) > 1:
            slope, intercept = np.polyfit(theta_arr**2, DB2_arr, 1)
            ax1.plot(theta_arr**2, slope * theta_arr**2 + intercept, 'r--', 
                    label=f'Linear fit: slope={slope:.3f}')
            ax1.legend()
        
        # Plot 2: Ratio DB²/θ²
        ax2.plot(theta_arr, ratio_arr, 'ro-', markersize=4)
        ax2.axhline(mean_ratio, color='blue', linestyle='--', 
                   label=f'Mean: {mean_ratio:.3f}±{std_ratio:.3f}')
        ax2.fill_between(theta_arr, mean_ratio - std_ratio, mean_ratio + std_ratio, 
                        alpha=0.2, color='blue')
        ax2.set_xlabel('θ (radians)')
        ax2.set_ylabel('D_B²/θ²')
        ax2.set_title(f'KS-1: Ratio Test (rel_std={relative_std:.3f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('figures/KS1_pure_state.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print results
        print(f"KS-1 Pure State Test Results:")
        print(f"Mean ratio DB²/θ²: {mean_ratio:.4f} ± {std_ratio:.4f}")
        print(f"Relative std dev: {relative_std:.3f}")
        print(f"Expected ratio ~0.25 for |+⟩ → Rz(θ)")
        print(f"KS-1 PASS: {pass_test}")
    
    return ratio_arr, mean_ratio, pass_test


def test_ks1_noisy_state(theta_max: float = 0.2,
                        n_points: int = 20,
                        T1: float = 50e-6,  # 50 μs
                        T2: float = 30e-6,  # 30 μs  
                        plot: bool = True) -> Tuple[np.ndarray, float, bool]:
    """
    Run KS-1 test for noisy pointer state
    
    Args:
        theta_max: Maximum angle to test  
        n_points: Number of test points
        T1, T2: Decoherence times
        plot: Whether to generate plots
        
    Returns:
        Tuple of (ratio_values, mean_ratio, pass_test)
    """
    theta_list = np.linspace(0.01, theta_max, n_points)
    
    # Test noisy state behavior
    theta_arr, DB2_arr, ratio_arr = bures_small_delta(
        theta_list,
        lambda theta: noisy_pointer_state(theta, T1=T1, T2=T2)
    )
    
    mean_ratio = np.mean(ratio_arr)
    std_ratio = np.std(ratio_arr) 
    relative_std = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')
    
    # Relaxed criterion for noisy case
    pass_test = relative_std < 0.2 and mean_ratio > 0.05
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(theta_arr**2, DB2_arr, 'go-', markersize=4)
        ax1.set_xlabel('θ² (radians²)')
        ax1.set_ylabel('D_B² (Bures distance²)')
        ax1.set_title(f'KS-1: Noisy State (T1={T1*1e6:.0f}μs, T2={T2*1e6:.0f}μs)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(theta_arr, ratio_arr, 'go-', markersize=4)
        ax2.axhline(mean_ratio, color='orange', linestyle='--',
                   label=f'Mean: {mean_ratio:.3f}±{std_ratio:.3f}')
        ax2.fill_between(theta_arr, mean_ratio - std_ratio, mean_ratio + std_ratio,
                        alpha=0.2, color='orange')
        ax2.set_xlabel('θ (radians)')
        ax2.set_ylabel('D_B²/θ²')
        ax2.set_title(f'Ratio (rel_std={relative_std:.3f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('figures/KS1_noisy_state.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"KS-1 Noisy State Test Results:")
        print(f"Mean ratio DB²/θ²: {mean_ratio:.4f} ± {std_ratio:.4f}")
        print(f"Relative std dev: {relative_std:.3f}")
        print(f"Noise parameters: T1={T1*1e6:.0f}μs, T2={T2*1e6:.0f}μs")
        print(f"KS-1 PASS: {pass_test}")
    
    return ratio_arr, mean_ratio, pass_test


def run_all_ks1_tests(save_figures: bool = True) -> dict:
    """
    Execute complete KS-1 test suite
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    print("=" * 60)
    print("KS-1: CLOCK QUADRATIC TEST SUITE")
    print("=" * 60)
    
    # Ensure figures directory exists
    if save_figures:
        import os
        os.makedirs('figures', exist_ok=True)
    
    # Test 1: Pure state
    print("\n1. Testing pure pointer state...")
    ratio_pure, mean_pure, pass_pure = test_ks1_pure_state(plot=save_figures)
    results['pure'] = {
        'ratios': ratio_pure,
        'mean_ratio': mean_pure, 
        'pass': pass_pure
    }
    
    # Test 2: Noisy state
    print("\n2. Testing noisy pointer state...")
    ratio_noisy, mean_noisy, pass_noisy = test_ks1_noisy_state(plot=save_figures)
    results['noisy'] = {
        'ratios': ratio_noisy,
        'mean_ratio': mean_noisy,
        'pass': pass_noisy
    }
    
    # Overall KS-1 verdict
    overall_pass = pass_pure and pass_noisy
    results['overall_pass'] = overall_pass
    
    print(f"\n" + "=" * 60)
    print(f"KS-1 OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Pure state: {'PASS' if pass_pure else 'FAIL'}")
    print(f"Noisy state: {'PASS' if pass_noisy else 'FAIL'}")
    print("=" * 60)
    
    return results
