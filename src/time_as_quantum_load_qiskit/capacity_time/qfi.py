"""
Quantum Fisher Information and Fidelity Utilities

This module provides tools to compute fidelity, Bures distance, and test
the small-delta law for quantum states. Critical for KS-1 experiments.

References:
- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- Helstrom, "Quantum Detection and Estimation Theory" 
- Paris, "Quantum estimation for quantum technology"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from typing import Union, List, Tuple, Callable
import scipy.linalg


def fidelity(state1: Union[Statevector, DensityMatrix, np.ndarray], 
             state2: Union[Statevector, DensityMatrix, np.ndarray]) -> float:
    """
    Compute quantum fidelity F(ρ,σ) between two quantum states.
    
    For pure states |ψ⟩, |φ⟩: F = |⟨ψ|φ⟩|²
    For mixed states ρ, σ: F = Tr(√(√ρ σ √ρ))²
    
    Args:
        state1, state2: Quantum states (Statevector, DensityMatrix, or numpy arrays)
        
    Returns:
        Fidelity value F ∈ [0,1]
    """
    # Convert to Qiskit objects if needed
    if isinstance(state1, np.ndarray):
        if state1.ndim == 1:  # Pure state vector
            state1 = Statevector(state1)
        else:  # Density matrix
            state1 = DensityMatrix(state1)
    
    if isinstance(state2, np.ndarray):
        if state2.ndim == 1:
            state2 = Statevector(state2)
        else:
            state2 = DensityMatrix(state2)
    
    return state_fidelity(state1, state2)


def bures_distance_squared(state1: Union[Statevector, DensityMatrix, np.ndarray],
                          state2: Union[Statevector, DensityMatrix, np.ndarray]) -> float:
    """
    Compute squared Bures distance: D_B² = 2(1 - √F)
    
    For pure states, this simplifies to D_B² = 2(1 - |⟨ψ|φ⟩|)
    
    Args:
        state1, state2: Quantum states
        
    Returns:
        Squared Bures distance D_B² ≥ 0
    """
    F = fidelity(state1, state2)
    return 2.0 * (1.0 - np.sqrt(F))


def bures_small_delta(theta_list: List[float], 
                     state_builder: Callable[[float], Union[Statevector, DensityMatrix]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test the small-delta law: D_B²(θ) ≈ C θ² for small θ
    
    Args:
        theta_list: List of small parameter values
        state_builder: Function θ → quantum_state(θ)
        
    Returns:
        Tuple of (theta_array, DB2_array, ratio_array) where
        ratio_array = DB2_array / theta_array²
    """
    theta_array = np.array(theta_list)
    DB2_array = np.zeros_like(theta_array)
    
    # Reference state at θ=0
    ref_state = state_builder(0.0)
    
    # Compute D_B² for each θ
    for i, theta in enumerate(theta_array):
        if theta == 0.0:
            DB2_array[i] = 0.0
        else:
            test_state = state_builder(theta)
            DB2_array[i] = bures_distance_squared(ref_state, test_state)
    
    # Compute ratio DB²/θ² (handle θ=0 case)
    ratio_array = np.zeros_like(theta_array)
    nonzero_mask = theta_array != 0
    ratio_array[nonzero_mask] = DB2_array[nonzero_mask] / (theta_array[nonzero_mask]**2)
    
    # For θ=0, estimate ratio via finite difference
    if len(theta_array) > 1 and theta_array[0] == 0.0:
        ratio_array[0] = ratio_array[1]  # Use next point as estimate
    
    return theta_array, DB2_array, ratio_array


def pure_state_qfi_generator(state: Statevector, generator: np.ndarray) -> float:
    """
    Compute Quantum Fisher Information for pure state under generator G:
    QFI = 4 * (⟨∂ψ|∂ψ⟩ - |⟨ψ|∂ψ⟩|²)
    
    For unitary evolution U(θ) = exp(-iθG), we have |∂ψ⟩ = -iG|ψ⟩
    
    Args:
        state: Pure quantum state |ψ⟩
        generator: Hermitian generator G (numpy array)
        
    Returns:
        Quantum Fisher Information
    """
    psi = state.data
    G_psi = generator @ psi
    
    # ⟨∂ψ|∂ψ⟩ = ⟨ψ|G†G|ψ⟩ = ⟨ψ|G²|ψ⟩ (since G is Hermitian)
    term1 = np.real(np.conj(psi) @ (generator @ G_psi))
    
    # |⟨ψ|∂ψ⟩|² = |⟨ψ|-iG|ψ⟩|² = |⟨ψ|G|ψ⟩|²
    expectation_G = np.conj(psi) @ G_psi
    term2 = np.real(expectation_G * np.conj(expectation_G))
    
    return 4.0 * (term1 - term2)


def estimate_demand_via_variance(state: Union[Statevector, DensityMatrix], 
                                local_generator: np.ndarray,
                                hbar: float = 1.0) -> float:
    """
    Estimate local demand D via generator variance: D ≈ 4 * Var(H_local) / ħ²
    
    This provides a proxy for the "computational cost" to maintain the state
    under local dynamics.
    
    Args:
        state: Quantum state (pure or mixed)
        local_generator: Local Hamiltonian/generator
        hbar: Reduced Planck constant (set to 1 for natural units)
        
    Returns:
        Demand estimate D ≥ 0
    """
    if isinstance(state, Statevector):
        # Pure state: Var(H) = ⟨H²⟩ - ⟨H⟩²
        psi = state.data
        expectation_H = np.real(np.conj(psi) @ (local_generator @ psi))
        expectation_H2 = np.real(np.conj(psi) @ (local_generator @ (local_generator @ psi)))
        variance = expectation_H2 - expectation_H**2
    else:
        # Mixed state: use density matrix formulation
        rho = state.data if hasattr(state, 'data') else state
        expectation_H = np.real(np.trace(rho @ local_generator))
        expectation_H2 = np.real(np.trace(rho @ (local_generator @ local_generator)))
        variance = expectation_H2 - expectation_H**2
    
    # Ensure non-negative (numerical precision issues)
    variance = max(0.0, variance)
    
    return 4.0 * variance / (hbar**2)


# Pauli matrices for common generators
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_I = np.eye(2, dtype=complex)

# Useful composite generators
def tensor_product_generator(local_generators: List[np.ndarray], target_qubit: int, n_qubits: int) -> np.ndarray:
    """
    Build tensor product generator acting on specific qubit in multi-qubit system.
    
    Args:
        local_generators: List of single-qubit generators
        target_qubit: Which qubit to act on (0-indexed)
        n_qubits: Total number of qubits
        
    Returns:
        Full system generator as numpy array
    """
    if len(local_generators) == 1:
        # Single generator - replicate across system
        generators = [PAULI_I] * n_qubits
        generators[target_qubit] = local_generators[0]
    else:
        # Multiple generators provided
        generators = local_generators
        
    result = generators[0]
    for i in range(1, len(generators)):
        result = np.kron(result, generators[i])
        
    return result
