"""
Demand Estimation and Capacity Models

This module implements demand proxies D, capacity models C, and the core
relationship N = C/D that controls local time dilation.

References:
- Holevo, "Quantum Systems, Channels, Information"
- Braunstein & Caves, "Statistical distance and the geometry of quantum states"
- Lloyd, "Ultimate physical limits to computation"
"""

import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from typing import Union, List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from .qfi import estimate_demand_via_variance, PAULI_X, PAULI_Y, PAULI_Z, PAULI_I


class DemandEstimator:
    """
    Estimates computational demand D for quantum states/subsystems
    
    Multiple methods available:
    - Variance-based: D ∝ Var(H_local)
    - Entanglement-based: D ∝ Von Neumann entropy of reduced state
    - Circuit-based: D ∝ gate depth/complexity
    """
    
    def __init__(self, method: str = 'variance'):
        """
        Initialize demand estimator
        
        Args:
            method: 'variance', 'entropy', 'circuit', or 'hybrid'
        """
        self.method = method.lower()
        valid_methods = ['variance', 'entropy', 'circuit', 'hybrid']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def estimate_local_demand(self, 
                             state: Union[Statevector, DensityMatrix],
                             subsystem: Union[int, List[int]],
                             generator: Optional[np.ndarray] = None) -> float:
        """
        Estimate demand for a local subsystem
        
        Args:
            state: Full system quantum state
            subsystem: Qubit index or list of indices for subsystem
            generator: Local Hamiltonian (default: sum of Pauli generators)
            
        Returns:
            Demand estimate D ≥ 0
        """
        if isinstance(subsystem, int):
            subsystem = [subsystem]
        
        if self.method == 'variance':
            return self._variance_demand(state, subsystem, generator)
        elif self.method == 'entropy':
            return self._entropy_demand(state, subsystem)
        elif self.method == 'circuit':
            return self._circuit_demand(state, subsystem)
        elif self.method == 'hybrid':
            # Weighted combination of methods
            d_var = self._variance_demand(state, subsystem, generator)
            d_ent = self._entropy_demand(state, subsystem)
            return 0.7 * d_var + 0.3 * d_ent
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _variance_demand(self, 
                        state: Union[Statevector, DensityMatrix],
                        subsystem: List[int],
                        generator: Optional[np.ndarray] = None) -> float:
        """Variance-based demand: D ∝ Var(H_local)"""
        
        # Get reduced state for subsystem
        if len(subsystem) == 1:
            # Single qubit - trace out others
            n_qubits = state.num_qubits
            other_qubits = [i for i in range(n_qubits) if i not in subsystem]
            
            if len(other_qubits) == 0:
                # Single-qubit system
                reduced_state = state
            else:
                reduced_state = partial_trace(state, other_qubits)
        else:
            # Multi-qubit subsystem  
            n_qubits = state.num_qubits
            other_qubits = [i for i in range(n_qubits) if i not in subsystem]
            
            if len(other_qubits) == 0:
                reduced_state = state
            else:
                reduced_state = partial_trace(state, other_qubits)
        
        # Default generator: sum of local Pauli matrices
        if generator is None:
            if len(subsystem) == 1:
                # Single qubit: H = σx + σy + σz
                generator = PAULI_X + PAULI_Y + PAULI_Z
            else:
                # Multi-qubit: sum of local terms
                dim = 2**len(subsystem)
                generator = np.zeros((dim, dim), dtype=complex)
                
                for i in range(len(subsystem)):
                    # Add Pauli-X, Y, Z on qubit i
                    pauli_ops = [PAULI_X, PAULI_Y, PAULI_Z]
                    for pauli in pauli_ops:
                        op_list = [PAULI_I] * len(subsystem)
                        op_list[i] = pauli
                        
                        full_op = op_list[0]
                        for j in range(1, len(op_list)):
                            full_op = np.kron(full_op, op_list[j])
                        
                        generator += full_op
        
        # Estimate demand via variance
        return estimate_demand_via_variance(reduced_state, generator)
    
    def _entropy_demand(self, 
                       state: Union[Statevector, DensityMatrix],
                       subsystem: List[int]) -> float:
        """Entanglement-based demand: D ∝ S(ρ_local)"""
        
        # Get reduced state
        n_qubits = state.num_qubits
        other_qubits = [i for i in range(n_qubits) if i not in subsystem]
        
        if len(other_qubits) == 0:
            reduced_state = state
        else:
            reduced_state = partial_trace(state, other_qubits)
        
        # Von Neumann entropy
        if isinstance(reduced_state, Statevector):
            # Pure state has zero entropy locally (unless entangled)
            if len(other_qubits) == 0:
                return 0.0
            else:
                # Convert to density matrix for entropy calculation
                rho = DensityMatrix(reduced_state)
                eigenvals = np.linalg.eigvals(rho.data)
        else:
            # Mixed state
            eigenvals = np.linalg.eigvals(reduced_state.data)
        
        # S = -Tr(ρ log ρ) = -Σ λᵢ log λᵢ
        eigenvals = eigenvals.real
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
        
        # Scale to match variance-based units (heuristic)
        return 2.0 * entropy
    
    def _circuit_demand(self, 
                       state: Union[Statevector, DensityMatrix],
                       subsystem: List[int]) -> float:
        """Circuit complexity proxy (simplified for emulator)"""
        
        # For emulator: use state complexity as proxy
        # More entangled → higher demand
        n_qubits = state.num_qubits
        other_qubits = [i for i in range(n_qubits) if i not in subsystem]
        
        if len(other_qubits) == 0:
            # No entanglement possible
            return 0.1  # Minimal demand
        
        # Use purity as complexity proxy: Tr(ρ²) 
        # Pure states have Tr(ρ²)=1, maximally mixed have Tr(ρ²)=1/d
        reduced_state = partial_trace(state, other_qubits)
        
        if isinstance(reduced_state, Statevector):
            purity = 1.0  # Pure state
        else:
            rho = reduced_state.data
            purity = np.real(np.trace(rho @ rho))
        
        # Demand inversely related to purity
        max_purity = 1.0
        min_purity = 1.0 / (2**len(subsystem))  # Maximally mixed
        
        normalized_impurity = (max_purity - purity) / (max_purity - min_purity)
        
        # Scale to reasonable demand units
        return 1.0 + 3.0 * normalized_impurity


class CapacityModel:
    """
    Model for computational capacity C
    
    In emulator, we keep C simple and fixed to isolate N = C/D effects
    """
    
    def __init__(self, base_capacity: float = 1.0, model_type: str = 'fixed'):
        """
        Initialize capacity model
        
        Args:
            base_capacity: Base capacity value
            model_type: 'fixed', 'site_dependent', or 'dynamic'
        """
        self.base_capacity = base_capacity
        self.model_type = model_type.lower()
    
    def get_capacity(self, 
                    site: Optional[int] = None,
                    time_step: Optional[int] = None,
                    system_state: Optional[Union[Statevector, DensityMatrix]] = None) -> float:
        """
        Get capacity value for given context
        
        Args:
            site: Spatial location (for site-dependent models)
            time_step: Time index (for dynamic models)  
            system_state: Current quantum state (for adaptive models)
            
        Returns:
            Capacity value C > 0
        """
        if self.model_type == 'fixed':
            return self.base_capacity
        
        elif self.model_type == 'site_dependent':
            if site is None:
                return self.base_capacity
            # Example: boundary sites have lower capacity
            if site == 0 or site is None:  # Handle edge cases
                return 0.8 * self.base_capacity
            else:
                return self.base_capacity
        
        elif self.model_type == 'dynamic':
            if time_step is None:
                return self.base_capacity
            # Example: capacity decreases over time (resource depletion)
            decay_rate = 0.01
            return self.base_capacity * np.exp(-decay_rate * time_step)
        
        else:
            return self.base_capacity


def compute_time_factor(demand: float, 
                       capacity: float,
                       min_factor: float = 0.1,
                       max_factor: float = 10.0) -> float:
    """
    Compute time dilation factor N = C/D with clipping
    
    Args:
        demand: Demand estimate D
        capacity: Capacity value C
        min_factor: Minimum allowed N (avoid division by zero)
        max_factor: Maximum allowed N (stability)
        
    Returns:
        Time factor N = C/D (clipped)
    """
    if demand <= 0:
        return max_factor
    
    N = capacity / demand
    
    # Clip to stable range
    N = np.clip(N, min_factor, max_factor)
    
    return N


def compute_demand_profile(states: List[Union[Statevector, DensityMatrix]],
                          demand_estimator: DemandEstimator,
                          n_qubits: int) -> np.ndarray:
    """
    Compute demand D across all sites and time steps
    
    Args:
        states: List of quantum states (time evolution)
        demand_estimator: DemandEstimator instance
        n_qubits: Number of qubits in system
        
    Returns:
        Array of shape [n_steps, n_qubits] with demand values
    """
    n_steps = len(states)
    demands = np.zeros((n_steps, n_qubits))
    
    for t, state in enumerate(states):
        for i in range(n_qubits):
            demands[t, i] = demand_estimator.estimate_local_demand(state, i)
    
    return demands


def compute_capacity_profile(n_steps: int,
                           n_qubits: int, 
                           capacity_model: CapacityModel) -> np.ndarray:
    """
    Compute capacity C across all sites and time steps
    
    Args:
        n_steps: Number of time steps
        n_qubits: Number of qubits
        capacity_model: CapacityModel instance
        
    Returns:
        Array of shape [n_steps, n_qubits] with capacity values
    """
    capacities = np.zeros((n_steps, n_qubits))
    
    for t in range(n_steps):
        for i in range(n_qubits):
            capacities[t, i] = capacity_model.get_capacity(site=i, time_step=t)
    
    return capacities


def compute_time_factor_profile(demands: np.ndarray,
                               capacities: np.ndarray,
                               min_factor: float = 0.1,
                               max_factor: float = 10.0) -> np.ndarray:
    """
    Compute time factor N = C/D profiles
    
    Args:
        demands: Demand array [n_steps, n_qubits]
        capacities: Capacity array [n_steps, n_qubits]
        min_factor, max_factor: Clipping bounds
        
    Returns:
        Time factor array N [n_steps, n_qubits]
    """
    # Avoid division by zero
    demands_safe = np.maximum(demands, 1e-8)
    
    N = capacities / demands_safe
    
    # Apply clipping
    N = np.clip(N, min_factor, max_factor)
    
    return N


def plot_demand_capacity_analysis(demands: np.ndarray,
                                 capacities: np.ndarray, 
                                 time_factors: np.ndarray,
                                 title: str = "Demand-Capacity Analysis",
                                 save_path: Optional[str] = None) -> None:
    """
    Visualize demand, capacity, and time factor profiles
    
    Args:
        demands: Demand array [n_steps, n_qubits]
        capacities: Capacity array [n_steps, n_qubits]
        time_factors: Time factor array [n_steps, n_qubits]
        title: Plot title
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Demand profile
    im1 = axes[0, 0].imshow(demands.T, aspect='auto', origin='lower', 
                           cmap='Reds', interpolation='nearest')
    axes[0, 0].set_title('Demand D(t, site)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Qubit Index')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Capacity profile  
    im2 = axes[0, 1].imshow(capacities.T, aspect='auto', origin='lower',
                           cmap='Blues', interpolation='nearest')
    axes[0, 1].set_title('Capacity C(t, site)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Qubit Index') 
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Time factor N = C/D
    im3 = axes[1, 0].imshow(time_factors.T, aspect='auto', origin='lower',
                           cmap='RdBu_r', vmin=0.5, vmax=2.0, interpolation='nearest')
    axes[1, 0].set_title('Time Factor N = C/D')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Qubit Index')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot 4: Statistics over time
    mean_demands = np.mean(demands, axis=1)
    mean_capacities = np.mean(capacities, axis=1)
    mean_time_factors = np.mean(time_factors, axis=1)
    
    time_steps = np.arange(len(mean_demands))
    axes[1, 1].plot(time_steps, mean_demands, 'r-', label='Mean Demand', linewidth=2)
    axes[1, 1].plot(time_steps, mean_capacities, 'b-', label='Mean Capacity', linewidth=2)
    axes[1, 1].plot(time_steps, mean_time_factors, 'g-', label='Mean N', linewidth=2)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Temporal Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_demand_vs_motion(moving_states: List[Union[Statevector, DensityMatrix]],
                           stationary_states: List[Union[Statevector, DensityMatrix]], 
                           demand_estimator: DemandEstimator,
                           n_qubits: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Compare demand profiles for moving vs stationary patterns (KS-2 helper)
    
    Args:
        moving_states: States from moving pattern evolution
        stationary_states: States from stationary pattern evolution
        demand_estimator: DemandEstimator instance
        n_qubits: Number of qubits
        
    Returns:
        Tuple of (moving_demands, stationary_demands, ks2_pass)
    """
    # Compute demand profiles
    D_moving = compute_demand_profile(moving_states, demand_estimator, n_qubits)
    D_stationary = compute_demand_profile(stationary_states, demand_estimator, n_qubits)
    
    # KS-2 criterion: moving pattern should show higher peak demand
    max_moving = np.max(D_moving)
    max_stationary = np.max(D_stationary)
    
    # Also check mean demand in active region (middle qubits)
    center_qubits = list(range(n_qubits//4, 3*n_qubits//4))
    if len(center_qubits) == 0:
        center_qubits = [n_qubits//2]
    
    mean_moving_center = np.mean(D_moving[:, center_qubits])
    mean_stationary_center = np.mean(D_stationary[:, center_qubits])
    
    # Pass criteria: moving > stationary by significant margin  
    peak_ratio = max_moving / (max_stationary + 1e-8)
    mean_ratio = mean_moving_center / (mean_stationary_center + 1e-8)
    
    ks2_pass = (peak_ratio > 1.5) and (mean_ratio > 1.2)
    
    print(f"KS-2 Demand Analysis:")
    print(f"Peak demand ratio (moving/stationary): {peak_ratio:.3f}")
    print(f"Mean center demand ratio: {mean_ratio:.3f}")
    print(f"KS-2 criterion: peak > 1.5 and mean > 1.2")
    print(f"KS-2 PASS: {ks2_pass}")
    
    return D_moving, D_stationary, ks2_pass


# Default configurations for experiments
def create_default_demand_estimator() -> DemandEstimator:
    """Create standard demand estimator for KS experiments"""
    return DemandEstimator(method='variance')


def create_default_capacity_model() -> CapacityModel:
    """Create standard capacity model for KS experiments"""
    return CapacityModel(base_capacity=1.0, model_type='fixed')
