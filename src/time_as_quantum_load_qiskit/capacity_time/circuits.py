"""
Small Quantum Circuits and Lattice Systems

This module creates small 1D lattice circuits for testing motion vs stationary
patterns (KS-2) and spatial response mapping (KS-4).

References:
- Preskill, "Quantum Computing in the NISQ era and beyond"
- Childs et al., "Universal computation by multiparticle quantum walk"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from typing import List, Tuple, Union, Dict, Optional
import matplotlib.pyplot as plt


class LatticeCircuit:
    """
    1D quantum lattice with nearest-neighbor interactions
    
    Supports various evolution patterns for KS-2 and KS-4 experiments
    """
    
    def __init__(self, n_qubits: int, coupling_type: str = 'cz'):
        """
        Initialize lattice circuit
        
        Args:
            n_qubits: Number of qubits in 1D chain (typically 4-8)
            coupling_type: Type of entangling gate ('cz', 'cx', 'xx')
        """
        self.n_qubits = n_qubits
        self.coupling_type = coupling_type.lower()
        self.history = []  # Store evolution history
        
        # Validate inputs
        if n_qubits < 2:
            raise ValueError("Need at least 2 qubits for lattice")
        if coupling_type not in ['cz', 'cx', 'xx']:
            raise ValueError("coupling_type must be 'cz', 'cx', or 'xx'")
    
    def initial_state(self, pattern: str = 'ground') -> QuantumCircuit:
        """
        Prepare initial state
        
        Args:
            pattern: 'ground' (all |0⟩), 'superposition' (all |+⟩), 
                    'excitation' (single |1⟩ in middle)
                    
        Returns:
            Quantum circuit preparing the state
        """
        qc = QuantumCircuit(self.n_qubits)
        
        if pattern == 'ground':
            pass  # |00...0⟩ is default
        elif pattern == 'superposition':
            qc.h(range(self.n_qubits))  # |++...+⟩
        elif pattern == 'excitation':
            # Single excitation in middle
            center = self.n_qubits // 2
            qc.x(center)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
            
        return qc
    
    def entangling_layer(self, strength: float = np.pi/4) -> QuantumCircuit:
        """
        Add entangling layer between nearest neighbors
        
        Args:
            strength: Coupling strength parameter
            
        Returns:
            Circuit implementing entangling layer
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply entangling gates to nearest-neighbor pairs
        for i in range(self.n_qubits - 1):
            if self.coupling_type == 'cz':
                qc.cz(i, i + 1)
            elif self.coupling_type == 'cx':
                qc.cx(i, i + 1)
            elif self.coupling_type == 'xx':
                # XX coupling via Ry-CX-Ry sequence
                qc.ry(strength, i)
                qc.ry(strength, i + 1)
                qc.cx(i, i + 1)
                qc.ry(-strength, i)
                qc.ry(-strength, i + 1)
                
        return qc
    
    def single_qubit_layer(self, angles: List[float]) -> QuantumCircuit:
        """
        Apply single-qubit rotations
        
        Args:
            angles: Rotation angles for each qubit
            
        Returns:
            Circuit with single-qubit gates
        """
        if len(angles) != self.n_qubits:
            raise ValueError(f"Need {self.n_qubits} angles, got {len(angles)}")
            
        qc = QuantumCircuit(self.n_qubits)
        for i, angle in enumerate(angles):
            qc.rz(angle, i)
            
        return qc
    
    def time_step(self, 
                  sq_angles: List[float], 
                  entangling_strength: float = np.pi/4) -> QuantumCircuit:
        """
        Single time evolution step: single-qubit + entangling
        
        Args:
            sq_angles: Single-qubit rotation angles
            entangling_strength: Coupling strength
            
        Returns:
            Complete time step circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Single-qubit layer first
        qc.compose(self.single_qubit_layer(sq_angles), inplace=True)
        
        # Then entangling layer
        qc.compose(self.entangling_layer(entangling_strength), inplace=True)
        
        return qc


def moving_pattern_schedule(n_qubits: int, n_steps: int, 
                           amplitude: float = 0.2) -> List[List[float]]:
    """
    Generate angle schedule for a "moving" excitation pattern
    
    Creates a localized rotation that translates across the chain
    
    Args:
        n_qubits: Number of qubits
        n_steps: Number of time steps
        amplitude: Rotation amplitude
        
    Returns:
        List of angle lists, one per time step
    """
    schedule = []
    
    for step in range(n_steps):
        angles = [0.0] * n_qubits
        
        # Moving gaussian-like envelope
        center = (step * (n_qubits - 1)) / (n_steps - 1) if n_steps > 1 else 0
        width = 1.0
        
        for i in range(n_qubits):
            distance = abs(i - center)
            weight = np.exp(-distance**2 / (2 * width**2))
            angles[i] = amplitude * weight
            
        schedule.append(angles)
    
    return schedule


def stationary_pattern_schedule(n_qubits: int, n_steps: int,
                               amplitude: float = 0.2,
                               center_qubit: Optional[int] = None) -> List[List[float]]:
    """
    Generate angle schedule for a "stationary" pattern
    
    Localized rotation that stays in place
    
    Args:
        n_qubits: Number of qubits
        n_steps: Number of time steps
        amplitude: Rotation amplitude  
        center_qubit: Center position (default: middle)
        
    Returns:
        List of angle lists, one per time step
    """
    if center_qubit is None:
        center_qubit = n_qubits // 2
    
    schedule = []
    
    for step in range(n_steps):
        angles = [0.0] * n_qubits
        
        # Stationary gaussian envelope
        width = 1.0
        
        for i in range(n_qubits):
            distance = abs(i - center_qubit)
            weight = np.exp(-distance**2 / (2 * width**2))
            angles[i] = amplitude * weight
            
        schedule.append(angles)
    
    return schedule


def uniform_pattern_schedule(n_qubits: int, n_steps: int,
                           amplitude: float = 0.1) -> List[List[float]]:
    """
    Generate uniform angle schedule (control pattern)
    
    Args:
        n_qubits: Number of qubits
        n_steps: Number of time steps  
        amplitude: Uniform rotation amplitude
        
    Returns:
        List of angle lists, one per time step
    """
    schedule = []
    
    for step in range(n_steps):
        angles = [amplitude] * n_qubits
        schedule.append(angles)
    
    return schedule


def evolve_circuit_schedule(lattice: LatticeCircuit,
                           initial_pattern: str,
                           angle_schedule: List[List[float]],
                           entangling_strength: float = np.pi/4) -> List[Statevector]:
    """
    Evolve lattice according to angle schedule and return state history
    
    Args:
        lattice: LatticeCircuit instance
        initial_pattern: Initial state pattern
        angle_schedule: List of angle lists for each step
        entangling_strength: Coupling strength
        
    Returns:
        List of state vectors, one per time step
    """
    # Prepare initial state
    qc = lattice.initial_state(initial_pattern)
    
    simulator = AerSimulator(method='statevector')
    states = []
    
    # Initial state
    result = simulator.run(qc, shots=1).result()
    states.append(result.get_statevector())
    
    # Evolve step by step
    for step, angles in enumerate(angle_schedule):
        # Add time step to circuit
        step_circuit = lattice.time_step(angles, entangling_strength)
        qc.compose(step_circuit, inplace=True)
        
        # Get evolved state
        result = simulator.run(qc, shots=1).result()
        states.append(result.get_statevector())
    
    return states


def measure_local_expectations(states: List[Statevector], 
                             observables: List[str] = ['X', 'Y', 'Z']) -> Dict[str, np.ndarray]:
    """
    Measure local expectation values across lattice sites and time
    
    Args:
        states: List of quantum states
        observables: List of Pauli observables to measure
        
    Returns:
        Dict mapping observable -> array[time_step, qubit_index]
    """
    from qiskit.quantum_info import Pauli
    
    n_steps = len(states)
    n_qubits = states[0].num_qubits
    
    results = {}
    
    for obs in observables:
        expectations = np.zeros((n_steps, n_qubits))
        
        for t, state in enumerate(states):
            for i in range(n_qubits):
                # Create local Pauli operator
                pauli_string = 'I' * n_qubits
                pauli_list = list(pauli_string)
                pauli_list[i] = obs
                pauli_op = Pauli(''.join(pauli_list))
                
                # Measure expectation
                expectations[t, i] = state.expectation_value(pauli_op).real
                
        results[obs] = expectations
    
    return results


def create_demand_bump(lattice: LatticeCircuit, 
                      bump_site: int,
                      bump_strength: float = 0.5) -> QuantumCircuit:
    """
    Create local "demand bump" by adding extra gate depth at specific site
    
    Used for KS-4 response kernel experiments
    
    Args:
        lattice: LatticeCircuit instance  
        bump_site: Qubit index for demand bump
        bump_strength: Strength of additional operations
        
    Returns:
        Circuit with local demand bump
    """
    qc = QuantumCircuit(lattice.n_qubits)
    
    # Extra single-qubit rotations (increases local demand)
    qc.rx(bump_strength, bump_site)
    qc.ry(bump_strength, bump_site) 
    qc.rz(bump_strength, bump_site)
    
    # Extra entangling if not at boundary
    if bump_site > 0:
        if lattice.coupling_type == 'cz':
            qc.cz(bump_site - 1, bump_site)
        elif lattice.coupling_type == 'cx':
            qc.cx(bump_site - 1, bump_site)
    
    if bump_site < lattice.n_qubits - 1:
        if lattice.coupling_type == 'cz':
            qc.cz(bump_site, bump_site + 1)
        elif lattice.coupling_type == 'cx':  
            qc.cx(bump_site, bump_site + 1)
    
    return qc


def plot_lattice_dynamics(expectations: Dict[str, np.ndarray], 
                         title: str = "Lattice Dynamics",
                         save_path: Optional[str] = None) -> None:
    """
    Visualize lattice dynamics as space-time plot
    
    Args:
        expectations: Output from measure_local_expectations
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, len(expectations), figsize=(15, 5))
    if len(expectations) == 1:
        axes = [axes]
    
    for i, (obs, data) in enumerate(expectations.items()):
        im = axes[i].imshow(data.T, aspect='auto', origin='lower', 
                           cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i].set_title(f'{obs} Expectation')
        axes[i].set_xlabel('Time Step') 
        axes[i].set_ylabel('Qubit Index')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# Predefined lattice configurations for experiments
def create_standard_lattice(n_qubits: int = 6) -> LatticeCircuit:
    """Create standard lattice for KS experiments"""
    return LatticeCircuit(n_qubits, coupling_type='cz')


def create_comparison_schedules(n_qubits: int = 6, 
                              n_steps: int = 10,
                              amplitude: float = 0.2) -> Dict[str, List[List[float]]]:
    """
    Create standard set of comparison schedules for KS-2
    
    Returns:
        Dict with 'moving', 'stationary', 'uniform' schedules
    """
    schedules = {
        'moving': moving_pattern_schedule(n_qubits, n_steps, amplitude),
        'stationary': stationary_pattern_schedule(n_qubits, n_steps, amplitude),
        'uniform': uniform_pattern_schedule(n_qubits, n_steps, amplitude/2)
    }
    
    return schedules
