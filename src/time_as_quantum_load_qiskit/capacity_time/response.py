"""
Response Kernel Mapping (KS-4)

This module implements spatial response kernel experiments: create a local
demand "bump" and measure how ln(N) proxies respond at other sites.

References:
- Martin & Schwinger, "Theory of Many-Particle Systems"
- Kadanoff & Baym, "Quantum Statistical Mechanics"
- Peskin & Schroeder, "An Introduction To Quantum Field Theory"
"""

import numpy as np
from qiskit.quantum_info import Statevector
from typing import List, Tuple, Dict, Optional, Callable
import matplotlib.pyplot as plt
from .circuits import LatticeCircuit, create_demand_bump, evolve_circuit_schedule
from .demand_capacity import (DemandEstimator, CapacityModel, compute_demand_profile, 
                             compute_time_factor_profile, compute_capacity_profile)


def create_response_perturbation(lattice: LatticeCircuit,
                                bump_site: int,
                                bump_strength: float = 0.5,
                                bump_type: str = 'gate_depth') -> Dict:
    """
    Create local demand perturbation for response kernel measurement
    
    Args:
        lattice: LatticeCircuit instance
        bump_site: Site index for demand bump  
        bump_strength: Strength of perturbation
        bump_type: Type of perturbation ('gate_depth', 'rotation', 'entangling')
        
    Returns:
        Dictionary with perturbation info
    """
    if bump_type == 'gate_depth':
        # Extra gate depth at bump site
        bump_circuit = create_demand_bump(lattice, bump_site, bump_strength)
    
    elif bump_type == 'rotation':
        # Enhanced single-qubit rotations
        from qiskit import QuantumCircuit
        bump_circuit = QuantumCircuit(lattice.n_qubits)
        bump_circuit.rx(bump_strength * np.pi/2, bump_site)
        bump_circuit.ry(bump_strength * np.pi/2, bump_site)
        bump_circuit.rz(bump_strength * np.pi/2, bump_site)
    
    elif bump_type == 'entangling':
        # Extra entangling gates involving bump site
        from qiskit import QuantumCircuit
        bump_circuit = QuantumCircuit(lattice.n_qubits)
        
        # Add extra entangling to neighbors
        if bump_site > 0:
            bump_circuit.cz(bump_site - 1, bump_site)
        if bump_site < lattice.n_qubits - 1:
            bump_circuit.cz(bump_site, bump_site + 1)
            
        # Add more complex pattern
        if bump_site > 1:
            bump_circuit.cx(bump_site - 2, bump_site)
        if bump_site < lattice.n_qubits - 2:
            bump_circuit.cx(bump_site, bump_site + 2)
    
    else:
        raise ValueError(f"Unknown bump_type: {bump_type}")
    
    perturbation = {
        'bump_site': bump_site,
        'bump_strength': bump_strength,
        'bump_type': bump_type,
        'bump_circuit': bump_circuit,
        'description': f"{bump_type} bump at site {bump_site} with strength {bump_strength}"
    }
    
    return perturbation


def measure_response_kernel(lattice: LatticeCircuit,
                          demand_estimator: DemandEstimator,
                          capacity_model: CapacityModel,
                          initial_pattern: str,
                          base_schedule: List[List[float]],
                          perturbation: Dict,
                          response_metric: str = 'ln_N') -> Dict:
    """
    Measure response kernel G(r) = Δresponse(r) / Δperturbation
    
    Args:
        lattice: LatticeCircuit for evolution
        demand_estimator: DemandEstimator instance
        capacity_model: CapacityModel instance  
        initial_pattern: Initial state pattern
        base_schedule: Base evolution schedule
        perturbation: Perturbation dict from create_response_perturbation
        response_metric: What to measure ('ln_N', 'demand', 'time_factor')
        
    Returns:
        Dictionary with response kernel data
    """
    # Run reference evolution (no perturbation)
    print(f"Running reference evolution...")
    ref_states = evolve_circuit_schedule(lattice, initial_pattern, base_schedule)
    
    # Run perturbed evolution
    print(f"Running perturbed evolution...")
    perturbed_states = evolve_circuit_schedule_with_bump(
        lattice, initial_pattern, base_schedule, perturbation
    )
    
    # Ensure same length
    min_length = min(len(ref_states), len(perturbed_states))
    ref_states = ref_states[:min_length]
    perturbed_states = perturbed_states[:min_length]
    
    # Compute response quantities
    n_qubits = lattice.n_qubits
    n_steps = len(ref_states)
    
    # Reference profiles
    ref_demands = compute_demand_profile(ref_states, demand_estimator, n_qubits)
    ref_capacities = compute_capacity_profile(n_steps, n_qubits, capacity_model)
    ref_time_factors = compute_time_factor_profile(ref_demands, ref_capacities)
    
    # Perturbed profiles
    pert_demands = compute_demand_profile(perturbed_states, demand_estimator, n_qubits)
    pert_capacities = compute_capacity_profile(n_steps, n_qubits, capacity_model)  # Same as reference
    pert_time_factors = compute_time_factor_profile(pert_demands, pert_capacities)
    
    # Compute response differences
    if response_metric == 'ln_N':
        # Δln(N) = ln(N_pert) - ln(N_ref)
        ref_ln_N = np.log(ref_time_factors + 1e-12)  # Avoid log(0)
        pert_ln_N = np.log(pert_time_factors + 1e-12)
        response_diff = pert_ln_N - ref_ln_N
        
    elif response_metric == 'demand':
        response_diff = pert_demands - ref_demands
        
    elif response_metric == 'time_factor':
        response_diff = pert_time_factors - ref_time_factors
        
    else:
        raise ValueError(f"Unknown response_metric: {response_metric}")
    
    # Average response over time steps (take latter half for steady state)
    steady_start = n_steps // 2
    steady_response = np.mean(response_diff[steady_start:], axis=0)
    
    # Compute response kernel G(r)
    bump_site = perturbation['bump_site']
    bump_strength = perturbation['bump_strength']
    
    distances = np.arange(n_qubits) - bump_site
    response_kernel = steady_response / bump_strength  # G(r) = Δresponse / Δbump
    
    kernel_data = {
        'distances': distances,
        'response_kernel': response_kernel,
        'steady_response': steady_response,
        'response_diff': response_diff,
        'ref_profile': {
            'demands': ref_demands,
            'time_factors': ref_time_factors,
            'ln_N': ref_ln_N if response_metric == 'ln_N' else np.log(ref_time_factors + 1e-12)
        },
        'perturbed_profile': {
            'demands': pert_demands,
            'time_factors': pert_time_factors, 
            'ln_N': pert_ln_N if response_metric == 'ln_N' else np.log(pert_time_factors + 1e-12)
        },
        'perturbation': perturbation,
        'response_metric': response_metric,
        'bump_site': bump_site,
        'bump_strength': bump_strength
    }
    
    return kernel_data


def evolve_circuit_schedule_with_bump(lattice: LatticeCircuit,
                                     initial_pattern: str,
                                     base_schedule: List[List[float]],
                                     perturbation: Dict) -> List[Statevector]:
    """
    Evolve circuit with local demand bump applied at specific time step
    
    Args:
        lattice: LatticeCircuit instance
        initial_pattern: Initial state pattern
        base_schedule: Base angle schedule
        perturbation: Perturbation dictionary
        
    Returns:
        List of evolved states
    """
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit
    
    # Initialize
    qc = lattice.initial_state(initial_pattern)
    simulator = AerSimulator(method='statevector')
    states = []
    
    # Initial state
    result = simulator.run(qc, shots=1).result()
    states.append(result.get_statevector())
    
    # Apply bump early in evolution (e.g., after 1st step)
    bump_applied_at_step = 1
    
    # Evolve with bump
    for step, angles in enumerate(base_schedule):
        # Regular time step
        step_circuit = lattice.time_step(angles)
        qc.compose(step_circuit, inplace=True)
        
        # Apply bump at specified step
        if step == bump_applied_at_step:
            qc.compose(perturbation['bump_circuit'], inplace=True)
        
        # Get evolved state
        result = simulator.run(qc, shots=1).result()
        states.append(result.get_statevector())
    
    return states


def analyze_response_symmetry(kernel_data: Dict) -> Tuple[bool, Dict]:
    """
    Test response kernel symmetry: G(+r) ≈ G(-r)
    
    Args:
        kernel_data: Output from measure_response_kernel
        
    Returns:
        Tuple of (symmetry_pass, symmetry_analysis)
    """
    distances = kernel_data['distances']
    response_kernel = kernel_data['response_kernel']
    bump_site = kernel_data['bump_site']
    
    # Extract left and right sides (exclude bump site itself)
    left_mask = distances < 0
    right_mask = distances > 0
    
    left_distances = distances[left_mask]
    left_response = response_kernel[left_mask]
    
    right_distances = distances[right_mask]
    right_response = response_kernel[right_mask]
    
    # Match up symmetric pairs
    symmetric_pairs = []
    symmetry_errors = []
    
    for r in range(1, min(len(left_response), len(right_response)) + 1):
        if bump_site - r >= 0 and bump_site + r < len(response_kernel):
            left_val = response_kernel[bump_site - r]
            right_val = response_kernel[bump_site + r]
            
            symmetric_pairs.append((r, left_val, right_val))
            symmetry_errors.append(abs(left_val - right_val))
    
    if len(symmetry_errors) == 0:
        symmetry_pass = False
        mean_asymmetry = float('inf')
        max_asymmetry = float('inf')
    else:
        mean_asymmetry = np.mean(symmetry_errors)
        max_asymmetry = np.max(symmetry_errors)
        
        # Pass criteria: reasonable symmetry
        symmetry_pass = (mean_asymmetry < 0.1) and (max_asymmetry < 0.2)
    
    symmetry_analysis = {
        'symmetric_pairs': symmetric_pairs,
        'symmetry_errors': symmetry_errors,
        'mean_asymmetry': mean_asymmetry,
        'max_asymmetry': max_asymmetry,
        'symmetry_pass': symmetry_pass
    }
    
    print(f"KS-4 Symmetry Analysis:")
    print(f"Mean asymmetry: {mean_asymmetry:.4f}")
    print(f"Max asymmetry: {max_asymmetry:.4f}")
    print(f"Symmetry criterion: mean < 0.1 AND max < 0.2")
    print(f"Symmetry PASS: {symmetry_pass}")
    
    return symmetry_pass, symmetry_analysis


def analyze_response_locality(kernel_data: Dict,
                             decay_threshold: float = 0.1) -> Tuple[bool, Dict]:
    """
    Test response kernel locality: G(r) → 0 as |r| increases
    
    Args:
        kernel_data: Output from measure_response_kernel
        decay_threshold: Threshold for considering response negligible
        
    Returns:
        Tuple of (locality_pass, locality_analysis)
    """
    distances = kernel_data['distances']
    response_kernel = kernel_data['response_kernel']
    bump_site = kernel_data['bump_site']
    
    # Get absolute distances and corresponding responses
    abs_distances = np.abs(distances)
    abs_response = np.abs(response_kernel)
    
    # Exclude bump site (distance 0)
    nonzero_mask = abs_distances > 0
    if np.sum(nonzero_mask) == 0:
        locality_pass = False
        locality_analysis = {'error': 'No data points away from bump site'}
        return locality_pass, locality_analysis
    
    r_vals = abs_distances[nonzero_mask]
    G_vals = abs_response[nonzero_mask]
    
    # Sort by distance
    sort_indices = np.argsort(r_vals)
    r_sorted = r_vals[sort_indices] 
    G_sorted = G_vals[sort_indices]
    
    # Check for decay: response at max distance should be smaller than at min distance
    if len(G_sorted) < 2:
        locality_pass = False
        locality_analysis = {'error': 'Insufficient data points'}
        return locality_pass, locality_analysis
    
    max_distance_response = G_sorted[-1]
    min_distance_response = G_sorted[0]
    
    # Decay ratio
    decay_ratio = max_distance_response / (min_distance_response + 1e-12)
    
    # Remote response should be below threshold
    remote_below_threshold = max_distance_response < decay_threshold
    
    # Overall decay trend (simple check)
    decay_trend = decay_ratio < 0.5
    
    locality_pass = remote_below_threshold and decay_trend
    
    locality_analysis = {
        'r_sorted': r_sorted,
        'G_sorted': G_sorted,
        'max_distance_response': max_distance_response,
        'min_distance_response': min_distance_response,
        'decay_ratio': decay_ratio,
        'remote_below_threshold': remote_below_threshold,
        'decay_trend': decay_trend,
        'locality_pass': locality_pass
    }
    
    print(f"KS-4 Locality Analysis:")
    print(f"Max distance response: {max_distance_response:.4f}")
    print(f"Decay ratio (far/near): {decay_ratio:.4f}")
    print(f"Remote below threshold ({decay_threshold}): {remote_below_threshold}")
    print(f"Locality criterion: remote < {decay_threshold} AND decay_ratio < 0.5")
    print(f"Locality PASS: {locality_pass}")
    
    return locality_pass, locality_analysis


def plot_response_kernel_analysis(kernel_data: Dict,
                                 symmetry_analysis: Optional[Dict] = None,
                                 locality_analysis: Optional[Dict] = None,
                                 save_path: Optional[str] = None) -> None:
    """
    Create comprehensive response kernel analysis plots
    
    Args:
        kernel_data: Response kernel data
        symmetry_analysis: Optional symmetry analysis results
        locality_analysis: Optional locality analysis results  
        save_path: Optional save path for figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    distances = kernel_data['distances'] 
    response_kernel = kernel_data['response_kernel']
    bump_site = kernel_data['bump_site']
    response_metric = kernel_data['response_metric']
    
    # Plot 1: Response kernel G(r)
    axes[0, 0].plot(distances, response_kernel, 'bo-', markersize=6, linewidth=2)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, 
                      label=f'Bump site ({bump_site})')
    axes[0, 0].axhline(0, color='black', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Distance from bump site')
    axes[0, 0].set_ylabel(f'Response G(r) [{response_metric}]')
    axes[0, 0].set_title('Response Kernel G(r)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Space-time response evolution
    response_diff = kernel_data['response_diff']
    im = axes[0, 1].imshow(response_diff.T, aspect='auto', origin='lower',
                          cmap='RdBu_r', interpolation='nearest')
    axes[0, 1].axhline(bump_site, color='yellow', linestyle='--', linewidth=2,
                      label=f'Bump site')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Qubit Index')
    axes[0, 1].set_title(f'Response Evolution Δ{response_metric}')
    axes[0, 1].legend()
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot 3: Symmetry analysis
    if symmetry_analysis and 'symmetric_pairs' in symmetry_analysis:
        pairs = symmetry_analysis['symmetric_pairs']
        if len(pairs) > 0:
            r_vals = [p[0] for p in pairs]
            left_vals = [p[1] for p in pairs]
            right_vals = [p[2] for p in pairs]
            
            axes[1, 0].plot(r_vals, left_vals, 'ro-', label='G(-r)', markersize=6)
            axes[1, 0].plot(r_vals, right_vals, 'bo-', label='G(+r)', markersize=6)
            axes[1, 0].plot(r_vals, np.array(left_vals) - np.array(right_vals), 
                           'go--', label='G(-r) - G(+r)', markersize=4)
            axes[1, 0].axhline(0, color='black', linestyle=':', alpha=0.5)
        
        axes[1, 0].set_xlabel('Distance |r|')
        axes[1, 0].set_ylabel('Response value') 
        axes[1, 0].set_title('Symmetry Test: G(-r) vs G(+r)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No symmetry analysis', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Plot 4: Locality analysis  
    if locality_analysis and 'r_sorted' in locality_analysis:
        r_sorted = locality_analysis['r_sorted']
        G_sorted = locality_analysis['G_sorted']
        
        axes[1, 1].semilogy(r_sorted, G_sorted, 'mo-', markersize=6, linewidth=2,
                          label='|G(r)| vs |r|')
        axes[1, 1].axhline(0.1, color='orange', linestyle='--', 
                          label='Decay threshold')
    else:
        # Fallback: plot absolute response vs distance
        abs_distances = np.abs(distances)
        abs_response = np.abs(response_kernel)
        nonzero_mask = abs_distances > 0
        
        if np.sum(nonzero_mask) > 0:
            axes[1, 1].semilogy(abs_distances[nonzero_mask], 
                              abs_response[nonzero_mask], 
                              'mo-', markersize=6, linewidth=2)
    
    axes[1, 1].set_xlabel('Distance |r|')
    axes[1, 1].set_ylabel('|Response| (log scale)')
    axes[1, 1].set_title('Locality Test: Decay vs Distance')  
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'KS-4: Response Kernel Analysis\n{kernel_data["perturbation"]["description"]}', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def run_ks4_experiment(lattice: LatticeCircuit,
                      demand_estimator: DemandEstimator, 
                      capacity_model: CapacityModel,
                      base_schedule: List[List[float]],
                      bump_site: Optional[int] = None,
                      bump_strength: float = 0.5,
                      bump_type: str = 'gate_depth',
                      initial_pattern: str = 'superposition',
                      save_plots: bool = True) -> Tuple[bool, Dict]:
    """
    Run complete KS-4 response kernel experiment
    
    Args:
        lattice: LatticeCircuit for evolution
        demand_estimator: DemandEstimator instance
        capacity_model: CapacityModel instance
        base_schedule: Base evolution schedule
        bump_site: Location of demand bump (default: center)
        bump_strength: Strength of perturbation
        bump_type: Type of perturbation
        initial_pattern: Initial state pattern
        save_plots: Whether to generate plots
        
    Returns:
        Tuple of (ks4_pass, results_dict)
    """
    print("=" * 60)
    print("KS-4: RESPONSE KERNEL EXPERIMENT")
    print("=" * 60)
    
    if bump_site is None:
        bump_site = lattice.n_qubits // 2
    
    # Create perturbation
    print(f"\n1. Creating demand bump at site {bump_site}...")
    perturbation = create_response_perturbation(lattice, bump_site, bump_strength, bump_type)
    
    # Measure response kernel
    print("2. Measuring response kernel...")
    kernel_data = measure_response_kernel(
        lattice, demand_estimator, capacity_model, 
        initial_pattern, base_schedule, perturbation, 
        response_metric='ln_N'
    )
    
    # Analyze symmetry
    print("3. Testing symmetry...")
    symmetry_pass, symmetry_analysis = analyze_response_symmetry(kernel_data)
    
    # Analyze locality  
    print("4. Testing locality...")
    locality_pass, locality_analysis = analyze_response_locality(kernel_data)
    
    # Overall KS-4 verdict
    ks4_pass = symmetry_pass and locality_pass
    
    # Generate plots
    if save_plots:
        print("5. Generating analysis plots...")
        import os
        os.makedirs('figures', exist_ok=True)
        
        plot_response_kernel_analysis(
            kernel_data, symmetry_analysis, locality_analysis,
            save_path='figures/KS4_response_kernel.png'
        )
    
    # Compile full results
    full_results = {
        'ks4_pass': ks4_pass,
        'kernel_data': kernel_data,
        'symmetry_analysis': symmetry_analysis,
        'locality_analysis': locality_analysis,
        'experiment_params': {
            'bump_site': bump_site,
            'bump_strength': bump_strength,
            'bump_type': bump_type, 
            'initial_pattern': initial_pattern,
            'lattice_n_qubits': lattice.n_qubits
        }
    }
    
    print(f"\n" + "=" * 60)
    print(f"KS-4 OVERALL RESULT: {'PASS' if ks4_pass else 'FAIL'}")
    print(f"Symmetry: {'PASS' if symmetry_pass else 'FAIL'}")
    print(f"Locality: {'PASS' if locality_pass else 'FAIL'}")
    print("=" * 60)
    
    return ks4_pass, full_results
