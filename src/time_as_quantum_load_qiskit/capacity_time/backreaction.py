"""
Back-Reaction Feedback Loop (KS-3)

This module implements the core feedback mechanism where demand D affects
the time evolution rate via N = C/D. Gate angles are scaled by N each step,
creating observable time dilation effects.

References:
- Wheeler & Feynman, "Interaction with the Absorber as the Mechanism of Radiation"
- Penrose, "The Road to Reality"  
- Susskind, "The Black Hole War"
"""

import numpy as np
from qiskit.quantum_info import Statevector
from typing import List, Tuple, Dict, Union, Optional, Callable
import matplotlib.pyplot as plt
from .circuits import LatticeCircuit, evolve_circuit_schedule, measure_local_expectations
from .demand_capacity import (DemandEstimator, CapacityModel, compute_time_factor, 
                             compute_demand_profile, compute_capacity_profile, 
                             compute_time_factor_profile)


class BackReactionLoop:
    """
    Implements the KS-3 back-reaction feedback loop
    
    At each time step:
    1. Evolve system with current gate schedule
    2. Measure local demand D at probe sites  
    3. Compute time factor N = C/D
    4. Scale next time step's gate angles by N
    5. Repeat and observe cumulative effects
    """
    
    def __init__(self,
                 lattice: LatticeCircuit,
                 demand_estimator: DemandEstimator,
                 capacity_model: CapacityModel,
                 probe_sites: Optional[List[int]] = None):
        """
        Initialize back-reaction loop
        
        Args:
            lattice: LatticeCircuit for evolution
            demand_estimator: DemandEstimator for computing D
            capacity_model: CapacityModel for computing C
            probe_sites: Sites to monitor for demand (default: all sites)
        """
        self.lattice = lattice
        self.demand_estimator = demand_estimator
        self.capacity_model = capacity_model
        
        if probe_sites is None:
            self.probe_sites = list(range(lattice.n_qubits))
        else:
            self.probe_sites = probe_sites
            
        # Evolution history
        self.states_history = []
        self.demands_history = []
        self.time_factors_history = []
        self.angles_history = []
    
    def run_feedback_evolution(self,
                              initial_pattern: str,
                              base_schedule: List[List[float]],
                              feedback_strength: float = 1.0,
                              min_factor: float = 0.1,
                              max_factor: float = 10.0) -> Dict:
        """
        Run complete back-reaction evolution
        
        Args:
            initial_pattern: Initial state pattern
            base_schedule: Base angle schedule (before scaling)
            feedback_strength: How strongly N affects gate scaling (0=off, 1=full)
            min_factor, max_factor: N clipping bounds
            
        Returns:
            Dictionary with evolution results
        """
        from qiskit_aer import AerSimulator
        
        # Initialize
        qc = self.lattice.initial_state(initial_pattern)
        simulator = AerSimulator(method='statevector')
        
        # Clear history
        self.states_history = []
        self.demands_history = []  
        self.time_factors_history = []
        self.angles_history = []
        
        # Initial state
        result = simulator.run(qc, shots=1).result()
        current_state = result.get_statevector()
        self.states_history.append(current_state)
        
        # Initial demand and time factor
        initial_demands = self._compute_local_demands(current_state)
        initial_time_factors = self._compute_time_factors(initial_demands, 0)
        
        self.demands_history.append(initial_demands)
        self.time_factors_history.append(initial_time_factors)
        self.angles_history.append([0.0] * self.lattice.n_qubits)  # No evolution yet
        
        # Evolution loop
        for step, base_angles in enumerate(base_schedule):
            # Compute current demand-dependent scaling
            current_demands = self._compute_local_demands(current_state)
            time_factors = self._compute_time_factors(current_demands, step + 1)
            
            # Scale angles by time factors (with feedback strength control)
            scaled_angles = self._scale_angles(base_angles, time_factors, feedback_strength)
            
            # Store for history
            self.demands_history.append(current_demands)
            self.time_factors_history.append(time_factors)
            self.angles_history.append(scaled_angles)
            
            # Apply time step with scaled angles
            step_circuit = self.lattice.time_step(scaled_angles)
            qc.compose(step_circuit, inplace=True)
            
            # Get new state
            result = simulator.run(qc, shots=1).result()
            current_state = result.get_statevector()
            self.states_history.append(current_state)
        
        # Compile results
        results = {
            'states': self.states_history,
            'demands': np.array(self.demands_history),
            'time_factors': np.array(self.time_factors_history), 
            'scaled_angles': np.array(self.angles_history),
            'base_schedule': np.array(base_schedule),
            'feedback_strength': feedback_strength,
            'lattice_params': {
                'n_qubits': self.lattice.n_qubits,
                'coupling_type': self.lattice.coupling_type
            }
        }
        
        return results
    
    def _compute_local_demands(self, state: Statevector) -> np.ndarray:
        """Compute demand at all probe sites"""
        demands = np.zeros(self.lattice.n_qubits)
        
        for i in self.probe_sites:
            demands[i] = self.demand_estimator.estimate_local_demand(state, i)
        
        return demands
    
    def _compute_time_factors(self, demands: np.ndarray, time_step: int) -> np.ndarray:
        """Compute time factors N = C/D for all sites"""
        time_factors = np.zeros(self.lattice.n_qubits)
        
        for i in range(self.lattice.n_qubits):
            capacity = self.capacity_model.get_capacity(site=i, time_step=time_step)
            time_factors[i] = compute_time_factor(demands[i], capacity)
        
        return time_factors
    
    def _scale_angles(self, 
                     base_angles: List[float], 
                     time_factors: np.ndarray, 
                     feedback_strength: float) -> List[float]:
        """Scale gate angles by time factors"""
        scaled = []
        
        for i, base_angle in enumerate(base_angles):
            if i < len(time_factors):
                # Apply feedback: angle → angle * (1 + feedback_strength * (N - 1))
                # When N < 1 (high demand), angles get smaller (time slows)
                # When N > 1 (low demand), angles get larger (time speeds up)
                scaling = 1.0 + feedback_strength * (time_factors[i] - 1.0)
                scaled_angle = base_angle * scaling
            else:
                scaled_angle = base_angle
            
            scaled.append(scaled_angle)
        
        return scaled


def run_control_evolution(lattice: LatticeCircuit,
                         initial_pattern: str,
                         schedule: List[List[float]]) -> Dict:
    """
    Run control evolution without back-reaction (fixed schedule)
    
    Args:
        lattice: LatticeCircuit for evolution
        initial_pattern: Initial state pattern  
        schedule: Fixed angle schedule
        
    Returns:
        Dictionary with evolution results (for comparison)
    """
    states = evolve_circuit_schedule(lattice, initial_pattern, schedule)
    
    # Measure expectations for comparison
    expectations = measure_local_expectations(states, ['X', 'Z'])
    
    results = {
        'states': states,
        'expectations': expectations,
        'schedule': np.array(schedule),
        'lattice_params': {
            'n_qubits': lattice.n_qubits,
            'coupling_type': lattice.coupling_type
        }
    }
    
    return results


def compare_backreaction_vs_control(feedback_results: Dict,
                                   control_results: Dict,
                                   observable: str = 'Z',
                                   probe_qubit: Optional[int] = None) -> Tuple[bool, Dict]:
    """
    Compare back-reaction evolution vs control to test KS-3
    
    Args:
        feedback_results: Results from BackReactionLoop.run_feedback_evolution
        control_results: Results from run_control_evolution  
        observable: Pauli observable to compare ('X', 'Y', 'Z')
        probe_qubit: Which qubit to monitor (default: middle qubit)
        
    Returns:
        Tuple of (ks3_pass, comparison_data)
    """
    n_qubits = feedback_results['lattice_params']['n_qubits']
    if probe_qubit is None:
        probe_qubit = n_qubits // 2
    
    # Get observable trajectories
    feedback_expectations = measure_local_expectations(feedback_results['states'], [observable])
    control_expectations = control_results['expectations']
    
    feedback_trajectory = feedback_expectations[observable][:, probe_qubit]
    control_trajectory = control_expectations[observable][:, probe_qubit]
    
    # Ensure same length (might differ by 1 due to initial state)
    min_length = min(len(feedback_trajectory), len(control_trajectory))
    feedback_trajectory = feedback_trajectory[:min_length]
    control_trajectory = control_trajectory[:min_length]
    
    # Compute divergence metric
    trajectory_diff = np.abs(feedback_trajectory - control_trajectory)
    max_divergence = np.max(trajectory_diff)
    mean_divergence = np.mean(trajectory_diff)
    
    # Final state difference
    final_diff = abs(feedback_trajectory[-1] - control_trajectory[-1])
    
    # KS-3 pass criteria: 
    # 1. Observable divergence between trajectories
    # 2. Back-reaction should create measurable difference
    ks3_pass = (max_divergence > 0.1) and (final_diff > 0.05)
    
    comparison_data = {
        'feedback_trajectory': feedback_trajectory,
        'control_trajectory': control_trajectory,
        'trajectory_diff': trajectory_diff,
        'max_divergence': max_divergence,
        'mean_divergence': mean_divergence,
        'final_diff': final_diff,
        'observable': observable,
        'probe_qubit': probe_qubit
    }
    
    print(f"KS-3 Back-Reaction Analysis:")
    print(f"Observable: {observable} on qubit {probe_qubit}")
    print(f"Maximum divergence: {max_divergence:.4f}")
    print(f"Mean divergence: {mean_divergence:.4f}")
    print(f"Final state difference: {final_diff:.4f}")
    print(f"KS-3 criterion: max_div > 0.1 AND final_diff > 0.05")
    print(f"KS-3 PASS: {ks3_pass}")
    
    return ks3_pass, comparison_data


def plot_backreaction_analysis(feedback_results: Dict,
                              control_results: Dict,
                              comparison_data: Dict,
                              save_path: Optional[str] = None) -> None:
    """
    Create comprehensive back-reaction analysis plots
    
    Args:
        feedback_results: Back-reaction evolution results
        control_results: Control evolution results
        comparison_data: Output from compare_backreaction_vs_control
        save_path: Optional save path for figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    n_steps = len(feedback_results['demands'])
    time_steps = np.arange(n_steps)
    
    # Plot 1: Demand evolution
    mean_demand = np.mean(feedback_results['demands'], axis=1)
    std_demand = np.std(feedback_results['demands'], axis=1)
    
    axes[0, 0].plot(time_steps, mean_demand, 'r-', linewidth=2, label='Mean Demand')
    axes[0, 0].fill_between(time_steps, mean_demand - std_demand, 
                           mean_demand + std_demand, alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Demand D')
    axes[0, 0].set_title('Demand Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Time factor evolution
    mean_N = np.mean(feedback_results['time_factors'], axis=1)
    std_N = np.std(feedback_results['time_factors'], axis=1)
    
    axes[0, 1].plot(time_steps, mean_N, 'b-', linewidth=2, label='Mean N = C/D')
    axes[0, 1].fill_between(time_steps, mean_N - std_N, mean_N + std_N, 
                           alpha=0.3, color='blue')
    axes[0, 1].axhline(1.0, color='black', linestyle='--', alpha=0.7, label='N = 1 (no dilation)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Time Factor N')
    axes[0, 1].set_title('Time Dilation Factor')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Observable trajectories comparison
    feedback_traj = comparison_data['feedback_trajectory']
    control_traj = comparison_data['control_trajectory']
    obs = comparison_data['observable']
    qubit = comparison_data['probe_qubit']
    
    traj_time = np.arange(len(feedback_traj))
    axes[1, 0].plot(traj_time, control_traj, 'k--', linewidth=2, 
                   label='Control (no back-reaction)', alpha=0.8)
    axes[1, 0].plot(traj_time, feedback_traj, 'g-', linewidth=2, 
                   label='Back-reaction')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel(f'⟨{obs}⟩ Expectation')
    axes[1, 0].set_title(f'Observable Comparison (Qubit {qubit})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Trajectory divergence
    trajectory_diff = comparison_data['trajectory_diff']
    max_div = comparison_data['max_divergence']
    
    axes[1, 1].plot(traj_time, trajectory_diff, 'purple', linewidth=2)
    axes[1, 1].axhline(max_div, color='red', linestyle=':', 
                      label=f'Max divergence: {max_div:.3f}')
    axes[1, 1].axhline(0.1, color='orange', linestyle='--', 
                      label='KS-3 threshold: 0.1')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('|Back-reaction - Control|')
    axes[1, 1].set_title('Trajectory Divergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('KS-3: Back-Reaction Analysis', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def run_ks3_experiment(lattice: LatticeCircuit,
                      demand_estimator: DemandEstimator,
                      capacity_model: CapacityModel,
                      base_schedule: List[List[float]],
                      initial_pattern: str = 'superposition',
                      feedback_strength: float = 1.0,
                      probe_qubit: Optional[int] = None,
                      save_plots: bool = True) -> Tuple[bool, Dict]:
    """
    Run complete KS-3 back-reaction experiment
    
    Args:
        lattice: LatticeCircuit for evolution
        demand_estimator: DemandEstimator instance
        capacity_model: CapacityModel instance
        base_schedule: Base angle schedule
        initial_pattern: Initial state pattern
        feedback_strength: Back-reaction strength (0-1)
        probe_qubit: Monitoring qubit (default: center)
        save_plots: Whether to generate plots
        
    Returns:
        Tuple of (ks3_pass, results_dict)
    """
    print("=" * 60)
    print("KS-3: BACK-REACTION EXPERIMENT")
    print("=" * 60)
    
    # Setup back-reaction loop
    loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
    
    # Run back-reaction evolution
    print("\n1. Running back-reaction evolution...")
    feedback_results = loop.run_feedback_evolution(
        initial_pattern, base_schedule, feedback_strength
    )
    
    # Run control evolution (no feedback)
    print("2. Running control evolution...")
    control_results = run_control_evolution(lattice, initial_pattern, base_schedule)
    
    # Compare results
    print("3. Comparing trajectories...")
    ks3_pass, comparison_data = compare_backreaction_vs_control(
        feedback_results, control_results, observable='Z', probe_qubit=probe_qubit
    )
    
    # Generate plots
    if save_plots:
        print("4. Generating analysis plots...")
        import os
        os.makedirs('figures', exist_ok=True)
        
        plot_backreaction_analysis(
            feedback_results, control_results, comparison_data,
            save_path='figures/KS3_backreaction.png'
        )
    
    # Compile full results
    full_results = {
        'ks3_pass': ks3_pass,
        'feedback_results': feedback_results,
        'control_results': control_results,
        'comparison_data': comparison_data,
        'experiment_params': {
            'initial_pattern': initial_pattern,
            'feedback_strength': feedback_strength,
            'base_schedule_shape': np.array(base_schedule).shape,
            'probe_qubit': probe_qubit
        }
    }
    
    print(f"\n" + "=" * 60)
    print(f"KS-3 OVERALL RESULT: {'PASS' if ks3_pass else 'FAIL'}")
    print(f"Back-reaction created measurable trajectory difference: {ks3_pass}")
    print("=" * 60)
    
    return ks3_pass, full_results
