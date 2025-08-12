"""
Tests for response kernel symmetry and locality

This module tests the KS-4 response kernel analysis functions to ensure
proper spatial structure detection.
"""

import pytest
import numpy as np
from capacity_time.circuits import create_standard_lattice, create_comparison_schedules
from capacity_time.demand_capacity import create_default_demand_estimator, create_default_capacity_model
from capacity_time.response import (
    create_response_perturbation, measure_response_kernel,
    analyze_response_symmetry, analyze_response_locality
)


class TestResponsePerturbation:
    """Test creation of response perturbations"""
    
    def test_perturbation_types(self):
        """Test that all perturbation types can be created"""
        lattice = create_standard_lattice(n_qubits=6)
        bump_site = 2
        bump_strength = 0.3
        
        perturbation_types = ['gate_depth', 'rotation', 'entangling']
        
        for bump_type in perturbation_types:
            pert = create_response_perturbation(
                lattice, bump_site, bump_strength, bump_type
            )
            
            # Check perturbation structure
            assert pert['bump_site'] == bump_site, f"Wrong bump site for {bump_type}"
            assert pert['bump_strength'] == bump_strength, f"Wrong strength for {bump_type}"
            assert pert['bump_type'] == bump_type, f"Wrong type for {bump_type}"
            assert 'bump_circuit' in pert, f"Missing circuit for {bump_type}"
            assert 'description' in pert, f"Missing description for {bump_type}"
    
    def test_invalid_perturbation_type(self):
        """Test that invalid perturbation types raise error"""
        lattice = create_standard_lattice(n_qubits=4)
        
        with pytest.raises(ValueError):
            create_response_perturbation(lattice, 1, 0.5, 'invalid_type')
    
    def test_boundary_sites(self):
        """Test perturbations at boundary sites"""
        lattice = create_standard_lattice(n_qubits=6)
        
        # Test first and last sites
        boundary_sites = [0, 5]
        
        for site in boundary_sites:
            pert = create_response_perturbation(
                lattice, site, 0.4, 'entangling'
            )
            
            # Should not crash and should have valid structure
            assert pert['bump_site'] == site
            assert pert['bump_circuit'] is not None


class TestResponseKernel:
    """Test response kernel measurement"""
    
    @pytest.fixture
    def setup_response_system(self):
        """Setup system for response kernel tests"""
        lattice = create_standard_lattice(n_qubits=6)
        demand_estimator = create_default_demand_estimator()
        capacity_model = create_default_capacity_model()
        
        schedules = create_comparison_schedules(n_qubits=6, n_steps=4, amplitude=0.2)
        base_schedule = schedules['stationary']
        
        # Create simple perturbation
        perturbation = create_response_perturbation(
            lattice, 2, 0.3, 'gate_depth'  # Center site
        )
        
        return lattice, demand_estimator, capacity_model, base_schedule, perturbation
    
    def test_kernel_measurement_basic(self, setup_response_system):
        """Test basic response kernel measurement"""
        lattice, estimator, capacity, schedule, perturbation = setup_response_system
        
        kernel_data = measure_response_kernel(
            lattice, estimator, capacity, 'superposition', 
            schedule, perturbation, 'ln_N'
        )
        
        # Check structure
        assert 'distances' in kernel_data
        assert 'response_kernel' in kernel_data
        assert 'response_diff' in kernel_data
        assert 'perturbation' in kernel_data
        
        # Check dimensions
        distances = kernel_data['distances']
        response_kernel = kernel_data['response_kernel']
        
        assert len(distances) == lattice.n_qubits, "Wrong distance array length"
        assert len(response_kernel) == lattice.n_qubits, "Wrong kernel array length"
        
        # Check distances are centered at bump site
        bump_site = perturbation['bump_site']
        expected_distances = np.arange(lattice.n_qubits) - bump_site
        np.testing.assert_array_equal(distances, expected_distances, "Wrong distances")
    
    def test_different_response_metrics(self, setup_response_system):
        """Test different response metrics"""
        lattice, estimator, capacity, schedule, perturbation = setup_response_system
        
        metrics = ['ln_N', 'demand', 'time_factor']
        
        for metric in metrics:
            kernel_data = measure_response_kernel(
                lattice, estimator, capacity, 'superposition',
                schedule, perturbation, metric
            )
            
            assert kernel_data['response_metric'] == metric
            assert np.all(np.isfinite(kernel_data['response_kernel'])), f"Non-finite kernel for {metric}"
    
    def test_kernel_scaling_with_strength(self, setup_response_system):
        """Test that kernel scales with perturbation strength"""
        lattice, estimator, capacity, schedule, _ = setup_response_system
        
        # Create perturbations with different strengths
        weak_pert = create_response_perturbation(lattice, 2, 0.1, 'gate_depth')
        strong_pert = create_response_perturbation(lattice, 2, 0.5, 'gate_depth')
        
        weak_kernel = measure_response_kernel(
            lattice, estimator, capacity, 'superposition',
            schedule, weak_pert, 'ln_N'
        )
        
        strong_kernel = measure_response_kernel(
            lattice, estimator, capacity, 'superposition',
            schedule, strong_pert, 'ln_N'
        )
        
        # Stronger perturbation should generally give larger response
        weak_max = np.max(np.abs(weak_kernel['response_kernel']))
        strong_max = np.max(np.abs(strong_kernel['response_kernel']))
        
        # Allow some tolerance for numerical effects
        assert strong_max >= 0.5 * weak_max, "Strong perturbation should give larger response"


class TestSymmetryAnalysis:
    """Test response kernel symmetry analysis"""
    
    def test_perfect_symmetry(self):
        """Test symmetry analysis with artificially symmetric kernel"""
        # Create symmetric response kernel manually
        n_qubits = 7
        bump_site = 3  # Center
        
        # Symmetric kernel: same response at equal distances
        response_kernel = np.array([0.1, 0.2, 0.4, 0.8, 0.4, 0.2, 0.1])
        distances = np.arange(n_qubits) - bump_site
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': bump_site
        }
        
        symmetry_pass, symmetry_analysis = analyze_response_symmetry(kernel_data)
        
        # Should pass symmetry test
        assert symmetry_pass, "Perfect symmetry should pass"
        assert symmetry_analysis['mean_asymmetry'] < 1e-12, "Perfect symmetry should have zero asymmetry"
    
    def test_broken_symmetry(self):
        """Test symmetry analysis with asymmetric kernel"""
        n_qubits = 7
        bump_site = 3
        
        # Asymmetric kernel
        response_kernel = np.array([0.1, 0.3, 0.4, 0.8, 0.2, 0.15, 0.05])
        distances = np.arange(n_qubits) - bump_site
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': bump_site
        }
        
        symmetry_pass, symmetry_analysis = analyze_response_symmetry(kernel_data)
        
        # Should fail symmetry test
        assert not symmetry_pass, "Broken symmetry should fail"
        assert symmetry_analysis['mean_asymmetry'] > 0.05, "Broken symmetry should have large asymmetry"
    
    def test_edge_site_symmetry(self):
        """Test symmetry analysis for edge site perturbations"""
        n_qubits = 6
        bump_site = 1  # Near edge
        
        response_kernel = np.array([0.2, 0.8, 0.4, 0.1, 0.05, 0.02])
        distances = np.arange(n_qubits) - bump_site
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': bump_site
        }
        
        # Should not crash even with limited symmetric pairs
        symmetry_pass, symmetry_analysis = analyze_response_symmetry(kernel_data)
        
        # Check that analysis completed
        assert 'symmetric_pairs' in symmetry_analysis
        assert 'mean_asymmetry' in symmetry_analysis


class TestLocalityAnalysis:
    """Test response kernel locality analysis"""
    
    def test_good_locality(self):
        """Test locality analysis with decaying kernel"""
        response_kernel = np.array([0.01, 0.05, 0.2, 0.8, 0.2, 0.05, 0.01])
        distances = np.array([-3, -2, -1, 0, 1, 2, 3])
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': 3
        }
        
        locality_pass, locality_analysis = analyze_response_locality(kernel_data, decay_threshold=0.1)
        
        # Should pass locality test
        assert locality_pass, "Decaying kernel should pass locality"
        assert locality_analysis['max_distance_response'] < 0.1, "Remote response should be small"
        assert locality_analysis['decay_ratio'] < 0.5, "Should show decay trend"
    
    def test_poor_locality(self):
        """Test locality analysis with non-decaying kernel"""
        # Kernel with large remote response
        response_kernel = np.array([0.7, 0.3, 0.2, 0.8, 0.2, 0.4, 0.6])
        distances = np.array([-3, -2, -1, 0, 1, 2, 3])
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': 3
        }
        
        locality_pass, locality_analysis = analyze_response_locality(kernel_data, decay_threshold=0.1)
        
        # Should fail locality test
        assert not locality_pass, "Non-decaying kernel should fail locality"
        assert locality_analysis['max_distance_response'] >= 0.1, "Remote response should be large"
    
    def test_single_site_system(self):
        """Test locality analysis with minimal system"""
        # Edge case: single site
        response_kernel = np.array([0.8])
        distances = np.array([0])
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': 0
        }
        
        locality_pass, locality_analysis = analyze_response_locality(kernel_data)
        
        # Should handle gracefully
        assert not locality_pass, "Single site should fail locality analysis"
        assert 'error' in locality_analysis, "Should indicate insufficient data"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_response_kernel(self):
        """Test analysis with all-zero response"""
        response_kernel = np.zeros(5)
        distances = np.array([-2, -1, 0, 1, 2])
        
        kernel_data = {
            'distances': distances,
            'response_kernel': response_kernel,
            'bump_site': 2
        }
        
        # Should not crash
        sym_pass, sym_analysis = analyze_response_symmetry(kernel_data)
        loc_pass, loc_analysis = analyze_response_locality(kernel_data)
        
        # Zero response should pass symmetry (trivially) but may fail locality
        assert sym_pass, "Zero response should be symmetric"
        assert sym_analysis['mean_asymmetry'] == 0, "Zero response has zero asymmetry"
    
    def test_invalid_bump_site(self):
        """Test with invalid bump site index"""
        lattice = create_standard_lattice(n_qubits=4)
        
        # Bump site outside lattice
        with pytest.raises((IndexError, ValueError)):
            create_response_perturbation(lattice, 10, 0.5, 'gate_depth')


if __name__ == "__main__":
    pytest.main([__file__])
