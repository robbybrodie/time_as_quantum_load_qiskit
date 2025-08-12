"""
Tests for back-reaction loop stability

This module tests that the KS-3 feedback loop remains stable and produces
finite, bounded results.
"""

import pytest
import numpy as np
from capacity_time.circuits import create_standard_lattice, create_comparison_schedules
from capacity_time.demand_capacity import create_default_demand_estimator, create_default_capacity_model
from capacity_time.backreaction import BackReactionLoop, compare_backreaction_vs_control


class TestBackReactionStability:
    """Test stability of back-reaction feedback loop"""
    
    @pytest.fixture
    def setup_system(self):
        """Create standard system components for testing"""
        lattice = create_standard_lattice(n_qubits=4)  # Small for fast tests
        demand_estimator = create_default_demand_estimator()
        capacity_model = create_default_capacity_model()
        
        schedules = create_comparison_schedules(n_qubits=4, n_steps=6, amplitude=0.2)
        base_schedule = schedules['moving']
        
        return lattice, demand_estimator, capacity_model, base_schedule
    
    def test_finite_time_factors(self, setup_system):
        """Test that time factors N remain finite"""
        lattice, demand_estimator, capacity_model, base_schedule = setup_system
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        # Run short evolution
        results = loop.run_feedback_evolution(
            initial_pattern='superposition',
            base_schedule=base_schedule[:4],  # Very short for stability test
            feedback_strength=1.0
        )
        
        time_factors = results['time_factors']
        
        # Check all time factors are finite
        assert np.all(np.isfinite(time_factors)), "Some time factors are not finite"
        
        # Check time factors are positive
        assert np.all(time_factors > 0), "Some time factors are not positive"
        
        # Check time factors are bounded
        assert np.all(time_factors < 100), "Some time factors are too large"
        assert np.all(time_factors > 0.01), "Some time factors are too small"
    
    def test_demand_positivity(self, setup_system):
        """Test that demands remain non-negative"""
        lattice, demand_estimator, capacity_model, base_schedule = setup_system
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        results = loop.run_feedback_evolution(
            initial_pattern='superposition',
            base_schedule=base_schedule[:4],
            feedback_strength=0.5
        )
        
        demands = results['demands']
        
        # Check all demands are non-negative
        assert np.all(demands >= 0), "Some demands are negative"
        
        # Check demands are finite
        assert np.all(np.isfinite(demands)), "Some demands are not finite"
    
    def test_clipping_bounds(self, setup_system):
        """Test that time factor clipping works correctly"""
        lattice, demand_estimator, capacity_model, base_schedule = setup_system
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        # Test with tight clipping bounds
        results = loop.run_feedback_evolution(
            initial_pattern='superposition',
            base_schedule=base_schedule[:3],
            feedback_strength=1.0,
            min_factor=0.5,
            max_factor=2.0
        )
        
        time_factors = results['time_factors']
        
        # Check clipping bounds are respected
        assert np.all(time_factors >= 0.5), "Time factors below min_factor"
        assert np.all(time_factors <= 2.0), "Time factors above max_factor"
    
    def test_feedback_strength_scaling(self, setup_system):
        """Test that feedback strength scales effects appropriately"""
        lattice, demand_estimator, capacity_model, base_schedule = setup_system
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        # Run with different feedback strengths
        results_weak = loop.run_feedback_evolution(
            initial_pattern='superposition',
            base_schedule=base_schedule[:3],
            feedback_strength=0.1
        )
        
        results_strong = loop.run_feedback_evolution(
            initial_pattern='superposition', 
            base_schedule=base_schedule[:3],
            feedback_strength=1.0
        )
        
        # Stronger feedback should create more variation in time factors
        var_weak = np.var(results_weak['time_factors'])
        var_strong = np.var(results_strong['time_factors'])
        
        assert var_strong >= var_weak, "Stronger feedback should create more variation"
    
    def test_no_explosion_long_evolution(self, setup_system):
        """Test that long evolution doesn't lead to explosion"""
        lattice, demand_estimator, capacity_model, base_schedule = setup_system
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        # Create longer schedule by repeating
        long_schedule = base_schedule * 2  # Double length
        
        results = loop.run_feedback_evolution(
            initial_pattern='superposition',
            base_schedule=long_schedule,
            feedback_strength=0.8  # Strong but not maximum
        )
        
        time_factors = results['time_factors']
        demands = results['demands']
        
        # Check final values are still reasonable
        final_N = time_factors[-1]
        final_D = demands[-1]
        
        assert np.all(np.isfinite(final_N)), "Final time factors not finite"
        assert np.all(np.isfinite(final_D)), "Final demands not finite"
        assert np.all(final_N > 0), "Final time factors not positive"
        assert np.all(final_D >= 0), "Final demands negative"


class TestBackReactionComparison:
    """Test back-reaction vs control comparison functions"""
    
    def test_comparison_basic_functionality(self):
        """Test that comparison function runs without error"""
        # Setup small system
        lattice = create_standard_lattice(n_qubits=4)
        demand_estimator = create_default_demand_estimator()
        capacity_model = create_default_capacity_model()
        
        schedules = create_comparison_schedules(n_qubits=4, n_steps=4, amplitude=0.15)
        base_schedule = schedules['stationary']
        
        # Run feedback evolution
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        feedback_results = loop.run_feedback_evolution(
            'superposition', base_schedule, feedback_strength=0.5
        )
        
        # Run control evolution
        from capacity_time.backreaction import run_control_evolution
        control_results = run_control_evolution(lattice, 'superposition', base_schedule)
        
        # Compare - should not raise exceptions
        try:
            ks3_pass, comparison_data = compare_backreaction_vs_control(
                feedback_results, control_results, observable='Z', probe_qubit=1
            )
            
            # Basic sanity checks
            assert isinstance(ks3_pass, bool), "KS3 pass should be boolean"
            assert 'max_divergence' in comparison_data, "Missing max_divergence"
            assert 'final_diff' in comparison_data, "Missing final_diff"
            
            # Values should be finite
            assert np.isfinite(comparison_data['max_divergence']), "Max divergence not finite"
            assert np.isfinite(comparison_data['final_diff']), "Final diff not finite"
            
        except Exception as e:
            pytest.fail(f"Comparison function raised exception: {e}")
    
    def test_zero_feedback_gives_small_difference(self):
        """Test that zero feedback gives minimal difference vs control"""
        lattice = create_standard_lattice(n_qubits=4)
        demand_estimator = create_default_demand_estimator()
        capacity_model = create_default_capacity_model()
        
        schedules = create_comparison_schedules(n_qubits=4, n_steps=4, amplitude=0.1)
        base_schedule = schedules['uniform']  # Simple pattern
        
        # Run with zero feedback (should be nearly identical to control)
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        feedback_results = loop.run_feedback_evolution(
            'superposition', base_schedule, feedback_strength=0.0  # No feedback
        )
        
        from capacity_time.backreaction import run_control_evolution
        control_results = run_control_evolution(lattice, 'superposition', base_schedule)
        
        ks3_pass, comparison_data = compare_backreaction_vs_control(
            feedback_results, control_results, observable='Z', probe_qubit=1
        )
        
        # With zero feedback, difference should be very small
        assert comparison_data['max_divergence'] < 0.05, "Zero feedback shows large divergence"
        assert comparison_data['final_diff'] < 0.03, "Zero feedback shows large final difference"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_step_evolution(self):
        """Test back-reaction with minimal evolution"""
        lattice = create_standard_lattice(n_qubits=3)
        demand_estimator = create_default_demand_estimator()
        capacity_model = create_default_capacity_model()
        
        # Single time step
        single_schedule = [[0.1, 0.0, 0.1]]
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        # Should not crash
        results = loop.run_feedback_evolution(
            'superposition', single_schedule, feedback_strength=0.5
        )
        
        assert len(results['states']) == 2, "Should have initial + 1 evolved state"
        assert len(results['demands']) == 2, "Should have 2 demand measurements"
    
    def test_all_zero_schedule(self):
        """Test with all-zero angle schedule"""
        lattice = create_standard_lattice(n_qubits=3)
        demand_estimator = create_default_demand_estimator()
        capacity_model = create_default_capacity_model()
        
        # All-zero schedule (no evolution)
        zero_schedule = [[0.0, 0.0, 0.0]] * 3
        
        loop = BackReactionLoop(lattice, demand_estimator, capacity_model)
        
        results = loop.run_feedback_evolution(
            'superposition', zero_schedule, feedback_strength=1.0
        )
        
        # Should complete without error
        assert len(results['states']) == 4, "Should have initial + 3 states"
        
        # Time factors should be finite (may be large due to low/zero demand)
        time_factors = results['time_factors']
        assert np.all(np.isfinite(time_factors)), "Time factors not finite with zero schedule"


if __name__ == "__main__":
    pytest.main([__file__])
