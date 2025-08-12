"""
Tests for QFI small-delta behavior

This module tests the fundamental small-delta law: DB²(θ) ≈ C θ² for small θ
"""

import pytest
import numpy as np
from capacity_time.qfi import bures_small_delta, bures_distance_squared, fidelity
from capacity_time.clocks import pointer_state


class TestSmallDeltaLaw:
    """Test quadratic small-delta behavior"""
    
    def test_pointer_state_quadratic(self):
        """Test that pointer state shows quadratic behavior"""
        # Small angles for testing
        theta_list = [0.01, 0.02, 0.05, 0.1]
        
        theta_arr, DB2_arr, ratio_arr = bures_small_delta(
            theta_list, lambda theta: pointer_state(theta)
        )
        
        # Check that ratios are approximately constant
        mean_ratio = np.mean(ratio_arr)
        std_ratio = np.std(ratio_arr)
        relative_std = std_ratio / mean_ratio
        
        # Assert ratio consistency (within 20% for numerical stability)
        assert relative_std < 0.2, f"Ratio variation too large: {relative_std:.3f}"
        
        # Assert reasonable magnitude (should be around 0.25 for pointer state)
        assert 0.1 < mean_ratio < 0.5, f"Mean ratio outside expected range: {mean_ratio:.3f}"
    
    def test_zero_angle_gives_zero_distance(self):
        """Test that θ=0 gives DB²=0"""
        state_0 = pointer_state(0.0)
        state_small = pointer_state(1e-6)
        
        # Distance to itself should be zero
        assert bures_distance_squared(state_0, state_0) < 1e-10
        
        # Very small angle should give very small distance
        db2_small = bures_distance_squared(state_0, state_small)
        assert db2_small < 1e-10
    
    def test_fidelity_bounds(self):
        """Test that fidelity stays in [0,1]"""
        angles = [0, 0.1, 0.5, 1.0, np.pi/2, np.pi]
        ref_state = pointer_state(0.0)
        
        for theta in angles:
            test_state = pointer_state(theta)
            F = fidelity(ref_state, test_state)
            
            assert 0 <= F <= 1, f"Fidelity {F:.4f} outside [0,1] for θ={theta:.3f}"
    
    def test_distance_symmetry(self):
        """Test that DB²(θ₁,θ₂) = DB²(θ₂,θ₁)"""
        theta1, theta2 = 0.1, 0.15
        state1 = pointer_state(theta1)
        state2 = pointer_state(theta2)
        
        db2_12 = bures_distance_squared(state1, state2)
        db2_21 = bures_distance_squared(state2, state1)
        
        assert abs(db2_12 - db2_21) < 1e-12, "Bures distance not symmetric"
    
    def test_triangle_inequality(self):
        """Test triangle inequality for Bures distance"""
        # Create three states
        state_a = pointer_state(0.0)
        state_b = pointer_state(0.1)
        state_c = pointer_state(0.2)
        
        # Compute distances (take square root for metric)
        d_ab = np.sqrt(bures_distance_squared(state_a, state_b))
        d_bc = np.sqrt(bures_distance_squared(state_b, state_c))
        d_ac = np.sqrt(bures_distance_squared(state_a, state_c))
        
        # Triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)
        assert d_ac <= d_ab + d_bc + 1e-10, "Triangle inequality violated"


class TestNumericalStability:
    """Test numerical stability of QFI calculations"""
    
    def test_very_small_angles(self):
        """Test behavior for very small angles"""
        tiny_angles = [1e-6, 1e-5, 1e-4]
        
        for theta in tiny_angles:
            state_0 = pointer_state(0.0)
            state_theta = pointer_state(theta)
            
            db2 = bures_distance_squared(state_0, state_theta)
            ratio = db2 / (theta**2)
            
            # Should be finite and positive
            assert np.isfinite(ratio), f"Ratio not finite for θ={theta}"
            assert ratio > 0, f"Ratio not positive for θ={theta}"
            assert ratio < 1.0, f"Ratio too large for θ={theta}"
    
    def test_large_angles(self):
        """Test behavior for large angles"""
        large_angles = [np.pi/2, np.pi, 2*np.pi]
        ref_state = pointer_state(0.0)
        
        for theta in large_angles:
            test_state = pointer_state(theta)
            
            F = fidelity(ref_state, test_state)
            db2 = bures_distance_squared(ref_state, test_state)
            
            # Should be finite
            assert np.isfinite(F), f"Fidelity not finite for θ={theta}"
            assert np.isfinite(db2), f"DB² not finite for θ={theta}"
            
            # Should be bounded
            assert 0 <= F <= 1, f"Fidelity out of bounds for θ={theta}"
            assert 0 <= db2 <= 2, f"DB² out of bounds for θ={theta}"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_identical_states(self):
        """Test distance between identical states"""
        state = pointer_state(0.5)
        
        F = fidelity(state, state)
        db2 = bures_distance_squared(state, state)
        
        assert abs(F - 1.0) < 1e-12, "Identical states should have fidelity 1"
        assert db2 < 1e-12, "Identical states should have zero distance"
    
    def test_orthogonal_states(self):
        """Test distance between orthogonal states"""
        # |+⟩ and |+⟩ rotated by π should be orthogonal
        state_plus = pointer_state(0.0)  # |+⟩
        state_minus = pointer_state(np.pi)  # Should be |-⟩
        
        F = fidelity(state_plus, state_minus)
        db2 = bures_distance_squared(state_plus, state_minus)
        
        # Orthogonal pure states have F=0, DB²=2
        assert F < 1e-10, "Orthogonal states should have zero fidelity"
        assert abs(db2 - 2.0) < 1e-10, "Orthogonal states should have DB²=2"


if __name__ == "__main__":
    pytest.main([__file__])
