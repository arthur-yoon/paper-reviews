# test_implementation.py
"""
Unit tests for the ELR Collapse implementation.
"""

import unittest
import numpy as np
from implementation import (
    calculate_effective_learning_rate,
    compute_collapse_error,
    fit_and_transfer_fsl,
    detect_delayed_acceleration,
    apply_hyperball_constraint,
    ELRTracker
)


class TestELRFunctions(unittest.TestCase):
    """Tests for core ELR mathematical functions."""
    
    def test_calculate_effective_learning_rate(self):
        """
        Tests the definition ELR = LR / Norm.
        """
        # Case 1: Standard values
        lr = 0.1
        norm = 2.0
        elr = calculate_effective_learning_rate(lr, norm)
        self.assertAlmostEqual(elr, 0.05, places=5)
        
        # Case 2: Zero norm (should return 0 to avoid inf/nan)
        elr_zero = calculate_effective_learning_rate(0.1, 0.0)
        self.assertEqual(elr_zero, 0.0)
        
        # Case 3: Very small norm
        elr_small = calculate_effective_learning_rate(0.1, 1e-9)
        self.assertGreater(elr_small, 0.0)
        
    def test_compute_collapse_error(self):
        """
        Tests the calculation of absolute and relative collapse errors.
        """
        loss_a = np.array([1.0, 2.0, 3.0, 4.0])
        loss_b = np.array([1.1, 2.2, 2.8, 4.4])
        
        metrics = compute_collapse_error(loss_a, loss_b)
        
        # Mean Absolute Error
        # Diffs: 0.1, 0.2, 0.2, 0.4
        # Mean: 0.9 / 4 = 0.225
        self.assertAlmostEqual(metrics['mean_absolute_error'], 0.225, places=5)
        
        # Max Relative Error
        # Rel Errs: 
        # t0: |1.0-1.1|/1.1 = 0.1/1.1 ~ 0.09
        # t1: |2.0-2.2|/2.2 = 0.2/2.2 ~ 0.09
        # t2: |3.0-2.8|/3.0 = 0.2/3.0 ~ 0.066
        # t3: |4.0-4.4|/4.4 = 0.4/4.4 ~ 0.09
        # Max should be ~0.09
        self.assertLess(metrics['max_relative_error'], 0.15)
        self.assertGreater(metrics['max_relative_error'], 0.05)
        
    def test_compute_collapse_error_mismatched_lengths(self):
        """
        Tests error handling for mismatched curve lengths.
        """
        loss_a = np.array([1.0, 2.0])
        loss_b = np.array([1.0])
        
        with self.assertRaises(ValueError):
            compute_collapse_error(loss_a, loss_b)
            
    def test_fit_and_transfer_fsl(self):
        """
        Tests the Fitted Scaling Law prediction.
        """
        # Generate synthetic data: Loss decreases with ELR (inverse relationship)
        # L = 1 / ELR
        elr_curve_1 = np.array([0.1, 0.2, 0.5, 1.0])
        loss_curve_1 = 1.0 / elr_curve_1 # [10, 5, 2, 1]
        
        # Predict for new ELRs
        elr_curve_2 = np.array([0.4, 0.8])
        
        predicted = fit_and_transfer_fsl(elr_curve_1, loss_curve_1, elr_curve_2)
        
        # The fit is a polynomial of degree 2. 
        # 1/x is not a polynomial, so the fit will be an approximation.
        # We just check that it returns an array of the correct shape and finite values.
        self.assertEqual(len(predicted), len(elr_curve_2))
        self.assertTrue(np.all(np.isfinite(predicted)))
        
        # Check that predictions are somewhat in the right range (positive)
        self.assertTrue(np.all(predicted > 0))
        
    def test_detect_delayed_acceleration(self):
        """
        Tests detection of sudden loss improvement.
        """
        # Slow decrease, then sudden drop
        loss_curve = np.array([10.0, 9.9, 9.8, 9.7, 9.6, 5.0, 4.9, 4.8])
        
        is_detected, step_idx = detect_delayed_acceleration(loss_curve)
        
        # We expect detection because of the drop from 9.6 to 5.0
        # Second diff around index 4-5: 
        # 9.6 - 2*9.7 + 9.8 = 0 (before drop)
        # 5.0 - 2*9.6 + 9.7 = 5.0 - 19.2 + 9.7 = -4.5 (large negative)
        
        if is_detected:
            # The step index should be near the drop (step 5 or 6 in 0-indexed original, 
            # but diff indices are shifted)
            self.assertGreater(step_idx, 0)
            self.assertLess(step_idx, len(loss_curve))
        else:
            # If not detected, it's a failure of the heuristic, but for this test
            # we want to ensure the function runs.
            self.fail("Acceleration should be detected")
            
    def test_apply_hyperball_constraint(self):
        """
        Tests the Hyperball norm clipping.
        """
        # Create weights with norm > 1.0
        weights = {
            "w1": np.array([1.0, 1.0, 1.0]), # Norm = sqrt(3) ~ 1.732
            "w2": np.array([0.0, 0.0, 0.0])
        }
        
        max_norm = 1.0
        constrained_weights = apply_hyperball_constraint(weights, max_norm)
        
        # Calculate new global norm
        total_sq = 0.0
        for w in constrained_weights.values():
            total_sq += np.sum(w ** 2)
        new_norm = np.sqrt(total_sq)
        
        # Norm should be <= max_norm
        self.assertLessEqual(new_norm, max_norm + 1e-5)
        
        # Weights should be scaled down
        self.assertLess(np.sum(constrained_weights["w1"] ** 2), np.sum(weights["w1"] ** 2))
        
        # Test case where norm is already within limit
        weights_small = {
            "w1": np.array([0.1, 0.1, 0.1]),
        }
        constrained_small = apply_hyperball_constraint(weights_small, max_norm)
        self.assertTrue(np.allclose(constrained_small["w1"], weights_small["w1"]))
        
    def test_elr_tracker(self):
        """
        Tests the ELRTracker data structure.
        """
        tracker = ELRTracker()
        
        for i in range(5):
            tracker.record(i, lr=0.1, norm=1.0, elr=0.1, loss=2.0 - 0.1*i)
            
        arrays = tracker.get_arrays()
        
        self.assertEqual(len(arrays['lr']), 5)
        self.assertEqual(len(arrays['loss']), 5)
        self.assertTrue(np.all(arrays['steps'] == np.arange(5)))
        self.assertAlmostEqual(np.mean(arrays['loss']), 2.0 - 0.1*2.0, places=5)


if __name__ == '__main__':
    unittest.main()
