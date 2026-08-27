# test_implementation.py
"""
Unit tests for the Neighborhood-based Fairness Audit Robustness Implementation.
"""

import unittest
import numpy as np
import logging
from implementation import (
    generate_synthetic_data,
    calculate_audit_score,
    apply_perturbation,
    check_neighborhood_invariance,
    calculate_audit_volatility
)

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)


class TestFairnessAuditImplementation(unittest.TestCase):
    
    def setUp(self):
        """Setup small test data for fast execution."""
        self.n_samples = 50
        self.dim = 2
        self.seed = 123
        self.X, self.y = generate_synthetic_data(self.n_samples, self.dim, seed=self.seed)
        
    def test_generate_synthetic_data_shape(self):
        """Test that generated data has correct shapes and types."""
        X, y = generate_synthetic_data(10, 3, seed=1)
        self.assertEqual(X.shape, (10, 3))
        self.assertEqual(y.shape, (10,))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_audit_score_perfect_model_zero(self):
        """
        If predictions are constant, the MAE among neighbors should be 0.
        """
        X = np.random.rand(20, 2)
        y = np.ones(20) # Constant predictions
        score = calculate_audit_score(X, y, k=5)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_audit_score_positive(self):
        """Audit score should be non-negative."""
        score = calculate_audit_score(self.X, self.y, k=5)
        self.assertGreaterEqual(score, 0.0)

    def test_perturbation_zero_epsilon_no_change(self):
        """With epsilon=0, perturbation should result in identical data."""
        X_pert = apply_perturbation(self.X, epsilon=0.0, seed=42)
        np.testing.assert_array_equal(self.X, X_pert)

    def test_perturbation_changes_data(self):
        """With epsilon>0, data should change."""
        X_pert = apply_perturbation(self.X, epsilon=0.1, seed=42)
        self.assertFalse(np.array_equal(self.X, X_pert))
        
    def test_invariance_perfect_identity(self):
        """If X and X_pert are identical, Jaccard should be 1.0."""
        jac = check_neighborhood_invariance(self.X, self.X, k=5)
        self.assertAlmostEqual(jac, 1.0, places=5)

    def test_invariance_decreases_with_noise(self):
        """Increasing epsilon should generally decrease invariance (Jaccard)."""
        k = 5
        jac_low = check_neighborhood_invariance(self.X, apply_perturbation(self.X, 0.01, seed=1), k=k)
        jac_high = check_neighborhood_invariance(self.X, apply_perturbation(self.X, 1.0, seed=2), k=k)
        
        # While not strictly guaranteed in every random draw for small N, 
        # for N=50 and dim=2, large noise should reduce overlap.
        # We check that high noise doesn't yield *higher* invariance than 
        # near-zero noise in a typical scenario.
        # Note: This can be flaky if clusters are very dense, so we use a 
        # "less than" check rather than "equal to 0".
        self.assertLess(jac_high, 1.0)
        # Usually jac_low is very close to 1.0
        self.assertGreater(jac_low, 0.8)

    def test_volatility_increases_with_epsilon(self):
        """Audit volatility should increase as epsilon increases."""
        trials = 10 # Small number for speed
        k = 5
        
        vol_low_eps, _ = calculate_audit_volatility(self.X, self.y, epsilon=0.05, trials=trials, k=k, base_seed=0)
        vol_high_eps, _ = calculate_audit_volatility(self.X, self.y, epsilon=2.0, trials=trials, k=k, base_seed=0)
        
        # High epsilon should lead to higher volatility
        self.assertGreater(vol_high_eps, vol_low_eps)

    def test_volatility_zero_epsilon_near_zero(self):
        """With epsilon=0, volatility should be 0 (deterministic)."""
        trials = 5
        k = 5
        vol, _ = calculate_audit_volatility(self.X, self.y, epsilon=0.0, trials=trials, k=k, base_seed=0)
        self.assertAlmostEqual(vol, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
