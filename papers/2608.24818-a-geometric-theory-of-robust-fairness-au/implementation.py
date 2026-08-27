# implementation.py
"""
Neighborhood-based Fairness Audit Robustness Analysis Implementation.

This module implements a geometric framework for analyzing the robustness of
neighborhood-based fairness audits under bounded perturbations, based on the
provided paper review.

Key Concepts:
- Audit Score: Measures local consistency of model predictions among k-nearest neighbors.
- Bounded Perturbation: Adds uniform noise to feature space within a bound epsilon.
- Neighborhood Invariance: Measures how much the k-NN sets change under perturbation (Jaccard Similarity).
- Audit Volatility: The standard deviation of the Audit Score across multiple perturbations.
"""

import logging
import numpy as np
from typing import List, Tuple

# --- Configuration & Constants ---

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default experimental parameters
DEFAULT_N_SAMPLES = 1000
DEFAULT_DIM = 2
DEFAULT_K_NEIGHBORS = 5
DEFAULT_EPSILON_STEPS = [0.0, 0.1, 0.5, 1.0]
DEFAULT_TRIALS = 100
RANDOM_SEED = 42


def generate_synthetic_data(n_samples: int = DEFAULT_N_SAMPLES, dim: int = DEFAULT_DIM, seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Section: Experimental Setup
    
    Generates synthetic data points and associated model predictions (y_pred) 
    for the fairness audit analysis.
    
    The data is generated as Gaussian clusters to simulate a non-trivial feature 
    distribution. Predictions are generated using a simple non-linear function 
    of the features to provide a baseline "model".
    
    Args:
        n_samples: Number of data points to generate.
        dim: Dimensionality of the feature space.
        seed: Random seed for reproducibility.
        
    Returns:
        X: numpy array of shape (n_samples, dim) containing feature data.
        y_pred: numpy array of shape (n_samples,) containing predicted values.
    """
    logger.info(f"Generating synthetic data: {n_samples} samples in {dim}D space. Seed={seed}")
    np.random.seed(seed)
    
    # Generate data from a standard normal distribution
    # This creates a relatively uniform distribution in the space
    X = np.random.randn(n_samples, dim)
    
    # Generate predictions using a simple non-linear model
    # y = sin(X[:,0]) + X[:,1] + small noise
    y_pred = np.sin(X[:, 0]) + X[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    logger.debug(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    logger.debug(f"y_pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    return X, y_pred


def calculate_audit_score(X: np.ndarray, y: np.ndarray, k: int = DEFAULT_K_NEIGHBORS) -> float:
    """
    Section: Neighborhood-based Fairness Definition
    
    Calculates the neighborhood-based fairness audit score.
    
    For each point, it finds its k-nearest neighbors (excluding self).
    It calculates the Mean Absolute Error (MAE) between the point's prediction 
    and its neighbors' predictions. The final score is the average of these 
    MAEs across all points.
    
    Lower score indicates better local fairness (predictions are consistent among neighbors).
    
    Args:
        X: Feature data (n_samples, dim).
        y: Predicted values (n_samples,).
        k: Number of nearest neighbors to consider.
        
    Returns:
        float: The average audit score (MAE across neighborhoods).
    """
    logger.info(f"Calculating Audit Score. k={k}, n_samples={X.shape[0]}")
    
    n_samples, dim = X.shape
    
    if k >= n_samples:
        logger.warning(f"k ({k}) is greater than or equal to n_samples ({n_samples}). Adjusting k to n_samples-1.")
        k = n_samples - 1
        
    # Calculate pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    # To save memory for large N, we do this in a vectorized way but careful about memory.
    # For n=1000, a 1000x1000 matrix is fine.
    
    # Squared distances
    # dist_sq[i, j] = ||X[i] - X[j]||^2
    X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
    dist_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
    
    # Ensure non-negative due to floating point errors
    dist_sq = np.maximum(dist_sq, 0)
    
    audit_scores = []
    
    for i in range(n_samples):
        # Get distances from i to all others
        distances = dist_sq[i, :].copy()
        distances[i] = np.inf # Exclude self
        
        # Get indices of k nearest neighbors
        # argsort returns indices that would sort the array
        neighbor_indices = np.argpartition(distances, k)[:k]
        
        # Get predictions of neighbors
        neighbor_preds = y[neighbor_indices]
        
        # Calculate MAE between y[i] and neighbor_preds
        mae = np.mean(np.abs(y[i] - neighbor_preds))
        audit_scores.append(mae)
        
        if i % 200 == 0 and i < n_samples:
            logger.debug(f"Processed sample {i}/{n_samples}, current avg score: {np.mean(audit_scores):.4f}")
    
    avg_score = np.mean(audit_scores)
    logger.info(f"Final Audit Score: {avg_score:.4f}")
    
    return float(avg_score)


def apply_perturbation(X: np.ndarray, epsilon: float, seed: int = None) -> np.ndarray:
    """
    Section: Bounded Perturbations
    
    Applies a bounded uniform perturbation to the feature data.
    
    Each feature value is perturbed by adding a random value drawn from
    Uniform(-epsilon, epsilon).
    
    Args:
        X: Original feature data (n_samples, dim).
        epsilon: The bound of the perturbation.
        seed: Random seed for this specific perturbation trial.
        
    Returns:
        X_perturbed: The perturbed feature data.
    """
    if seed is not None:
        np.random.seed(seed)
        
    logger.debug(f"Applying perturbation with epsilon={epsilon}.")
    
    # Generate uniform noise
    noise = np.random.uniform(low=-epsilon, high=epsilon, size=X.shape)
    
    X_perturbed = X + noise
    
    logger.debug(f"Perturbed data range: [{X_perturbed.min():.4f}, {X_perturbed.max():.4f}]")
    
    return X_perturbed


def check_neighborhood_invariance(X_orig: np.ndarray, X_perturbed: np.ndarray, k: int = DEFAULT_K_NEIGHBORS) -> float:
    """
    Section: Neighborhood Invariance Conditions
    
    Checks the stability of the k-neighborhood sets under perturbation.
    
    For each point, it compares the set of k-nearest neighbors in the original 
    space vs the perturbed space. It calculates the Jaccard Similarity of these 
    two sets. The result is the average Jaccard Similarity across all points.
    
    Args:
        X_orig: Original feature data.
        X_perturbed: Perturbed feature data.
        k: Number of neighbors.
        
    Returns:
        float: Average Jaccard Similarity (0.0 to 1.0). 1.0 means perfect invariance.
    """
    logger.info(f"Checking Neighborhood Invariance. k={k}")
    
    n_samples, dim = X_orig.shape
    if n_samples != X_perturbed.shape[0]:
        raise ValueError("X_orig and X_perturbed must have the same number of samples.")
    
    # Helper to get k-NN indices
    def get_knn_indices(X: np.ndarray) -> List[np.ndarray]:
        X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
        dist_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
        dist_sq = np.maximum(dist_sq, 0)
        
        indices_list = []
        for i in range(n_samples):
            distances = dist_sq[i, :].copy()
            distances[i] = np.inf
            indices = np.argpartition(distances, k)[:k]
            indices_list.append(indices)
        return indices_list
    
    orig_indices = get_knn_indices(X_orig)
    pert_indices = get_knn_indices(X_perturbed)
    
    jaccard_scores = []
    
    for i in range(n_samples):
        set_orig = set(orig_indices[i])
        set_pert = set(pert_indices[i])
        
        # Jaccard Similarity = |A ∩ B| / |A ∪ B|
        intersection = len(set_orig.intersection(set_pert))
        union = len(set_orig.union(set_pert))
        
        if union == 0:
            jaccard = 1.0
        else:
            jaccard = intersection / union
            
        jaccard_scores.append(jaccard)
        
    avg_jaccard = np.mean(jaccard_scores)
    logger.info(f"Average Neighborhood Invariance (Jaccard): {avg_jaccard:.4f}")
    
    return float(avg_jaccard)


def calculate_audit_volatility(X: np.ndarray, y: np.ndarray, epsilon: float, trials: int = DEFAULT_TRIALS, k: int = DEFAULT_K_NEIGHBORS, base_seed: int = RANDOM_SEED) -> Tuple[float, float]:
    """
    Section: Audit Volatility Definition
    
    Calculates the Audit Volatility, defined as the standard deviation of the 
    Audit Score over multiple perturbations with a given epsilon.
    
    It also returns the average Neighborhood Invariance for the context.
    
    Args:
        X: Original feature data.
        y: Predicted values.
        epsilon: Perturbation magnitude.
        trials: Number of perturbation trials.
        k: Number of neighbors.
        base_seed: Base seed for reproducibility.
        
    Returns:
        Tuple[float, float]: (Audit Volatility (Std Dev), Average Jaccard Invariance)
    """
    logger.info(f"Calculating Audit Volatility. Epsilon={epsilon}, Trials={trials}")
    
    scores = []
    invariance_scores = []
    
    for t in range(trials):
        # Generate unique seed for each trial to ensure different noise
        current_seed = base_seed + t
        
        X_pert = apply_perturbation(X, epsilon, seed=current_seed)
        
        # Calculate audit score on perturbed data
        # Note: y remains the same, only X changes. This tests the robustness 
        # of the *neighborhood structure* and thus the audit result.
        score_t = calculate_audit_score(X_pert, y, k=k)
        scores.append(score_t)
        
        # Calculate invariance for this specific perturbation (optional, expensive)
        # For efficiency in the demo, we might sample fewer invariance checks 
        # or just check invariance once per epsilon block in the main loop.
        # However, to be rigorous, let's calculate it.
        jac_t = check_neighborhood_invariance(X, X_pert, k=k)
        invariance_scores.append(jac_t)
        
        if (t + 1) % 10 == 0:
            logger.debug(f"  Trial {t+1}/{trials}: Score={score_t:.4f}, Jaccard={jac_t:.4f}")
    
    volatility = np.std(scores)
    avg_invariance = np.mean(invariance_scores)
    
    logger.info(f"Result for Epsilon={epsilon}: Volatility={volatility:.4f}, Avg Invariance={avg_invariance:.4f}")
    
    return float(volatility), float(avg_invariance)


def run_experiments(X: np.ndarray, y: np.ndarray, epsilon_steps: List[float], trials: int = DEFAULT_TRIALS, k: int = DEFAULT_K_NEIGHBORS) -> List[dict]:
    """
    Section: Main Results / Stability Analysis
    
    Runs the robustness analysis by varying the perturbation magnitude (epsilon).
    
    Args:
        X: Feature data.
        y: Predictions.
        epsilon_steps: List of epsilon values to test.
        trials: Number of trials per epsilon.
        k: Number of neighbors.
        
    Returns:
        List[dict]: Results for each epsilon step.
    """
    logger.info("Starting Robustness Analysis...")
    logger.info("-" * 50)
    logger.info(f"{'Epsilon':<10} | {'Avg Invariance (Jaccard)':<25} | {'Audit Volatility (Std)':<25}")
    logger.info("-" * 50)
    
    results = []
    
    # Baseline score with no perturbation
    base_score = calculate_audit_score(X, y, k=k)
    logger.info(f"Base Audit Score (Epsilon=0.0, single run): {base_score:.4f}")
    logger.info("-" * 50)
    
    for eps in epsilon_steps:
        volatility, invariance = calculate_audit_volatility(X, y, eps, trials=trials, k=k)
        
        results.append({
            'epsilon': eps,
            'volatility': volatility,
            'invariance': invariance
        })
        
        # Format and log the table row
        logger.info(f"{eps:<10.2f} | {invariance:<25.4f} | {volatility:<25.4f}")
        
    logger.info("-" * 50)
    logger.info("Robustness Analysis Completed.")
    
    return results


def main():
    """
    Main entry point for the demo execution.
    """
    logger.info("Initializing Robust Fairness Audit Demo")
    
    # 1. Data Generation
    n_samples = DEFAULT_N_SAMPLES
    dim = DEFAULT_DIM
    X, y = generate_synthetic_data(n_samples, dim)
    
    # 2. Run Experiments
    epsilon_steps = DEFAULT_EPSILON_STEPS
    trials = DEFAULT_TRIALS
    k = DEFAULT_K_NEIGHBORS
    
    results = run_experiments(X, y, epsilon_steps, trials=trials, k=k)
    
    # 3. Summary
    logger.info("Summary of Results:")
    for res in results:
        logger.info(f"  Epsilon={res['epsilon']:.2f}: Volatility={res['volatility']:.4f}, Invariance={res['invariance']:.4f}")


if __name__ == "__main__":
    main()
