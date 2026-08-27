import numpy as np

def isotonic_regression(x, y, increasing=True):
    """
    Performs isotonic regression using the Pool Adjacent Violators Algorithm (PAVA).
    Assumes x is sorted in ascending order.
    """
    n = len(y)
    if n == 0:
        return np.array([])
    
    # Initialize weights and means
    w = np.ones(n)
    y_hat = y.copy()
    
    # Stack to keep track of indices, or just work with arrays.
    # PAVA implementation
    # We maintain a list of blocks: [start_index, end_index, mean_value, weight]
    # For simplicity and O(N) average case with list, we can use a stack.
    
    blocks = []
    for i in range(n):
        # Start new block
        curr_w = w[i]
        curr_y = y_hat[i]
        
        # Merge with previous blocks if necessary
        while blocks:
            prev_w, prev_y, _ = blocks[-1]
            if (increasing and prev_y > curr_y) or (not increasing and prev_y < curr_y):
                # Violation: merge
                new_w = prev_w + curr_w
                new_y = (prev_w * prev_y + curr_w * curr_y) / new_w
                blocks.pop()
                curr_w = new_w
                curr_y = new_y
            else:
                break
        
        blocks.append((curr_w, curr_y, i))
    
    # Construct the output array
    y_iso = np.zeros(n)
    # Reconstruct blocks with start/end indices
    # The above simple stack doesn't track indices well for back-filling.
    # Let's use a more robust implementation with indices.
    
    # Reset
    blocks = [] # list of [start, end, mean, weight]
    for i in range(n):
        blocks.append([i, i, y[i], w[i]])
        
        while len(blocks) > 1:
            b_prev = blocks[-2]
            b_curr = blocks[-1]
            if (increasing and b_prev[2] > b_curr[2]) or (not increasing and b_prev[2] < b_curr[2]):
                # Merge
                total_w = b_prev[3] + b_curr[3]
                new_mean = (b_prev[2] * b_prev[3] + b_curr[2] * b_curr[3]) / total_w
                new_start = b_prev[0]
                new_end = b_curr[1]
                
                # Pop current
                blocks.pop()
                # Replace previous with merged
                blocks[-1] = [new_start, new_end, new_mean, total_w]
            else:
                break
        
    # Fill y_iso
    for start, end, mean, weight in blocks:
        y_iso[start:end+1] = mean
        
    return y_iso

def simulate_offline_rl_data(N=10000, dim=5, seed=42):
    """
    Simulates an offline RL scenario to generate:
    1. Initial importance weights (w_hat)
    2. True importance weights (w_true) for ground truth
    3. Rewards
    4. Bellman residuals for calibration target
    """
    np.random.seed(seed)
    
    # Simulate states
    states = np.random.randn(N, dim)
    
    # Simulate a "true" occupancy ratio distribution for target policy vs behavior policy
    # Let's assume a non-linear relationship to test isotonic capability.
    # w_true = f(state)
    # Let f be a smooth, non-decreasing function of a linear combination of states.
    score = np.dot(states, np.random.randn(dim))
    w_true = 1.0 + 2.0 * np.exp(score) + 0.1 * score**2
    
    # Initial estimate w_hat is noisy and has a bias (systematic error)
    # Noise
    noise = np.random.normal(0, 0.5, N)
    # Systematic bias: e.g., w_hat tends to be higher for high scores
    bias = 0.5 * score
    w_hat = w_true + noise + bias
    
    # Ensure weights are positive
    w_hat = np.abs(w_hat) + 0.1
    w_true = np.abs(w_true) + 0.1
    
    # Simulate rewards. Let reward depend on state and action (simplified)
    rewards = np.random.normal(0, 1, N) + 0.5 * score
    
    # Simulate "Bellman Target" or Residual Target.
    # In the context of the review, we want to find g such that g(w_hat) approximates the "ideal" weight
    # that satisfies the Bellman equation structure.
    # For this simulation, let's assume the "ideal" weight for calibration is w_true.
    # The review says: "isotonic regression ... dependent variable ... theoretical value (or residual minimization target)".
    # Let's use w_true as the target for the isotonic regression to "calibrate" w_hat.
    
    return w_hat, w_true, rewards, states

def compute_policy_value(w, rewards):
    """
    Estimates policy value using weighted average of rewards.
    V = sum(w * r) / sum(w)
    """
    if np.sum(w) == 0:
        return 0.0
    return np.sum(w * rewards) / np.sum(w)

def main():
    # 1. Generate Data
    N = 20000
    dim = 5
    w_hat, w_true, rewards, states = simulate_offline_rl_data(N, dim)
    
    # 2. Initial Policy Value Estimate
    V_initial = compute_policy_value(w_hat, rewards)
    
    # 3. Isotonic Bellman Calibration (IBC)
    # Sort w_hat to apply isotonic regression.
    # Isotonic regression requires the independent variable (x) to be sorted.
    sort_idx = np.argsort(w_hat)
    w_hat_sorted = w_hat[sort_idx]
    
    # The target for calibration. 
    # According to the review, we want to minimize Bellman Residual. 
    # In a simplified setup, if we had a proxy for the "correct" weight that satisfies Bellman eq, 
    # we would regress against that. Here, w_true serves as that ground truth proxy for demonstration.
    # Note: In a real scenario, we wouldn't have w_true. We'd have a target derived from Bellman residuals.
    # For this simulation, we use w_true mapped to the sorted w_hat indices.
    w_target_sorted = w_true[sort_idx]
    
    # Apply Isotonic Regression
    # g(w_hat) should approximate w_target (or a value consistent with Bellman structure)
    w_calibrated_sorted = isotonic_regression(w_hat_sorted, w_target_sorted, increasing=True)
    
    # Unsort to get calibrated weights in original order
    w_calibrated = np.zeros_like(w_calibrated_sorted)
    w_calibrated[sort_idx] = w_calibrated_sorted
    
    # 4. Calibrated Policy Value Estimate
    V_calibrated = compute_policy_value(w_calibrated, rewards)
    
    # 5. Ground Truth Value (using true weights)
    V_true = compute_policy_value(w_true, rewards)
    
    # 6. Metrics
    # Calculate MSE of weights (just for demonstration of "smoothing")
    mse_initial = np.mean((w_hat - w_true)**2)
    mse_calibrated = np.mean((w_calibrated - w_true)**2)
    
    # Calculate Error in Value Estimate
    err_initial = abs(V_initial - V_true)
    err_calibrated = abs(V_calibrated - V_true)
    
    # Print Results
    print(f"--- Isotonic Bellman Calibration (IBC) Simulation ---")
    print(f"Sample Size: {N}")
    print(f"")
    print(f"Policy Value Estimates:")
    print(f"  Initial Estimate V: {V_initial:.4f}")
    print(f"  Calibrated V:       {V_calibrated:.4f}")
    print(f"  Ground Truth V:     {V_true:.4f}")
    print(f"")
    print(f"Errors (|Estimate - True|):")
    print(f"  Initial Error:      {err_initial:.4f}")
    print(f"  Calibrated Error:   {err_calibrated:.4f}")
    print(f"")
    print(f"Weight MSE (vs Ground Truth Weights):")
    print(f"  Initial Weight MSE: {mse_initial:.4f}")
    print(f"  Calibrated Weight MSE: {mse_calibrated:.4f}")
    
    # Demonstration of distribution change
    print(f"")
    print(f"Weight Distribution Statistics:")
    print(f"  Initial: Min={np.min(w_hat):.2f}, Max={np.max(w_hat):.2f}, Mean={np.mean(w_hat):.2f}")
    print(f"  Calibrated: Min={np.min(w_calibrated):.2f}, Max={np.max(w_calibrated):.2f}, Mean={np.mean(w_calibrated):.2f}")

if __name__ == "__main__":
    main()
