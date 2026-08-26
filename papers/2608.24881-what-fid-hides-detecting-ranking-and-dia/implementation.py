```python
import numpy as np

def _compute_rise_score(ref_feats, gen_feats):
    """
    Computes the Rank Graph (RISE) based score.
    Uses a 1D projection (PCA) to compute ranks and the sum of squared differences of ranks.
    """
    # Combine features
    all_feats = np.vstack([ref_feats, gen_feats])
    n_ref = len(ref_feats)
    
    # Center and PCA (top 1 component for stability in rank calculation)
    mean = np.mean(all_feats, axis=0)
    centered = all_feats - mean
    # Use SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto the first principal component
    projections = centered @ Vt[0]
    
    ref_proj = projections[:n_ref]
    gen_proj = projections[n_ref:]
    
    # Compute ranks (1-based)
    # np.argsort(np.argsort(x)) gives 0-based ranks. Adding 1 makes it 1-based.
    ref_ranks = np.argsort(np.argsort(ref_proj)) + 1
    gen_ranks = np.argsort(np.argsort(gen_proj)) + 1
    
    # Calculate the sum of squared differences of ranks (normalized by n^2 for scale invariance roughly)
    # A common metric in rank-based tests (like Wilcoxon or similar) is based on rank differences.
    # Here we use the sum of squared differences between the rank vectors.
    # If distributions are identical, the rank vectors should be similar in structure.
    # A simple distance: ||r_ref - r_gen||^2
    diff = ref_ranks - gen_ranks
    rise_score = np.sum(diff**2) / (n_ref * n_ref)
    
    return rise_score

def _compute_gpk_score(ref_feats, gen_feats, bandwidth=None):
    """
    Computes the Gaussian Kernel (GPK) based score.
    Uses MMD (Maximum Mean Discrepancy) with a Gaussian kernel.
    """
    if bandwidth is None:
        # Median heuristic for bandwidth
        dists = []
        for i in range(len(ref_feats)):
            for j in range(len(ref_feats)):
                if i != j:
                    dists.append(np.linalg.norm(ref_feats[i] - ref_feats[j]))
        med_dist = np.median(dists) if len(dists) > 0 else 1.0
        bandwidth = med_dist / np.sqrt(2 * np.log(len(ref_feats)))
        
    def gauss_kernel(x, y, bw):
        # x: (n, d), y: (m, d)
        # dist^2 = ||x||^2 + ||y||^2 - 2xy^T
        x_sq = np.sum(x**2, axis=1, keepdims=True)
        y_sq = np.sum(y**2, axis=1, keepdims=True).T
        dist_sq = x_sq + y_sq - 2 * np.dot(x, y.T)
        dist_sq = np.maximum(dist_sq, 0) # Numerical stability
        return np.exp(-dist_sq / (2 * bandwidth**2))
    
    K_rr = gauss_kernel(ref_feats, ref_feats, bandwidth)
    K_gg = gauss_kernel(gen_feats, gen_feats, bandwidth)
    K_rg = gauss_kernel(ref_feats, gen_feats, bandwidth)
    
    n_r = len(ref_feats)
    n_g = len(gen_feats)
    
    # MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    term1 = np.sum(K_rr) / (n_r * n_r)
    term2 = 2 * np.sum(K_rg) / (n_r * n_g)
    term3 = np.sum(K_gg) / (n_g * n_g)
    
    gpk_score = term1 - term2 + term3
    
    # Ensure non-negative
    return max(gpk_score, 0.0)

def _compute_dispersion_ratio(ref_feats, gen_feats):
    """
    Computes a dispersion ratio to determine the sign of the deviation.
    Compares the effective volume (variance) of the generated set vs reference set.
    """
    # Use the trace of the covariance matrix as a measure of dispersion
    cov_ref = np.cov(ref_feats, rowvar=False)
    cov_gen = np.cov(gen_feats, rowvar=False)
    
    # Handle case where cov is not full rank or scalar
    if np.ndim(cov_ref) == 0:
        disp_ref = float(cov_ref)
    else:
        disp_ref = np.trace(cov_ref)
        
    if np.ndim(cov_gen) == 0:
        disp_gen = float(cov_gen)
    else:
        disp_gen = np.trace(cov_gen)
        
    if disp_ref == 0:
        return 0.0
        
    ratio = disp_gen / disp_ref
    return ratio

def zid(reference_features, generated_features, n_permutations=1000, alpha=0.05):
    """
    Computes the Z-resolved Integrated Diagnostic (ZID) metric.
    
    Args:
        reference_features: (N, D) numpy array of reference features.
        generated_features: (M, D) numpy array of generated features.
        n_permutations: Number of permutations for p-value calculation.
        alpha: Significance level (not directly used in output tuple, but for context).
        
    Returns:
        A tuple (ranking_index, p_value, signed_dispersion_readout).
        - ranking_index: A scalar float representing the magnitude of deviation.
        - p_value: A float between 0 and 1 indicating statistical significance.
        - signed_dispersion_readout: A string '+' (over-dispersion), '-' (under-dispersion), or '0' (neutral).
    """
    # 1. Compute the 6 standardized measurement arms
    # The prompt mentions 6 arms based on RISE and GPK. 
    # We will construct 6 scores by combining RISE and GPK with different bandwidths/projections.
    
    # Arm 1: RISE Score
    rise_arm = _compute_rise_score(reference_features, generated_features)
    
    # Arm 2: GPK Score (default bandwidth)
    gpk_arm_default = _compute_gpk_score(reference_features, generated_features)
    
    # Arm 3: GPK Score (small bandwidth - more sensitive to local modes)
    gpk_arm_small = _compute_gpk_score(reference_features, generated_features, bandwidth=0.1)
    
    # Arm 4: GPK Score (large bandwidth - more sensitive to global shape)
    gpk_arm_large = _compute_gpk_score(reference_features, generated_features, bandwidth=10.0)
    
    # Arm 5: Variance Ratio (Dispersion measure)
    disp_ratio = _compute_dispersion_ratio(reference_features, generated_features)
    
    # Arm 6: Distance of Means (Location shift)
    mean_diff = np.linalg.norm(np.mean(reference_features, axis=0) - np.mean(generated_features, axis=0))
    # Normalize mean diff by the standard deviation of the reference set
    std_ref = np.std(reference_features, axis=0)
    std_ref_mean = np.mean(std_ref[std_ref > 1e-8])
    if std_ref_mean > 1e-8:
        mean_diff_normalized = mean_diff / std_ref_mean
    else:
        mean_diff_normalized = 0.0

    # Combine the 6 arms into a single Ranking Index
    # We use a weighted sum. Weights can be tuned, but for a general implementation,
    # we assume equal importance or specific weights defined by the "standardized" nature.
    # Let's use a simple average for the magnitude score, but scale them appropriately.
    # RISE is O(1), GPK is O(1), Disp Ratio is O(1), Mean Diff is O(1).
    
    # To make them comparable, we can just sum them or use a robust combination.
    # Let's define the Ranking Index as the root mean square of the 6 normalized scores.
    # First, we need to "standardize" them. Since we don't have a test set to compute global stats,
    # we assume the scores are already on a comparable scale (0.0 to 1.0 or similar).
    # RISE: 0 if identical, 1 if completely different (max rank diff).
    # GPK: 0 if identical, increases with difference.
    # Disp Ratio: 1.0 if identical dispersion. Deviation from 1.0 indicates change.
    # Mean Diff: 0 if identical means.
    
    # Let's transform Disp Ratio to a deviation score: |ratio - 1|
    disp_dev_score = np.abs(disp_ratio - 1.0)
    
    # List of 6 scores
    scores = [
        rise_arm,
        gpk_arm_default,
        gpk_arm_small,
        gpk_arm_large,
        disp_dev_score,
        mean_diff_normalized
    ]
    
    # Calculate Ranking Index (Magnitude)
    # We take the average of these scores as the composite index.
    ranking_index = np.mean(scores)
    
    # 2. Compute Permutation p-value
    # Null hypothesis: reference and generated features come from the same distribution.
    # Test statistic: The composite ranking index (or a specific component like GPK).
    # We will use the GPK score as the test statistic for the permutation test, as it is a common non-parametric test statistic.
    
    observed_stat = gpk_arm_default
    
    # Combine all features
    all_features = np.vstack([reference_features, generated_features])
    n_ref = len(reference_features)
    n_gen = len(generated_features)
    
    count = 0
    # To ensure reproducibility in a demo, we could set a seed, but the prompt asks for a function.
    # We'll use a fixed number of permutations.
    
    for _ in range(n_permutations):
        # Shuffle the labels (indices)
        indices = np.random.permutation(len(all_features))
        perm_ref = all_features[indices[:n_ref]]
        perm_gen = all_features[indices[n_ref:]]
        
        # Calculate the statistic for the permuted samples
        perm_stat = _compute_gpk_score(perm_ref, perm_gen)
        
        if perm_stat >= observed_stat:
            count += 1
            
    # Add 1 to both numerator and denominator to avoid zero division and ensure p-value is in [0, 1]
    p_value = (count + 1) / (n_permutations + 1)
    
    # 3. Compute Signed Dispersion Readout
    # Determine the sign based on the dispersion ratio
    if disp_ratio > 1.05:  # Threshold for "over-dispersion"
        sign = '+'
    elif disp_ratio < 0.95: # Threshold for "under-dispersion"
        sign = '-'
    else:
        sign = '0'
        
    return ranking_index, p_value, sign

# Example usage to demonstrate execution
if __name__ == "__main__":
    # Generate dummy data to simulate the scenario described in the review
    np.random.seed(42)
    
    # Reference set: Well-distributed images (standard normal)
    n_samples = 500
    dim_features = 10
    
    reference_feats = np.random.randn(n_samples, dim_features)
    
    # Case 1: Low guidance (High diversity, Over-dispersion)
    # Generated features have higher variance
    generated_feats_over = np.random.randn(n_samples, dim_features) * 2.0
    
    # Case 2: High guidance (Low diversity, Under-dispersion/Mode Collapse)
    # Generated features have lower variance and are clustered
    generated_feats_under = np.random.randn(n_samples, dim_features) * 0.3
    
    print("Calculating ZID for Over-dispersion case...")
    idx1, p1, sign1 = zid(reference_feats, generated_feats_over)
    print(f"Ranking Index: {idx1:.4f}, P-value: {p1:.4f}, Signed Dispersion: {sign1}")
    print("Interpretation: High ranking index, low p-value, '+' sign indicates Over-dispersion.")
    
    print("\nCalculating ZID for Under-dispersion case...")
    idx2, p2, sign2 = zid(reference_feats, generated_feats_under)
    print(f"Ranking Index: {idx2:.4f}, P-value: {p2:.4f}, Signed Dispersion: {sign2}")
    print("Interpretation: High ranking index, low p-value, '-' sign indicates Under-dispersion (Mode Collapse).")
    
    print("\nCalculating ZID for Identical case (Control)...")
    generated_feats_same = np.random.randn(n_samples, dim_features) # Similar distribution
    idx3, p3, sign3 = zid(reference_feats, generated_feats_same)
    print(f"Ranking Index: {idx3:.4f}, P-value: {p3:.4f}, Signed Dispersion: {sign3}")
    print("Interpretation: Low ranking index, high p-value, '0' or small sign indicates similar distribution.")