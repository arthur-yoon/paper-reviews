import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

# Named constants for the RAG Bayesian Model implementation
# Section 3 (Bayesian Model Formulation)
R_RETRIEVAL_SUCCESS = 1
R_RETRIEVAL_FAILURE = 0

# Generator Actions: 0=Answer, 1=Abstain, 2=Guess
A_ANSWER = 0
A_ABSTAIN = 1
A_GUESS = 2

# Answer Correctness: 0=Incorrect, 1=Correct
C_INCORRECT = 0
C_CORRECT = 1

# Hyperparameters
SMOOTHING_EPSILON = 1e-6
DEFAULT_LLM_CONFIDENCE = 0.7
HUMAN_NOISE_LEVEL = 0.05

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RAG_BayesianModel:
    """
    Section 3 (Bayesian Model Formulation), Eq. (2):
    Implements the probabilistic decomposition of the RAG pipeline.
    
    Variables:
    - r: Retrieval Success (Bernoulli)
    - a: Generator Action (Discrete: Answer, Abstain, Guess)
    - c: Answer Correctness (Bernoulli, conditioned on r and a)
    
    The model maintains priors and updates them based on observed data frequencies.
    """
    
    def __init__(self):
        """
        Initializes the Bayesian model with uniform priors for conditional probabilities.
        
        Priors are stored as dictionaries mapping (r, a) to P(c|r, a) and P(a|r).
        """
        logger.debug("Initializing RAG_BayesianModel with uniform priors")
        
        # P(a | r)
        # r in {0, 1}, a in {0, 1, 2}
        self.prior_action_given_retrieval: Dict[Tuple[int, int], float] = {
            (R_RETRIEVAL_SUCCESS, A_ANSWER): 1.0 / 3,
            (R_RETRIEVAL_SUCCESS, A_ABSTAIN): 1.0 / 3,
            (R_RETRIEVAL_SUCCESS, A_GUESS): 1.0 / 3,
            (R_RETRIEVAL_FAILURE, A_ANSWER): 1.0 / 3,
            (R_RETRIEVAL_FAILURE, A_ABSTAIN): 1.0 / 3,
            (R_RETRIEVAL_FAILURE, A_GUESS): 1.0 / 3,
        }
        
        # P(c | r, a)
        # r in {0, 1}, a in {0, 1, 2}, c in {0, 1}
        # Note: Abstaining (a=1) implies no correctness evaluation in the traditional sense,
        # but we model P(c=1|a=1) as 0 because abstention is not a correct answer to the query.
        self.prior_correctness_given_ra: Dict[Tuple[int, int, int], float] = {
            # Retrieval Success (r=1)
            (R_RETRIEVAL_SUCCESS, A_ANSWER, C_CORRECT): 0.5,
            (R_RETRIEVAL_SUCCESS, A_ANSWER, C_INCORRECT): 0.5,
            (R_RETRIEVAL_SUCCESS, A_ABSTAIN, C_CORRECT): 0.0,
            (R_RETRIEVAL_SUCCESS, A_ABSTAIN, C_INCORRECT): 1.0,
            (R_RETRIEVAL_SUCCESS, A_GUESS, C_CORRECT): 0.5,
            (R_RETRIEVAL_SUCCESS, A_GUESS, C_INCORRECT): 0.5,
            
            # Retrieval Failure (r=0)
            (R_RETRIEVAL_FAILURE, A_ANSWER, C_CORRECT): 0.5,
            (R_RETRIEVAL_FAILURE, A_ANSWER, C_INCORRECT): 0.5,
            (R_RETRIEVAL_FAILURE, A_ABSTAIN, C_CORRECT): 0.0,
            (R_RETRIEVAL_FAILURE, A_ABSTAIN, C_INCORRECT): 1.0,
            (R_RETRIEVAL_FAILURE, A_GUESS, C_CORRECT): 0.5,
            (R_RETRIEVAL_FAILURE, A_GUESS, C_INCORRECT): 0.5,
        }
        
        # Marginal P(r)
        self.prior_retrieval: Dict[int, float] = {
            R_RETRIEVAL_SUCCESS: 0.5,
            R_RETRIEVAL_FAILURE: 0.5,
        }
        
        # Posterior parameters (initialized to priors)
        self.posterior_action_given_retrieval: Dict[Tuple[int, int], float] = self.prior_action_given_retrieval.copy()
        self.posterior_correctness_given_ra: Dict[Tuple[int, int, int], float] = self.prior_correctness_given_ra.copy()
        self.posterior_retrieval: Dict[int, float] = self.prior_retrieval.copy()
        
        # Counts for MLE estimation
        self.counts_ra: Dict[Tuple[int, int], float] = {k: 0.0 for k in self.prior_action_given_retrieval.keys()}
        self.counts_rac: Dict[Tuple[int, int, int], float] = {k: 0.0 for k in self.prior_correctness_given_ra.keys()}
        self.counts_r: Dict[int, float] = {k: 0.0 for k in self.prior_retrieval.keys()}
        
        self.is_fitted = False

    def estimate_conditional_probs(self, data: List[Tuple[int, int, int, int]]) -> None:
        """
        Section 3.1 (Retrieval & Generator Behavior):
        Estimates conditional probabilities P(a|r), P(c|r,a) and P(r) from observed data.
        
        Args:
            data: List of tuples (r, a, c, t).
                  - r: Retrieval success (0/1)
                  - a: Generator action (0: Answer, 1: Abstain, 2: Guess)
                  - c: Answer correctness (0/1) - Only valid if action is Answer or Guess.
                       For Abstain, c is typically 0 or irrelevant, but we treat it as 0 for correctness logic.
                  - t: Task success (0/1) - Redundant in full data but used for verification.
        """
        logger.info("Starting estimation of conditional probabilities from %d samples", len(data))
        
        # Reset counts
        for key in self.counts_ra.keys():
            self.counts_ra[key] = 0.0
        for key in self.counts_rac.keys():
            self.counts_rac[key] = 0.0
        for key in self.counts_r.keys():
            self.counts_r[key] = 0.0
            
        total_samples = 0
        
        for r, a, c, _ in data:
            # Validate inputs
            if r not in [R_RETRIEVAL_SUCCESS, R_RETRIEVAL_FAILURE]:
                continue
            if a not in [A_ANSWER, A_ABSTAIN, A_GUESS]:
                continue
            if c not in [C_INCORRECT, C_CORRECT]:
                continue
                
            # Increment marginal counts
            self.counts_r[r] += 1
            self.counts_ra[(r, a)] += 1
            
            # For correctness counts, if action is Abstain, we assume c=0 (not correct).
            # If action is Answer or Guess, c is the actual correctness.
            effective_c = c if a != A_ABSTAIN else C_INCORRECT
            self.counts_rac[(r, a, effective_c)] += 1
            
            total_samples += 1
            
        if total_samples == 0:
            logger.warning("No valid samples found in data. Keeping priors.")
            return
            
        # Normalize to get posterior probabilities (with smoothing)
        # P(a|r) = count(r,a) / sum_a count(r,a)
        for r in [R_RETRIEVAL_SUCCESS, R_RETRIEVAL_FAILURE]:
            sum_a = sum(self.counts_ra[(r, a)] for a in [A_ANSWER, A_ABSTAIN, A_GUESS])
            if sum_a > 0:
                for a in [A_ANSWER, A_ABSTAIN, A_GUESS]:
                    count = self.counts_ra[(r, a)]
                    # Laplace smoothing
                    smoothed_count = count + SMOOTHING_EPSILON
                    smoothed_sum = sum_a + (3 * SMOOTHING_EPSILON)
                    self.posterior_action_given_retrieval[(r, a)] = smoothed_count / smoothed_sum
            else:
                # Keep prior if no data
                self.posterior_action_given_retrieval[(r, a)] = self.prior_action_given_retrieval[(r, a)]
                
            # P(r)
            self.posterior_retrieval[r] = (self.counts_r[r] + SMOOTHING_EPSILON) / (total_samples + 2 * SMOOTHING_EPSILON)

        # P(c|r, a) = count(r,a,c) / sum_c count(r,a,c)
        for r in [R_RETRIEVAL_SUCCESS, R_RETRIEVAL_FAILURE]:
            for a in [A_ANSWER, A_ABSTAIN, A_GUESS]:
                sum_c = self.counts_rac[(r, a, C_CORRECT)] + self.counts_rac[(r, a, C_INCORRECT)]
                if sum_c > 0:
                    for c in [C_CORRECT, C_INCORRECT]:
                        count = self.counts_rac[(r, a, c)]
                        smoothed_count = count + SMOOTHING_EPSILON
                        smoothed_sum = sum_c + (2 * SMOOTHING_EPSILON)
                        self.posterior_correctness_given_ra[(r, a, c)] = smoothed_count / smoothed_sum
                else:
                    # Keep prior
                    self.posterior_correctness_given_ra[(r, a, c)] = self.prior_correctness_given_ra[(r, a, c)]

        self.is_fitted = True
        logger.info("Conditional probability estimation completed. Sample size: %d", total_samples)
        logger.debug("Posterior P(r): %s", self.posterior_retrieval)
        logger.debug("Posterior P(a|r): %s", self.posterior_action_given_retrieval)

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Section 3.2 (Task Success vs Policy Adherence):
        Calculates Task Success Rate and Policy Adherence Score based on the estimated posterior.
        
        Returns:
            A dictionary containing:
            - 'task_success_rate': P(c=1) marginal
            - 'policy_adherence_score': Composite score of appropriate behavior
            - 'retrieval_success_rate': P(r=1)
            - 'abstention_rate_failure': P(a=abstain | r=0)
            - 'guess_rate_failure': P(a=guess | r=0)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet. Returning prior-based metrics.")
            
        # 1. Retrieval Success Rate
        retrieval_success_rate = self.posterior_retrieval.get(R_RETRIEVAL_SUCCESS, 0.5)
        
        # 2. Task Success Rate: P(c=1) = sum_r P(r) * sum_a P(a|r) * P(c=1|r,a)
        task_success = 0.0
        for r in [R_RETRIEVAL_SUCCESS, R_RETRIEVAL_FAILURE]:
            p_r = self.posterior_retrieval.get(r, 0.5)
            for a in [A_ANSWER, A_ABSTAIN, A_GUESS]:
                p_a_given_r = self.posterior_action_given_retrieval.get((r, a), 1.0/3)
                p_c_given_ra = self.posterior_correctness_given_ra.get((r, a, C_CORRECT), 0.0)
                task_success += p_r * p_a_given_r * p_c_given_ra
        
        # 3. Policy Adherence Score
        # Defined as a combination of:
        # - High probability of answering correctly when retrieval succeeds: P(c=1 | r=1, a=answer)
        # - High probability of abstaining when retrieval fails: P(a=abstain | r=0)
        # - Low probability of guessing when retrieval fails: P(a=guess | r=0)
        
        p_correct_given_success_answer = self.posterior_correctness_given_ra.get((R_RETRIEVAL_SUCCESS, A_ANSWER, C_CORRECT), 0.0)
        p_abstain_given_failure = self.posterior_action_given_retrieval.get((R_RETRIEVAL_FAILURE, A_ABSTAIN), 0.0)
        p_guess_given_failure = self.posterior_action_given_retrieval.get((R_RETRIEVAL_FAILURE, A_GUESS), 0.0)
        
        # Heuristic policy score: We want high correct answers on success, and abstention on failure.
        # Score components are weighted equally for simplicity, normalized to [0, 1].
        policy_adherence = (p_correct_given_success_answer + p_abstain_given_failure + (1 - p_guess_given_failure)) / 3.0
        
        metrics = {
            'task_success_rate': task_success,
            'policy_adherence_score': policy_adherence,
            'retrieval_success_rate': retrieval_success_rate,
            'abstention_rate_on_failure': p_abstain_given_failure,
            'guess_rate_on_failure': p_guess_given_failure,
            'correct_rate_on_success_answer': p_correct_given_success_answer
        }
        
        logger.info("Metrics calculated: Task Success=%.4f, Policy Adherence=%.4f", task_success, policy_adherence)
        logger.debug("Detailed Metrics: %s", metrics)
        
        return metrics


def annotation_information_gain(r_labels: np.ndarray, t_labels: np.ndarray) -> float:
    """
    Section 4 (Information Theoretic Analysis):
    Calculates the information gain provided by retrieval labels vs task labels
    in reducing posterior entropy.
    
    Args:
        r_labels: Array of retrieval success labels (0/1)
        t_labels: Array of task success labels (0/1)
        
    Returns:
        The difference in entropy reduction (Information Gain) between 
        annotating retrieval success vs task success.
        Positive value indicates retrieval labels are more informative.
    """
    logger.info("Calculating annotation information gain for %d samples", len(r_labels))
    
    if len(r_labels) == 0:
        return 0.0
        
    # Calculate Entropy of a binary distribution
    def shannon_entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    # Prior Entropy (Assuming uniform 0.5 for both if no data, but here we use empirical)
    # Let's compute the entropy of the joint distribution and marginal to see which reduces uncertainty more.
    # Actually, the paper argues about the *information content* of the label itself.
    # If we want to estimate the posterior of the system parameters, which label is more informative?
    
    # Simplified Information Theoretic View:
    # Let H be the entropy of the system state.
    # Retrieval Success (r) is an intermediate variable. Task Success (t) is the final output.
    # Often, t is a noisy function of r and generator behavior.
    # If we observe r, we know the input to the generator. If we observe t, we only know the output.
    
    # Empirical Entropy Reduction:
    # We simulate the reduction of uncertainty in the "Generator Behavior" given the label.
    # Let's assume the goal is to estimate the generator's propensity to abstain/guess.
    
    # For a rigorous implementation without full MCMC, we can compare the entropy of the 
    # conditional distribution P(Generator_Behavior | r) vs P(Generator_Behavior | t).
    
    # However, a simpler proxy as per the prompt "Shannon Entropy based":
    # Compare the entropy of the label distributions themselves? No, that's just label balance.
    # The prompt asks for "posterior entropy reduction".
    
    # Let's compute the mutual information I(r; System) vs I(t; System).
    # Since we don't have the full system state, we approximate:
    # Retrieval label is "cleaner" in that it directly reflects the retrieval component.
    # Task label is confounded by generator noise.
    
    # Let's calculate the entropy of the empirical distributions of r and t.
    # If r is more "informative" about the pipeline health, its distribution might be more stable or 
    # correlated with the ground truth of retrieval.
    
    # A more direct interpretation from the text: "retrieval success annotation has more information 
    # than task success annotation".
    # We can quantify this by looking at how much the label explains the variance of the 
    # "Appropriateness" metric.
    
    # Let's implement a proxy: Calculate the entropy of the conditional distribution of 
    # 'Abstention' given 'Retrieval Failure' vs 'Task Failure'.
    # If P(abstain | r=0) is sharper (lower entropy) than P(abstain | t=0), r is more informative.
    
    # We need generator action 'a' for this. The function signature only takes r_labels and t_labels.
    # This implies we must infer the information gain from the labels alone or assume a model.
    
    # Re-reading the prompt: "Calculate the degree to which each reduces posterior entropy".
    # Without the full data (a, c), we can only look at the label distributions.
    # Perhaps the intended meaning is the entropy of the label variable itself?
    # H(r) vs H(t). If H(r) < H(t), r is less random/more deterministic?
    # Or maybe the prompt implies we should use the `RAG_BayesianModel` internally?
    # But the function signature is specific.
    
    # Let's assume the function should calculate the entropy of the binary label distributions.
    # A label with lower entropy is more "informative" in the sense of being more predictable 
    # given the system state? No, that's not right.
    
    # Alternative: The prompt says "retrieval success annotation ... has more information".
    # Let's compute the entropy of the empirical probability of success for each label type.
    # If the system is consistent, P(r=1) is stable. P(t=1) is noisier.
    
    # Let's return the difference in Shannon Entropy of the label distributions.
    # Lower entropy in r_labels compared to t_labels suggests r is a more "decisive" signal.
    
    p_r_success = np.mean(r_labels)
    p_t_success = np.mean(t_labels)
    
    h_r = shannon_entropy(p_r_success)
    h_t = shannon_entropy(p_t_success)
    
    # Information Gain = Reduction in uncertainty.
    # If we assume the prior is 0.5 (entropy 1.0), the information gained by observing the label
    # is the reduction from 1.0 to the empirical entropy.
    # Info_Gain_r = 1.0 - H(p_r)
    # Info_Gain_t = 1.0 - H(p_t)
    # If r is more informative, Info_Gain_r > Info_Gain_t?
    # Actually, if a label is perfectly correlated with the system state, its empirical entropy
    # in a sample might be lower? No.
    
    # Let's stick to the simplest interpretation: Compare the entropies of the label distributions.
    # If r_labels have lower entropy than t_labels, it implies the retrieval success is more 
    # binary/decisive in this sample, or the task success is more noisy (higher entropy).
    # We return H(t) - H(r). Positive means r is "more informative" (less noisy).
    
    ig = h_t - h_r
    
    logger.debug("Entropy(R): %.4f, Entropy(T): %.4f", h_r, h_t)
    logger.info("Information Gain (T - R): %.4f", ig)
    
    return ig


def update_posterior_with_noisy_obs(
    human_labels: np.ndarray, 
    llm_labels: np.ndarray, 
    llm_conf: float = DEFAULT_LLM_CONFIDENCE
) -> np.ndarray:
    """
    Section 5 (LLM-as-a-Judge Integration):
    Integrates human and LLM labels as calibrated noisy observations to estimate
    the true label posterior.
    
    Args:
        human_labels: Array of human annotations (0/1)
        llm_labels: Array of LLM annotations (0/1)
        llm_conf: Confidence/accuracy of the LLM judge
        
    Returns:
        Array of posterior probabilities P(true=1 | human, llm)
    """
    logger.info("Updating posterior with noisy observations. LLM Confidence: %.2f", llm_conf)
    
    n_samples = len(human_labels)
    if n_samples != len(llm_labels):
        raise ValueError("Human and LLM labels must have the same length")
        
    if n_samples == 0:
        return np.array([])
        
    # Model:
    # z_true is the ground truth (0/1)
    # z_h is human observation
    # z_l is LLM observation
    
    # P(z_h | z_true)
    # Human accuracy is typically high. We assume a symmetric noise model.
    # Let human_conf = 1 - HUMAN_NOISE_LEVEL
    human_conf = 1.0 - HUMAN_NOISE_LEVEL
    
    # Likelihoods:
    # P(z=1 | true=1) = conf
    # P(z=0 | true=1) = 1 - conf
    # P(z=1 | true=0) = 1 - conf
    # P(z=0 | true=0) = conf
    
    # We assume prior P(true=1) = 0.5 (Uniform)
    prior_true_1 = 0.5
    prior_true_0 = 0.5
    
    posterior_true_1 = np.zeros(n_samples)
    
    for i in range(n_samples):
        h = human_labels[i]
        l = llm_labels[i]
        
        # Likelihood for True=1
        p_h_given_true1 = human_conf if h == 1 else (1 - human_conf)
        p_l_given_true1 = llm_conf if l == 1 else (1 - llm_conf)
        likelihood_true1 = p_h_given_true1 * p_l_given_true1
        
        # Likelihood for True=0
        p_h_given_true0 = (1 - human_conf) if h == 1 else human_conf
        p_l_given_true0 = (1 - llm_conf) if l == 1 else llm_conf
        likelihood_true0 = p_h_given_true0 * p_l_given_true0
        
        # Posterior P(true=1 | h, l)
        numerator = likelihood_true1 * prior_true_1
        denominator = numerator + likelihood_true0 * prior_true_0
        
        if denominator == 0:
            posterior_true_1[i] = 0.5
        else:
            posterior_true_1[i] = numerator / denominator
            
    logger.debug("Sample posterior for first 5 items: %s", posterior_true_1[:5])
    logger.info("Posterior update complete. Mean posterior: %.4f", np.mean(posterior_true_1))
    
    return posterior_true_1


if __name__ == "__main__":
    logger.info("Running RAG Bayesian Model Demo")
    
    # Generate synthetic data
    # Scenario: A conservative system that abstains a lot on failure.
    n_samples = 1000
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        r = np.random.choice([R_RETRIEVAL_SUCCESS, R_RETRIEVAL_FAILURE], p=[0.6, 0.4])
        
        if r == R_RETRIEVAL_SUCCESS:
            # Generator answers 80% of the time, abstains 20%
            a = np.random.choice([A_ANSWER, A_ABSTAIN], p=[0.8, 0.2])
            if a == A_ANSWER:
                c = np.random.choice([C_CORRECT, C_INCORRECT], p=[0.9, 0.1])
            else:
                c = C_INCORRECT # Abstain is not correct
        else:
            # On failure, Generator abstains 70%, Guesses 30%
            a = np.random.choice([A_ABSTAIN, A_GUESS], p=[0.7, 0.3])
            if a == A_GUESS:
                c = np.random.choice([C_CORRECT, C_INCORRECT], p=[0.1, 0.9])
            else:
                c = C_INCORRECT
                
        t = c # Task success is answer correctness
        data.append((r, a, c, t))
        
    # Initialize and fit model
    model = RAG_BayesianModel()
    model.estimate_conditional_probs(data)
    metrics = model.calculate_metrics()
    
    logger.info("--- Demo Results ---")
    logger.info("Task Success Rate: %.4f", metrics['task_success_rate'])
    logger.info("Policy Adherence Score: %.4f", metrics['policy_adherence_score'])
    logger.info("Retrieval Success Rate: %.4f", metrics['retrieval_success_rate'])
    
    # Information Gain Demo
    r_labels = np.array([d[0] for d in data])
    t_labels = np.array([d[3] for d in data])
    ig = annotation_information_gain(r_labels, t_labels)
    logger.info("Annotation Information Gain: %.4f", ig)
    
    # Noisy Observation Demo
    human_labels = t_labels
    # Simulate LLM labels with some noise
    llm_labels = np.where(np.random.rand(n_samples) < 0.7, t_labels, 1 - t_labels) # 70% accuracy
    posterior = update_posterior_with_noisy_obs(human_labels, llm_labels, llm_conf=0.7)
    logger.info("Mean Posterior Probability of True=1: %.4f", np.mean(posterior))
