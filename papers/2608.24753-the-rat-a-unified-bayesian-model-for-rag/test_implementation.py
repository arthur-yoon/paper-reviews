import unittest
import numpy as np
from implementation import (
    RAG_BayesianModel, 
    annotation_information_gain, 
    update_posterior_with_noisy_obs,
    R_RETRIEVAL_SUCCESS, R_RETRIEVAL_FAILURE,
    A_ANSWER, A_ABSTAIN, A_GUESS,
    C_CORRECT, C_INCORRECT
)


class TestRAGBayesianModel(unittest.TestCase):

    def setUp(self):
        """Initializes a fresh model instance for each test."""
        self.model = RAG_BayesianModel()

    def test_estimate_conditional_probs_deterministic_data(self):
        """
        Tests that the model correctly estimates probabilities from deterministic data.
        Data:
        - 10 samples with r=1, a=0, c=1 (Success, Answer, Correct)
        - 10 samples with r=1, a=1, c=0 (Success, Abstain, Incorrect)
        - 10 samples with r=0, a=1, c=0 (Failure, Abstain, Incorrect)
        - 10 samples with r=0, a=2, c=1 (Failure, Guess, Correct)
        
        Expected:
        - P(r=1) = 0.5
        - P(a=0|r=1) = 0.5, P(a=1|r=1) = 0.5
        - P(a=1|r=0) = 0.5, P(a=2|r=0) = 0.5
        - P(c=1|r=1,a=0) = 1.0
        - P(c=1|r=0,a=2) = 1.0
        """
        data = []
        # 10 samples: r=1, a=0, c=1
        for _ in range(10):
            data.append((R_RETRIEVAL_SUCCESS, A_ANSWER, C_CORRECT, 1))
        # 10 samples: r=1, a=1, c=0
        for _ in range(10):
            data.append((R_RETRIEVAL_SUCCESS, A_ABSTAIN, C_INCORRECT, 0))
        # 10 samples: r=0, a=1, c=0
        for _ in range(10):
            data.append((R_RETRIEVAL_FAILURE, A_ABSTAIN, C_INCORRECT, 0))
        # 10 samples: r=0, a=2, c=1
        for _ in range(10):
            data.append((R_RETRIEVAL_FAILURE, A_GUESS, C_CORRECT, 1))

        self.model.estimate_conditional_probs(data)

        # Check P(r)
        self.assertAlmostEqual(self.model.posterior_retrieval[R_RETRIEVAL_SUCCESS], 0.5, places=4)
        self.assertAlmostEqual(self.model.posterior_retrieval[R_RETRIEVAL_FAILURE], 0.5, places=4)

        # Check P(a|r)
        self.assertAlmostEqual(self.model.posterior_action_given_retrieval[(R_RETRIEVAL_SUCCESS, A_ANSWER)], 0.5, places=4)
        self.assertAlmostEqual(self.model.posterior_action_given_retrieval[(R_RETRIEVAL_SUCCESS, A_ABSTAIN)], 0.5, places=4)
        self.assertAlmostEqual(self.model.posterior_action_given_retrieval[(R_RETRIEVAL_FAILURE, A_ABSTAIN)], 0.5, places=4)
        self.assertAlmostEqual(self.model.posterior_action_given_retrieval[(R_RETRIEVAL_FAILURE, A_GUESS)], 0.5, places=4)

        # Check P(c|r,a)
        self.assertAlmostEqual(self.model.posterior_correctness_given_ra[(R_RETRIEVAL_SUCCESS, A_ANSWER, C_CORRECT)], 1.0, places=4)
        self.assertAlmostEqual(self.model.posterior_correctness_given_ra[(R_RETRIEVAL_FAILURE, A_GUESS, C_CORRECT)], 1.0, places=4)
        # Abstain should have 0 correctness
        self.assertAlmostEqual(self.model.posterior_correctness_given_ra[(R_RETRIEVAL_SUCCESS, A_ABSTAIN, C_CORRECT)], 0.0, places=4)

    def test_calculate_metrics_separation(self):
        """
        Tests that Task Success and Policy Adherence are calculated correctly and separately.
        Scenario: A system that always retrieves successfully and always answers correctly.
        Data:
        - All samples: r=1, a=0, c=1
        Expected:
        - Task Success: 1.0
        - Policy Adherence: High (because it answers correctly on success).
          Formula: (P(c=1|r=1,a=ans) + P(a=abstain|r=0) + (1-P(a=guess|r=0))) / 3
          Since r=0 has no data, it uses prior (0.333 for abstain, 0.333 for guess).
          P(c=1|r=1,a=ans) = 1.0
          Score = (1.0 + 0.333 + (1 - 0.333)) / 3 = (1.0 + 0.333 + 0.667) / 3 = 2.0 / 3 = 0.666
        """
        data = [(R_RETRIEVAL_SUCCESS, A_ANSWER, C_CORRECT, 1) for _ in range(100)]
        self.model.estimate_conditional_probs(data)
        metrics = self.model.calculate_metrics()

        self.assertAlmostEqual(metrics['task_success_rate'], 1.0, places=4)
        self.assertAlmostEqual(metrics['retrieval_success_rate'], 1.0, places=4)
        
        # Policy Adherence calculation:
        # p_correct_given_success_answer = 1.0
        # p_abstain_given_failure = prior (1/3)
        # p_guess_given_failure = prior (1/3)
        # Score = (1.0 + 1/3 + (1 - 1/3)) / 3 = (1.0 + 1/3 + 2/3) / 3 = 2/3
        self.assertAlmostEqual(metrics['policy_adherence_score'], 2.0 / 3.0, places=4)

    def test_annotation_information_gain(self):
        """
        Tests that information gain is calculated based on entropy difference.
        If r_labels are perfectly split (0.5/0.5), entropy is 1.0.
        If t_labels are all 1 (1.0/0.0), entropy is 0.0.
        IG = H(t) - H(r) = 0.0 - 1.0 = -1.0.
        """
        r_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        t_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        ig = annotation_information_gain(r_labels, t_labels)
        
        # H(r) for 0.5 is 1.0. H(t) for 1.0 is 0.0.
        # IG = 0.0 - 1.0 = -1.0
        self.assertAlmostEqual(ig, -1.0, places=4)

    def test_update_posterior_with_noisy_obs(self):
        """
        Tests the Bayesian update of true labels given human and LLM observations.
        Case: Human says 1, LLM says 1. LLM conf 0.9, Human conf 0.95.
        P(true=1|1,1) should be high, close to 1.
        Case: Human says 1, LLM says 0.
        P(true=1|1,0) should be lower.
        """
        human_labels = np.array([1, 1])
        llm_labels = np.array([1, 0])
        llm_conf = 0.9
        # Default human conf is 0.95 (1 - 0.05)
        
        posterior = update_posterior_with_noisy_obs(human_labels, llm_labels, llm_conf=llm_conf)
        
        # Case 1: Both 1.
        # L(true=1) = 0.95 * 0.9 = 0.855
        # L(true=0) = 0.05 * 0.1 = 0.005
        # P(true=1) = 0.855 / (0.855 + 0.005) = 0.855 / 0.86 ≈ 0.994
        
        # Case 2: Human 1, LLM 0.
        # L(true=1) = 0.95 * 0.1 = 0.095
        # L(true=0) = 0.05 * 0.9 = 0.045
        # P(true=1) = 0.095 / (0.095 + 0.045) = 0.095 / 0.14 ≈ 0.678
        
        self.assertGreater(posterior[0], 0.99)
        self.assertLess(posterior[0], 1.0)
        self.assertGreater(posterior[1], 0.6)
        self.assertLess(posterior[1], 0.7)


if __name__ == '__main__':
    unittest.main()
