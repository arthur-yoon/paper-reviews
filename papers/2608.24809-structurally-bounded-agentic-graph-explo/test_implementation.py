# test_implementation.py
import unittest
import logging
import numpy as np
from implementation import CraseAgent, PRUNING_THRESHOLD_DEFAULT

# Configure logging for tests to capture output if needed
logging.basicConfig(level=logging.DEBUG)

class TestCraseAgent(unittest.TestCase):
    """
    Test suite for the CraseAgent implementation.
    """

    def setUp(self):
        """Set up the agent with a small corpus."""
        self.agent = CraseAgent(corpus_size=100, pruning_threshold=0.5)
        # Access the mock corpus directly to control test data
        self.mock_corpus = self.agent._mock_corpus

    def test_fetch_seeds(self):
        """Test that fetch_seeds returns relevant papers for a query."""
        query = "large language model attention"
        seeds = self.agent.fetch_seeds(query, max_seeds=5)
        
        self.assertGreater(len(seeds), 0, "Should find at least one seed")
        self.assertLessEqual(len(seeds), 5, "Should not exceed max_seeds")
        
        # Verify that the returned seeds are in the corpus
        for seed in seeds:
            self.assertIn(seed, self.mock_corpus)

    def test_prune_edges_similarity(self):
        """
        Test that low similarity edges are pruned and high similarity edges are kept.
        We will manually construct a scenario using the agent's internal logic.
        """
        # Create a controlled graph
        # P_0001: "large language model attention mechanism"
        # P_0002: "large language model attention" (High similarity)
        # P_0003: "quantization of neural networks" (Low similarity to P_0001)
        
        # Ensure these papers exist in the mock corpus with specific abstracts
        self.mock_corpus["P_0001"] = {
            "title": "Paper 1",
            "abstract": "large language model attention mechanism transformer",
            "year": 2023,
            "citations": ["P_0002", "P_0003"]
        }
        self.mock_corpus["P_0002"] = {
            "title": "Paper 2",
            "abstract": "large language model attention",
            "year": 2024,
            "citations": []
        }
        self.mock_corpus["P_0003"] = {
            "title": "Paper 3",
            "abstract": "quantization low precision inference",
            "year": 2020,
            "citations": []
        }

        # Construct a simple graph dict
        graph = {
            "P_0001": ["P_0002", "P_0003"],
            "P_0002": [],
            "P_0003": []
        }
        
        pruned_graph, reasons = self.agent.prune_edges(graph, ["P_0001"])
        
        # P_0001 -> P_0002 should be kept (high similarity)
        self.assertIn("P_0002", pruned_graph.get("P_0001", []), "High similarity edge should be kept")
        
        # P_0001 -> P_0003 should be pruned (low similarity)
        self.assertNotIn("P_0003", pruned_graph.get("P_0001", []), "Low similarity edge should be pruned")

    def test_rank_papers_recency_awareness(self):
        """
        Test that the random walk ranks papers, and ideally that recency
        plays a role (though strict testing of recency vs degree is complex,
        we can verify that the function returns valid scores for connected nodes).
        """
        # Create a simple chain: Seed -> Paper A (Recent) , Seed -> Paper B (Old)
        # Paper A and B cite nothing further.
        
        self.mock_corpus["Seed"] = {
            "title": "Seed",
            "abstract": "common words here",
            "year": 2024,
            "citations": ["Recent_Paper", "Old_Paper"]
        }
        self.mock_corpus["Recent_Paper"] = {
            "title": "Recent",
            "abstract": "common words here", # High similarity to Seed
            "year": 2024,
            "citations": []
        }
        self.mock_corpus["Old_Paper"] = {
            "title": "Old",
            "abstract": "common words here", # High similarity to Seed
            "year": 2010,
            "citations": []
        }

        graph = {
            "Seed": ["Recent_Paper", "Old_Paper"],
            "Recent_Paper": [],
            "Old_Paper": []
        }

        # Prune (should keep both due to high similarity)
        pruned_graph, _ = self.agent.prune_edges(graph, ["Seed"])
        
        # Rank
        results = self.agent.rank_papers(pruned_graph, ["Seed"], top_k=5)
        
        # Should return 3 nodes
        self.assertEqual(len(results), 3)
        
        # Check that Seed has a high score (it's the start)
        seed_score = next((score for pid, score in results if pid == "Seed"), -1)
        self.assertGreater(seed_score, 0)
        
        # Check that Recent_Paper and Old_Paper have scores
        recent_score = next((score for pid, score in results if pid == "Recent_Paper"), 0)
        old_score = next((score for pid, score in results if pid == "Old_Paper"), 0)
        
        self.assertGreater(recent_score, 0)
        self.assertGreater(old_score, 0)
        
        # Due to recency-awareness, Recent_Paper (2024) should generally have a 
        # higher or equal score than Old_Paper (2010) if all else is equal.
        # The decay factor exp(lambda * (year - base)) makes recent > old.
        # Transition probability to Recent is boosted.
        # Thus, Recent_Paper should be ranked higher or equal.
        self.assertGreaterEqual(recent_score, old_score, "Recent paper should have higher or equal score due to recency bias")

    def test_evaluate_metrics(self):
        """Test the evaluation metrics calculation."""
        ground_truth = {"P1", "P2", "P3"}
        retrieved = ["P1", "P4", "P2", "P5"]
        
        metrics = self.agent.evaluate(ground_truth, retrieved, k=3)
        
        # Top 3: P1, P4, P2
        # Relevant in top 3: P1, P2
        # Precision: 2/3
        # Recall: 2/3
        self.assertAlmostEqual(metrics["precision@k"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(metrics["recall@k"], 2.0 / 3.0, places=4)

    def test_run_pipeline_integration(self):
        """Test the full pipeline runs without error and returns expected structure."""
        query = "test query"
        result = self.agent.run_pipeline(query)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("results", result)
        self.assertIn("evaluation", result)
        self.assertIsInstance(result["results"], list)
        
        if len(result["results"]) > 0:
            first_result = result["results"][0]
            self.assertIn("id", first_result)
            self.assertIn("score", first_result)
            self.assertIn("year", first_result)

if __name__ == '__main__':
    unittest.main()
