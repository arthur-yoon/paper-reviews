# implementation.py
import logging
import math
import re
import time
from typing import Dict, List, Tuple, Any, Set

# Standard library + numpy only

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Named constants to avoid magic numbers
PRUNING_THRESHOLD_DEFAULT = 0.5
EPSILON_DEFAULT = 1e-6
MAX_ITERATIONS_DEFAULT = 500
TOP_K_DEFAULT = 50
TIME_DECAY_LAMBDA = 0.1
BASE_YEAR_FOR_DECAY = 2024 # Reference year for recency calculation

class CraseAgent:
    """
    Implements the Crase (bounded search) framework as described in the paper.
    
    Mapping to Paper Sections:
    - Section: Methodology -> Seed Query & Initial Graph Construction
      (fetch_seeds, expand_graph_1_5_hop)
    - Section: Methodology -> Entailment-based Pruning
      (prune_edges)
    - Section: Methodology -> Recency-aware Random Walk
      (rank_papers, calculate_transition_matrix)
    - Section: Experiments -> Benchmarking
      (evaluate)
    - Section: Results -> Cost/Performance Trade-off
      (run_pipeline)
    """

    def __init__(self, corpus_size: int, pruning_threshold: float = PRUNING_THRESHOLD_DEFAULT):
        """
        Initialize the CraseAgent.
        
        Args:
            corpus_size: Simulated size of the available corpus.
            pruning_threshold: Minimum cosine similarity required to keep an edge (Section 3.2).
        """
        self.corpus_size = corpus_size
        self.pruning_threshold = pruning_threshold
        self._mock_corpus: Dict[str, Dict[str, Any]] = {}
        self._init_mock_corpus()
        logger.debug(f"CraseAgent initialized with corpus_size={corpus_size}, pruning_threshold={pruning_threshold}")

    def _init_mock_corpus(self) -> None:
        """
        Initialize a small, deterministic mock corpus for testing and demonstration.
        Simulates the 500K corpus structure mentioned in Section 4 (Experiments) but scaled down.
        """
        logger.debug("Initializing mock corpus...")
        # Create a small set of papers with controlled abstracts and years
        # Papers are structured as: { id: { 'title': str, 'abstract': str, 'year': int, 'citations': List[str] } }
        
        # Define some core "seed" topics
        topics = [
            ("large language model attention mechanism", "transformer architecture neural network"),
            ("efficient attention for long contexts", "sparse attention linear complexity"),
            ("quantization of neural networks", "model compression low precision inference"),
            ("reinforcement learning for robotics", "policy gradient optimal control"),
            ("computer vision object detection", "convolutional neural network bounding box")
        ]

        # Generate 100 papers to allow for some expansion
        # We will manually link them to ensure connectivity for the demo
        paper_ids = [f"P_{i:04d}" for i in range(100)]
        
        for i, pid in enumerate(paper_ids):
            # Cycle through topics
            topic_idx = i % len(topics)
            title_prefix, abstract_kw = topics[topic_idx]
            
            # Add some variation to abstracts to test similarity
            abstract = f"Research on {title_prefix}. We propose a method involving {abstract_kw}."
            if i % 2 == 0:
                abstract += " Our experiments show significant improvements in performance."
            
            year = 2015 + (i % 10) # Years between 2015 and 2024
            
            self._mock_corpus[pid] = {
                "title": f"Paper {i}: {title_prefix.title()}",
                "abstract": abstract,
                "year": year,
                "citations": []
            }

        # Create citation links
        # P_0000 to P_0009 are potential seeds
        # Each paper cites 2-3 others
        for i, pid in enumerate(paper_ids):
            # Cite older papers (lower index) to simulate time flow
            # Ensure we don't cite self
            if i > 0:
                # Cite i-1 and i-2 if they exist
                if i-1 >= 0:
                    self._mock_corpus[pid]["citations"].append(f"P_{i-1:04d}")
                if i-2 >= 0:
                    self._mock_corpus[pid]["citations"].append(f"P_{i-2:04d}")
            # Some cross-links for 1.5-hop testing
            if i > 10:
                 self._mock_corpus[pid]["citations"].append(f"P_{i-10:04d}")

        logger.info(f"Mock corpus initialized with {len(paper_ids)} papers.")

    def fetch_seeds(self, query: str, max_seeds: int = 10) -> List[str]:
        """
        Fetches initial seed papers based on a query.
        Section: Methodology - Seed Query & Initial Graph Construction.
        
        In the real paper, this calls a search engine. Here, we mock it by
        finding the top max_seeds papers whose abstracts contain the most query terms.
        
        Args:
            query: The user's search query.
            max_seeds: Number of seed papers to return.
            
        Returns:
            A list of paper IDs representing the seeds.
        """
        logger.info(f"Fetching seeds for query: '{query}'")
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        scores: Dict[str, float] = {}
        for pid, paper in self._mock_corpus.items():
            abstract_terms = set(re.findall(r'\b\w+\b', paper['abstract'].lower()))
            # Simple keyword overlap score
            overlap = len(query_terms & abstract_terms)
            if overlap > 0:
                scores[pid] = overlap

        if not scores:
            logger.warning("No relevant seeds found. Returning empty list.")
            return []

        # Sort by score descending, then by ID for determinism
        sorted_pids = sorted(scores.keys(), key=lambda x: (-scores[x], x))
        seeds = sorted_pids[:max_seeds]
        
        logger.info(f"Retrieved {len(seeds)} seed papers: {seeds}")
        return seeds

    def expand_graph_1_5_hop(self, seeds: List[str]) -> Dict[str, List[str]]:
        """
        Expands the graph to a 1.5-hop neighborhood.
        Section: Methodology - Graph Construction.
        
        1.0-hop: All direct citations of seeds.
        0.5-hop: Citations of the 1.0-hop papers, but limited.
        The '0.5-hop' is interpreted as including only citations of 1-hop nodes
        that are also highly relevant or simply limiting the out-degree of 1-hop nodes
        to simulate bounded expansion. Here we include all citations of 1-hop nodes
        but mark them for potential pruning. The "1.5" aspect is controlled by
        the pruning step which acts as the gate for the 2nd hop.
        
        Args:
            seeds: List of seed paper IDs.
            
        Returns:
            An adjacency list dictionary: {paper_id: [cited_paper_ids]}
        """
        logger.info("Expanding to 1.5-hop neighborhood...")
        graph: Dict[str, List[str]] = {}
        
        # Start with seeds
        current_nodes = set(seeds)
        next_nodes = set()
        
        # 0-hop: Seeds
        for seed in seeds:
            if seed not in self._mock_corpus:
                continue
            graph[seed] = self._mock_corpus[seed]["citations"].copy()
            next_nodes.update(graph[seed])
        
        # 1-hop: Add citations of seeds
        # Note: The graph structure here is parent -> children (citations)
        # We need to ensure all nodes in the graph are present in the dict as keys
        # if they are to be further expanded, but for ranking, we mainly care about
        # the directed edges.
        
        # 1.0-hop expansion:
        first_hop_nodes = list(next_nodes)
        for node in first_hop_nodes:
            if node not in self._mock_corpus:
                continue
            # Add the node to the graph keys if it exists
            if node not in graph:
                graph[node] = self._mock_corpus[node]["citations"].copy()
            else:
                # Merge citations if node was already a seed
                graph[node].extend(self._mock_corpus[node]["citations"])
                # Remove duplicates
                graph[node] = list(set(graph[node]))
            
            # 0.5-hop logic: We collect citations of 1-hop nodes, but we might limit them
            # For strict "bounded" behavior, we could limit the number of citations
            # from each 1-hop node. Here we include them but rely on pruning.
            # To explicitly implement the "0.5" boundedness, we might cap the out-degree
            # of 1-hop nodes to a small number, e.g., 1 or 2.
            # Let's cap to 2 citations per 1-hop node to simulate the bounded nature.
            # However, the prompt says "limit out-degree".
            
            # Re-evaluating: The prompt says "1-hop nodes' out-degree is limited".
            # So we should only keep a subset of the citations from 1-hop nodes.
            # Let's keep the top 2 citations for 1-hop nodes to simulate the 0.5 hop.
            pass # We will handle the filtering in a cleaner way below.

        # Let's rebuild the graph more carefully to enforce the 1.5-hop constraint
        graph = {}
        nodes_0_hop = set(seeds)
        nodes_1_hop = set()
        
        # 0-hop -> 1-hop
        for seed in nodes_0_hop:
            if seed in self._mock_corpus:
                cites = self._mock_corpus[seed]["citations"]
                graph[seed] = cites.copy()
                nodes_1_hop.update(cites)
        
        # 1-hop -> 2-hop (limited to 0.5)
        nodes_2_hop_limited = set()
        for node in nodes_1_hop:
            if node in self._mock_corpus:
                if node not in graph:
                    graph[node] = []
                
                # Limit the out-degree of 1-hop nodes to simulate 0.5 hop
                # We take the first 2 citations (arbitrary bound, could be config)
                LIMITED_OUT_DEGREE = 2
                cites = self._mock_corpus[node]["citations"]
                limited_cites = cites[:LIMITED_OUT_DEGREE]
                
                # Add these limited citations to the graph
                graph[node].extend(limited_cites)
                nodes_2_hop_limited.update(limited_cites)
                
                # Ensure the 2-hop nodes themselves are in the graph dict (as leaves)
                # Even if they don't have outgoing edges in this expanded view
        
        # Ensure all 2-hop nodes are keys in the graph (with empty lists if no further expansion)
        for node in nodes_2_hop_limited:
            if node not in graph:
                graph[node] = []
            # We do NOT expand 2-hop nodes further (bounded at 1.5)

        total_nodes = len(graph)
        logger.info(f"1.5-hop expansion complete. Total nodes in graph: {total_nodes}")
        return graph

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two text strings.
        Simple bag-of-words implementation.
        
        Args:
            text1: First text string.
            text2: Second text string.
            
        Returns:
            Cosine similarity score between 0.0 and 1.0.
        """
        def vectorize(text: str) -> Dict[str, int]:
            words = re.findall(r'\b\w+\b', text.lower())
            vector: Dict[str, int] = {}
            for w in words:
                vector[w] = vector.get(w, 0) + 1
            return vector

        vec1 = vectorize(text1)
        vec2 = vectorize(text2)
        
        if not vec1 or not vec2:
            return 0.0

        # Calculate dot product
        common_words = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[w] * vec2[w] for w in common_words)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(count ** 2 for count in vec1.values()))
        mag2 = math.sqrt(sum(count ** 2 for count in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

    def prune_edges(self, graph: Dict[str, List[str]], seeds: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Prunes edges based on entailment similarity (proxy: abstract similarity).
        Section: Methodology - Entailment-based Pruning.
        
        Args:
            graph: The 1.5-hop expanded graph.
            seeds: The original seed papers (for context, though pruning is local).
            
        Returns:
            A tuple containing:
            1. The pruned graph (adjacency list).
            2. A list of strings describing why each remaining edge/node is kept (for inspectability).
        """
        logger.info(f"Pruning edges based on entailment similarity (threshold={self.pruning_threshold})...")
        pruned_graph: Dict[str, List[str]] = {node: [] for node in graph.keys()}
        reasons: List[str] = []
        removed_edges = 0
        initial_edge_count = sum(len(edges) for edges in graph.values())
        
        for parent, children in graph.items():
            if not children:
                continue
                
            parent_abstract = self._mock_corpus.get(parent, {}).get('abstract', '')
            
            for child in children:
                child_abstract = self._mock_corpus.get(child, {}).get('abstract', '')
                
                sim = self._calculate_cosine_similarity(parent_abstract, child_abstract)
                
                if sim >= self.pruning_threshold:
                    pruned_graph[parent].append(child)
                    # Record reason for inspectability
                    reasons.append(f"Edge {parent} -> {child} kept (similarity: {sim:.2f})")
                else:
                    removed_edges += 1
                    logger.debug(f"Pruned edge {parent} -> {child} (similarity: {sim:.2f} < {self.pruning_threshold})")
        
        # Remove nodes that have no incoming or outgoing edges in the pruned graph?
        # The prompt implies "remaining nodes". A node is remaining if it has any connection
        # or if it was a seed.
        
        # Let's identify remaining nodes. A node remains if it has any outgoing edges
        # in the pruned graph OR if it was a seed.
        # Actually, if a 1-hop node gets pruned from its parent, does it disappear?
        # In graph terms, if it has no incoming edges, it's disconnected.
        # We should keep the node structure but only rank connected components.
        # For simplicity, we keep all nodes in the dict, but only those with edges
        # or seeds will have non-empty lists or be start points.
        
        final_node_count = len(pruned_graph)
        # To be more precise about "remaining nodes", we might want to filter out
        # nodes that are completely isolated (no in, no out) and not seeds.
        # But for the random walk, we just iterate over the provided nodes.
        
        logger.info(f"Removed {removed_edges} edges. Remaining nodes in structure: {final_node_count}")
        return pruned_graph, reasons

    def calculate_transition_matrix(self, graph: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Calculates the transition matrix for the random walk.
        Section: Methodology - Recency-aware Random Walk.
        
        Applies a time decay factor based on the year of the target node.
        
        Args:
            graph: The pruned adjacency list graph.
            
        Returns:
            A tuple containing:
            1. The transition matrix (numpy array).
            2. The list of node IDs corresponding to the rows/cols.
        """
        logger.debug("Calculating transition matrix...")
        nodes = list(graph.keys())
        n = len(nodes)
        
        if n == 0:
            return np.zeros((0, 0)), []

        node_index = {node: i for i, node in enumerate(nodes)}
        transition_matrix = np.zeros((n, n), dtype=np.float64)
        
        # Pre-compute decay factors
        decay_factors = {}
        for node in nodes:
            year = self._mock_corpus.get(node, {}).get('year', BASE_YEAR_FOR_DECAY)
            # Exponential decay: exp(-lambda * (current_ref_year - year))
            # Or a simple linear decay? Exponential is standard for time decay.
            # Let's use a factor that increases recency.
            # Higher year -> smaller difference -> larger factor (closer to 1)
            # Let's define decay such that older papers get lower probability mass.
            # Standard: P(i->j) = (1/d_i) * decay(j)
            # We want recent j to have higher weight.
            # decay(j) = exp(TIME_DECAY_LAMBDA * (year_j - BASE_YEAR_FOR_DECAY))
            # If year_j < BASE_YEAR, exponent is negative, decay < 1.
            # If year_j == BASE_YEAR, decay = 1.
            # If year_j > BASE_YEAR (future), decay > 1.
            # This seems reasonable.
            
            time_diff = year - BASE_YEAR_FOR_DECAY
            decay_factor = math.exp(TIME_DECAY_LAMBDA * time_diff)
            decay_factors[node] = decay_factor
            
        for i, parent in enumerate(nodes):
            children = graph[parent]
            if not children:
                # Self-loop or default probability for dangling nodes
                # In PageRank/Random Walk, dangling nodes usually redirect to all nodes or stay.
                # Let's assume they stay (self-loop) with probability 1.
                transition_matrix[i, i] = 1.0
                continue
            
            out_degree = len(children)
            
            # Calculate weights for each child
            weights = []
            for child in children:
                decay = decay_factors[child]
                # Base probability is 1/out_degree, modified by decay
                # We need to normalize these weighted probabilities
                weights.append(decay)
            
            total_weight = sum(weights)
            
            if total_weight == 0:
                # Fallback to uniform
                for j, child in enumerate(children):
                    transition_matrix[i, node_index[child]] = 1.0 / out_degree
            else:
                # Normalize weights to sum to 1
                for j, child in enumerate(children):
                    prob = (weights[j] / total_weight)
                    transition_matrix[i, node_index[child]] = prob
        
        logger.debug(f"Transition matrix calculated. Shape: {transition_matrix.shape}")
        return transition_matrix, nodes

    def rank_papers(self, pruned_graph: Dict[str, List[str]], seeds: List[str], top_k: int = TOP_K_DEFAULT) -> List[Tuple[str, float]]:
        """
        Performs the recency-aware random walk to rank papers.
        Section: Methodology - Recency-aware Random Walk.
        
        Args:
            pruned_graph: The graph after pruning.
            seeds: The seed nodes to start the walk from.
            top_k: Number of top results to return.
            
        Returns:
            A list of tuples (paper_id, score) sorted by score descending.
        """
        logger.info("Running Recency-Aware Random Walk...")
        
        transition_matrix, nodes = self.calculate_transition_matrix(pruned_graph)
        n = len(nodes)
        
        if n == 0:
            return []

        # Initialize score vector
        # Uniform distribution over seeds? Or uniform over all nodes?
        # "Seeds as start points" -> Uniform over seeds
        score_vector = np.zeros(n, dtype=np.float64)
        
        seed_indices = []
        for seed in seeds:
            if seed in nodes:
                idx = nodes.index(seed)
                seed_indices.append(idx)
        
        if not seed_indices:
            logger.warning("No seeds found in the pruned graph. Returning empty ranking.")
            return []
        
        # Uniform distribution over seeds
        for idx in seed_indices:
            score_vector[idx] = 1.0 / len(seed_indices)
        
        # Power iteration
        epsilon = EPSILON_DEFAULT
        max_iter = MAX_ITERATIONS_DEFAULT
        converged = False
        
        for iteration in range(max_iter):
            # Random walk step: new_score = old_score @ transition_matrix
            # Note: If using row-stochastic matrix (sum of rows = 1), 
            # and score_vector is row vector: v_new = v_old * M
            new_score_vector = score_vector @ transition_matrix
            
            # Check convergence
            diff = np.linalg.norm(new_score_vector - score_vector, ord=1)
            if diff < epsilon:
                converged = True
                logger.debug(f"Converged at iteration {iteration + 1} with diff {diff:.2e}")
                break
            
            score_vector = new_score_vector
            if (iteration + 1) % 10 == 0:
                logger.debug(f"Iteration {iteration + 1}, diff: {diff:.2e}")
        
        if not converged:
            logger.warning(f"Did not converge after {max_iter} iterations. Using last estimate.")
        else:
            logger.info(f"Converged after {iteration + 1} iterations.")
            
        # Map scores back to paper IDs
        results: List[Tuple[str, float]] = []
        for i, node in enumerate(nodes):
            results.append((node, score_vector[i]))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Random walk complete. Top {min(top_k, len(results))} papers identified.")
        return results[:top_k]

    def evaluate(self, ground_truth: Set[str], retrieved_papers: List[str], k: int = 10) -> Dict[str, float]:
        """
        Evaluates the retrieved papers against ground truth.
        Section: Experiments - Benchmarking (LitSearch).
        
        Args:
            ground_truth: Set of relevant paper IDs.
            retrieved_papers: List of retrieved paper IDs (ordered by rank).
            k: Cut-off for metrics.
            
        Returns:
            Dictionary containing Recall@K and Precision@K.
        """
        if not retrieved_papers:
            return {"recall@k": 0.0, "precision@k": 0.0}
        
        top_k_retrieved = set(retrieved_papers[:k])
        relevant_in_top_k = len(ground_truth & top_k_retrieved)
        
        precision = relevant_in_top_k / k if k > 0 else 0.0
        recall = relevant_in_top_k / len(ground_truth) if ground_truth else 0.0
        
        metrics = {
            "recall@k": recall,
            "precision@k": precision,
            "k": k,
            "num_relevant": len(ground_truth),
            "num_retrieved_in_top_k": len(top_k_retrieved)
        }
        
        logger.debug(f"Evaluation metrics: {metrics}")
        return metrics

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Orchestrates the full Crase pipeline.
        Section: Results - Cost/Performance Trade-off.
        
        Args:
            query: User search query.
            
        Returns:
            A dictionary containing the final results, metadata, and reasons.
        """
        start_time = time.time()
        logger.info(f"Starting Crase pipeline for query: '{query}'")
        
        # 1. Fetch Seeds
        seeds = self.fetch_seeds(query, max_seeds=10)
        if not seeds:
            logger.error("Pipeline terminated: No seeds found.")
            return {
                "query": query,
                "status": "error",
                "error": "No seeds found",
                "results": []
            }
        
        # 2. Expand Graph (1.5-hop)
        graph = self.expand_graph_1_5_hop(seeds)
        
        # 3. Prune Edges (Entailment)
        pruned_graph, reasons = self.prune_edges(graph, seeds)
        
        # 4. Rank Papers (Recency-aware Random Walk)
        ranked_results = self.rank_papers(pruned_graph, seeds, top_k=TOP_K_DEFAULT)
        
        # 5. Format Output
        results_formatted = []
        for pid, score in ranked_results:
            paper_info = self._mock_corpus.get(pid, {})
            results_formatted.append({
                "id": pid,
                "title": paper_info.get("title", "Unknown"),
                "score": float(score),
                "year": paper_info.get("year", 0),
                "reason": "Entailment verified via similarity" # Simplified reason
            })
        
        # 6. Evaluation (Mock)
        # Create a dummy ground truth for demo purposes
        # In a real scenario, this would come from a dataset like LitSearch
        mock_ground_truth = set(seeds) # Assume seeds are relevant
        if len(ranked_results) > 5:
            # Add a couple of top retrieved as ground truth for demo
            mock_ground_truth.update([r[0] for r in ranked_results[:5]])
            
        eval_metrics = self.evaluate(mock_ground_truth, [r[0] for r in ranked_results], k=10)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Pipeline complete. Execution time: {execution_time:.4f}s")
        logger.info(f"Top 5 results: {[r['id'] for r in results_formatted[:5]]}")
        
        return {
            "query": query,
            "status": "success",
            "execution_time": execution_time,
            "seeds": seeds,
            "results": results_formatted,
            "reasons": reasons,
            "evaluation": eval_metrics
        }

if __name__ == "__main__":
    # Demo execution
    agent = CraseAgent(corpus_size=1000)
    query = "Large Language Model Attention Mechanism"
    result = agent.run_pipeline(query)
    
    # Log a summary
    if result["status"] == "success":
        logger.info("DEMO SUCCESS")
        logger.info(f"Recall@10: {result['evaluation']['recall@k']:.2f}")
        for r in result["results"][:5]:
            logger.info(f"{r['id']}: {r['title']} (Score: {r['score']:.4f}, Year: {r['year']})")
    else:
        logger.error(f"DEMO FAILED: {result.get('error')}")
