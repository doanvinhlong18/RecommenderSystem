"""
Evaluation Module for Advanced Hybrid Anime Recommender System
================================================================
Evaluates the 3-component hybrid recommender using ranking-based metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)

Compares performance across different weight configurations (α, β, γ).

Author: Senior ML Engineer
Date: February 2026
"""

import os
import io
import sys
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Dict
from Main import AdvancedHybridRecommender


class AdvancedRecommenderEvaluator:
    """
    Evaluator for Advanced Hybrid Anime Recommender System.

    Evaluates the 3-component hybrid model:
    - TF-IDF (lexical similarity)
    - SBERT (semantic similarity)
    - Score normalization (quality)

    Uses simulated user preferences based on anime themes.
    """

    def __init__(self, recommender: AdvancedHybridRecommender):
        """
        Initialize evaluator with a trained recommender.

        Args:
            recommender: Trained AdvancedHybridRecommender instance
        """
        self.recommender = recommender
        self.df = recommender.df

        # Extract unique themes for simulation
        self.available_themes = self._extract_themes()
        print(f"Found {len(self.available_themes)} unique themes for evaluation")

    def _extract_themes(self) -> List[str]:
        """
        Extract unique themes from the dataset.

        Returns:
            List of unique theme strings
        """
        if 'themes' not in self.df.columns:
            raise ValueError("Dataset must have 'themes' column for evaluation")

        # Get all unique themes (themes column may have comma-separated values)
        all_themes = set()
        for themes_str in self.df['themes'].dropna():
            if themes_str and themes_str != 'Unknown':
                # Split by comma and clean up
                themes = [t.strip() for t in str(themes_str).split(',')]
                all_themes.update(themes)

        # Filter out empty strings
        all_themes = [t for t in all_themes if t and t != 'Unknown']
        return all_themes

    def simulate_user_preferences(self, theme: str = None) -> Tuple[str, Set[int]]:
        """
        Simulate user preferences based on a theme.

        Creates a synthetic user profile by:
        1. Randomly selecting a theme (or using provided one)
        2. Finding all anime with that theme as "relevant" items

        Args:
            theme: Optional specific theme to use. If None, randomly select.

        Returns:
            Tuple of (selected_theme, set of relevant anime indices)
        """
        # Select theme randomly if not provided
        if theme is None:
            theme = np.random.choice(self.available_themes)

        # Find all anime containing this theme (these are "relevant" to the user)
        relevant_indices = set()
        for idx, themes_str in enumerate(self.df['themes']):
            if pd.notna(themes_str) and theme.lower() in str(themes_str).lower():
                relevant_indices.add(idx)

        return theme, relevant_indices

    def precision_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K = (# relevant items in top K) / K

        Measures the proportion of recommended items that are relevant.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k <= 0:
            return 0.0

        # Get top K recommendations
        top_k = recommended_indices[:k]

        # Count how many are relevant
        relevant_in_top_k = len(set(top_k) & relevant_indices)

        return relevant_in_top_k / k

    def recall_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# relevant items in top K) / (Total relevant items)

        Measures the proportion of relevant items that were recommended.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if len(relevant_indices) == 0:
            return 0.0

        # Get top K recommendations
        top_k = recommended_indices[:k]

        # Count how many relevant items are in top K
        relevant_in_top_k = len(set(top_k) & relevant_indices)

        return relevant_in_top_k / len(relevant_indices)

    def accuracy_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Accuracy@K.

        Accuracy@K = (# correct predictions) / (Total predictions)

        In recommendation context, accuracy measures how well the model
        correctly identifies relevant items among the top K recommendations.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            Accuracy@K score (0.0 to 1.0)
        """
        if k <= 0:
            return 0.0

        # Get top K recommendations
        top_k = recommended_indices[:k]

        # Count correct predictions (relevant items in top K)
        correct = len(set(top_k) & relevant_indices)

        return correct / k

    def hit_rate_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K (also known as Hit Ratio).

        Hit Rate@K = 1 if at least one relevant item is in top K, else 0

        Binary metric indicating whether any relevant item was recommended.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if k <= 0 or len(relevant_indices) == 0:
            return 0.0

        top_k = set(recommended_indices[:k])

        # Check if any relevant item is in top K
        if top_k & relevant_indices:
            return 1.0
        return 0.0

    def f1_score_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate F1 Score@K.

        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

        Harmonic mean of precision and recall, balancing both metrics.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            F1 Score@K (0.0 to 1.0)
        """
        precision = self.precision_at_k(recommended_indices, relevant_indices, k)
        recall = self.recall_at_k(recommended_indices, relevant_indices, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def mean_reciprocal_rank(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / rank of first relevant item

        Measures how early the first relevant item appears in recommendations.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user

        Returns:
            MRR score (0.0 to 1.0)
        """
        if len(relevant_indices) == 0:
            return 0.0

        for i, idx in enumerate(recommended_indices):
            if idx in relevant_indices:
                return 1.0 / (i + 1)

        return 0.0

    def average_precision(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Average Precision@K.

        AP@K = (1/min(K, |relevant|)) * Σ (Precision@i * rel_i) for i = 1 to K

        Averages precision at each position where a relevant item is found.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            AP@K score (0.0 to 1.0)
        """
        if len(relevant_indices) == 0:
            return 0.0

        top_k = recommended_indices[:k]
        score = 0.0
        num_hits = 0

        for i, idx in enumerate(top_k):
            if idx in relevant_indices:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                score += precision_at_i

        # Normalize by the minimum of K and total relevant items
        num_relevant = min(k, len(relevant_indices))
        if num_relevant == 0:
            return 0.0

        return score / num_relevant

    def dcg_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = Σ (rel_i / log2(i + 1)) for i = 1 to K

        where rel_i = 1 if item at position i is relevant, 0 otherwise.

        Gives higher weight to relevant items appearing earlier in the list.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            DCG@K score
        """
        dcg = 0.0
        top_k = recommended_indices[:k]

        for i, idx in enumerate(top_k):
            # Position is 1-indexed for the formula
            position = i + 1

            # rel_i is 1 if relevant, 0 otherwise
            rel_i = 1.0 if idx in relevant_indices else 0.0

            # DCG formula: rel_i / log2(position + 1)
            dcg += rel_i / np.log2(position + 1)

        return dcg

    def ndcg_at_k(
        self,
        recommended_indices: List[int],
        relevant_indices: Set[int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        where IDCG@K is the ideal DCG (if all top K items were relevant).

        Normalizes DCG to a 0-1 scale for comparison across queries.

        Args:
            recommended_indices: List of recommended anime indices (ordered by rank)
            relevant_indices: Set of indices considered relevant to the user
            k: Number of top recommendations to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        # Calculate actual DCG
        dcg = self.dcg_at_k(recommended_indices, relevant_indices, k)

        # Calculate ideal DCG (IDCG)
        # Ideal case: all positions up to min(k, |relevant|) are relevant
        num_relevant = min(k, len(relevant_indices))

        if num_relevant == 0:
            return 0.0

        # IDCG: sum of 1/log2(i+1) for i = 1 to num_relevant
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _get_recommendation_indices(
        self,
        query_title: str,
        k: int,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> List[int]:
        """
        Get indices of recommended anime for a query.

        Args:
            query_title: Title of the anime to get recommendations for
            k: Number of recommendations
            alpha: TF-IDF weight
            beta: SBERT weight
            gamma: Score weight

        Returns:
            List of recommended anime indices
        """
        # Suppress output during evaluation
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Get recommendations using new 3-weight API
        recs = self.recommender.recommend(
            title=query_title,
            top_k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            verbose=False
        )

        sys.stdout = old_stdout

        if recs.empty:
            return []

        # Get indices by matching names back to original dataframe
        indices = []
        for name in recs['name']:
            # Find index in original dataframe
            idx = self.df[self.df['name'] == name].index
            if len(idx) > 0:
                indices.append(idx[0])

        return indices

    def evaluate_single_query(
        self,
        query_idx: int,
        relevant_indices: Set[int],
        k: int,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single query.

        Args:
            query_idx: Index of the query anime
            relevant_indices: Set of relevant anime indices
            k: Number of recommendations to evaluate
            alpha: TF-IDF weight
            beta: SBERT weight
            gamma: Score weight
            verbose: Whether to print detailed output

        Returns:
            Dictionary with all evaluation metrics
        """
        # Get query anime title
        query_title = self.df.iloc[query_idx]['name']

        if verbose:
            print(f"\n   Query: {query_title}")

        # Get recommendations
        rec_indices = self._get_recommendation_indices(query_title, k, alpha, beta, gamma)

        if len(rec_indices) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'ndcg': 0.0,
                'accuracy': 0.0,
                'f1': 0.0,
                'hit_rate': 0.0,
                'mrr': 0.0,
                'ap': 0.0
            }

        # Calculate all metrics
        precision = self.precision_at_k(rec_indices, relevant_indices, k)
        recall = self.recall_at_k(rec_indices, relevant_indices, k)
        ndcg = self.ndcg_at_k(rec_indices, relevant_indices, k)
        accuracy = self.accuracy_at_k(rec_indices, relevant_indices, k)
        f1 = self.f1_score_at_k(rec_indices, relevant_indices, k)
        hit_rate = self.hit_rate_at_k(rec_indices, relevant_indices, k)
        mrr = self.mean_reciprocal_rank(rec_indices, relevant_indices)
        ap = self.average_precision(rec_indices, relevant_indices, k)

        if verbose:
            print(f"   Precision@{k}: {precision:.4f}")
            print(f"   Recall@{k}: {recall:.4f}")
            print(f"   Accuracy@{k}: {accuracy:.4f}")
            print(f"   F1@{k}: {f1:.4f}")
            print(f"   Hit Rate@{k}: {hit_rate:.4f}")
            print(f"   MRR: {mrr:.4f}")
            print(f"   AP@{k}: {ap:.4f}")
            print(f"   NDCG@{k}: {ndcg:.4f}")

        return {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'accuracy': accuracy,
            'f1': f1,
            'hit_rate': hit_rate,
            'mrr': mrr,
            'ap': ap
        }

    def evaluate_model(
        self,
        num_trials: int = 50,
        k: int = 5,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the recommender model over multiple trials.

        Procedure:
        1. For each trial:
           a. Simulate user preferences (random theme)
           b. Pick a random anime with that theme as query
           c. Generate top K recommendations
           d. Compute all metrics
        2. Report average metrics across all trials

        Args:
            num_trials: Number of evaluation trials
            k: Number of recommendations per query
            alpha: TF-IDF weight
            beta: SBERT weight
            gamma: Score weight
            verbose: Whether to print progress

        Returns:
            Dictionary with all average metrics
        """
        if verbose:
            print("=" * 60)
            print(f"EVALUATING RECOMMENDER MODEL")
            print(f"   Trials: {num_trials}, K: {k}")
            print(f"   Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
            print("=" * 60)

        # Store metrics for each trial
        precisions = []
        recalls = []
        ndcgs = []
        accuracies = []
        f1_scores = []
        hit_rates = []
        mrrs = []
        aps = []

        successful_trials = 0

        for trial in range(num_trials):
            if verbose and (trial + 1) % 10 == 0:
                print(f"   Progress: {trial + 1}/{num_trials} trials completed...")

            # Simulate user preferences
            theme, relevant_indices = self.simulate_user_preferences()

            # Skip if too few relevant items (need at least 2: 1 for query, 1 for evaluation)
            if len(relevant_indices) < 2:
                continue

            # Pick a random anime with this theme as the query
            relevant_list = list(relevant_indices)
            query_idx = np.random.choice(relevant_list)

            # Remove query from relevant set (can't recommend the query itself)
            eval_relevant = relevant_indices - {query_idx}

            if len(eval_relevant) == 0:
                continue

            # Evaluate this query
            metrics = self.evaluate_single_query(
                query_idx=query_idx,
                relevant_indices=eval_relevant,
                k=k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                verbose=False
            )

            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            ndcgs.append(metrics['ndcg'])
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1'])
            hit_rates.append(metrics['hit_rate'])
            mrrs.append(metrics['mrr'])
            aps.append(metrics['ap'])
            successful_trials += 1

        # Calculate averages
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
        avg_mrr = np.mean(mrrs) if mrrs else 0.0
        avg_ap = np.mean(aps) if aps else 0.0

        # Calculate standard deviations
        std_precision = np.std(precisions) if precisions else 0.0
        std_recall = np.std(recalls) if recalls else 0.0
        std_ndcg = np.std(ndcgs) if ndcgs else 0.0
        std_accuracy = np.std(accuracies) if accuracies else 0.0
        std_f1 = np.std(f1_scores) if f1_scores else 0.0
        std_hit_rate = np.std(hit_rates) if hit_rates else 0.0
        std_mrr = np.std(mrrs) if mrrs else 0.0
        std_ap = np.std(aps) if aps else 0.0

        if verbose:
            print("\n" + "=" * 60)
            print(f"EVALUATION RESULTS (K={k})")
            print("=" * 60)
            print(f"   Successful trials: {successful_trials}/{num_trials}")
            print("-" * 50)
            print(f"   Average Precision@{k}:  {avg_precision:.4f} (+/-{std_precision:.4f})")
            print(f"   Average Recall@{k}:     {avg_recall:.4f} (+/-{std_recall:.4f})")
            print(f"   Average Accuracy@{k}:   {avg_accuracy:.4f} (+/-{std_accuracy:.4f})")
            print(f"   Average F1@{k}:         {avg_f1:.4f} (+/-{std_f1:.4f})")
            print(f"   Average Hit Rate@{k}:   {avg_hit_rate:.4f} (+/-{std_hit_rate:.4f})")
            print(f"   Average MRR:            {avg_mrr:.4f} (+/-{std_mrr:.4f})")
            print(f"   Average AP@{k}:         {avg_ap:.4f} (+/-{std_ap:.4f})")
            print(f"   Average NDCG@{k}:       {avg_ndcg:.4f} (+/-{std_ndcg:.4f})")
            print("=" * 60)

        return {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_ndcg': avg_ndcg,
            'avg_accuracy': avg_accuracy,
            'avg_f1': avg_f1,
            'avg_hit_rate': avg_hit_rate,
            'avg_mrr': avg_mrr,
            'avg_ap': avg_ap,
            'std_precision': std_precision,
            'std_recall': std_recall,
            'std_ndcg': std_ndcg,
            'std_accuracy': std_accuracy,
            'std_f1': std_f1,
            'std_hit_rate': std_hit_rate,
            'std_mrr': std_mrr,
            'std_ap': std_ap,
            'successful_trials': successful_trials
        }

    def compare_weight_configurations(
        self,
        configurations: List[Dict[str, float]] = None,
        num_trials: int = 50,
        k: int = 5
    ) -> pd.DataFrame:
        """
        Compare model performance across different weight configurations.

        This is the key comparison for the 3-component hybrid model.

        Args:
            configurations: List of dicts with 'alpha', 'beta', 'gamma' keys
            num_trials: Number of trials per configuration
            k: Number of recommendations

        Returns:
            DataFrame with metrics for each configuration
        """
        if configurations is None:
            # Default configurations to test
            configurations = [
                {'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1, 'name': 'TF-IDF Heavy'},
                {'alpha': 0.1, 'beta': 0.8, 'gamma': 0.1, 'name': 'SBERT Heavy'},
                {'alpha': 0.1, 'beta': 0.1, 'gamma': 0.8, 'name': 'Score Heavy'},
                {'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2, 'name': 'Balanced (Default)'},
                {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.2, 'name': 'Semantic Focus'},
                {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2, 'name': 'Lexical Focus'},
                {'alpha': 0.33, 'beta': 0.33, 'gamma': 0.34, 'name': 'Equal Weights'},
            ]

        print("\n" + "=" * 70)
        print("WEIGHT CONFIGURATION COMPARISON STUDY")
        print("   Comparing TF-IDF (alpha) vs SBERT (beta) vs Score (gamma)")
        print("=" * 70)

        results = []

        for config in configurations:
            alpha = config['alpha']
            beta = config['beta']
            gamma = config['gamma']
            name = config.get('name', f'alpha={alpha}, beta={beta}, gamma={gamma}')

            print(f"\n Testing: {name}")
            print(f"   Weights: alpha={alpha}, beta={beta}, gamma={gamma}")

            metrics = self.evaluate_model(
                num_trials=num_trials,
                k=k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                verbose=False
            )

            results.append({
                'Configuration': name,
                'alpha (TF-IDF)': alpha,
                'beta (SBERT)': beta,
                'gamma (Score)': gamma,
                'Precision@K': metrics['avg_precision'],
                'Recall@K': metrics['avg_recall'],
                'Accuracy@K': metrics['avg_accuracy'],
                'F1@K': metrics['avg_f1'],
                'Hit Rate@K': metrics['avg_hit_rate'],
                'MRR': metrics['avg_mrr'],
                'AP@K': metrics['avg_ap'],
                'NDCG@K': metrics['avg_ndcg']
            })

            print(f"   Precision@{k}: {metrics['avg_precision']:.4f}")
            print(f"   Recall@{k}: {metrics['avg_recall']:.4f}")
            print(f"   Accuracy@{k}: {metrics['avg_accuracy']:.4f}")
            print(f"   F1@{k}: {metrics['avg_f1']:.4f}")
            print(f"   Hit Rate@{k}: {metrics['avg_hit_rate']:.4f}")
            print(f"   MRR: {metrics['avg_mrr']:.4f}")
            print(f"   AP@{k}: {metrics['avg_ap']:.4f}")
            print(f"   NDCG@{k}: {metrics['avg_ndcg']:.4f}")

        df_results = pd.DataFrame(results)

        print("\n" + "-" * 70)
        print("Summary Table:")
        print(df_results.to_string(index=False))

        # Find best configuration
        best_idx = df_results['NDCG@K'].idxmax()
        best_config = df_results.iloc[best_idx]['Configuration']
        print(f"\n Best configuration by NDCG: {best_config}")

        return df_results

    def evaluate_k_comparison(
        self,
        k_values: List[int] = [3, 5, 10, 15],
        num_trials: int = 50,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> pd.DataFrame:
        """
        Compare model performance across different K values.

        Args:
            k_values: List of K values to compare
            num_trials: Number of trials per K
            alpha: TF-IDF weight
            beta: SBERT weight
            gamma: Score weight

        Returns:
            DataFrame with metrics for each K
        """
        print("\n" + "=" * 60)
        print("K VALUE COMPARISON STUDY")
        print(f"   Fixed weights: alpha={alpha}, beta={beta}, gamma={gamma}")
        print("=" * 60)

        results = []

        for k in k_values:
            print(f"\n Evaluating K = {k}")
            metrics = self.evaluate_model(
                num_trials=num_trials,
                k=k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                verbose=False
            )

            results.append({
                'K': k,
                'Precision@K': metrics['avg_precision'],
                'Recall@K': metrics['avg_recall'],
                'Accuracy@K': metrics['avg_accuracy'],
                'F1@K': metrics['avg_f1'],
                'Hit Rate@K': metrics['avg_hit_rate'],
                'MRR': metrics['avg_mrr'],
                'AP@K': metrics['avg_ap'],
                'NDCG@K': metrics['avg_ndcg']
            })

            print(f"   Precision@{k}: {metrics['avg_precision']:.4f}")
            print(f"   Recall@{k}: {metrics['avg_recall']:.4f}")
            print(f"   Accuracy@{k}: {metrics['avg_accuracy']:.4f}")
            print(f"   F1@{k}: {metrics['avg_f1']:.4f}")
            print(f"   Hit Rate@{k}: {metrics['avg_hit_rate']:.4f}")
            print(f"   MRR: {metrics['avg_mrr']:.4f}")
            print(f"   AP@{k}: {metrics['avg_ap']:.4f}")
            print(f"   NDCG@{k}: {metrics['avg_ndcg']:.4f}")

        df_results = pd.DataFrame(results)

        print("\n" + "-" * 40)
        print("Summary Table:")
        print(df_results.to_string(index=False))

        return df_results


def main():
    """
    Main evaluation function for Advanced Hybrid Recommender.

    Runs comprehensive evaluation including:
    - Main evaluation with default weights
    - Weight configuration comparison (TF-IDF vs SBERT vs Score)
    - K value comparison
    """
    print("=" * 70)
    print("ADVANCED HYBRID ANIME RECOMMENDER - EVALUATION")
    print("   TF-IDF (alpha) + SBERT (beta) + Score (gamma)")
    print("=" * 70)

    # Load the trained model
    MODEL_PATH = "advanced_recommender_model.pkl"

    if not os.path.exists(MODEL_PATH):
        print("No saved model found. Please run demo.py first to create the model.")
        return None

    # Load recommender
    print("\nLoading trained model...")
    recommender = AdvancedHybridRecommender.from_saved_model(MODEL_PATH)

    if recommender is None:
        print("Failed to load model.")
        return None

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = AdvancedRecommenderEvaluator(recommender)

    # ==========================================================================
    # EVALUATION 1: Main evaluation with default weights
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MAIN EVALUATION (50 trials, K=5)")
    print("   Default weights: alpha=0.4, beta=0.4, gamma=0.2")
    print("=" * 70)

    main_results = evaluator.evaluate_model(
        num_trials=50,
        k=5,
        alpha=0.4,
        beta=0.4,
        gamma=0.2,
        verbose=True
    )

    # ==========================================================================
    # EVALUATION 2: Weight configuration comparison
    # ==========================================================================
    print("\n")
    weight_results = evaluator.compare_weight_configurations(
        num_trials=30,
        k=5
    )

    # ==========================================================================
    # EVALUATION 3: K value comparison
    # ==========================================================================
    print("\n")
    k_results = evaluator.evaluate_k_comparison(
        k_values=[3, 5, 10, 15],
        num_trials=30,
        alpha=0.4,
        beta=0.4,
        gamma=0.2
    )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print("\nMain Evaluation Results (K=5, alpha=0.4, beta=0.4, gamma=0.2):")
    print(f"   Precision@5:  {main_results['avg_precision']:.4f}")
    print(f"   Recall@5:     {main_results['avg_recall']:.4f}")
    print(f"   Accuracy@5:   {main_results['avg_accuracy']:.4f}")
    print(f"   F1@5:         {main_results['avg_f1']:.4f}")
    print(f"   Hit Rate@5:   {main_results['avg_hit_rate']:.4f}")
    print(f"   MRR:          {main_results['avg_mrr']:.4f}")
    print(f"   AP@5:         {main_results['avg_ap']:.4f}")
    print(f"   NDCG@5:       {main_results['avg_ndcg']:.4f}")

    print("\nBest Weight Configuration:")
    best_config_idx = weight_results['NDCG@K'].idxmax()
    best_config = weight_results.iloc[best_config_idx]
    print(f"   {best_config['Configuration']}")
    print(f"   NDCG@5: {best_config['NDCG@K']:.4f}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)

    return {
        'evaluator': evaluator,
        'main_results': main_results,
        'weight_results': weight_results,
        'k_results': k_results
    }


if __name__ == "__main__":
    results = main()
