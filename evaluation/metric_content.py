"""
Evaluation Metrics for Recommender Systems.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderMetric:
    """
    Evaluation metrics for recommender systems.

    Implements:
    - Precision@K
    - Recall@K
    - MAP (Mean Average Precision)
    - NDCG (Normalized Discounted Cumulative Gain)
    - Hit Rate
    - Coverage
    - RMSE / MAE (for rating prediction)
    """

    @staticmethod
    def precision_at_k(
        recommended: List[int], relevant: Set[int], k: int = 10
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K = |recommended ∩ relevant| / K

        Args:
            recommended: List of recommended item IDs (ordered)
            relevant: Set of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0

        recommended_k = recommended[:k]
        n_relevant = len(set(recommended_k) & relevant)

        return n_relevant / k

    @staticmethod
    def recall_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
        """
        Calculate Recall@K.

        Recall@K = |recommended ∩ relevant| / |relevant|

        Args:
            recommended: List of recommended item IDs (ordered)
            relevant: Set of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0

        recommended_k = recommended[:k]
        n_relevant = len(set(recommended_k) & relevant)

        return n_relevant / len(relevant)

    @staticmethod
    def average_precision(
        recommended: List[int], relevant: Set[int], k: int = None
    ) -> float:
        """
        Calculate Average Precision.

        AP = (1/|relevant|) * Σ(Precision@i * rel(i))

        Args:
            recommended: List of recommended item IDs (ordered)
            relevant: Set of relevant item IDs
            k: Max number of recommendations (None = all)

        Returns:
            Average Precision score
        """
        if len(relevant) == 0:
            return 0.0

        if k is not None:
            recommended = recommended[:k]

        score = 0.0
        n_hits = 0

        for i, item_id in enumerate(recommended):
            if item_id in relevant:
                n_hits += 1
                precision_at_i = n_hits / (i + 1)
                score += precision_at_i

        return score / len(relevant)

    @staticmethod
    def mean_average_precision(
        recommendations: Dict[int, List[int]],
        relevant_items: Dict[int, Set[int]],
        k: int = None,
    ) -> float:
        """
        Calculate Mean Average Precision across users.

        MAP = (1/|U|) * Σ AP(u)

        Args:
            recommendations: Dict of user_id -> recommended item list
            relevant_items: Dict of user_id -> set of relevant items
            k: Max number of recommendations per user

        Returns:
            MAP score
        """
        if len(recommendations) == 0:
            return 0.0

        total_ap = 0.0

        for user_id, rec_list in recommendations.items():
            relevant = relevant_items.get(user_id, set())
            ap = RecommenderMetric.average_precision(rec_list, relevant, k)
            total_ap += ap

        return total_ap / len(recommendations)

    @staticmethod
    def dcg_at_k(
        recommended: List[int],
        relevant: Set[int],
        k: int = 10,
        relevance_scores: Dict[int, float] = None,
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = Σ (rel_i / log2(i + 1))

        Args:
            recommended: List of recommended item IDs (ordered)
            relevant: Set of relevant item IDs
            k: Number of top recommendations
            relevance_scores: Optional dict of item_id -> relevance score

        Returns:
            DCG@K score
        """
        dcg = 0.0

        for i, item_id in enumerate(recommended[:k]):
            if item_id in relevant:
                # Binary relevance or use provided scores
                rel = relevance_scores.get(item_id, 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed

        return dcg

    @staticmethod
    def ndcg_at_k(
        recommended: List[int],
        relevant: Set[int],
        k: int = 10,
        relevance_scores: Dict[int, float] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        Args:
            recommended: List of recommended item IDs (ordered)
            relevant: Set of relevant item IDs
            k: Number of top recommendations
            relevance_scores: Optional dict of item_id -> relevance score

        Returns:
            NDCG@K score
        """
        if len(relevant) == 0:
            return 0.0

        # Calculate DCG
        dcg = RecommenderMetric.dcg_at_k(recommended, relevant, k, relevance_scores)

        # Calculate ideal DCG (perfect ranking)
        if relevance_scores:
            ideal_order = sorted(
                relevant, key=lambda x: relevance_scores.get(x, 1.0), reverse=True
            )
        else:
            ideal_order = list(relevant)

        idcg = RecommenderMetric.dcg_at_k(ideal_order, relevant, k, relevance_scores)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(
        recommendations: Dict[int, List[int]],
        relevant_items: Dict[int, Set[int]],
        k: int = 10,
    ) -> float:
        """
        Calculate Hit Rate at K.

        Hit Rate = |users with at least one hit| / |users|

        Args:
            recommendations: Dict of user_id -> recommended item list
            relevant_items: Dict of user_id -> set of relevant items
            k: Number of top recommendations

        Returns:
            Hit Rate score
        """
        if len(recommendations) == 0:
            return 0.0

        n_hits = 0

        for user_id, rec_list in recommendations.items():
            relevant = relevant_items.get(user_id, set())
            if len(set(rec_list[:k]) & relevant) > 0:
                n_hits += 1

        return n_hits / len(recommendations)

    @staticmethod
    def coverage(
        recommendations: Dict[int, List[int]], all_items: Set[int], k: int = 10
    ) -> float:
        """
        Calculate catalog coverage.

        Coverage = |unique recommended items| / |all items|

        Args:
            recommendations: Dict of user_id -> recommended item list
            all_items: Set of all item IDs
            k: Number of top recommendations per user

        Returns:
            Coverage score
        """
        if len(all_items) == 0:
            return 0.0

        recommended_items = set()
        for rec_list in recommendations.values():
            recommended_items.update(rec_list[:k])

        return len(recommended_items) / len(all_items)

    @staticmethod
    def f1_at_k(precision: float, recall: float) -> float:
        """F1 = harmonic mean of precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def mrr(recommended: List[int], relevant: Set[int]) -> float:
        """
        Mean Reciprocal Rank — vị trí đầu tiên có item relevant.
        Truyền recommended đã slice theo K từ caller.
        """
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def map_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        AP@K — chia cho min(|relevant|, K) thay vì |relevant|.
        Phản ánh đúng hơn khi K < |relevant|: thưởng model tìm được nhiều
        relevant items trong top-K chứ không phạt vì relevant set lớn.
        Khác với average_precision() chia cho |relevant|.
        """
        if not relevant or not recommended:
            return 0.0
        hits, sum_p = 0, 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                hits += 1
                sum_p += hits / (i + 1)
        return sum_p / min(len(relevant), k)

    @classmethod
    def calculate_all_metrics(
        cls,
        recommended: List[int],
        relevant: Set[int],
        k: int,
    ) -> Dict[str, float]:
        """
        Compute tất cả metrics cho một user tại một K.
        Dùng trong evaluation loop thay vì gọi từng hàm riêng.
        """
        prec = cls.precision_at_k(recommended, relevant, k)
        rec = cls.recall_at_k(recommended, relevant, k)
        return {
            "precision": prec,
            "recall": rec,
            "f1": cls.f1_at_k(prec, rec),
            "hit_rate": 1.0 if set(recommended[:k]) & relevant else 0.0,
            "mrr": cls.mrr(recommended[:k], relevant),
            "ndcg": cls.ndcg_at_k(recommended, relevant, k),
            "map": cls.map_at_k(recommended, relevant, k),
        }

    @staticmethod
    def rmse(predictions: List[Tuple[int, int, float, float]]) -> float:
        """
        Calculate Root Mean Square Error.

        RMSE = sqrt(mean((predicted - actual)^2))

        Args:
            predictions: List of (user_id, item_id, actual_rating, predicted_rating)

        Returns:
            RMSE score
        """
        if len(predictions) == 0:
            return 0.0

        errors = [(actual - pred) ** 2 for _, _, actual, pred in predictions]
        return np.sqrt(np.mean(errors))

    @staticmethod
    def mae(predictions: List[Tuple[int, int, float, float]]) -> float:
        """
        Calculate Mean Absolute Error.

        MAE = mean(|predicted - actual|)

        Args:
            predictions: List of (user_id, item_id, actual_rating, predicted_rating)

        Returns:
            MAE score
        """
        if len(predictions) == 0:
            return 0.0

        errors = [abs(actual - pred) for _, _, actual, pred in predictions]
        return np.mean(errors)

    @classmethod
    def evaluate_recommendations(
        cls,
        recommendations: Dict[int, List[int]],
        relevant_items: Dict[int, Set[int]],
        all_items: Set[int] = None,
        k_values: List[int] = [5, 10, 20],
    ) -> Dict:
        """
        Comprehensive evaluation of recommendations.

        Args:
            recommendations: Dict of user_id -> recommended item list
            relevant_items: Dict of user_id -> set of relevant items
            all_items: Set of all item IDs (for coverage)
            k_values: List of K values to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        for k in k_values:
            # Precision@K
            precisions = []
            for user_id, rec_list in recommendations.items():
                relevant = relevant_items.get(user_id, set())
                precisions.append(cls.precision_at_k(rec_list, relevant, k))
            results[f"Precision@{k}"] = np.mean(precisions)

            # Recall@K
            recalls = []
            for user_id, rec_list in recommendations.items():
                relevant = relevant_items.get(user_id, set())
                recalls.append(cls.recall_at_k(rec_list, relevant, k))
            results[f"Recall@{k}"] = np.mean(recalls)

            # NDCG@K
            ndcgs = []
            for user_id, rec_list in recommendations.items():
                relevant = relevant_items.get(user_id, set())
                ndcgs.append(cls.ndcg_at_k(rec_list, relevant, k))
            results[f"NDCG@{k}"] = np.mean(ndcgs)

            # Hit Rate@K
            results[f"HitRate@{k}"] = cls.hit_rate_at_k(
                recommendations, relevant_items, k
            )

            # Coverage@K
            if all_items:
                results[f"Coverage@{k}"] = cls.coverage(recommendations, all_items, k)

        # MAP
        results["MAP"] = cls.mean_average_precision(recommendations, relevant_items)

        return results

    @classmethod
    def compare_models(
        cls,
        model_recommendations: Dict[str, Dict[int, List[int]]],
        relevant_items: Dict[int, Set[int]],
        all_items: Set[int] = None,
        k_values: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Compare multiple recommendation models.

        Args:
            model_recommendations: Dict of model_name -> recommendations
            relevant_items: Dict of user_id -> set of relevant items
            all_items: Set of all item IDs
            k_values: List of K values

        Returns:
            DataFrame comparing model performance
        """
        results = {}

        for model_name, recommendations in model_recommendations.items():
            results[model_name] = cls.evaluate_recommendations(
                recommendations, relevant_items, all_items, k_values
            )

        df = pd.DataFrame(results).T
        df.index.name = "Model"

        return df


if __name__ == "__main__":
    # Test metrics
    print("Testing Recommender Metrics...")

    # Sample data
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = {2, 4, 6, 8, 10}

    print(f"\nRecommended: {recommended}")
    print(f"Relevant: {relevant}")

    print(
        f"\nPrecision@5: {RecommenderMetric.precision_at_k(recommended, relevant, 5):.4f}"
    )
    print(
        f"Precision@10: {RecommenderMetric.precision_at_k(recommended, relevant, 10):.4f}"
    )
    print(f"Recall@5: {RecommenderMetric.recall_at_k(recommended, relevant, 5):.4f}")
    print(f"Recall@10: {RecommenderMetric.recall_at_k(recommended, relevant, 10):.4f}")
    print(f"AP: {RecommenderMetric.average_precision(recommended, relevant):.4f}")
    print(f"NDCG@5: {RecommenderMetric.ndcg_at_k(recommended, relevant, 5):.4f}")
    print(f"NDCG@10: {RecommenderMetric.ndcg_at_k(recommended, relevant, 10):.4f}")

    # Multi-user evaluation
    recommendations = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10], 3: [2, 4, 6, 8, 10]}
    relevant_items = {1: {2, 4}, 2: {7, 8, 11}, 3: {2, 4, 6, 8, 10}}

    print("\n=== Multi-user Evaluation ===")
    results = RecommenderMetric.evaluate_recommendations(
        recommendations, relevant_items, k_values=[3, 5]
    )
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
