"""
Evaluation script for individual recommendation models.

Evaluates each model separately:
- Content-Based Recommender (TF-IDF + SBERT)
- Collaborative Filtering (Matrix Factorization)
- Implicit Feedback (ALS)
- Popularity-Based
- Hybrid Engine (Combined)

FIXES APPLIED:
1. Train/Test split now keeps ALL interactions in train, only relevant items in test
2. Content-Based evaluation uses multiple items to build user profile
3. Implicit ALS evaluation uses correct index mapping
4. Hybrid evaluation uses consistent ID space

Usage:
    python evaluate_models.py [--sample-users 500] [--k 5 10 20] [--output results.json]
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR
from models.content import ContentBasedRecommender
from models.collaborative import MatrixFactorization
from models.implicit import ALSImplicit
from models.popularity import PopularityModel
from models.hybrid import HybridEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading with Memory Optimization
# =============================================================================

def load_ratings_for_evaluation(sample_size: int = 1000000) -> pd.DataFrame:
    """
    Load ratings data with memory optimization.

    Args:
        sample_size: Number of ratings to sample

    Returns:
        DataFrame with user_id, anime_id, rating columns
    """
    from config import DATASET_PATH

    rating_file = DATASET_PATH / "rating_complete.csv"

    if not rating_file.exists():
        raise FileNotFoundError(f"Rating file not found: {rating_file}")

    logger.info(f"Loading ratings (sample: {sample_size:,})...")

    chunks = []
    total_rows = 0

    # Use smaller dtype to save memory
    dtype = {
        'user_id': 'int32',
        'anime_id': 'int32',
        'rating': 'int8'
    }

    try:
        for chunk in pd.read_csv(
            rating_file,
            usecols=['user_id', 'anime_id', 'rating'],
            dtype=dtype,
            chunksize=500000
        ):
            # Keep ALL ratings (including negative) - FIX #1
            # Only filter out invalid ratings (-1 means not rated)
            chunk = chunk[chunk['rating'] >= 0]
            chunks.append(chunk)
            total_rows += len(chunk)

            if total_rows >= sample_size:
                break

        ratings_df = pd.concat(chunks, ignore_index=True)

        if len(ratings_df) > sample_size:
            ratings_df = ratings_df.sample(n=sample_size, random_state=42)

        logger.info(f"Loaded {len(ratings_df):,} ratings")
        return ratings_df

    except Exception as e:
        logger.error(f"Error loading ratings: {e}")
        raise


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K = |recommended ∩ relevant| / K"""
    if k <= 0 or not recommended:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K = |recommended ∩ relevant| / |relevant|"""
    if not relevant or not recommended:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    """F1@K = 2 * Precision * Recall / (Precision + Recall)"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Hit Rate@K = 1 if any relevant item in top K, else 0"""
    if not relevant or not recommended:
        return 0.0
    rec_k = set(recommended[:k])
    return 1.0 if rec_k & relevant else 0.0


def mrr(recommended: List[int], relevant: Set[int]) -> float:
    """MRR = 1 / rank of first relevant item"""
    if not relevant or not recommended:
        return 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K"""
    if not relevant or not recommended:
        return 0.0

    rec_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    # IDCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """MAP@K = mean of precision at each relevant position"""
    if not relevant or not recommended:
        return 0.0

    rec_k = recommended[:k]
    hits = 0
    sum_precision = 0.0

    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(len(relevant), k)


def calculate_all_metrics(recommended: List[int], relevant: Set[int], k: int) -> Dict[str, float]:
    """Calculate all metrics for a single prediction."""
    prec = precision_at_k(recommended, relevant, k)
    rec = recall_at_k(recommended, relevant, k)

    return {
        'precision': prec,
        'recall': rec,
        'f1': f1_at_k(prec, rec),
        'hit_rate': hit_rate_at_k(recommended, relevant, k),
        'mrr': mrr(recommended[:k], relevant),
        'ndcg': ndcg_at_k(recommended, relevant, k),
        'map': map_at_k(recommended, relevant, k)
    }


# =============================================================================
# Data Preparation - FIXED
# =============================================================================

def prepare_evaluation_data(
    ratings_df: pd.DataFrame,
    min_train_items: int = 10,
    min_test_items: int = 3,
    test_ratio: float = 0.2,
    relevance_threshold: float = 7.0,
    leave_one_out: bool = False
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Set[int]], List[int]]:
    """
    Prepare train/test split for evaluation - FIXED VERSION.

    FIX #1: Train set contains ALL interactions (positive + negative)
            This is critical for CF/ALS to learn user preferences properly.

    FIX #2: Test set contains ONLY relevant items (rating >= threshold)
            We evaluate whether the model can predict items the user will LIKE.

    FIX #3: Split is done per-user to ensure fair evaluation.

    Args:
        ratings_df: DataFrame with user_id, anime_id, rating
        min_train_items: Minimum items in training set per user
        min_test_items: Minimum relevant items in test set per user
        test_ratio: Ratio of relevant items for test set
        relevance_threshold: Rating threshold to consider item as "relevant"
        leave_one_out: If True, use Leave-One-Out evaluation (1 item in test)

    Returns:
        user_train: Dict of user_id -> {anime_id: rating} (ALL interactions)
        user_test: Dict of user_id -> set of anime_ids (relevant items only)
        eval_users: List of user_ids eligible for evaluation
    """
    logger.info("Preparing evaluation data (FIXED logic)...")
    logger.info(f"  - Relevance threshold: {relevance_threshold}")
    logger.info(f"  - Leave-One-Out mode: {leave_one_out}")

    user_train = {}  # user_id -> {anime_id: rating}
    user_test = {}   # user_id -> set of test anime_ids (relevant only)
    eval_users = []

    # Group by user
    user_groups = ratings_df.groupby('user_id')

    for user_id, group in user_groups:
        # Get ALL user interactions (for training)
        all_items = dict(zip(group['anime_id'], group['rating']))

        # Get only relevant items (for test set selection)
        relevant_items = group[group['rating'] >= relevance_threshold]['anime_id'].tolist()

        # Skip users with insufficient data
        if len(all_items) < min_train_items:
            continue
        if len(relevant_items) < min_test_items + 1:  # Need at least 1 for train
            continue

        # Shuffle relevant items for random split
        np.random.shuffle(relevant_items)

        if leave_one_out:
            # Leave-One-Out: Put exactly 1 relevant item in test
            test_items = set([relevant_items[0]])
            # Train contains ALL items (including the test item's rating context)
            # But we exclude test item from train for fair evaluation
            train_items = {k: v for k, v in all_items.items() if k not in test_items}
        else:
            # Standard split: Put test_ratio of relevant items in test
            n_test = max(min_test_items, int(len(relevant_items) * test_ratio))
            n_test = min(n_test, len(relevant_items) - 1)  # Keep at least 1 for train

            test_items = set(relevant_items[:n_test])
            # Train contains ALL items except test items
            train_items = {k: v for k, v in all_items.items() if k not in test_items}

        # Verify we have enough data
        if len(train_items) < min_train_items:
            continue
        if len(test_items) < (1 if leave_one_out else min_test_items):
            continue

        user_train[user_id] = train_items
        user_test[user_id] = test_items
        eval_users.append(user_id)

    logger.info(f"Prepared {len(eval_users)} users for evaluation")
    logger.info(f"  - Avg train items: {np.mean([len(v) for v in user_train.values()]):.1f}")
    logger.info(f"  - Avg test items: {np.mean([len(v) for v in user_test.values()]):.1f}")

    return user_train, user_test, eval_users


def prepare_implicit_evaluation_data(
    ratings_df: pd.DataFrame,
    implicit_model: ALSImplicit,
    min_train_items: int = 5,
    min_test_items: int = 1,
    test_ratio: float = 0.2
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], List[int]]:
    """
    Prepare train/test split specifically for Implicit ALS evaluation.

    KEY DIFFERENCE FROM EXPLICIT EVALUATION:
    - No rating threshold (all interactions are "positive" implicit feedback)
    - Only users that exist in the implicit model AND have valid indices are included
    - Ground truth = any item the user interacted with

    Args:
        ratings_df: DataFrame with user_id, anime_id, rating
        implicit_model: Trained ALSImplicit model (to get valid users)
        min_train_items: Minimum items in training set per user
        min_test_items: Minimum items in test set per user
        test_ratio: Ratio of items for test set

    Returns:
        user_train: Dict of user_id -> set of training anime_ids
        user_test: Dict of user_id -> set of test anime_ids
        eval_users: List of user_ids eligible for evaluation
    """
    logger.info("Preparing IMPLICIT evaluation data...")

    if implicit_model is None:
        logger.error("Implicit model is None")
        return {}, {}, []

    # Get users and items from model
    model_users = set(implicit_model.user_to_idx.keys())
    model_items = set(implicit_model.anime_to_idx.keys())

    # Determine actual factor dimensions (handle swapped factors)
    n_users_in_mapping = len(implicit_model.user_to_idx)
    n_items_in_mapping = len(implicit_model.anime_to_idx)

    if implicit_model.user_factors.shape[0] == n_items_in_mapping:
        # Factors are swapped - item_factors contains users
        actual_n_users = implicit_model.item_factors.shape[0]
    else:
        actual_n_users = implicit_model.user_factors.shape[0]

    logger.info(f"  Model has {len(model_users)} users in mapping, {actual_n_users} in factors")
    logger.info(f"  Model has {len(model_items)} items")

    user_train = {}  # user_id -> set of anime_ids
    user_test = {}   # user_id -> set of anime_ids
    eval_users = []

    # Group ratings by user
    user_groups = ratings_df.groupby('user_id')

    for user_id, group in user_groups:
        # Skip users not in implicit model
        if user_id not in model_users:
            continue

        # Check if user index is within bounds
        user_idx = implicit_model.user_to_idx[user_id]
        if user_idx >= actual_n_users:
            continue

        # For implicit: ALL interactions are positive (rating > 0)
        # Filter to items that exist in the model
        user_items = group[group['rating'] > 0]['anime_id'].tolist()
        user_items = [aid for aid in user_items if aid in model_items]

        if len(user_items) < min_train_items + min_test_items:
            continue

        # Random split
        np.random.shuffle(user_items)
        n_test = max(min_test_items, int(len(user_items) * test_ratio))
        n_test = min(n_test, len(user_items) - min_train_items)

        test_items = set(user_items[:n_test])
        train_items = set(user_items[n_test:])

        if len(train_items) < min_train_items or len(test_items) < min_test_items:
            continue

        user_train[user_id] = train_items
        user_test[user_id] = test_items
        eval_users.append(user_id)

    if eval_users:
        logger.info(f"Prepared {len(eval_users)} users for implicit evaluation")
        logger.info(f"  - Avg train items: {np.mean([len(v) for v in user_train.values()]):.1f}")
        logger.info(f"  - Avg test items: {np.mean([len(v) for v in user_test.values()]):.1f}")
    else:
        logger.warning("No users prepared for implicit evaluation!")

    return user_train, user_test, eval_users


# =============================================================================
# Content-Based Evaluation - FIXED
# =============================================================================

def evaluate_content_model(
    content_model: ContentBasedRecommender,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500,
    num_profile_items: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Content-Based Recommender - FIXED VERSION.

    FIX #2: Build user profile from MULTIPLE top-rated items, not just one.
            This gives a more representative view of user preferences.

    Strategy:
    1. Get user's top-rated items from training set
    2. Get recommendations for each item
    3. Aggregate scores across all recommendations (voting/averaging)
    4. Compare aggregated recommendations with test set

    Args:
        content_model: Trained ContentBasedRecommender
        user_train: Dict of user_id -> {anime_id: rating}
        user_test: Dict of user_id -> set of test anime_ids
        eval_users: List of users to evaluate
        k_values: List of K values for metrics
        max_users: Maximum users to evaluate
        num_profile_items: Number of top-rated items to use for profile
    """
    logger.info("Evaluating Content-Based Model (FIXED - multi-item profile)...")

    if content_model is None:
        logger.warning("Content model not available")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_no_profile = 0
    skipped_no_recs = 0

    sample_users = eval_users[:max_users] if len(eval_users) > max_users else eval_users

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not train_items or not test_items:
            continue

        try:
            # FIX: Get TOP-RATED items to build user profile (not random)
            sorted_items = sorted(train_items.items(), key=lambda x: x[1], reverse=True)
            profile_items = [item_id for item_id, rating in sorted_items[:num_profile_items]]

            if len(profile_items) == 0:
                skipped_no_profile += 1
                continue

            # Aggregate recommendations from multiple profile items
            item_scores = defaultdict(float)
            item_counts = defaultdict(int)

            for query_anime_id in profile_items:
                try:
                    recommendations = content_model.get_similar_anime(
                        query_anime_id,
                        top_k=max(k_values) * 2  # Get more to aggregate
                    )

                    if not recommendations:
                        continue

                    # Weight by user's rating of the query item
                    query_rating = train_items.get(query_anime_id, 7.0)
                    weight = query_rating / 10.0  # Normalize to [0, 1]

                    for rec in recommendations:
                        rec_id = rec['mal_id']
                        # Skip items already in training set
                        if rec_id in train_items:
                            continue
                        similarity = rec.get('similarity', 0.5)
                        item_scores[rec_id] += similarity * weight
                        item_counts[rec_id] += 1

                except Exception:
                    continue

            if not item_scores:
                skipped_no_recs += 1
                continue

            # Average scores and sort
            avg_scores = {
                item_id: score / item_counts[item_id]
                for item_id, score in item_scores.items()
            }
            sorted_recs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            rec_ids = [item_id for item_id, score in sorted_recs]

            # Calculate metrics
            for k in k_values:
                metrics = calculate_all_metrics(rec_ids, test_items, k)
                for metric, value in metrics.items():
                    results[k][metric].append(value)

            evaluated += 1

        except Exception as e:
            logger.debug(f"Error evaluating user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Content: {i + 1}/{len(sample_users)} processed, {evaluated} evaluated")

    logger.info(f"  Content evaluation: {evaluated} users evaluated")
    logger.info(f"  Skipped (no profile): {skipped_no_profile}, (no recs): {skipped_no_recs}")

    # Aggregate results
    summary = {}
    for k in k_values:
        summary[f'K={k}'] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in results[k].items()
        }
        summary[f'K={k}']['evaluated_users'] = evaluated

    return summary


# =============================================================================
# Collaborative Filtering Evaluation
# =============================================================================

def evaluate_collaborative_model(
    collab_model: MatrixFactorization,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Collaborative Filtering Model.

    Strategy: Get recommendations for each user, excluding training items.
    """
    logger.info("Evaluating Collaborative Filtering Model...")

    if collab_model is None:
        logger.warning("Collaborative model not available")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_not_in_model = 0

    sample_users = eval_users[:max_users] if len(eval_users) > max_users else eval_users

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        try:
            # Check if user exists in model
            if hasattr(collab_model, 'user_to_idx'):
                if user_id not in collab_model.user_to_idx:
                    skipped_not_in_model += 1
                    continue

            recommendations = collab_model.recommend_for_user(
                user_id,
                top_k=max(k_values) + len(train_items)  # Get extra to filter
            )

            if not recommendations:
                continue

            # Filter out training items
            rec_ids = [r['mal_id'] for r in recommendations if r['mal_id'] not in train_items]
            rec_ids = rec_ids[:max(k_values)]

            if not rec_ids:
                continue

            for k in k_values:
                metrics = calculate_all_metrics(rec_ids, test_items, k)
                for metric, value in metrics.items():
                    results[k][metric].append(value)

            evaluated += 1

        except Exception as e:
            logger.debug(f"Error evaluating user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Collaborative: {i + 1}/{len(sample_users)} processed")

    logger.info(f"  Collaborative evaluation: {evaluated} users")
    logger.info(f"  Skipped (not in model): {skipped_not_in_model}")

    summary = {}
    for k in k_values:
        summary[f'K={k}'] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in results[k].items()
        }
        summary[f'K={k}']['evaluated_users'] = evaluated

    return summary


# =============================================================================
# Implicit ALS Evaluation - FIXED FOR IMPLICIT FEEDBACK
# =============================================================================

def evaluate_implicit_model(
    implicit_model: ALSImplicit,
    ratings_df: pd.DataFrame,
    k_values: List[int],
    max_users: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Implicit Feedback Model (ALS) - FIXED FOR IMPLICIT FEEDBACK.

    KEY FIXES:
    1. Use model's own user_to_idx and anime_to_idx mappings
    2. Ground truth = ANY interaction in test set (binary relevance, no threshold)
    3. Explicit bounds checking before accessing model factors
    4. Proper index conversion: user_id -> user_idx, item_idx -> anime_id
    5. Never silently skip - log all skip reasons
    6. Uses separate data preparation for implicit feedback

    For implicit feedback:
    - We don't use rating threshold (all interactions are "positive")
    - Ground truth = items the user interacted with in test set
    - Metrics: HitRate, Recall, NDCG are most meaningful
    """
    logger.info("Evaluating Implicit Feedback Model (FIXED for implicit feedback)...")

    if implicit_model is None:
        logger.warning("Implicit model not available")
        return {}

    # Validate model has required attributes
    if not implicit_model.user_to_idx or not implicit_model.anime_to_idx:
        logger.error("Implicit model missing user_to_idx or anime_to_idx mappings")
        return {}

    if implicit_model.user_factors is None or implicit_model.item_factors is None:
        logger.error("Implicit model factors not initialized")
        return {}

    n_model_users = implicit_model.user_factors.shape[0]
    n_model_items = implicit_model.item_factors.shape[0]

    logger.info(f"  Model dimensions: {n_model_users} users x {n_model_items} items")
    logger.info(f"  Mappings: {len(implicit_model.user_to_idx)} users, {len(implicit_model.anime_to_idx)} items")

    # Prepare implicit-specific evaluation data
    implicit_train, implicit_test, implicit_users = prepare_implicit_evaluation_data(
        ratings_df,
        implicit_model,
        min_train_items=5,
        min_test_items=1,
        test_ratio=0.2
    )

    if not implicit_users:
        logger.error("No users available for implicit evaluation")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0

    # Track skip reasons
    skip_reasons = {
        'index_out_of_bounds': 0,
        'no_test_items': 0,
        'no_recommendations': 0,
        'exception': 0
    }

    # Sample users
    sample_users = implicit_users[:max_users] if len(implicit_users) > max_users else implicit_users
    logger.info(f"  Evaluating {len(sample_users)} users")

    for i, user_id in enumerate(sample_users):
        train_items = implicit_train.get(user_id, set())
        test_items = implicit_test.get(user_id, set())

        if not test_items:
            skip_reasons['no_test_items'] += 1
            continue

        try:
            # Check user index bounds
            user_idx = implicit_model.user_to_idx[user_id]
            if user_idx >= n_model_users:
                skip_reasons['index_out_of_bounds'] += 1
                logger.debug(f"User {user_id} index {user_idx} >= {n_model_users}")
                continue

            # Get recommendations using model's recommend_for_user
            recommendations = implicit_model.recommend_for_user(
                user_id,
                top_k=max(k_values) * 3,  # Get extra to account for filtering
                exclude_known=True,
                known_items=train_items
            )

            if not recommendations:
                skip_reasons['no_recommendations'] += 1
                continue

            # Extract anime_ids from recommendations
            rec_anime_ids = []
            for rec in recommendations:
                anime_id = rec.get('mal_id')
                if anime_id is not None and anime_id not in train_items:
                    rec_anime_ids.append(anime_id)

            if not rec_anime_ids:
                skip_reasons['no_recommendations'] += 1
                continue

            # Calculate metrics - for implicit, all test interactions are relevant
            for k in k_values:
                rec_k = rec_anime_ids[:k]
                metrics = calculate_all_metrics(rec_k, test_items, k)
                for metric, value in metrics.items():
                    results[k][metric].append(value)

            evaluated += 1

        except Exception as e:
            skip_reasons['exception'] += 1
            logger.debug(f"Exception for user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Implicit: {i + 1}/{len(sample_users)} processed, {evaluated} evaluated")

    # Log skip summary
    logger.info(f"  Implicit evaluation complete:")
    logger.info(f"    Evaluated: {evaluated}")
    logger.info(f"    Skipped (index OOB): {skip_reasons['index_out_of_bounds']}")
    logger.info(f"    Skipped (no test items): {skip_reasons['no_test_items']}")
    logger.info(f"    Skipped (no recs): {skip_reasons['no_recommendations']}")
    logger.info(f"    Skipped (exception): {skip_reasons['exception']}")

    # Build summary
    summary = {}
    for k in k_values:
        if results[k]['hit_rate']:  # Check if we have any results
            summary[f'K={k}'] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in results[k].items()
            }
        else:
            summary[f'K={k}'] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'hit_rate': 0.0,
                'mrr': 0.0,
                'ndcg': 0.0,
                'map': 0.0
            }
        summary[f'K={k}']['evaluated_users'] = evaluated

    return summary


# =============================================================================
# Popularity Model Evaluation
# =============================================================================

def evaluate_popularity_model(
    popularity_model: PopularityModel,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Popularity-Based Model (baseline).

    Note: Popular items are the same for all users, but we exclude
    items already in each user's training set for fair comparison.
    """
    logger.info("Evaluating Popularity-Based Model...")

    if popularity_model is None:
        logger.warning("Popularity model not available")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0

    sample_users = eval_users[:max_users] if len(eval_users) > max_users else eval_users

    # Get popular items once
    try:
        popular_recs = popularity_model.get_popular(
            top_k=max(k_values) * 5,  # Get extra for filtering
            popularity_type='top_rated'
        )
        all_popular_ids = [r['mal_id'] for r in popular_recs]
    except Exception as e:
        logger.error(f"Failed to get popular recommendations: {e}")
        return {}

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        # Filter out user's training items
        rec_ids = [pid for pid in all_popular_ids if pid not in train_items]
        rec_ids = rec_ids[:max(k_values)]

        if not rec_ids:
            continue

        for k in k_values:
            metrics = calculate_all_metrics(rec_ids, test_items, k)
            for metric, value in metrics.items():
                results[k][metric].append(value)

        evaluated += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  Popularity: {i + 1}/{len(sample_users)} processed")

    logger.info(f"  Popularity evaluation: {evaluated} users")

    summary = {}
    for k in k_values:
        summary[f'K={k}'] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in results[k].items()
        }
        summary[f'K={k}']['evaluated_users'] = evaluated

    return summary


# =============================================================================
# Hybrid Model Evaluation - FIXED
# =============================================================================

def evaluate_hybrid_model(
    hybrid_engine: HybridEngine,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Hybrid Engine - FIXED VERSION.

    FIX #4: Ensure consistent ID space and proper exclusion of watched items.

    The hybrid model combines multiple sub-models, so we must ensure:
    1. All sub-models use the same ID space (anime_id)
    2. Training items are properly excluded
    3. Cold-start users are handled gracefully
    """
    logger.info("Evaluating Hybrid Engine (FIXED - consistent ID space)...")

    if hybrid_engine is None:
        logger.warning("Hybrid engine not available")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_cold_start = 0

    sample_users = eval_users[:max_users] if len(eval_users) > max_users else eval_users

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        try:
            # Get recommendations with exclude_watched=True
            recommendations = hybrid_engine.recommend_for_user(
                user_id,
                top_k=max(k_values) + len(train_items),
                exclude_watched=True
            )

            if not recommendations:
                # Try without exclusion for cold-start users
                recommendations = hybrid_engine.recommend_for_user(
                    user_id,
                    top_k=max(k_values) * 2,
                    exclude_watched=False
                )
                if not recommendations:
                    skipped_cold_start += 1
                    continue

            # FIX: Ensure anime_ids and filter training items
            rec_ids = []
            for rec in recommendations:
                anime_id = rec.get('mal_id') or rec.get('anime_id')
                if anime_id and anime_id not in train_items:
                    rec_ids.append(anime_id)

            rec_ids = rec_ids[:max(k_values)]

            if not rec_ids:
                continue

            for k in k_values:
                metrics = calculate_all_metrics(rec_ids, test_items, k)
                for metric, value in metrics.items():
                    results[k][metric].append(value)

            evaluated += 1

        except Exception as e:
            logger.debug(f"Error evaluating user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Hybrid: {i + 1}/{len(sample_users)} processed, {evaluated} evaluated")

    logger.info(f"  Hybrid evaluation: {evaluated} users")
    if skipped_cold_start > 0:
        logger.info(f"  Skipped (cold start): {skipped_cold_start}")

    summary = {}
    for k in k_values:
        summary[f'K={k}'] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in results[k].items()
        }
        summary[f'K={k}']['evaluated_users'] = evaluated

    return summary


# =============================================================================
# Reporting Functions
# =============================================================================

def print_model_results(model_name: str, results: Dict[str, Dict[str, float]]):
    """Print results for a single model."""
    if not results:
        print(f"\n{model_name}: No results available")
        return

    print(f"\n{'=' * 60}")
    print(f"{model_name} Results")
    print('=' * 60)

    for k_key, metrics in results.items():
        print(f"\n  {k_key}:")
        print(f"    {'Metric':<12} {'Value':>10}")
        print(f"    {'-' * 24}")
        for metric, value in metrics.items():
            if metric != 'evaluated_users':
                print(f"    {metric:<12} {value:>10.4f}")
        print(f"    {'evaluated':<12} {metrics.get('evaluated_users', 0):>10}")


def print_comparison_table(all_results: Dict[str, Dict], k: int):
    """Print comparison table for all models at specific K."""
    print(f"\n{'=' * 85}")
    print(f"Model Comparison at K={k}")
    print('=' * 85)

    metrics = ['precision', 'recall', 'f1', 'hit_rate', 'mrr', 'ndcg', 'map']

    # Header
    header = f"{'Model':<15}"
    for metric in metrics:
        header += f"{metric:<10}"
    header += f"{'users':>8}"
    print(header)
    print('-' * 85)

    # Each model
    model_order = ['Content', 'Collaborative', 'Implicit', 'Popularity', 'Hybrid']
    for model_name in model_order:
        if model_name not in all_results:
            continue

        model_results = all_results[model_name]
        k_key = f'K={k}'

        if k_key not in model_results:
            continue

        row = f"{model_name:<15}"
        for metric in metrics:
            value = model_results[k_key].get(metric, 0)
            row += f"{value:<10.4f}"
        row += f"{model_results[k_key].get('evaluated_users', 0):>8}"
        print(row)

    print('=' * 85)


def find_best_model(all_results: Dict[str, Dict], k: int, metric: str = 'ndcg'):
    """Find the best model for a specific metric."""
    best_model = None
    best_value = -1

    k_key = f'K={k}'

    for model_name, results in all_results.items():
        if k_key in results:
            value = results[k_key].get(metric, 0)
            if value > best_value:
                best_value = value
                best_model = model_name

    return best_model, best_value


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Individual Recommendation Models")
    parser.add_argument("--sample-users", type=int, default=500, help="Number of users to sample")
    parser.add_argument("--k", type=int, nargs='+', default=[5, 10, 20], help="K values for evaluation")
    parser.add_argument("--output", type=str, default="model_evaluation_results.json", help="Output file")
    parser.add_argument("--skip-content", action="store_true", help="Skip content model evaluation")
    parser.add_argument("--skip-collaborative", action="store_true", help="Skip collaborative model evaluation")
    parser.add_argument("--skip-implicit", action="store_true", help="Skip implicit model evaluation")
    parser.add_argument("--skip-popularity", action="store_true", help="Skip popularity model evaluation")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip hybrid model evaluation")
    parser.add_argument("--rating-sample", type=int, default=1000000, help="Number of ratings to sample")
    parser.add_argument("--leave-one-out", action="store_true", help="Use Leave-One-Out evaluation")
    parser.add_argument("--relevance-threshold", type=float, default=7.0, help="Rating threshold for relevance")

    args = parser.parse_args()

    total_start = time.time()

    print("=" * 60)
    print("ANIME RECOMMENDATION MODELS - EVALUATION (FIXED)")
    print("=" * 60)
    print("\nFIXES APPLIED:")
    print("  1. Train/Test split keeps ALL interactions in train")
    print("  2. Content-Based uses multi-item user profile")
    print("  3. Implicit ALS uses correct index mapping")
    print("  4. Hybrid uses consistent ID space")
    print("=" * 60)

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n[1/3] Loading Trained Models...")
    print("-" * 40)

    model_path = MODELS_DIR / "hybrid"

    content_model = None
    collaborative_model = None
    implicit_model = None
    popularity_model = None
    hybrid_engine = None

    # Load each model
    if not args.skip_content:
        try:
            content_model = ContentBasedRecommender()
            content_model.load(model_path / "content_model.pkl")
            print("  [OK] Content-Based Model loaded")
        except Exception as e:
            print(f"  [SKIP] Content-Based Model: {e}")

    if not args.skip_collaborative:
        try:
            collaborative_model = MatrixFactorization()
            collaborative_model.load(model_path / "collaborative_model.pkl")
            print("  [OK] Collaborative Filtering Model loaded")
        except Exception as e:
            print(f"  [SKIP] Collaborative Filtering Model: {e}")

    if not args.skip_implicit:
        try:
            implicit_model = ALSImplicit()
            implicit_model.load(model_path / "implicit_model.pkl")
            print("  [OK] Implicit Feedback Model loaded")
        except Exception as e:
            print(f"  [SKIP] Implicit Feedback Model: {e}")

    if not args.skip_popularity:
        try:
            popularity_model = PopularityModel()
            popularity_model.load(model_path / "popularity_model.pkl")
            print("  [OK] Popularity Model loaded")
        except Exception as e:
            print(f"  [SKIP] Popularity Model: {e}")

    if not args.skip_hybrid:
        try:
            hybrid_engine = HybridEngine()
            hybrid_engine.load(model_path)
            print("  [OK] Hybrid Engine loaded")
        except Exception as e:
            print(f"  [SKIP] Hybrid Engine: {e}")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[2/3] Loading Evaluation Data...")
    print("-" * 40)

    try:
        ratings_df = load_ratings_for_evaluation(args.rating_sample)
        print(f"  Ratings loaded: {len(ratings_df):,} records")
    except Exception as e:
        print(f"  Error loading ratings: {e}")
        print("  Trying with smaller sample...")
        ratings_df = load_ratings_for_evaluation(500000)
        print(f"  Ratings loaded: {len(ratings_df):,} records")

    # Prepare train/test split with FIXED logic
    user_train, user_test, eval_users = prepare_evaluation_data(
        ratings_df,
        relevance_threshold=args.relevance_threshold,
        leave_one_out=args.leave_one_out
    )

    # Sample users
    if len(eval_users) > args.sample_users:
        eval_users = list(np.random.choice(eval_users, args.sample_users, replace=False))

    print(f"  Evaluation users: {len(eval_users)}")
    print(f"  K values: {args.k}")
    print(f"  Leave-One-Out: {args.leave_one_out}")

    # =========================================================================
    # Evaluate Models
    # =========================================================================
    print("\n[3/3] Evaluating Models...")
    print("-" * 40)

    all_results = {}

    if content_model is not None:
        start = time.time()
        all_results['Content'] = evaluate_content_model(
            content_model, user_train, user_test, eval_users, args.k, args.sample_users
        )
        print(f"  Content evaluation time: {time.time() - start:.1f}s")

    if collaborative_model is not None:
        start = time.time()
        all_results['Collaborative'] = evaluate_collaborative_model(
            collaborative_model, user_train, user_test, eval_users, args.k, args.sample_users
        )
        print(f"  Collaborative evaluation time: {time.time() - start:.1f}s")

    if implicit_model is not None:
        start = time.time()
        # Implicit uses its own data preparation (no rating threshold)
        all_results['Implicit'] = evaluate_implicit_model(
            implicit_model, ratings_df, args.k, args.sample_users
        )
        print(f"  Implicit evaluation time: {time.time() - start:.1f}s")

    if popularity_model is not None:
        start = time.time()
        all_results['Popularity'] = evaluate_popularity_model(
            popularity_model, user_train, user_test, eval_users, args.k, args.sample_users
        )
        print(f"  Popularity evaluation time: {time.time() - start:.1f}s")

    if hybrid_engine is not None:
        start = time.time()
        all_results['Hybrid'] = evaluate_hybrid_model(
            hybrid_engine, user_train, user_test, eval_users, args.k, args.sample_users
        )
        print(f"  Hybrid evaluation time: {time.time() - start:.1f}s")

    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for model_name, results in all_results.items():
        print_model_results(model_name, results)

    for k in args.k:
        print_comparison_table(all_results, k)

    # Find best models
    print("\n" + "=" * 60)
    print("BEST MODEL PER METRIC")
    print("=" * 60)

    for k in args.k:
        print(f"\nAt K={k}:")
        for metric in ['precision', 'recall', 'f1', 'hit_rate', 'mrr', 'ndcg', 'map']:
            best_model, best_value = find_best_model(all_results, k, metric)
            if best_model:
                print(f"  {metric:<12}: {best_model} ({best_value:.4f})")

    # =========================================================================
    # Save Results
    # =========================================================================
    total_time = time.time() - total_start

    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'sample_users': args.sample_users,
            'k_values': args.k,
            'leave_one_out': args.leave_one_out,
            'relevance_threshold': args.relevance_threshold
        },
        'fixes_applied': [
            'Train/Test split keeps ALL interactions in train',
            'Content-Based uses multi-item user profile',
            'Implicit ALS uses correct index mapping',
            'Hybrid uses consistent ID space'
        ],
        'results': all_results,
        'total_time': f"{total_time:.2f}s"
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Evaluation Complete! Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = main()
