"""
Training script for the Hybrid Anime Recommendation System.

This script trains all recommendation models:
1. Content-Based (TF-IDF + SBERT)
2. Collaborative Filtering (Item-Based CF + Matrix Factorization)
3. Implicit Feedback (ALS)
4. Popularity-Based

Usage:
    python train.py [--skip-sbert] [--skip-collaborative] [--skip-implicit] [--sample-size SIZE]

The ratings split is created once before training and then persisted so
evaluation can reuse the exact same train/test boundary.
"""
import argparse
import logging
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, SPLITS_DIR, data_config, eval_config, model_config
from preprocessing import (
    DataLoader,
    MatrixBuilder,
    create_ratings_user_split,
    filter_holdout_interactions,
    load_ratings_user_split,
    save_ratings_user_split,
    split_to_ratings_df,
)
from models.content import ContentBasedRecommender
from models.collaborative import ItemBasedCF, MatrixFactorization
from models.implicit import ALSImplicit
from models.popularity import PopularityModel
from models.hybrid import HybridEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_content_model(
    anime_df,
    use_sbert: bool = True
) -> ContentBasedRecommender:
    """Train content-based model."""
    logger.info("=" * 50)
    logger.info("Training Content-Based Model")
    logger.info("=" * 50)

    start_time = time.time()

    model = ContentBasedRecommender()
    model.fit(anime_df, use_tfidf=True, use_sbert=use_sbert, use_faiss=True)

    elapsed = time.time() - start_time
    logger.info(f"Content-Based Model trained in {elapsed:.2f} seconds")

    return model


def train_collaborative_model(
    user_item_matrix,
    anime_to_idx,
    idx_to_anime,
    user_to_idx,
    idx_to_user,
    method: str = "svd"
) -> MatrixFactorization:
    """Train collaborative filtering model."""
    logger.info("=" * 50)
    logger.info(f"Training Collaborative Filtering Model ({method.upper()})")
    logger.info("=" * 50)

    start_time = time.time()

    model = MatrixFactorization(
        n_factors=model_config.svd_factors,
        n_epochs=model_config.svd_epochs,
        method=method
    )
    model.fit(
        user_item_matrix,
        anime_to_idx,
        idx_to_anime,
        user_to_idx,
        idx_to_user
    )

    elapsed = time.time() - start_time
    logger.info(f"Collaborative Model trained in {elapsed:.2f} seconds")

    return model


def train_item_based_cf(
    user_item_matrix,
    anime_to_idx,
    idx_to_anime,
    user_to_idx,
    idx_to_user
) -> ItemBasedCF:
    """Train item-based collaborative filtering model."""
    logger.info("=" * 50)
    logger.info("Training Item-Based CF Model")
    logger.info("=" * 50)

    start_time = time.time()

    model = ItemBasedCF(k_neighbors=50)
    model.fit(
        user_item_matrix,
        anime_to_idx,
        idx_to_anime,
        user_to_idx,
        idx_to_user,
        compute_full_similarity=False  # Use FAISS instead
    )

    elapsed = time.time() - start_time
    logger.info(f"Item-Based CF trained in {elapsed:.2f} seconds")

    return model


def train_implicit_model(
    implicit_matrix,
    anime_to_idx,
    idx_to_anime,
    user_to_idx,
    idx_to_user
) -> ALSImplicit:
    """Train implicit feedback model."""
    logger.info("=" * 50)
    logger.info("Training Implicit Feedback Model (ALS)")
    logger.info("=" * 50)

    start_time = time.time()

    model = ALSImplicit(
        n_factors=model_config.implicit_factors,
        n_iterations=model_config.implicit_iterations,
        regularization=model_config.implicit_regularization
    )
    model.fit(
        implicit_matrix,
        anime_to_idx,
        idx_to_anime,
        user_to_idx,
        idx_to_user
    )

    elapsed = time.time() - start_time
    logger.info(f"Implicit Model trained in {elapsed:.2f} seconds")

    return model


def train_popularity_model(
    anime_df,
    ratings_df,
    animelist_df
) -> PopularityModel:
    """Train popularity model."""
    logger.info("=" * 50)
    logger.info("Training Popularity Model")
    logger.info("=" * 50)

    start_time = time.time()

    model = PopularityModel()
    model.fit(anime_df, ratings_df, animelist_df)

    elapsed = time.time() - start_time
    logger.info(f"Popularity Model trained in {elapsed:.2f} seconds")

    return model


def load_or_create_training_split(
    loader: DataLoader,
    split_path: Path,
    force_resplit: bool = False,
    test_ratio: float = None,
    relevance_threshold: float = 7.0,
    min_train_items: int = 10,
    min_test_items: int = 3,
    leave_one_out: bool = False,
):
    """
    Load an existing split artifact or create it once before training.

    Returns:
        Tuple of (split_artifact, train_ratings_df)
    """
    if split_path.exists() and not force_resplit:
        split_artifact = load_ratings_user_split(split_path)
        logger.info("Reusing persisted split: %s", split_path)
    else:
        ratings_df = loader.load_ratings(sample=True)
        split_artifact = create_ratings_user_split(
            ratings_df,
            min_train_items=min_train_items,
            min_test_items=min_test_items,
            test_ratio=test_ratio if test_ratio is not None else eval_config.test_size,
            relevance_threshold=relevance_threshold,
            leave_one_out=leave_one_out,
            random_state=eval_config.random_state,
        )
        save_ratings_user_split(split_artifact, split_path)

    train_ratings_df = split_to_ratings_df(split_artifact.user_train)
    logger.info(
        "Training will use %s users and %s ratings from the persisted train split",
        len(split_artifact.user_train),
        len(train_ratings_df),
    )

    return split_artifact, train_ratings_df


def main():
    parser = argparse.ArgumentParser(description="Train Anime Recommendation Models")
    parser.add_argument("--skip-sbert", action="store_true", help="Skip SBERT embeddings")
    parser.add_argument("--skip-collaborative", action="store_true", help="Skip collaborative filtering")
    parser.add_argument("--skip-implicit", action="store_true", help="Skip implicit feedback model")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for ratings")
    parser.add_argument("--cf-method", type=str, default="svd", choices=["svd", "als"], help="CF method")
    parser.add_argument(
        "--split-path",
        type=str,
        default=str(SPLITS_DIR / "ratings_user_split.pkl"),
        help="Path to the persisted train/eval split artifact",
    )
    parser.add_argument("--force-resplit", action="store_true", help="Regenerate the split before training")
    parser.add_argument("--test-ratio", type=float, default=eval_config.test_size, help="Held-out ratio for the split")
    parser.add_argument("--relevance-threshold", type=float, default=7.0, help="Minimum rating to be eligible for user_test")
    parser.add_argument("--min-train-items", type=int, default=10, help="Minimum train items per user after the split")
    parser.add_argument("--min-test-items", type=int, default=3, help="Minimum test items per user")
    parser.add_argument("--leave-one-out", action="store_true", help="Create a leave-one-out split")

    args = parser.parse_args()

    total_start = time.time()

    # Update config if sample size provided
    if args.sample_size:
        data_config.rating_sample_size = args.sample_size
        data_config.animelist_sample_size = args.sample_size

    # ===== LOAD DATA =====
    logger.info("=" * 50)
    logger.info("Loading Data")
    logger.info("=" * 50)

    loader = DataLoader()
    split_path = Path(args.split_path)

    # Load anime data
    anime_df = loader.get_merged_anime_data()
    logger.info(f"Anime data: {len(anime_df)} records")

    # Create the split once before any model sees interaction data.
    split_artifact, train_ratings_df = load_or_create_training_split(
        loader,
        split_path=split_path,
        force_resplit=args.force_resplit,
        test_ratio=args.test_ratio,
        relevance_threshold=args.relevance_threshold,
        min_train_items=args.min_train_items,
        min_test_items=args.min_test_items,
        leave_one_out=args.leave_one_out,
    )
    logger.info(
        "Persisted split ready at %s (%s eval users, %s held-out items)",
        split_path,
        split_artifact.metadata.get("eval_users", len(split_artifact.eval_users)),
        split_artifact.metadata.get("test_interactions", 0),
    )

    # Load animelist
    animelist_df = loader.load_animelist(sample=True)
    logger.info(f"Animelist: {len(animelist_df):,} records")
    train_animelist_df = filter_holdout_interactions(animelist_df, split_artifact.user_test)
    logger.info(f"Animelist after removing held-out pairs: {len(train_animelist_df):,} records")

    # ===== BUILD MATRICES =====
    logger.info("=" * 50)
    logger.info("Building Matrices")
    logger.info("=" * 50)

    matrix_builder = MatrixBuilder()
    matrix_builder.build_rating_matrix(train_ratings_df)
    matrix_builder.build_implicit_matrix(train_animelist_df)

    # ===== TRAIN MODELS =====

    # 1. Content-Based
    content_model = train_content_model(anime_df, use_sbert=not args.skip_sbert)

    # 2. Collaborative Filtering
    if not args.skip_collaborative:
        collaborative_model = train_collaborative_model(
            matrix_builder.user_item_matrix,
            matrix_builder.anime_to_idx,
            matrix_builder.idx_to_anime,
            matrix_builder.user_to_idx,
            matrix_builder.idx_to_user,
            method=args.cf_method
        )
    else:
        collaborative_model = None
        logger.info("Skipping Collaborative Filtering")

    # 3. Implicit Feedback
    if not args.skip_implicit:
        implicit_model = train_implicit_model(
            matrix_builder.implicit_matrix,
            matrix_builder.anime_to_idx,
            matrix_builder.idx_to_anime,
            matrix_builder.user_to_idx,
            matrix_builder.idx_to_user
        )
    else:
        implicit_model = None
        logger.info("Skipping Implicit Feedback Model")

    # 4. Popularity
    popularity_model = train_popularity_model(anime_df, train_ratings_df, train_animelist_df)

    # ===== CREATE HYBRID ENGINE =====
    logger.info("=" * 50)
    logger.info("Creating Hybrid Engine")
    logger.info("=" * 50)

    hybrid_engine = HybridEngine(
        content_model=content_model,
        collaborative_model=collaborative_model,
        implicit_model=implicit_model,
        popularity_model=popularity_model
    )
    hybrid_engine.set_anime_info(anime_df)

    # ===== SAVE MODELS =====
    logger.info("=" * 50)
    logger.info("Saving Models")
    logger.info("=" * 50)

    save_dir = MODELS_DIR / "hybrid"
    hybrid_engine.save(save_dir)

    # Also save matrix builder for later use
    matrix_builder.save(MODELS_DIR / "matrices")

    total_elapsed = time.time() - total_start
    logger.info("=" * 50)
    logger.info(f"Training Complete! Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    logger.info(f"Models saved to: {save_dir}")
    logger.info("=" * 50)

    # ===== QUICK TEST =====
    logger.info("\n" + "=" * 50)
    logger.info("Quick Test")
    logger.info("=" * 50)

    # Test anime similarity
    test_anime = "Naruto"
    print(f"\nRecommendations similar to '{test_anime}':")
    recs = hybrid_engine.recommend_similar_anime(test_anime, top_k=5)
    for rec in recs:
        print(f"  - {rec['name']} (Score: {rec['score']}, Hybrid: {rec['hybrid_score']:.4f})")

    # Test user recommendation
    test_user = list(matrix_builder.user_to_idx.keys())[0]
    test_user_ratings = matrix_builder.get_user_ratings(test_user)
    hybrid_engine.set_user_history(
        test_user,
        ratings=test_user_ratings,
        watched=set(test_user_ratings.keys()),
    )
    print(f"\nRecommendations for user {test_user}:")
    user_recs = hybrid_engine.recommend_for_user(test_user, top_k=5)
    for rec in user_recs:
        print(f"  - {rec['name']} (Score: {rec['score']}, Hybrid: {rec['hybrid_score']:.4f})")

    # Test popular
    print("\nTop Rated Anime:")
    popular = popularity_model.get_top_rated(5)
    for p in popular:
        print(f"  - {p['name']} (Score: {p['score']})")

    return hybrid_engine


if __name__ == "__main__":
    engine = main()
