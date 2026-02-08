"""
Training script for the Hybrid Anime Recommendation System.

This script trains all recommendation models:
1. Content-Based (TF-IDF + SBERT)
2. Collaborative Filtering (Item-Based CF + Matrix Factorization)
3. Implicit Feedback (ALS)
4. Popularity-Based

Usage:
    python train.py [--skip-sbert] [--skip-collaborative] [--skip-implicit] [--sample-size SIZE]
"""
import argparse
import logging
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, data_config, model_config
from preprocessing import DataLoader, MatrixBuilder
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


def main():
    parser = argparse.ArgumentParser(description="Train Anime Recommendation Models")
    parser.add_argument("--skip-sbert", action="store_true", help="Skip SBERT embeddings")
    parser.add_argument("--skip-collaborative", action="store_true", help="Skip collaborative filtering")
    parser.add_argument("--skip-implicit", action="store_true", help="Skip implicit feedback model")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for ratings")
    parser.add_argument("--cf-method", type=str, default="svd", choices=["svd", "als"], help="CF method")

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

    # Load anime data
    anime_df = loader.get_merged_anime_data()
    logger.info(f"Anime data: {len(anime_df)} records")

    # Load ratings
    ratings_df = loader.load_ratings(sample=True)
    logger.info(f"Ratings: {len(ratings_df):,} records")

    # Load animelist
    animelist_df = loader.load_animelist(sample=True)
    logger.info(f"Animelist: {len(animelist_df):,} records")

    # ===== BUILD MATRICES =====
    logger.info("=" * 50)
    logger.info("Building Matrices")
    logger.info("=" * 50)

    matrix_builder = MatrixBuilder()
    matrix_builder.build_rating_matrix(ratings_df)
    matrix_builder.build_implicit_matrix(animelist_df)

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
    popularity_model = train_popularity_model(anime_df, ratings_df, animelist_df)

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
