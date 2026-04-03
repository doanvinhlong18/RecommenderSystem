"""
Training script for the Hybrid Anime Recommendation System.

GPU-accelerated version with automatic device detection.

This script trains all recommendation models:
1. Content-Based (SBERT + Structured Features) - GPU accelerated
2. Collaborative Filtering (Item-Based CF + Matrix Factorization) - GPU with PyTorch
3. Implicit Feedback (ALS) - GPU with implicit library
4. Popularity-Based
5. LearnedHybridEngine (meta-model LightGBM) — chạy sau khi 1-4 xong

Tất cả models được lưu vào 1 thư mục: saved_models/learned_hybrid/
(không còn saved_models/hybrid/ riêng biệt)

Usage:
    python train.py [--skip-sbert] [--skip-collaborative] [--skip-implicit]
    python train.py --force-cpu
    python train.py --skip-learned-hybrid  # bỏ qua bước train meta-model
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, SPLITS_DIR, data_config, eval_config, model_config

from preprocessing import (
    DataLoader,
    MatrixBuilder,
    TextProcessor,
    ContentFeatureBuilder,
    create_ratings_user_split,
    filter_holdout_interactions,
    load_ratings_user_split,
    save_ratings_user_split,
    split_to_ratings_df,
)
from device_config import init_device, get_device, log_gpu_memory, clear_gpu_cache
from models.content import ContentBasedRecommender
from models.collaborative import ItemBasedCF, MatrixFactorization
from models.implicit import ALSImplicit
from models.popularity import PopularityModel

# LearnedHybridEngine là engine duy nhất — không import HybridEngine nữa
from models.hybrid.learned_hybrid import LearnedHybridEngine, train_learned_hybrid

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_content_model(
    anime_df, use_sbert: bool = True, device: str = None
) -> ContentBasedRecommender:
    """Train content-based model: SBERT + structured features → fused embedding."""
    logger.info("=" * 50)
    logger.info("Training Content-Based Model")
    logger.info("=" * 50)

    start_time = time.time()
    device = device or get_device()

    processor = TextProcessor()

    if use_sbert:
        logger.info("Step 1: Generating SBERT embeddings...")
        combined_text = processor.combine_text_features(anime_df)
        sbert_emb = processor.fit_sbert(
            combined_text, anime_ids=anime_df["MAL_ID"].tolist()
        )
        logger.info(f"  SBERT embeddings shape: {sbert_emb.shape}")
    else:
        logger.warning("Step 1: Skipping SBERT — using zero embeddings")
        sbert_emb = np.zeros((len(anime_df), 384), dtype=np.float32)

    logger.info("Step 2: Building structured features...")
    builder = ContentFeatureBuilder()
    builder.fit(anime_df)

    if use_sbert:
        w_struct, w_text, w_tmp = 0.3, 0.55, 0.15
        logger.info(
            f"  Fusing SBERT ({w_text*100:.0f}%) "
            f"+ Structured ({w_struct*100:.0f}%) "
            f"+ Member ({w_tmp*100:.0f}%)..."
        )
        final_emb = builder.transform(
            anime_df, sbert_emb, w_struct=w_struct, w_text=w_text, w_tmp=w_tmp
        )
    else:
        logger.info("  Using structured features only...")
        final_emb = builder.transform(
            anime_df, sbert_emb, w_struct=0.5, w_text=0.0, w_tmp=0.5
        )

    logger.info(f"  Final embedding shape: {final_emb.shape}")

    logger.info("Step 3: Fitting ContentBasedRecommender...")
    model = ContentBasedRecommender()
    model.fit(anime_df, final_emb)

    elapsed = time.time() - start_time
    logger.info(f"✓ Content-Based Model trained in {elapsed:.2f}s")
    log_gpu_memory("After Content-Based training: ")

    return model


def train_collaborative_model(
    user_item_matrix,
    anime_to_idx,
    idx_to_anime,
    user_to_idx,
    idx_to_user,
    method: str = "bpr",
    implicit_matrix=None,
    implicit_model=None,
    positive_rating_threshold: float = None,
) -> MatrixFactorization:
    """Train collaborative filtering model with GPU support."""
    logger.info("=" * 50)
    logger.info(f"Training Collaborative Filtering Model ({method.upper()})")
    logger.info("=" * 50)

    start_time = time.time()

    if method == "bpr":
        model = MatrixFactorization(
            n_factors=model_config.bpr_factors,
            n_epochs=model_config.bpr_iterations,
            learning_rate=model_config.bpr_learning_rate,
            regularization=model_config.bpr_regularization,
            method=method,
            rating_positive_threshold=positive_rating_threshold,
            verify_negative_samples=model_config.bpr_verify_negative_samples,
            use_implicit_signal=model_config.bpr_use_implicit_signal,
            warm_start_from_als=model_config.bpr_warm_start_from_als,
        )
    else:
        model = MatrixFactorization(
            n_factors=model_config.svd_factors,
            n_epochs=model_config.svd_epochs,
            learning_rate=model_config.svd_lr,
            regularization=model_config.svd_reg,
            method=method,
        )

    model.fit(
        user_item_matrix,
        anime_to_idx,
        idx_to_anime,
        user_to_idx,
        idx_to_user,
        implicit_matrix=implicit_matrix,
        implicit_model=implicit_model,
    )

    elapsed = time.time() - start_time
    logger.info(f"Collaborative Model trained in {elapsed:.2f} seconds")
    log_gpu_memory("After Collaborative training: ")

    return model


def train_implicit_model(
    implicit_matrix,
    anime_to_idx,
    idx_to_anime,
    user_to_idx,
    idx_to_user,
    device: str = None,
) -> ALSImplicit:
    """Train implicit feedback model with GPU support."""
    logger.info("=" * 50)
    logger.info("Training Implicit Feedback Model (ALS)")
    logger.info("=" * 50)

    start_time = time.time()
    device = device or get_device()

    model = ALSImplicit(
        n_factors=model_config.implicit_factors,
        n_iterations=model_config.implicit_iterations,
        regularization=model_config.implicit_regularization,
        device=device,
    )
    model.fit(implicit_matrix, anime_to_idx, idx_to_anime, user_to_idx, idx_to_user)

    elapsed = time.time() - start_time
    logger.info(f"Implicit Model trained in {elapsed:.2f} seconds")
    log_gpu_memory("After Implicit training: ")

    return model


def train_popularity_model(anime_df, ratings_df, animelist_df) -> PopularityModel:
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
    parser.add_argument(
        "--skip-sbert", action="store_true", help="Skip SBERT embeddings"
    )
    parser.add_argument(
        "--skip-collaborative", action="store_true", help="Skip collaborative filtering"
    )
    parser.add_argument(
        "--skip-implicit", action="store_true", help="Skip implicit feedback model"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Sample size for ratings"
    )
    parser.add_argument(
        "--cf-method",
        type=str,
        default="bpr",
        choices=["bpr", "svd", "als"],
        help="CF method",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--split-path",
        type=str,
        default=str(SPLITS_DIR / "ratings_user_split.pkl"),
        help="Path to the persisted train/eval split artifact",
    )
    parser.add_argument(
        "--force-resplit", action="store_true", help="Regenerate the split"
    )
    parser.add_argument("--test-ratio", type=float, default=eval_config.test_size)
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=7.0,
        help="Minimum rating to be eligible for user_test",
    )
    parser.add_argument("--min-train-items", type=int, default=10)
    parser.add_argument("--min-test-items", type=int, default=3)
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument(
        "--skip-learned-hybrid",
        action="store_true",
        help="Bỏ qua bước train meta-model (tiết kiệm thời gian khi test)",
    )

    args = parser.parse_args()
    total_start = time.time()

    # ===== INITIALIZE DEVICE =====
    logger.info("=" * 50)
    logger.info("Initializing Device")
    logger.info("=" * 50)
    device = init_device(force_cpu=args.force_cpu)
    logger.info(
        "🚀 GPU acceleration enabled!"
        if device == "cuda"
        else "Running in CPU fallback mode"
    )

    if args.sample_size:
        data_config.rating_sample_size = args.sample_size
        data_config.animelist_sample_size = args.sample_size

    # ===== LOAD DATA =====
    logger.info("=" * 50)
    logger.info("Loading Data")
    logger.info("=" * 50)

    loader = DataLoader()
    split_path = Path(args.split_path)

    anime_df = loader.get_merged_anime_data()
    logger.info(f"Anime data: {len(anime_df)} records")

    content_anime_df = loader.get_content_base_dataframe()
    logger.info(f"Content anime data: {len(content_anime_df)} records")

    # Load ratings một lần duy nhất — dùng lại cho cả train và train_learned_hybrid
    ratings_df = loader.load_ratings(sample=True)
    logger.info(f"Ratings loaded: {len(ratings_df):,} records")

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

    animelist_df = loader.load_animelist(sample=True)
    logger.info(f"Animelist: {len(animelist_df):,} records")
    train_animelist_df = filter_holdout_interactions(
        animelist_df, split_artifact.user_test
    )
    logger.info(
        f"Animelist after removing held-out pairs: {len(train_animelist_df):,} records"
    )

    # ===== BUILD MATRICES =====
    logger.info("=" * 50)
    logger.info("Building Matrices")
    logger.info("=" * 50)

    matrix_builder = MatrixBuilder()
    matrix_builder.build_rating_matrix(train_ratings_df)
    matrix_builder.build_implicit_matrix(train_animelist_df)

    # ===== TRAIN MODELS =====

    # 1. Content-Based
    content_model = train_content_model(
        content_anime_df, use_sbert=not args.skip_sbert, device=device
    )

    # 2. Implicit Feedback
    if not args.skip_implicit:
        implicit_model = train_implicit_model(
            matrix_builder.implicit_matrix,
            matrix_builder.anime_to_idx,
            matrix_builder.idx_to_anime,
            matrix_builder.user_to_idx,
            matrix_builder.idx_to_user,
            device=device,
        )
    else:
        implicit_model = None
        logger.info("Skipping Implicit Feedback Model")

    # 3. Collaborative Filtering
    if not args.skip_collaborative:
        collaborative_model = train_collaborative_model(
            matrix_builder.user_item_matrix,
            matrix_builder.anime_to_idx,
            matrix_builder.idx_to_anime,
            matrix_builder.user_to_idx,
            matrix_builder.idx_to_user,
            method=args.cf_method,
            implicit_matrix=matrix_builder.implicit_matrix,
            implicit_model=implicit_model,
            positive_rating_threshold=args.relevance_threshold,
        )
    else:
        collaborative_model = None
        logger.info("Skipping Collaborative Filtering")

    # 4. Popularity
    popularity_model = train_popularity_model(
        anime_df, train_ratings_df, train_animelist_df
    )

    # ===== TRAIN LEARNED HYBRID ENGINE =====
    # Đây là engine duy nhất — lưu tất cả models vào saved_models/learned_hybrid/
    logger.info("=" * 50)
    logger.info("Training Learned Hybrid Engine (meta-model)")
    logger.info("=" * 50)

    save_dir = MODELS_DIR / "learned_hybrid"

    if not args.skip_learned_hybrid:
        test_ratings_df = split_to_ratings_df(split_artifact.user_test)

        learned_engine = train_learned_hybrid(
            content_model=content_model,
            collaborative_model=collaborative_model,
            implicit_model=implicit_model,
            popularity_model=popularity_model,
            anime_df=anime_df,
            train_ratings_df=train_ratings_df,
            test_ratings_df=test_ratings_df,
            all_ratings_df=ratings_df,  # dùng lại ratings đã load — không load lại
            save_dir=save_dir,
            min_ratings_to_train=20,
            top_users=3000,
            relevance_threshold=args.relevance_threshold,
        )

        fi = learned_engine.get_feature_importance()
        if fi:
            logger.info("Feature importance (meta-model):")
            for name, val in sorted(fi.items(), key=lambda x: -x[1]):
                logger.info(f"  {name:<30s} {val:.4f}  {'█' * int(val * 40)}")
    else:
        # Skip meta-model: chỉ save sub-models với fallback weights
        logger.info("Skipping meta-model training (--skip-learned-hybrid)")
        logger.info("Saving sub-models only (fallback weighted sum sẽ được dùng)")
        learned_engine = LearnedHybridEngine(
            content_model=content_model,
            collaborative_model=collaborative_model,
            implicit_model=implicit_model,
            popularity_model=popularity_model,
        )
        learned_engine.set_anime_info(anime_df)
        learned_engine.save(save_dir)

    # Save matrix builder cho các script khác dùng
    matrix_builder.save(MODELS_DIR / "matrices")

    total_elapsed = time.time() - total_start
    logger.info("=" * 50)
    logger.info(
        f"Training Complete! Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)"
    )
    logger.info(f"Models saved to: {save_dir}")
    logger.info("=" * 50)

    # ===== QUICK TEST =====
    logger.info("\n" + "=" * 50)
    logger.info("Quick Test")
    logger.info("=" * 50)

    test_anime = "Naruto"
    print(f"\nRecommendations similar to '{test_anime}':")
    recs = learned_engine.recommend_similar_anime(test_anime, top_k=5)
    for rec in recs:
        print(
            f"  - {rec['name']} (Score: {rec['score']}, Hybrid: {rec['hybrid_score']:.4f})"
        )

    test_user = list(matrix_builder.user_to_idx.keys())[0]
    test_user_ratings = matrix_builder.get_user_ratings(test_user)
    learned_engine.set_user_history(
        test_user,
        ratings=test_user_ratings,
        watched=set(test_user_ratings.keys()),
    )
    print(f"\nRecommendations for user {test_user}:")
    user_recs = learned_engine.recommend_for_user(test_user, top_k=5)
    for rec in user_recs:
        print(
            f"  - {rec['name']} (Hybrid: {rec['hybrid_score']:.4f}, strategy: {rec.get('strategy','?')})"
        )

    print("\nTop Rated Anime:")
    popular = popularity_model.get_top_rated(5)
    for p in popular:
        print(f"  - {p['name']} (Score: {p['score']})")

    return learned_engine


if __name__ == "__main__":
    engine = main()
