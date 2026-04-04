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
    create_ratings_disk_split,
    create_ratings_user_split,
    extract_holdout_ratings_df,
    filter_animelist_to_disk,
    filter_holdout_interactions,
    get_ratings_disk_split,
    load_ratings_disk_split,
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
from models.hybrid.learned_hybrid import (
    LearnedHybridEngine,
    train_learned_hybrid,
    train_learned_hybrid_from_csv,
)

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
    device: str = None,
) -> MatrixFactorization:
    """Train collaborative filtering model with GPU support."""
    logger.info("=" * 50)
    logger.info(f"Training Collaborative Filtering Model ({method.upper()})")
    logger.info("=" * 50)

    start_time = time.time()
    device = device or get_device()

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
            device=device,
        )
    else:
        model = MatrixFactorization(
            n_factors=model_config.svd_factors,
            n_epochs=model_config.svd_epochs,
            learning_rate=model_config.svd_lr,
            regularization=model_config.svd_reg,
            method=method,
            device=device,
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


def train_popularity_model_from_csv(
    anime_df,
    ratings_csv_path: Path,
    animelist_csv_path: Path,
    chunk_size: int = 500_000,
) -> PopularityModel:
    """Train popularity model from streamed CSV inputs."""
    logger.info("=" * 50)
    logger.info("Training Popularity Model")
    logger.info("=" * 50)

    start_time = time.time()
    model = PopularityModel()
    model.fit_from_csv(
        anime_df,
        ratings_csv_path=ratings_csv_path,
        animelist_csv_path=animelist_csv_path,
        chunk_size=chunk_size,
    )

    elapsed = time.time() - start_time
    logger.info(f"Popularity Model trained in {elapsed:.2f} seconds")

    return model


def load_or_create_training_split(
    loader: DataLoader,
    split_path: Path,
    ratings_df=None,
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
        ratings_df = ratings_df if ratings_df is not None else loader.load_ratings(sample=False)
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


def load_or_create_training_disk_split(
    loader: DataLoader,
    split_path: Path,
    force_resplit: bool = False,
    test_ratio: float = None,
    relevance_threshold: float = 7.0,
    min_train_items: int = 10,
    min_test_items: int = 3,
    leave_one_out: bool = False,
    chunk_size: int = 500_000,
):
    """Create or reuse a streaming-friendly split stored as CSV artifacts on disk."""
    expected_split = get_ratings_disk_split(split_path)
    split_exists = (
        expected_split.manifest_path.exists()
        and expected_split.train_ratings_path.exists()
        and expected_split.test_ratings_path.exists()
        and expected_split.train_animelist_path.exists()
    )

    if split_exists and not force_resplit:
        disk_split = load_ratings_disk_split(split_path)
        logger.info("Reusing persisted on-disk split: %s", split_path)
        return disk_split

    ratings_csv_path = loader.dataset_path / data_config.rating_file
    animelist_csv_path = loader.dataset_path / data_config.animelist_file

    logger.info("Creating persisted on-disk split at %s", split_path)
    disk_split = create_ratings_disk_split(
        ratings_csv_path=ratings_csv_path,
        split_path=split_path,
        min_train_items=min_train_items,
        min_test_items=min_test_items,
        test_ratio=test_ratio if test_ratio is not None else eval_config.test_size,
        relevance_threshold=relevance_threshold,
        leave_one_out=leave_one_out,
        random_state=eval_config.random_state,
        chunk_size=chunk_size,
    )

    filter_animelist_to_disk(
        animelist_csv_path=animelist_csv_path,
        test_ratings_path=disk_split.test_ratings_path,
        output_csv_path=disk_split.train_animelist_path,
        chunk_size=chunk_size,
    )
    return load_ratings_disk_split(split_path)


def save_training_checkpoint(
    save_dir: Path,
    anime_df,
    content_model=None,
    collaborative_model=None,
    implicit_model=None,
    popularity_model=None,
    learned_engine=None,
    reason: str = "",
):
    """Persist the currently available models so training can resume later."""
    if reason:
        logger.info("Saving checkpoint after %s...", reason)
    else:
        logger.info("Saving checkpoint...")

    engine = learned_engine
    if engine is None:
        engine = LearnedHybridEngine(
            content_model=content_model,
            collaborative_model=collaborative_model,
            implicit_model=implicit_model,
            popularity_model=popularity_model,
        )
        if anime_df is not None:
            engine.set_anime_info(anime_df)
    elif anime_df is not None and not getattr(engine, "_anime_info", None):
        engine.set_anime_info(anime_df)

    engine.save(save_dir)
    logger.info("Checkpoint saved to %s", save_dir)
    return engine


def load_existing_models_for_hybrid(save_dir: Path) -> LearnedHybridEngine:
    """Load previously trained base models for a hybrid-only stage."""
    if not save_dir.exists():
        raise FileNotFoundError(
            f"{save_dir} not found. Train the base models first before running hybrid-only."
        )

    engine = LearnedHybridEngine()
    engine.load(save_dir)
    if not any(
        [
            engine.content_model is not None,
            engine.collaborative_model is not None,
            engine.implicit_model is not None,
            engine.popularity_model is not None,
        ]
    ):
        raise ValueError(
            f"No base models were found in {save_dir}. Train the base stage first."
        )
    return engine


def run_quick_test(
    learned_engine: LearnedHybridEngine,
    matrix_builder: MatrixBuilder,
    popularity_model: PopularityModel,
):
    """Small post-train sanity check."""
    if learned_engine is None or matrix_builder is None or popularity_model is None:
        logger.info("Skipping quick test because the required artifacts are not available")
        return
    if not matrix_builder.user_to_idx:
        logger.info("Skipping quick test because matrix mappings are empty")
        return

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


def main():
    default_split_path = str(SPLITS_DIR / "ratings_user_split.pkl")
    default_full_split_path = SPLITS_DIR / "full_train_split.json"

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
        "--implicit-chunk-size",
        type=int,
        default=500_000,
        help="Chunk size when building the implicit matrix",
    )
    parser.add_argument(
        "--stream-chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for CSV streaming and on-disk split generation",
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
        default=default_split_path,
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
        "--learned-hybrid-min-ratings",
        type=int,
        default=20,
        help="Minimum ratings per user to include when training the learned hybrid",
    )
    parser.add_argument(
        "--learned-hybrid-top-users",
        type=int,
        default=3000,
        help="Maximum number of active users used to train the learned hybrid",
    )
    parser.add_argument(
        "--skip-learned-hybrid",
        action="store_true",
        help="Bỏ qua bước train meta-model (tiết kiệm thời gian khi test)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "base", "hybrid"],
        help="Train all stages, only base models, or only the learned hybrid stage",
    )

    args = parser.parse_args()
    total_start = time.time()

    if args.stage == "hybrid" and args.skip_learned_hybrid:
        raise ValueError("--stage hybrid cannot be used together with --skip-learned-hybrid")

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

    use_sample = args.sample_size is not None
    if use_sample:
        data_config.rating_sample_size = args.sample_size
        data_config.animelist_sample_size = args.sample_size
        logger.info("Sample mode enabled: %s rows", args.sample_size)
    else:
        data_config.rating_sample_size = None
        data_config.animelist_sample_size = None
        logger.info("Full-data mode enabled")

    train_base_stage = args.stage in {"all", "base"}
    train_hybrid_stage = args.stage in {"all", "hybrid"} and not args.skip_learned_hybrid
    save_dir = MODELS_DIR / "learned_hybrid"
    matrix_dir = MODELS_DIR / "matrices"

    # ===== LOAD DATA =====
    logger.info("=" * 50)
    logger.info("Loading Data")
    logger.info("=" * 50)

    loader = DataLoader()
    split_path = Path(args.split_path)
    if not use_sample and args.split_path == default_split_path:
        split_path = default_full_split_path
        logger.info(
            "Using full-data split artifact path: %s",
            split_path,
        )
    if args.stage == "hybrid" and args.force_resplit:
        raise ValueError(
            "Hybrid-only stage cannot regenerate the split. Reuse the same split from the base-model run."
        )

    anime_df = loader.get_merged_anime_data()
    logger.info(f"Anime data: {len(anime_df)} records")

    content_anime_df = loader.get_content_base_dataframe()
    logger.info(f"Content anime data: {len(content_anime_df)} records")

    ratings_df = None
    train_ratings_df = None
    train_animelist_df = None
    split_artifact = None
    disk_split = None
    all_ratings_csv_path = loader.dataset_path / data_config.rating_file

    if args.stage == "hybrid" and not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found. Train the base stage first so hybrid uses the same split."
        )

    if use_sample:
        # Sample path keeps the existing in-memory workflow for quick iteration.
        ratings_df = loader.load_ratings(sample=True)
        logger.info(f"Ratings loaded: {len(ratings_df):,} records")

        split_artifact, train_ratings_df = load_or_create_training_split(
            loader,
            split_path=split_path,
            ratings_df=ratings_df,
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
    else:
        # Full-data path uses streamed CSV splits written to disk, so we never keep
        # the full ratings/animelist tables in memory at once.
        disk_split = load_or_create_training_disk_split(
            loader,
            split_path=split_path,
            force_resplit=args.force_resplit,
            test_ratio=args.test_ratio,
            relevance_threshold=args.relevance_threshold,
            min_train_items=args.min_train_items,
            min_test_items=args.min_test_items,
            leave_one_out=args.leave_one_out,
            chunk_size=args.stream_chunk_size,
        )
        logger.info(
            "Persisted on-disk split ready at %s (%s eval users, %s held-out items)",
            disk_split.manifest_path,
            disk_split.metadata.get("eval_users", 0),
            disk_split.metadata.get("test_interactions", 0),
        )

    matrix_builder = None
    content_model = None
    collaborative_model = None
    implicit_model = None
    popularity_model = None
    learned_engine = None
    matrices_saved = False

    try:
        if train_base_stage:
            # ===== BUILD MATRICES =====
            logger.info("=" * 50)
            logger.info("Building Matrices")
            logger.info("=" * 50)

            matrix_builder = MatrixBuilder()
            if use_sample:
                matrix_builder.build_rating_matrix(train_ratings_df)
            else:
                matrix_builder.build_rating_matrix_from_csv(
                    disk_split.train_ratings_path,
                    chunk_size=args.stream_chunk_size,
                )

            need_implicit_matrix = (not args.skip_implicit) or (
                not args.skip_collaborative and args.cf_method == "bpr"
            )
            if need_implicit_matrix:
                if use_sample:
                    matrix_builder.build_implicit_matrix(
                        train_animelist_df,
                        chunk_size=args.implicit_chunk_size,
                    )
                else:
                    matrix_builder.build_implicit_matrix_from_csv(
                        disk_split.train_animelist_path,
                        chunk_size=args.implicit_chunk_size,
                    )
            else:
                logger.info("Skipping implicit matrix build (not needed by selected models)")
                matrix_builder.implicit_matrix = None

            matrix_builder.save(matrix_dir)
            matrices_saved = True

            # ===== TRAIN BASE MODELS =====
            content_model = train_content_model(
                content_anime_df, use_sbert=not args.skip_sbert, device=device
            )
            clear_gpu_cache()
            learned_engine = save_training_checkpoint(
                save_dir,
                anime_df,
                content_model=content_model,
                reason="content model",
            )

            if not args.skip_implicit:
                # Dùng implicit_anime_to_idx (superset của anime_to_idx) để implicit model
                # bao gồm cả anime ít explicit rating như One Piece (đang phát sóng)
                implicit_anime_to_idx = getattr(matrix_builder, "implicit_anime_to_idx", matrix_builder.anime_to_idx)
                implicit_idx_to_anime = getattr(matrix_builder, "implicit_idx_to_anime", matrix_builder.idx_to_anime)
                implicit_model = train_implicit_model(
                    matrix_builder.implicit_matrix,
                    implicit_anime_to_idx,
                    implicit_idx_to_anime,
                    matrix_builder.user_to_idx,
                    matrix_builder.idx_to_user,
                    device=device,
                )
                clear_gpu_cache()
                learned_engine = save_training_checkpoint(
                    save_dir,
                    anime_df,
                    content_model=content_model,
                    implicit_model=implicit_model,
                    reason="implicit model",
                )
            else:
                logger.info("Skipping Implicit Feedback Model")

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
                    device=device,
                )
                clear_gpu_cache()
                learned_engine = save_training_checkpoint(
                    save_dir,
                    anime_df,
                    content_model=content_model,
                    collaborative_model=collaborative_model,
                    implicit_model=implicit_model,
                    reason="collaborative model",
                )
            else:
                logger.info("Skipping Collaborative Filtering")

            if use_sample:
                popularity_model = train_popularity_model(
                    anime_df, train_ratings_df, train_animelist_df
                )
                del animelist_df
                del train_animelist_df
            else:
                popularity_model = train_popularity_model_from_csv(
                    anime_df,
                    ratings_csv_path=disk_split.train_ratings_path,
                    animelist_csv_path=disk_split.train_animelist_path,
                    chunk_size=args.stream_chunk_size,
                )

            learned_engine = save_training_checkpoint(
                save_dir,
                anime_df,
                content_model=content_model,
                collaborative_model=collaborative_model,
                implicit_model=implicit_model,
                popularity_model=popularity_model,
                reason="popularity model",
            )
        else:
            logger.info("Loading existing base models for hybrid-only stage...")
            learned_engine = load_existing_models_for_hybrid(save_dir)
            content_model = learned_engine.content_model
            collaborative_model = learned_engine.collaborative_model
            implicit_model = learned_engine.implicit_model
            popularity_model = learned_engine.popularity_model

            if matrix_dir.exists():
                matrix_builder = MatrixBuilder()
                matrix_builder.load(matrix_dir)
                matrices_saved = True
            else:
                logger.warning(
                    "Saved matrices not found in %s. Quick test may be skipped.",
                    matrix_dir,
                )

        if train_hybrid_stage:
            logger.info("=" * 50)
            logger.info("Training Learned Hybrid Engine (meta-model)")
            logger.info("=" * 50)

            if use_sample:
                test_ratings_df = extract_holdout_ratings_df(
                    ratings_df,
                    split_artifact.user_test,
                )

                learned_engine = train_learned_hybrid(
                    content_model=content_model,
                    collaborative_model=collaborative_model,
                    implicit_model=implicit_model,
                    popularity_model=popularity_model,
                    anime_df=anime_df,
                    train_ratings_df=train_ratings_df,
                    test_ratings_df=test_ratings_df,
                    all_ratings_df=ratings_df,
                    save_dir=save_dir,
                    min_ratings_to_train=args.learned_hybrid_min_ratings,
                    top_users=args.learned_hybrid_top_users,
                    relevance_threshold=args.relevance_threshold,
                )
            else:
                learned_engine = train_learned_hybrid_from_csv(
                    content_model=content_model,
                    collaborative_model=collaborative_model,
                    implicit_model=implicit_model,
                    popularity_model=popularity_model,
                    anime_df=anime_df,
                    train_ratings_csv_path=disk_split.train_ratings_path,
                    test_ratings_csv_path=disk_split.test_ratings_path,
                    all_ratings_csv_path=all_ratings_csv_path,
                    save_dir=save_dir,
                    min_ratings_to_train=args.learned_hybrid_min_ratings,
                    top_users=args.learned_hybrid_top_users,
                    relevance_threshold=args.relevance_threshold,
                    chunk_size=args.stream_chunk_size,
                )

            fi = learned_engine.get_feature_importance()
            if fi:
                logger.info("Feature importance (meta-model):")
                for name, val in sorted(fi.items(), key=lambda x: -x[1]):
                    logger.info(f"  {name:<30s} {val:.4f}  {'█' * int(val * 40)}")
        elif learned_engine is None:
            learned_engine = save_training_checkpoint(
                save_dir,
                anime_df,
                content_model=content_model,
                collaborative_model=collaborative_model,
                implicit_model=implicit_model,
                popularity_model=popularity_model,
                reason="base stage completion",
            )

        total_elapsed = time.time() - total_start
        logger.info("=" * 50)
        logger.info(
            f"Training Complete! Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)"
        )
        logger.info(f"Models saved to: {save_dir}")
        logger.info("=" * 50)

        run_quick_test(learned_engine, matrix_builder, popularity_model)
        return learned_engine

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving available checkpoints...")
        try:
            if (
                matrix_builder is not None
                and matrix_builder.user_item_matrix is not None
                and not matrices_saved
            ):
                matrix_builder.save(matrix_dir)
            if any(
                model is not None
                for model in [
                    content_model,
                    collaborative_model,
                    implicit_model,
                    popularity_model,
                    learned_engine,
                ]
            ):
                save_training_checkpoint(
                    save_dir,
                    anime_df,
                    content_model=content_model,
                    collaborative_model=collaborative_model,
                    implicit_model=implicit_model,
                    popularity_model=popularity_model,
                    learned_engine=learned_engine,
                    reason="interrupt",
                )
        except Exception as checkpoint_error:
            logger.exception("Failed to save checkpoint after interrupt: %s", checkpoint_error)
        raise SystemExit(130)


if __name__ == "__main__":
    engine = main()
