"""
Standalone script to train Content-Based Recommendation Model.

Usage:
    python train_content.py [--skip-sbert]
"""

import argparse
import logging
import time
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, data_config, model_config
from preprocessing import DataLoader, TextProcessor, ContentFeatureBuilder
from models.content import ContentBasedRecommender

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_anime_data():
    """
    Load anime metadata, synopsis, và extract year/episodes.

    FIX: Dùng get_content_base_dataframe() thay vì load_anime() để đảm bảo
    cột 'year' (trích từ 'Aired') có mặt — ContentFeatureBuilder.fit() cần cột này.
    """
    logger.info("Loading anime data...")

    loader = DataLoader()
    # FIX: load_anime() trả về raw data, thiếu cột 'year'
    # get_content_base_dataframe() merge synopsis + extract year + clean episodes
    anime_df = loader.get_content_base_dataframe()

    logger.info(f"Loaded {len(anime_df)} anime")
    return anime_df


def train_content_model(anime_df, use_sbert: bool = True) -> ContentBasedRecommender:
    """
    Train content-based model: SBERT + structured features → fused embedding.

    Args:
        anime_df: DataFrame với anime metadata (phải có cột 'year')
        use_sbert: Có dùng SBERT embeddings không (nếu False thì dùng structured only)

    Returns:
        Trained ContentBasedRecommender model
    """
    logger.info("=" * 70)
    logger.info("TRAINING CONTENT-BASED MODEL")
    logger.info("=" * 70)

    start_time = time.time()

    # Step 1: Generate text embeddings
    processor = TextProcessor()

    if use_sbert:
        logger.info("Step 1: Generating SBERT embeddings...")
        combined_text = processor.combine_text_features(anime_df)
        sbert_emb = processor.fit_sbert(
            combined_text, anime_ids=anime_df["MAL_ID"].tolist()
        )
        logger.info(f"  SBERT embeddings shape: {sbert_emb.shape}")
    else:
        import numpy as np

        logger.warning("Step 1: Skipping SBERT — using zero embeddings")
        sbert_emb = np.zeros((len(anime_df), 384), dtype=np.float32)

    # Step 2: Build structured features + fusion
    logger.info("Step 2: Building structured features...")
    builder = ContentFeatureBuilder()
    builder.fit(anime_df)

    if use_sbert:
        w_struct, w_text, w_member = 0.2, 0.5, 0.3
        logger.info(
            f"  Fusing SBERT ({w_text*100:.0f}%) "
            f"+ Structured ({w_struct*100:.0f}%) "
            f"+ Member ({w_member*100:.0f}%)..."
        )
        final_emb = builder.transform(
            anime_df, sbert_emb, w_struct=w_struct, w_text=w_text, w_member=w_member
        )
    else:
        logger.info("  Using structured features only...")
        final_emb = builder.transform(
            anime_df, sbert_emb, w_struct=0.5, w_text=0.0, w_member=0.5
        )

    logger.info(f"  Final embedding shape: {final_emb.shape}")

    # Step 3: Fit recommender
    logger.info("Step 3: Fitting ContentBasedRecommender...")
    model = ContentBasedRecommender()
    model.fit(anime_df, final_emb)

    elapsed = time.time() - start_time
    logger.info(f"✓ Content Model trained in {elapsed:.2f}s")

    return model


def save_model(model, output_path):
    """Save trained model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(output_path)
    logger.info(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Content-Based Recommendation Model"
    )
    parser.add_argument(
        "--skip-sbert",
        action="store_true",
        help="Skip SBERT encoding (use structured features only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS_DIR / "hybrid" / "content_model.pkl"),
        help="Path to save trained model",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CONTENT-BASED RECOMMENDATION MODEL TRAINER")
    print("=" * 70)

    # Load data
    anime_df = load_anime_data()

    # Train model
    model = train_content_model(anime_df, use_sbert=not args.skip_sbert)

    # Save model
    save_model(model, args.output)

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    print(f"\nModel configuration:")
    print(f"  - SBERT: {'Enabled' if not args.skip_sbert else 'Disabled'}")
    print(f"  - Anime count: {len(anime_df)}")
    print(f"  - Saved to: {args.output}")


if __name__ == "__main__":
    main()
