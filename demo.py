"""
Demo Script for Advanced Hybrid Anime Recommender System
=========================================================
Demonstrates training and using the 3-component hybrid recommender:
- TF-IDF (lexical similarity)
- SBERT (semantic similarity)
- Score weighting (quality)

Usage:
    python demo.py
"""

import os
from Main import AdvancedHybridRecommender


def main():
    """
    Main demo function for the Advanced Hybrid Recommender.
    """
    print("=" * 70)
    print("🎌 ADVANCED HYBRID ANIME RECOMMENDER - DEMO")
    print("   TF-IDF + SBERT + Score Weighting")
    print("=" * 70)

    MODEL_PATH = "advanced_recommender_model.pkl"

    # Initialize recommender
    recommender = AdvancedHybridRecommender()

    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        print("\n📌 Found existing model, loading...")
        if recommender.load_model(MODEL_PATH):
            print("✅ Using cached model")
        else:
            print("⚠️ Failed to load, will retrain...")
            recommender = AdvancedHybridRecommender()

    # If not loaded, train from scratch
    if not recommender.is_fitted:
        print("\n📌 Training new model...")
        recommender.fit()

        # Save for future use
        print("\n📌 Saving model...")
        recommender.save_model(MODEL_PATH)

    # ==========================================================================
    # DEMO: Recommendations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("📌 DEMO: Generating Recommendations")
    print("=" * 70)

    # Example 1: Basic recommendation with default weights
    print("\n🔹 Example 1: Recommendations for 'Sousou no Frieren'")
    print("   (α=0.4 TF-IDF, β=0.4 SBERT, γ=0.2 Score)")
    recs = recommender.recommend(
        title="Sousou no Frieren",
        top_k=5,
        alpha=0.4,
        beta=0.4,
        gamma=0.2
    )
    if not recs.empty:
        print("\n📊 Top 5 Recommendations:")
        print(recs.to_string(index=False))

    # Example 2: More weight on semantic similarity
    print("\n🔹 Example 2: Higher SBERT weight (semantic focus)")
    print("   (α=0.2 TF-IDF, β=0.6 SBERT, γ=0.2 Score)")
    recs = recommender.recommend(
        title="Sousou no Frieren",
        top_k=5,
        alpha=0.2,
        beta=0.6,
        gamma=0.2
    )
    if not recs.empty:
        print("\n📊 Top 5 Recommendations:")
        print(recs.to_string(index=False))

    # Example 3: Different anime
    print("\n🔹 Example 3: Recommendations for 'Death Note'")
    recs = recommender.recommend(
        title="Death Note",
        top_k=5
    )
    if not recs.empty:
        print("\n📊 Top 5 Recommendations:")
        print(recs.to_string(index=False))

    # Example 4: Filtered recommendations
    print("\n🔹 Example 4: TV only, min score 8.0")
    recs = recommender.recommend(
        title="Attack on Titan",
        top_k=5,
        filter_type="TV",
        min_score=8.0
    )
    if not recs.empty:
        print("\n📊 Top 5 Recommendations:")
        print(recs.to_string(index=False))

    # Example 5: Search functionality
    print("\n🔹 Example 5: Search for 'naruto'")
    search_results = recommender.search_anime("naruto")
    if not search_results.empty:
        print(search_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)

    return recommender


if __name__ == "__main__":
    recommender = main()

