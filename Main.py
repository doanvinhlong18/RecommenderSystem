"""
Advanced Hybrid Anime Recommender System
=========================================
A 3-component hybrid recommender combining:
1. TF-IDF Content Similarity (lexical matching)
2. SBERT Semantic Similarity (deep meaning)
3. Normalized Score Weighting (quality/popularity)

Optimized for memory efficiency using sparse matrices and FAISS.

Author: Senior ML Engineer
Date: February 2026
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from scipy.sparse import csr_matrix, save_npz, load_npz

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Sentence Transformers for SBERT
from sentence_transformers import SentenceTransformer

# FAISS for efficient similarity search
import faiss


class AdvancedHybridRecommender:
    """
    Advanced Hybrid Anime Recommender combining:
    - TF-IDF for lexical/keyword similarity
    - SBERT for semantic/meaning similarity
    - Normalized scores for quality weighting

    Hybrid Score Formula:
    FinalScore = α * TF-IDF_sim + β * SBERT_sim + γ * Score_norm

    Default weights: α=0.4, β=0.4, γ=0.2
    """

    def __init__(self, sbert_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the recommender.

        Args:
            sbert_model: Name of the sentence-transformers model to use
        """
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.title_to_index: Dict[str, int] = {}

        # TF-IDF components
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[csr_matrix] = None

        # SBERT components
        self.sbert_model_name = sbert_model
        self.sbert_model: Optional[SentenceTransformer] = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Score normalization
        self.score_scaler: Optional[MinMaxScaler] = None
        self.normalized_scores: Optional[np.ndarray] = None

        # Model metadata
        self.model_version: str = "2.0"
        self.is_fitted: bool = False

    # =========================================================================
    # STEP 1: Data Loading and Preprocessing
    # =========================================================================

    def prepare_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load and preprocess the anime dataset.

        Steps:
        1. Load from HuggingFace or local file
        2. Remove null values in combined_features
        3. Keep only rows with valid scores
        4. Create title-to-index mapping

        Args:
            filepath: Optional local file path. If None, loads from HuggingFace.

        Returns:
            Cleaned DataFrame
        """
        print("=" * 60)
        print("📥 STEP 1: Loading and Preparing Data")
        print("=" * 60)

        if filepath and os.path.exists(filepath):
            print(f"   Loading from local file: {filepath}")
            self.df = pd.read_csv(filepath)
        else:
            print("   Loading from HuggingFace...")
            from datasets import load_dataset
            anime_ds = load_dataset(
                "csv",
                data_files="hf://datasets/victor-odunsi/anime-recommender-artifacts/anime_data.csv"
            )
            self.df = anime_ds["train"].to_pandas()

        original_size = len(self.df)
        print(f"   Original dataset size: {original_size}")

        # Remove rows with null combined_features
        self.df = self.df.dropna(subset=['combined_features'])
        print(f"   After removing null features: {len(self.df)}")

        # Keep only rows with valid scores
        self.df = self.df.dropna(subset=['score'])
        self.df = self.df[self.df['score'] > 0]
        print(f"   After filtering valid scores: {len(self.df)}")

        # Reset index for consistent indexing
        self.df = self.df.reset_index(drop=True)

        # Create title-to-index mapping (case-insensitive)
        self.title_to_index = {
            name.lower(): idx for idx, name in enumerate(self.df['name'])
        }

        # Fill missing optional columns
        for col in ['themes', 'type', 'demographics']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')

        print(f"✅ Data preparation complete: {len(self.df)} anime loaded")

        return self.df

    # =========================================================================
    # STEP 2: TF-IDF Similarity
    # =========================================================================

    def build_tfidf_matrix(self) -> csr_matrix:
        """
        Build TF-IDF matrix from combined_features.

        Uses sparse matrix representation for memory efficiency.
        Does NOT compute full similarity matrix to avoid O(n²) memory.

        Returns:
            Sparse TF-IDF matrix
        """
        print("\n" + "=" * 60)
        print("🔧 STEP 2: Building TF-IDF Matrix")
        print("=" * 60)

        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            dtype=np.float32  # Use float32 for memory efficiency
        )

        # Fit and transform - result is already sparse
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df['combined_features']
        )

        print(f"   TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"   Sparse matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4f}")
        print(f"   Memory usage: {self.tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB")
        print("✅ TF-IDF matrix built (sparse format)")

        return self.tfidf_matrix

    def get_tfidf_similarities(self, query_idx: int, top_k: int = None) -> np.ndarray:
        """
        Compute TF-IDF cosine similarities for a single query.

        Memory efficient: computes similarities on-the-fly instead of
        storing full N×N matrix.

        Args:
            query_idx: Index of the query anime
            top_k: If provided, only compute for potential candidates

        Returns:
            Array of similarity scores for all anime
        """
        query_vector = self.tfidf_matrix[query_idx]

        # Compute cosine similarity with all other anime
        # This is O(n) instead of O(n²) by computing one row at a time
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        return similarities

    # =========================================================================
    # STEP 3: SBERT Semantic Embeddings
    # =========================================================================

    def build_sbert_embeddings(self, batch_size: int = 64) -> np.ndarray:
        """
        Generate SBERT embeddings for combined_features.

        Uses sentence-transformers model for semantic understanding.
        Builds FAISS index for efficient similarity search.

        Args:
            batch_size: Batch size for encoding

        Returns:
            SBERT embeddings matrix
        """
        print("\n" + "=" * 60)
        print("🧠 STEP 3: Building SBERT Embeddings")
        print("=" * 60)

        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Load SBERT model
        print(f"   Loading model: {self.sbert_model_name}")
        self.sbert_model = SentenceTransformer(self.sbert_model_name)

        # Get texts for encoding
        texts = self.df['combined_features'].tolist()

        print(f"   Encoding {len(texts)} texts...")
        print(f"   (This may take a few minutes...)")

        # Encode with progress
        self.sbert_embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        print(f"   Embeddings shape: {self.sbert_embeddings.shape}")
        print(f"   Memory usage: {self.sbert_embeddings.nbytes / 1024 / 1024:.2f} MB")

        # Build FAISS index for efficient search
        self._build_faiss_index()

        print("✅ SBERT embeddings built with FAISS index")

        return self.sbert_embeddings

    def _build_faiss_index(self):
        """
        Build FAISS index for efficient similarity search.

        Uses Inner Product (IP) index since embeddings are normalized,
        which is equivalent to cosine similarity.
        """
        print("   Building FAISS index...")

        embedding_dim = self.sbert_embeddings.shape[1]

        # Use Inner Product index (equivalent to cosine sim for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

        # Add embeddings to index
        self.faiss_index.add(self.sbert_embeddings.astype(np.float32))

        print(f"   FAISS index built with {self.faiss_index.ntotal} vectors")

    def get_sbert_similarities(self, query_idx: int, top_k: int = None) -> np.ndarray:
        """
        Get SBERT similarities using FAISS for efficiency.

        Args:
            query_idx: Index of the query anime
            top_k: Number of nearest neighbors (if None, returns all)

        Returns:
            Array of similarity scores for all anime
        """
        query_embedding = self.sbert_embeddings[query_idx:query_idx+1]

        if top_k is None or top_k >= len(self.df):
            # Return similarities for all
            similarities = np.dot(self.sbert_embeddings, query_embedding.T).flatten()
        else:
            # Use FAISS for top-k search (much faster for large datasets)
            similarities = np.zeros(len(self.df))
            scores, indices = self.faiss_index.search(
                query_embedding.astype(np.float32),
                min(top_k * 2, len(self.df))  # Get extra for safety
            )
            for score, idx in zip(scores[0], indices[0]):
                similarities[idx] = score

        return similarities

    # =========================================================================
    # STEP 4: Score Normalization
    # =========================================================================

    def normalize_scores(self) -> np.ndarray:
        """
        Normalize anime scores to [0, 1] range using Min-Max scaling.

        Formula: score_norm = (score - min) / (max - min)

        Returns:
            Normalized scores array
        """
        print("\n" + "=" * 60)
        print("📊 STEP 4: Normalizing Scores")
        print("=" * 60)

        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        scores = self.df['score'].values.reshape(-1, 1)

        # Use MinMaxScaler for normalization
        self.score_scaler = MinMaxScaler(feature_range=(0, 1))
        self.normalized_scores = self.score_scaler.fit_transform(scores).flatten()

        # Store in dataframe
        self.df['normalized_score'] = self.normalized_scores

        print(f"   Original score range: {self.df['score'].min():.2f} - {self.df['score'].max():.2f}")
        print(f"   Normalized range: {self.normalized_scores.min():.4f} - {self.normalized_scores.max():.4f}")
        print("✅ Score normalization complete")

        return self.normalized_scores

    # =========================================================================
    # STEP 5 & 6: Hybrid Recommendation
    # =========================================================================

    def fit(self, filepath: str = None):
        """
        Fit the complete recommender pipeline.

        Args:
            filepath: Optional local data file path
        """
        print("\n" + "=" * 60)
        print("🚀 FITTING ADVANCED HYBRID RECOMMENDER")
        print("=" * 60)

        # Step 1: Prepare data
        self.prepare_data(filepath)

        # Step 2: Build TF-IDF
        self.build_tfidf_matrix()

        # Step 3: Build SBERT embeddings
        self.build_sbert_embeddings()

        # Step 4: Normalize scores
        self.normalize_scores()

        self.is_fitted = True

        print("\n" + "=" * 60)
        print("✅ RECOMMENDER FULLY FITTED AND READY!")
        print("=" * 60)

    def recommend(
        self,
        title: str,
        top_k: int = 5,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        filter_type: Optional[str] = None,
        min_score: Optional[float] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations.

        Hybrid Score Formula:
        FinalScore = α * TF-IDF_sim + β * SBERT_sim + γ * Score_norm

        Args:
            title: Name of the anime to get recommendations for
            top_k: Number of recommendations to return
            alpha: Weight for TF-IDF similarity (default: 0.4)
            beta: Weight for SBERT similarity (default: 0.4)
            gamma: Weight for normalized score (default: 0.2)
            filter_type: Optional filter by anime type
            min_score: Optional minimum score threshold
            verbose: Whether to print details

        Returns:
            DataFrame with top-K recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Validate weights
        if abs(alpha + beta + gamma - 1.0) > 0.001:
            print(f"⚠️ Warning: Weights sum to {alpha + beta + gamma:.2f}, not 1.0")

        # Find anime index
        anime_idx = self._get_anime_index(title)
        if anime_idx is None:
            if verbose:
                print(f"❌ Anime '{title}' not found.")
            return pd.DataFrame()

        source_anime = self.df.iloc[anime_idx]

        if verbose:
            print(f"\n🎬 Recommendations for: '{source_anime['name']}'")
            print(f"   Score: {source_anime['score']:.2f} | Type: {source_anime.get('type', 'N/A')}")
            print(f"   Weights: α={alpha}, β={beta}, γ={gamma}")

        # Get similarity scores
        tfidf_sim = self.get_tfidf_similarities(anime_idx)
        sbert_sim = self.get_sbert_similarities(anime_idx)

        # Compute hybrid scores
        # FinalScore = α * TF-IDF + β * SBERT + γ * Score_norm
        hybrid_scores = (
            alpha * tfidf_sim +
            beta * sbert_sim +
            gamma * self.normalized_scores
        )

        # Create results dataframe
        results = self.df.copy()
        results['tfidf_sim'] = tfidf_sim
        results['sbert_sim'] = sbert_sim
        results['hybrid_score'] = hybrid_scores

        # Remove query anime
        results = results.drop(anime_idx)

        # Apply filters
        if filter_type and 'type' in results.columns:
            results = results[results['type'].str.lower() == filter_type.lower()]

        if min_score is not None:
            results = results[results['score'] >= min_score]

        # Get top-K
        results = results.nlargest(top_k, 'hybrid_score')

        # Select output columns
        output_cols = ['name', 'score', 'hybrid_score', 'tfidf_sim', 'sbert_sim']
        if 'type' in results.columns:
            output_cols.insert(2, 'type')
        if 'themes' in results.columns:
            output_cols.append('themes')
        if 'image_url' in results.columns:
            output_cols.append('image_url')
        if 'anime_url' in results.columns:
            output_cols.append('anime_url')

        return results[output_cols].reset_index(drop=True)

    def _get_anime_index(self, title: str) -> Optional[int]:
        """Get anime index by title (case-insensitive with partial match)."""
        lower_title = title.lower()

        # Exact match
        if lower_title in self.title_to_index:
            return self.title_to_index[lower_title]

        # Partial match
        for name, idx in self.title_to_index.items():
            if lower_title in name or name in lower_title:
                return idx

        return None

    def search_anime(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search for anime by title."""
        if self.df is None:
            raise ValueError("Data not prepared.")

        matches = self.df[
            self.df['name'].str.lower().str.contains(query.lower(), na=False)
        ]

        # Select columns, including image_url and anime_url if available
        columns = ['name', 'score', 'type']
        if 'image_url' in self.df.columns:
            columns.append('image_url')
        if 'anime_url' in self.df.columns:
            columns.append('anime_url')

        return matches[columns].head(limit)

    # =========================================================================
    # Model Persistence
    # =========================================================================

    def save_model(self, filepath: str = "advanced_recommender_model.pkl"):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")

        print(f"💾 Saving model to '{filepath}'...")

        model_state = {
            'version': self.model_version,
            'df': self.df,
            'title_to_index': self.title_to_index,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'sbert_model_name': self.sbert_model_name,
            'sbert_embeddings': self.sbert_embeddings,
            'score_scaler': self.score_scaler,
            'normalized_scores': self.normalized_scores,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ Model saved! Size: {file_size_mb:.2f} MB")

    def load_model(self, filepath: str = "advanced_recommender_model.pkl") -> bool:
        """Load a previously saved model."""
        if not os.path.exists(filepath):
            print(f"❌ Model file '{filepath}' not found.")
            return False

        print(f"📂 Loading model from '{filepath}'...")

        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)

            self.df = model_state['df']
            self.title_to_index = model_state['title_to_index']
            self.tfidf_vectorizer = model_state['tfidf_vectorizer']
            self.tfidf_matrix = model_state['tfidf_matrix']
            self.sbert_model_name = model_state['sbert_model_name']
            self.sbert_embeddings = model_state['sbert_embeddings']
            self.score_scaler = model_state['score_scaler']
            self.normalized_scores = model_state['normalized_scores']
            self.is_fitted = model_state['is_fitted']

            # Rebuild FAISS index
            self._build_faiss_index()

            print(f"✅ Model loaded! {len(self.df)} anime available")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    @classmethod
    def from_saved_model(cls, filepath: str = "advanced_recommender_model.pkl"):
        """Create instance from saved model."""
        recommender = cls()
        if recommender.load_model(filepath):
            return recommender
        return None


# =============================================================================
# Utility Functions
# =============================================================================

def quick_recommend(
    title: str,
    top_k: int = 5,
    model_path: str = "advanced_recommender_model.pkl"
) -> pd.DataFrame:
    """Quick function to get recommendations from saved model."""
    recommender = AdvancedHybridRecommender.from_saved_model(model_path)
    if recommender is None:
        print("❌ No saved model found. Run training first.")
        return pd.DataFrame()
    return recommender.recommend(title, top_k)
