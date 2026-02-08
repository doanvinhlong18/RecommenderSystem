"""
Item-Based Collaborative Filtering.
"""
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering recommender.

    Computes item-item similarity based on user rating patterns.

    Attributes:
        item_similarity: Item-item similarity matrix
        user_item_matrix: Original user-item rating matrix
        anime_to_idx: Anime ID to matrix index mapping
        idx_to_anime: Matrix index to anime ID mapping
    """

    def __init__(self, k_neighbors: int = 50):
        """
        Initialize ItemBasedCF.

        Args:
            k_neighbors: Number of neighbors to consider
        """
        self.k_neighbors = k_neighbors
        self.item_similarity: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[csr_matrix] = None
        self.mean_ratings: Optional[np.ndarray] = None

        # Mappings
        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}

        # For FAISS
        self.faiss_index = None

    def fit(
        self,
        user_item_matrix: csr_matrix,
        anime_to_idx: Dict[int, int],
        idx_to_anime: Dict[int, int],
        user_to_idx: Dict[int, int] = None,
        idx_to_user: Dict[int, int] = None,
        compute_full_similarity: bool = False
    ) -> "ItemBasedCF":
        """
        Fit the Item-Based CF model.

        Args:
            user_item_matrix: Sparse user-item rating matrix
            anime_to_idx: Anime ID to index mapping
            idx_to_anime: Index to anime ID mapping
            user_to_idx: User ID to index mapping
            idx_to_user: Index to user ID mapping
            compute_full_similarity: Whether to compute full similarity matrix

        Returns:
            Self for chaining
        """
        self.user_item_matrix = user_item_matrix
        self.anime_to_idx = anime_to_idx
        self.idx_to_anime = idx_to_anime
        self.user_to_idx = user_to_idx or {}
        self.idx_to_user = idx_to_user or {}

        logger.info(f"Fitting Item-Based CF on matrix {user_item_matrix.shape}...")

        # Compute mean ratings per item
        item_sums = np.array(user_item_matrix.sum(axis=0)).flatten()
        item_counts = np.array((user_item_matrix > 0).sum(axis=0)).flatten()
        item_counts[item_counts == 0] = 1  # Avoid division by zero
        self.mean_ratings = item_sums / item_counts

        # Item-user matrix (transpose)
        item_user_matrix = user_item_matrix.T.tocsr()

        if compute_full_similarity:
            # Compute full item-item similarity (memory intensive)
            logger.info("Computing full item-item similarity matrix...")
            self.item_similarity = cosine_similarity(item_user_matrix)
            logger.info(f"Item similarity matrix shape: {self.item_similarity.shape}")
        else:
            # Build FAISS index for on-demand similarity
            self._build_faiss_index(item_user_matrix)

        logger.info("Item-Based CF fitted successfully")
        return self

    def _build_faiss_index(self, item_user_matrix: csr_matrix) -> None:
        """Build FAISS index for efficient similarity search."""
        try:
            import faiss

            logger.info("Building FAISS index for item similarity...")

            # Convert to dense and normalize
            item_vectors = item_user_matrix.toarray().astype(np.float32)
            norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            item_vectors = item_vectors / norms

            # Build index
            d = item_vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)
            self.faiss_index.add(item_vectors)

            logger.info(f"FAISS index built with {self.faiss_index.ntotal} items")

        except ImportError:
            logger.warning("FAISS not installed. Will use sklearn for similarity.")
            # Fallback: compute similarity on-demand
            self.item_vectors_normalized = None

    def get_similar_items(
        self,
        anime_id: int,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get similar anime based on collaborative filtering.

        Args:
            anime_id: Anime ID
            top_k: Number of similar items

        Returns:
            List of similar anime dictionaries
        """
        if anime_id not in self.anime_to_idx:
            logger.warning(f"Anime ID {anime_id} not in training data")
            return []

        idx = self.anime_to_idx[anime_id]

        if self.item_similarity is not None:
            # Use precomputed similarity
            similarities = self.item_similarity[idx]
            similar_indices = similarities.argsort()[::-1][1:top_k+1]
            similar_scores = similarities[similar_indices]
        elif self.faiss_index is not None:
            # Use FAISS
            item_user_matrix = self.user_item_matrix.T.tocsr()
            query = item_user_matrix[idx].toarray().astype(np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            scores, indices = self.faiss_index.search(query, top_k + 1)
            # Filter out self
            mask = indices[0] != idx
            similar_indices = indices[0][mask][:top_k]
            similar_scores = scores[0][mask][:top_k]
        else:
            # Compute on-demand
            item_user_matrix = self.user_item_matrix.T.tocsr()
            query = item_user_matrix[idx]
            similarities = cosine_similarity(query, item_user_matrix).flatten()
            similar_indices = similarities.argsort()[::-1][1:top_k+1]
            similar_scores = similarities[similar_indices]

        results = []
        for sim_idx, score in zip(similar_indices, similar_scores):
            results.append({
                'mal_id': self.idx_to_anime[sim_idx],
                'similarity': float(score),
                'mean_rating': float(self.mean_ratings[sim_idx])
            })

        return results

    def predict_rating(
        self,
        user_id: int,
        anime_id: int
    ) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User ID
            anime_id: Anime ID

        Returns:
            Predicted rating
        """
        if user_id not in self.user_to_idx or anime_id not in self.anime_to_idx:
            return self.mean_ratings.mean() if self.mean_ratings is not None else 5.0

        user_idx = self.user_to_idx[user_id]
        item_idx = self.anime_to_idx[anime_id]

        # Get user's rated items
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return self.mean_ratings[item_idx]

        # Get similarities with rated items
        if self.item_similarity is not None:
            similarities = self.item_similarity[item_idx, rated_items]
        else:
            # Compute on-demand
            item_user_matrix = self.user_item_matrix.T.tocsr()
            query = item_user_matrix[item_idx]
            all_sims = cosine_similarity(query, item_user_matrix[rated_items]).flatten()
            similarities = all_sims

        # Get top-k neighbors
        if len(similarities) > self.k_neighbors:
            top_k_indices = similarities.argsort()[::-1][:self.k_neighbors]
            similarities = similarities[top_k_indices]
            rated_items = rated_items[top_k_indices]

        # Weighted average
        if similarities.sum() > 0:
            prediction = np.dot(similarities, user_ratings[rated_items]) / similarities.sum()
        else:
            prediction = self.mean_ratings[item_idx]

        return float(np.clip(prediction, 1, 10))

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_rated: Whether to exclude already rated items

        Returns:
            List of recommendation dictionaries
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User ID {user_id} not in training data")
            return []

        user_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = set(np.where(user_ratings > 0)[0])

        # Predict ratings for all unrated items
        predictions = []

        for item_idx in range(self.user_item_matrix.shape[1]):
            if exclude_rated and item_idx in rated_items:
                continue

            anime_id = self.idx_to_anime[item_idx]
            pred_rating = self.predict_rating(user_id, anime_id)
            predictions.append((item_idx, pred_rating))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        results = []
        for item_idx, pred_rating in predictions[:top_k]:
            results.append({
                'mal_id': self.idx_to_anime[item_idx],
                'predicted_rating': pred_rating,
                'mean_rating': float(self.mean_ratings[item_idx])
            })

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'k_neighbors': self.k_neighbors,
            'item_similarity': self.item_similarity,
            'user_item_matrix': self.user_item_matrix,
            'mean_ratings': self.mean_ratings,
            'anime_to_idx': self.anime_to_idx,
            'idx_to_anime': self.idx_to_anime,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"ItemBasedCF saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "ItemBasedCF":
        """Load model from file."""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.k_neighbors = state['k_neighbors']
        self.item_similarity = state['item_similarity']
        self.user_item_matrix = state['user_item_matrix']
        self.mean_ratings = state['mean_ratings']
        self.anime_to_idx = state['anime_to_idx']
        self.idx_to_anime = state['idx_to_anime']
        self.user_to_idx = state['user_to_idx']
        self.idx_to_user = state['idx_to_user']

        # Rebuild FAISS index if needed
        if self.item_similarity is None:
            item_user_matrix = self.user_item_matrix.T.tocsr()
            self._build_faiss_index(item_user_matrix)

        logger.info(f"ItemBasedCF loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Test Item-Based CF
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader, MatrixBuilder

    loader = DataLoader()
    loader.load_ratings(sample=True)

    builder = MatrixBuilder()
    builder.build_rating_matrix(loader.ratings_df)

    cf = ItemBasedCF(k_neighbors=20)
    cf.fit(
        builder.user_item_matrix,
        builder.anime_to_idx,
        builder.idx_to_anime,
        builder.user_to_idx,
        builder.idx_to_user
    )

    # Test similar items
    test_anime_id = list(builder.anime_to_idx.keys())[0]
    print(f"\nSimilar items to anime {test_anime_id}:")
    similar = cf.get_similar_items(test_anime_id, top_k=5)
    for item in similar:
        print(f"  Anime {item['mal_id']}: similarity={item['similarity']:.4f}")

    # Test user recommendation
    test_user_id = list(builder.user_to_idx.keys())[0]
    print(f"\nRecommendations for user {test_user_id}:")
    recs = cf.recommend_for_user(test_user_id, top_k=5)
    for rec in recs:
        print(f"  Anime {rec['mal_id']}: predicted={rec['predicted_rating']:.2f}")
