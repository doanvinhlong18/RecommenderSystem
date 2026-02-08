"""
ALS Implicit Feedback Model.
"""
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
from scipy.sparse import csr_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ALSImplicit:
    """
    Alternating Least Squares for Implicit Feedback.

    Uses watching behavior (episodes watched, status) as implicit feedback.
    Based on "Collaborative Filtering for Implicit Feedback Datasets" (Hu et al., 2008)

    Attributes:
        user_factors: User latent factor matrix
        item_factors: Item latent factor matrix
    """

    def __init__(
        self,
        n_factors: int = None,
        n_iterations: int = None,
        regularization: float = None,
        alpha: float = 40.0
    ):
        """
        Initialize ALS Implicit model.

        Args:
            n_factors: Number of latent factors
            n_iterations: Number of ALS iterations
            regularization: Regularization parameter
            alpha: Confidence scaling factor
        """
        self.n_factors = n_factors or model_config.implicit_factors
        self.n_iterations = n_iterations or model_config.implicit_iterations
        self.regularization = regularization or model_config.implicit_regularization
        self.alpha = alpha

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        # External library model
        self._implicit_model = None

        # Mappings
        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}

    def fit(
        self,
        implicit_matrix: csr_matrix,
        anime_to_idx: Dict[int, int],
        idx_to_anime: Dict[int, int],
        user_to_idx: Dict[int, int] = None,
        idx_to_user: Dict[int, int] = None,
        use_gpu: bool = False
    ) -> "ALSImplicit":
        """
        Fit the ALS model on implicit feedback data.

        Args:
            implicit_matrix: User-item matrix with implicit feedback scores
            anime_to_idx: Anime ID to index mapping
            idx_to_anime: Index to anime ID mapping
            user_to_idx: User ID to index mapping
            idx_to_user: Index to user ID mapping
            use_gpu: Whether to use GPU acceleration

        Returns:
            Self for chaining
        """
        self.anime_to_idx = anime_to_idx
        self.idx_to_anime = idx_to_anime
        self.user_to_idx = user_to_idx or {}
        self.idx_to_user = idx_to_user or {}

        n_users, n_items = implicit_matrix.shape
        logger.info(f"Fitting ALS Implicit on {n_users} users x {n_items} items...")

        try:
            # Try using the implicit library
            self._fit_with_implicit_library(implicit_matrix, use_gpu)
        except ImportError:
            logger.warning("implicit library not found. Using custom implementation.")
            self._fit_custom(implicit_matrix, n_users, n_items)

        logger.info("ALS Implicit fitted successfully")
        return self

    def _fit_with_implicit_library(
        self,
        matrix: csr_matrix,
        use_gpu: bool
    ) -> None:
        """Fit using the implicit library."""
        from implicit.als import AlternatingLeastSquares

        logger.info("Using implicit library for ALS...")

        self._implicit_model = AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.n_iterations,
            regularization=self.regularization,
            use_gpu=use_gpu
        )

        # implicit expects item-user matrix
        item_user_matrix = matrix.T.tocsr()

        # Scale by confidence
        item_user_matrix.data = 1 + self.alpha * item_user_matrix.data

        self._implicit_model.fit(item_user_matrix, show_progress=True)

        self.user_factors = self._implicit_model.user_factors
        self.item_factors = self._implicit_model.item_factors

    def _fit_custom(
        self,
        matrix: csr_matrix,
        n_users: int,
        n_items: int
    ) -> None:
        """Custom ALS implementation."""
        logger.info("Using custom ALS implementation...")

        # Initialize factors
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Confidence matrix: C = 1 + alpha * R
        # Preference matrix: P = (R > 0)

        lambda_eye = self.regularization * np.eye(self.n_factors)

        for iteration in range(self.n_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}")

            # Update user factors
            item_factors_T = self.item_factors.T
            YtY = item_factors_T @ self.item_factors

            for u in range(n_users):
                # Get items for this user
                item_indices = matrix[u].indices
                confidences = 1 + self.alpha * matrix[u].data

                # Y_u^T C_u Y_u
                Y_u = self.item_factors[item_indices]
                Cu_diag = np.diag(confidences - 1)  # C_u - I

                A = YtY + Y_u.T @ Cu_diag @ Y_u + lambda_eye

                # Y_u^T C_u p_u
                p_u = np.ones(len(item_indices))  # preferences = 1 for observed
                b = Y_u.T @ (confidences * p_u)

                self.user_factors[u] = np.linalg.solve(A, b)

            # Update item factors
            matrix_T = matrix.T.tocsr()
            user_factors_T = self.user_factors.T
            XtX = user_factors_T @ self.user_factors

            for i in range(n_items):
                # Get users for this item
                user_indices = matrix_T[i].indices
                if len(user_indices) == 0:
                    continue

                confidences = 1 + self.alpha * matrix_T[i].data

                # X_i^T C_i X_i
                X_i = self.user_factors[user_indices]
                Ci_diag = np.diag(confidences - 1)

                A = XtX + X_i.T @ Ci_diag @ X_i + lambda_eye

                # X_i^T C_i p_i
                p_i = np.ones(len(user_indices))
                b = X_i.T @ (confidences * p_i)

                self.item_factors[i] = np.linalg.solve(A, b)

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_known: bool = True,
        known_items: set = None
    ) -> List[Dict]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_known: Whether to exclude known items
            known_items: Set of known anime IDs

        Returns:
            List of recommendation dictionaries
        """
        if user_id not in self.user_to_idx:
            logger.debug(f"User {user_id} not in training data")
            return []

        user_idx = self.user_to_idx[user_id]

        # Determine which factors to use based on dimensions
        # implicit library transposes the matrix, so factors may be swapped
        n_users_in_mapping = len(self.user_to_idx)
        n_items_in_mapping = len(self.anime_to_idx)

        # Check if factors are swapped (common with implicit library)
        if self.user_factors.shape[0] == n_items_in_mapping and self.item_factors.shape[0] == n_users_in_mapping:
            # Factors are swapped - user_factors contains items, item_factors contains users
            actual_user_factors = self.item_factors
            actual_item_factors = self.user_factors
        else:
            actual_user_factors = self.user_factors
            actual_item_factors = self.item_factors

        # Bounds check
        if user_idx >= actual_user_factors.shape[0]:
            logger.debug(f"User index {user_idx} out of bounds (max: {actual_user_factors.shape[0]})")
            return []

        if self._implicit_model is not None:
            # Use implicit library
            try:
                filter_items = None
                if exclude_known and known_items:
                    filter_items = [
                        self.anime_to_idx[aid]
                        for aid in known_items
                        if aid in self.anime_to_idx
                    ]

                item_ids, scores = self._implicit_model.recommend(
                    user_idx,
                    None,  # No user_items needed for filtering
                    N=top_k * 2,  # Get extra to account for filtering
                    filter_already_liked_items=False
                )

                results = []
                for idx, score in zip(item_ids, scores):
                    if idx not in self.idx_to_anime:
                        continue
                    anime_id = self.idx_to_anime[idx]
                    if exclude_known and known_items and anime_id in known_items:
                        continue
                    results.append({
                        'mal_id': anime_id,
                        'score': float(score)
                    })
                    if len(results) >= top_k:
                        break

                return results
            except Exception as e:
                logger.debug(f"implicit recommend failed: {e}, using custom")

        # Custom recommendation using corrected factors
        user_vec = actual_user_factors[user_idx]
        scores = actual_item_factors @ user_vec

        # Exclude known items
        if exclude_known and known_items:
            for aid in known_items:
                if aid in self.anime_to_idx:
                    item_idx = self.anime_to_idx[aid]
                    if item_idx < len(scores):
                        scores[item_idx] = -np.inf

        top_indices = scores.argsort()[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            if idx >= len(scores) or scores[idx] == -np.inf:
                continue
            if idx not in self.idx_to_anime:
                continue
            anime_id = self.idx_to_anime[idx]
            if exclude_known and known_items and anime_id in known_items:
                continue
            results.append({
                'mal_id': anime_id,
                'score': float(scores[idx])
            })
            if len(results) >= top_k:
                break

        return results

        return results

    def get_similar_items(
        self,
        anime_id: int,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get similar items based on latent factors.

        Args:
            anime_id: Anime ID
            top_k: Number of similar items

        Returns:
            List of similar anime dictionaries
        """
        if anime_id not in self.anime_to_idx:
            return []

        item_idx = self.anime_to_idx[anime_id]

        if self._implicit_model is not None:
            try:
                similar_items = self._implicit_model.similar_items(item_idx, N=top_k + 1)

                results = []
                for idx, score in similar_items:
                    if idx == item_idx:
                        continue
                    results.append({
                        'mal_id': self.idx_to_anime[idx],
                        'similarity': float(score)
                    })
                    if len(results) >= top_k:
                        break

                return results
            except Exception:
                pass

        # Custom similarity
        item_vec = self.item_factors[item_idx]

        # Cosine similarity
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1
        normalized = self.item_factors / norms[:, np.newaxis]

        item_norm = np.linalg.norm(item_vec)
        if item_norm > 0:
            item_vec_norm = item_vec / item_norm
        else:
            item_vec_norm = item_vec

        similarities = normalized @ item_vec_norm
        similar_indices = similarities.argsort()[::-1]

        results = []
        for idx in similar_indices:
            if idx == item_idx:
                continue
            if len(results) >= top_k:
                break
            results.append({
                'mal_id': self.idx_to_anime[idx],
                'similarity': float(similarities[idx])
            })

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'n_factors': self.n_factors,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'anime_to_idx': self.anime_to_idx,
            'idx_to_anime': self.idx_to_anime,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"ALSImplicit saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "ALSImplicit":
        """Load model from file."""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.n_factors = state['n_factors']
        self.n_iterations = state['n_iterations']
        self.regularization = state['regularization']
        self.alpha = state['alpha']
        self.user_factors = state['user_factors']
        self.item_factors = state['item_factors']
        self.anime_to_idx = state['anime_to_idx']
        self.idx_to_anime = state['idx_to_anime']
        self.user_to_idx = state['user_to_idx']
        self.idx_to_user = state['idx_to_user']

        logger.info(f"ALSImplicit loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Test ALS Implicit
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader, MatrixBuilder

    loader = DataLoader()
    loader.load_ratings(sample=True)
    loader.load_animelist(sample=True)

    builder = MatrixBuilder()
    builder.build_rating_matrix(loader.ratings_df)
    builder.build_implicit_matrix(loader.animelist_df)

    als = ALSImplicit(n_factors=30, n_iterations=10)
    als.fit(
        builder.implicit_matrix,
        builder.anime_to_idx,
        builder.idx_to_anime,
        builder.user_to_idx,
        builder.idx_to_user
    )

    # Test recommendations
    test_user = list(builder.user_to_idx.keys())[0]
    print(f"\nImplicit recommendations for user {test_user}:")
    recs = als.recommend_for_user(test_user, top_k=5)
    for rec in recs:
        print(f"  Anime {rec['mal_id']}: score={rec['score']:.4f}")

    # Test similar items
    test_anime = list(builder.anime_to_idx.keys())[0]
    print(f"\nSimilar items to anime {test_anime}:")
    similar = als.get_similar_items(test_anime, top_k=5)
    for item in similar:
        print(f"  Anime {item['mal_id']}: similarity={item['similarity']:.4f}")
