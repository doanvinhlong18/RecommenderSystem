"""
Matrix builder for constructing sparse user-item matrices.
"""
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from scipy.sparse import csr_matrix, save_npz, load_npz

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CACHE_DIR, data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixBuilder:
    """
    Builds sparse matrices for collaborative filtering.

    Attributes:
        user_item_matrix: Sparse user-item rating matrix
        item_user_matrix: Transposed matrix for item-based CF
        user_to_idx: User ID to matrix index mapping
        idx_to_user: Matrix index to user ID mapping
        anime_to_idx: Anime ID to matrix index mapping
        idx_to_anime: Matrix index to anime ID mapping
    """

    def __init__(self):
        """Initialize MatrixBuilder."""
        self.user_item_matrix: Optional[csr_matrix] = None
        self.item_user_matrix: Optional[csr_matrix] = None
        self.implicit_matrix: Optional[csr_matrix] = None

        # ID mappings
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}

        # Stats
        self.n_users: int = 0
        self.n_items: int = 0
        self.n_ratings: int = 0

    def build_rating_matrix(
        self,
        ratings_df: pd.DataFrame,
        min_user_ratings: int = None,
        min_anime_ratings: int = None
    ) -> csr_matrix:
        """
        Build sparse user-item rating matrix.

        Args:
            ratings_df: DataFrame with user_id, anime_id, rating columns
            min_user_ratings: Minimum ratings per user to include
            min_anime_ratings: Minimum ratings per anime to include

        Returns:
            Sparse CSR matrix of ratings
        """
        min_user_ratings = min_user_ratings or data_config.min_user_ratings
        min_anime_ratings = min_anime_ratings or data_config.min_anime_ratings

        logger.info(f"Building rating matrix from {len(ratings_df):,} ratings...")

        # Filter by minimum ratings
        df = ratings_df.copy()

        if min_user_ratings > 0:
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            df = df[df['user_id'].isin(valid_users)]
            logger.info(f"After user filter: {len(df):,} ratings")

        if min_anime_ratings > 0:
            anime_counts = df['anime_id'].value_counts()
            valid_anime = anime_counts[anime_counts >= min_anime_ratings].index
            df = df[df['anime_id'].isin(valid_anime)]
            logger.info(f"After anime filter: {len(df):,} ratings")

        # Create ID mappings
        unique_users = df['user_id'].unique()
        unique_anime = df['anime_id'].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.anime_to_idx = {aid: idx for idx, aid in enumerate(unique_anime)}
        self.idx_to_anime = {idx: aid for aid, idx in self.anime_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_anime)
        self.n_ratings = len(df)

        logger.info(f"Matrix dimensions: {self.n_users:,} users x {self.n_items:,} items")

        # Build sparse matrix
        row_indices = df['user_id'].map(self.user_to_idx).values
        col_indices = df['anime_id'].map(self.anime_to_idx).values
        ratings = df['rating'].values.astype(np.float32)

        self.user_item_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(self.n_users, self.n_items)
        )

        # Also create item-user matrix (transpose)
        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        logger.info(f"Rating matrix built: {self.user_item_matrix.nnz:,} non-zero entries")
        logger.info(f"Sparsity: {1 - self.user_item_matrix.nnz / (self.n_users * self.n_items):.4%}")

        return self.user_item_matrix

    def build_implicit_matrix(
        self,
        animelist_df: pd.DataFrame,
        watching_status_df: Optional[pd.DataFrame] = None
    ) -> csr_matrix:
        """
        Build implicit feedback matrix from watch data.

        Converts watching status and episodes watched into confidence scores.

        Args:
            animelist_df: DataFrame with user_id, anime_id, watching_status, watched_episodes
            watching_status_df: Optional status mapping DataFrame

        Returns:
            Sparse CSR matrix of implicit feedback
        """
        logger.info(f"Building implicit matrix from {len(animelist_df):,} records...")

        df = animelist_df.copy()

        # Calculate implicit score based on:
        # 1. Watching status (completed = higher weight)
        # 2. Episodes watched

        # Status weights (1=watching, 2=completed, 3=on hold, 4=dropped, 6=plan to watch)
        status_weights = {
            1: 0.8,   # Currently watching
            2: 1.0,   # Completed
            3: 0.5,   # On hold
            4: 0.2,   # Dropped
            6: 0.1    # Plan to watch
        }

        df['status_weight'] = df['watching_status'].map(status_weights).fillna(0.1)

        # Normalize episodes watched (log scale)
        df['episode_weight'] = np.log1p(df['watched_episodes'].fillna(0)) / 10
        df['episode_weight'] = df['episode_weight'].clip(0, 1)

        # Combined implicit score
        df['implicit_score'] = (df['status_weight'] * 0.6 + df['episode_weight'] * 0.4)

        # Use existing mappings or create new ones
        if not self.user_to_idx:
            unique_users = df['user_id'].unique()
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}

        if not self.anime_to_idx:
            unique_anime = df['anime_id'].unique()
            self.anime_to_idx = {aid: idx for idx, aid in enumerate(unique_anime)}
            self.idx_to_anime = {idx: aid for aid, idx in self.anime_to_idx.items()}

        # Filter to known users and items
        df = df[df['user_id'].isin(self.user_to_idx.keys())]
        df = df[df['anime_id'].isin(self.anime_to_idx.keys())]

        # Build sparse matrix
        row_indices = df['user_id'].map(self.user_to_idx).values
        col_indices = df['anime_id'].map(self.anime_to_idx).values
        scores = df['implicit_score'].values.astype(np.float32)

        n_users = len(self.user_to_idx)
        n_items = len(self.anime_to_idx)

        self.implicit_matrix = csr_matrix(
            (scores, (row_indices, col_indices)),
            shape=(n_users, n_items)
        )

        logger.info(f"Implicit matrix built: {self.implicit_matrix.nnz:,} entries")

        return self.implicit_matrix

    def get_user_ratings(self, user_id: int) -> Dict[int, float]:
        """
        Get all ratings for a user.

        Args:
            user_id: User ID

        Returns:
            Dictionary of anime_id -> rating
        """
        if user_id not in self.user_to_idx:
            return {}

        user_idx = self.user_to_idx[user_id]
        user_row = self.user_item_matrix[user_idx].toarray().flatten()

        ratings = {}
        for item_idx, rating in enumerate(user_row):
            if rating > 0:
                anime_id = self.idx_to_anime[item_idx]
                ratings[anime_id] = rating

        return ratings

    def get_item_ratings(self, anime_id: int) -> Dict[int, float]:
        """
        Get all ratings for an anime.

        Args:
            anime_id: Anime ID

        Returns:
            Dictionary of user_id -> rating
        """
        if anime_id not in self.anime_to_idx:
            return {}

        item_idx = self.anime_to_idx[anime_id]
        item_col = self.item_user_matrix[item_idx].toarray().flatten()

        ratings = {}
        for user_idx, rating in enumerate(item_col):
            if rating > 0:
                user_id = self.idx_to_user[user_idx]
                ratings[user_id] = rating

        return ratings

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[csr_matrix, csr_matrix]:
        """
        Split rating matrix into train and test sets.

        Args:
            test_size: Fraction of ratings for test set
            random_state: Random seed

        Returns:
            Tuple of (train_matrix, test_matrix)
        """
        if self.user_item_matrix is None:
            raise ValueError("Rating matrix not built. Call build_rating_matrix first.")

        np.random.seed(random_state)

        # Get non-zero entries
        rows, cols = self.user_item_matrix.nonzero()
        data = self.user_item_matrix.data.copy()

        n_ratings = len(data)
        n_test = int(n_ratings * test_size)

        # Random split
        test_indices = np.random.choice(n_ratings, size=n_test, replace=False)
        train_mask = np.ones(n_ratings, dtype=bool)
        train_mask[test_indices] = False

        # Create train matrix
        train_data = data.copy()
        train_data[test_indices] = 0
        train_matrix = csr_matrix(
            (train_data[train_mask], (rows[train_mask], cols[train_mask])),
            shape=self.user_item_matrix.shape
        )

        # Create test matrix
        test_matrix = csr_matrix(
            (data[test_indices], (rows[test_indices], cols[test_indices])),
            shape=self.user_item_matrix.shape
        )

        logger.info(f"Train: {train_matrix.nnz:,} ratings, Test: {test_matrix.nnz:,} ratings")

        return train_matrix, test_matrix

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save matrices and mappings.

        Args:
            directory: Directory to save files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save sparse matrices
        if self.user_item_matrix is not None:
            save_npz(directory / "user_item_matrix.npz", self.user_item_matrix)
        if self.item_user_matrix is not None:
            save_npz(directory / "item_user_matrix.npz", self.item_user_matrix)
        if self.implicit_matrix is not None:
            save_npz(directory / "implicit_matrix.npz", self.implicit_matrix)

        # Save mappings
        mappings = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'anime_to_idx': self.anime_to_idx,
            'idx_to_anime': self.idx_to_anime,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_ratings': self.n_ratings
        }

        with open(directory / "mappings.pkl", 'wb') as f:
            pickle.dump(mappings, f)

        logger.info(f"MatrixBuilder saved to {directory}")

    def load(self, directory: Union[str, Path]) -> None:
        """
        Load matrices and mappings.

        Args:
            directory: Directory with saved files
        """
        directory = Path(directory)

        # Load sparse matrices
        if (directory / "user_item_matrix.npz").exists():
            self.user_item_matrix = load_npz(directory / "user_item_matrix.npz")
        if (directory / "item_user_matrix.npz").exists():
            self.item_user_matrix = load_npz(directory / "item_user_matrix.npz")
        if (directory / "implicit_matrix.npz").exists():
            self.implicit_matrix = load_npz(directory / "implicit_matrix.npz")

        # Load mappings
        with open(directory / "mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)

        self.user_to_idx = mappings['user_to_idx']
        self.idx_to_user = mappings['idx_to_user']
        self.anime_to_idx = mappings['anime_to_idx']
        self.idx_to_anime = mappings['idx_to_anime']
        self.n_users = mappings['n_users']
        self.n_items = mappings['n_items']
        self.n_ratings = mappings['n_ratings']

        logger.info(f"MatrixBuilder loaded from {directory}")


if __name__ == "__main__":
    # Test matrix builder
    from data_loader import DataLoader

    loader = DataLoader()
    loader.load_ratings(sample=True)
    loader.load_animelist(sample=True)

    builder = MatrixBuilder()

    # Build rating matrix
    builder.build_rating_matrix(loader.ratings_df)

    # Build implicit matrix
    builder.build_implicit_matrix(loader.animelist_df)

    # Test train-test split
    train, test = builder.get_train_test_split()

    # Save
    builder.save(CACHE_DIR / "matrices")

    print("\nMatrix Builder Test Complete!")
