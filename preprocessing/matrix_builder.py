"""
Matrix builder for constructing sparse user-item matrices.
"""
from collections import defaultdict

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

_RATING_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "rating": "int8",
}

_ANIMELIST_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "watching_status": "int8",
    "watched_episodes": "int32",
}


class MatrixBuilder:
    """
    Builds sparse matrices for collaborative filtering.

    Attributes:
        user_item_matrix: Sparse user-item rating matrix
        item_user_matrix: Transposed matrix for item-based CF
        implicit_matrix: Sparse user-item implicit feedback matrix
        user_to_idx / idx_to_user: User ID <-> matrix index mappings
        anime_to_idx / idx_to_anime: Anime ID <-> matrix index mappings
    """

    def __init__(self):
        self.user_item_matrix: Optional[csr_matrix] = None
        self.item_user_matrix: Optional[csr_matrix] = None
        self.implicit_matrix: Optional[csr_matrix] = None

        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}

        self.n_users: int = 0
        self.n_items: int = 0
        self.n_ratings: int = 0

    # ------------------------------------------------------------------
    # build_rating_matrix  (unchanged logic, minor dtype tweak)
    # ------------------------------------------------------------------

    def build_rating_matrix(
        self,
        ratings_df: pd.DataFrame,
        min_user_ratings: int = None,
        min_anime_ratings: int = None,
    ) -> csr_matrix:
        """
        Build sparse user-item rating matrix.

        Args:
            ratings_df: DataFrame with user_id, anime_id, rating columns.
            min_user_ratings: Minimum ratings per user to include.
            min_anime_ratings: Minimum ratings per anime to include.

        Returns:
            Sparse CSR matrix of ratings.
        """
        min_user_ratings = min_user_ratings or data_config.min_user_ratings
        min_anime_ratings = min_anime_ratings or data_config.min_anime_ratings

        logger.info(f"Building rating matrix from {len(ratings_df):,} ratings...")

        df = ratings_df

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

        unique_users = df['user_id'].unique()
        unique_anime = df['anime_id'].unique()

        self.user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: int(uid) for uid, idx in self.user_to_idx.items()}
        self.anime_to_idx = {int(aid): idx for idx, aid in enumerate(unique_anime)}
        self.idx_to_anime = {idx: int(aid) for aid, idx in self.anime_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_anime)
        self.n_ratings = len(df)

        logger.info(f"Matrix dimensions: {self.n_users:,} users x {self.n_items:,} items")

        if self.n_users == 0 or self.n_items == 0:
            raise ValueError(
                "Rating matrix is empty after applying the minimum user/item filters. "
                "Use a larger sample or lower min_user_ratings/min_anime_ratings."
            )

        row_indices = df['user_id'].map(self.user_to_idx).values
        col_indices = df['anime_id'].map(self.anime_to_idx).values
        ratings = df['rating'].values.astype(np.float32)

        self.user_item_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(self.n_users, self.n_items),
        )
        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        logger.info(f"Rating matrix built: {self.user_item_matrix.nnz:,} non-zero entries")
        logger.info(
            f"Sparsity: {1 - self.user_item_matrix.nnz / (self.n_users * self.n_items):.4%}"
        )
        return self.user_item_matrix

    def build_rating_matrix_from_csv(
        self,
        filepath: Union[str, Path],
        min_user_ratings: int = None,
        min_anime_ratings: int = None,
        chunk_size: int = 500_000,
    ) -> csr_matrix:
        """
        Build the explicit rating matrix directly from CSV, without loading the
        whole file into a pandas DataFrame.
        """
        filepath = Path(filepath)
        min_user_ratings = min_user_ratings or data_config.min_user_ratings
        min_anime_ratings = min_anime_ratings or data_config.min_anime_ratings

        logger.info("Building rating matrix from CSV: %s", filepath)

        user_counts: Dict[int, int] = defaultdict(int)
        anime_counts: Dict[int, int] = defaultdict(int)

        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                filepath,
                chunksize=chunk_size,
                usecols=["user_id", "anime_id", "rating"],
                dtype=_RATING_DTYPES,
            ),
            start=1,
        ):
            chunk = chunk.loc[chunk["rating"] > 0, ["user_id", "anime_id", "rating"]]
            if chunk.empty:
                continue

            user_vc = chunk["user_id"].value_counts(sort=False)
            anime_vc = chunk["anime_id"].value_counts(sort=False)

            for user_id, count in user_vc.items():
                user_counts[int(user_id)] += int(count)
            for anime_id, count in anime_vc.items():
                anime_counts[int(anime_id)] += int(count)

            if chunk_idx % 20 == 0:
                logger.info("  Counted ratings for %d chunks...", chunk_idx)

        valid_users = {
            user_id
            for user_id, count in user_counts.items()
            if count >= min_user_ratings
        }
        valid_anime = {
            anime_id
            for anime_id, count in anime_counts.items()
            if count >= min_anime_ratings
        }

        self.user_to_idx = {
            int(uid): idx for idx, uid in enumerate(sorted(valid_users))
        }
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.anime_to_idx = {
            int(aid): idx for idx, aid in enumerate(sorted(valid_anime))
        }
        self.idx_to_anime = {idx: aid for aid, idx in self.anime_to_idx.items()}

        self.n_users = len(self.user_to_idx)
        self.n_items = len(self.anime_to_idx)

        logger.info(
            "After CSV filters: %d users x %d items",
            self.n_users,
            self.n_items,
        )
        if self.n_users == 0 or self.n_items == 0:
            raise ValueError(
                "Rating matrix is empty after applying the minimum user/item filters. "
                "Use a larger split or lower min_user_ratings/min_anime_ratings."
            )

        valid_user_set = set(self.user_to_idx.keys())
        valid_anime_set = set(self.anime_to_idx.keys())
        row_parts = []
        col_parts = []
        rating_parts = []
        total_rows = 0

        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                filepath,
                chunksize=chunk_size,
                usecols=["user_id", "anime_id", "rating"],
                dtype=_RATING_DTYPES,
            ),
            start=1,
        ):
            chunk = chunk.loc[chunk["rating"] > 0, ["user_id", "anime_id", "rating"]]
            if chunk.empty:
                continue

            chunk = chunk.loc[
                chunk["user_id"].isin(valid_user_set)
                & chunk["anime_id"].isin(valid_anime_set)
            ]
            if chunk.empty:
                continue

            row_parts.append(
                chunk["user_id"].map(self.user_to_idx).to_numpy(dtype=np.int32, copy=False)
            )
            col_parts.append(
                chunk["anime_id"].map(self.anime_to_idx).to_numpy(dtype=np.int32, copy=False)
            )
            rating_parts.append(
                chunk["rating"].to_numpy(dtype=np.float32, copy=False)
            )
            total_rows += len(chunk)

            if chunk_idx % 20 == 0:
                logger.info("  Materialized ratings for %d chunks...", chunk_idx)

        if not rating_parts:
            raise ValueError("No valid explicit interactions were found in the split CSV.")

        row_indices = np.concatenate(row_parts)
        col_indices = np.concatenate(col_parts)
        ratings = np.concatenate(rating_parts)

        self.n_ratings = int(total_rows)
        self.user_item_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(self.n_users, self.n_items),
        )
        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        logger.info(f"Rating matrix built: {self.user_item_matrix.nnz:,} non-zero entries")
        logger.info(
            f"Sparsity: {1 - self.user_item_matrix.nnz / (self.n_users * self.n_items):.4%}"
        )
        return self.user_item_matrix

    # ------------------------------------------------------------------
    # build_implicit_matrix  — KEY FIX: no .copy() on full 100M+ df
    # ------------------------------------------------------------------

    def build_implicit_matrix(
        self,
        animelist_df: pd.DataFrame,
        watching_status_df: Optional[pd.DataFrame] = None,
        chunk_size: int = 2_000_000,
        min_implicit_anime_interactions: int = 50,
    ) -> csr_matrix:
        """
        Build implicit feedback matrix from watch data.

        Converts watching status and episodes watched into confidence scores.
        Processes the animelist in chunks so it never allocates a full copy
        of the 100M+ row DataFrame in memory.

        Anime với >= min_implicit_anime_interactions watch entries sẽ được thêm vào
        implicit_anime_to_idx ngay cả khi chúng không có đủ explicit ratings
        (ví dụ One Piece — đang phát sóng).

        Args:
            animelist_df: DataFrame with user_id, anime_id,
                          watching_status, watched_episodes.
            watching_status_df: Optional status mapping (unused currently).
            chunk_size: Number of rows processed per pass. Tune down if OOM.

        Returns:
            Sparse CSR matrix of implicit feedback [n_users x n_items].
        """
        logger.info(f"Building implicit matrix from {len(animelist_df):,} records...")

        user_set = set(self.user_to_idx.keys()) if self.user_to_idx else None
        explicit_anime_set = set(self.anime_to_idx.keys()) if self.anime_to_idx else set()

        # Đếm implicit interactions per anime để mở rộng anime set
        implicit_anime_counts: Dict[int, int] = defaultdict(int)
        filtered = animelist_df if user_set is None else animelist_df[animelist_df["user_id"].isin(user_set)]
        for aid, cnt in filtered["anime_id"].value_counts(sort=False).items():
            implicit_anime_counts[int(aid)] += int(cnt)

        implicit_only = {
            aid
            for aid, cnt in implicit_anime_counts.items()
            if cnt >= min_implicit_anime_interactions and aid not in explicit_anime_set
        }
        logger.info(
            "Implicit-only anime additions: %d (threshold=%d)",
            len(implicit_only), min_implicit_anime_interactions,
        )

        all_implicit_anime = sorted(explicit_anime_set | implicit_only)
        self.implicit_anime_to_idx: Dict[int, int] = {aid: i for i, aid in enumerate(all_implicit_anime)}
        self.implicit_idx_to_anime: Dict[int, int] = {i: aid for aid, i in self.implicit_anime_to_idx.items()}
        anime_set = set(self.implicit_anime_to_idx.keys())

        # Status weights: 1=watching, 2=completed, 3=on-hold, 4=dropped, 6=plan-to-watch
        status_weights = {
            1: 0.8,
            2: 1.0,
            3: 0.5,
            4: 0.2,
            6: 0.1,
        }

        # Accumulators for COO data — built incrementally, no full copy
        all_rows: list = []
        all_cols: list = []
        all_scores: list = []

        n_total = len(animelist_df)
        n_chunks = (n_total + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_total)

            # iloc slice returns a view (not a copy) — negligible extra RAM
            chunk = animelist_df.iloc[start:end]

            # --- Filter to known users/items ---
            if user_set is not None:
                chunk = chunk[chunk['user_id'].isin(user_set)]
            if anime_set is not None:
                chunk = chunk[chunk['anime_id'].isin(anime_set)]

            if chunk.empty:
                continue

            # --- Compute implicit score (vectorized, no copy of full df) ---
            status_w = chunk['watching_status'].map(status_weights).fillna(0.1)
            episode_w = (
                np.log1p(chunk['watched_episodes'].fillna(0).astype(np.float32)) / 10
            ).clip(0, 1)
            scores = (status_w * 0.6 + episode_w * 0.4).astype(np.float32)

            rows = chunk['user_id'].map(self.user_to_idx).values
            cols = chunk['anime_id'].map(self.implicit_anime_to_idx).values

            # Drop any unmapped rows (NaN from map — happens when mappings absent)
            valid = ~(pd.isnull(rows) | pd.isnull(cols))
            if not valid.all():
                rows = rows[valid]
                cols = cols[valid]
                scores = scores.values[valid]

            all_rows.append(rows.astype(np.int32))
            all_cols.append(cols.astype(np.int32))
            all_scores.append(scores if isinstance(scores, np.ndarray) else scores.values)

            if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == n_chunks:
                logger.info(
                    f"  Processed chunk {chunk_idx + 1}/{n_chunks} "
                    f"(rows {start:,}–{end:,})"
                )

        # --- Build user mappings if not already set ---
        if not self.user_to_idx:
            unique_users = animelist_df['user_id'].unique()
            self.user_to_idx = {int(u): i for i, u in enumerate(unique_users)}
            self.idx_to_user = {i: int(u) for u, i in self.user_to_idx.items()}

        # --- Concatenate and build sparse matrix ---
        logger.info("Concatenating chunks and building sparse matrix...")
        all_rows_np = np.concatenate(all_rows).astype(np.int32)
        all_cols_np = np.concatenate(all_cols).astype(np.int32)
        all_scores_np = np.concatenate(all_scores).astype(np.float32)

        del all_rows, all_cols, all_scores  # Free list RAM

        n_users = len(self.user_to_idx)
        n_items = len(self.implicit_anime_to_idx)

        self.implicit_matrix = csr_matrix(
            (all_scores_np, (all_rows_np, all_cols_np)),
            shape=(n_users, n_items),
        )

        logger.info(f"Implicit matrix built: {self.implicit_matrix.nnz:,} entries ({n_items} items)")
        return self.implicit_matrix

    def build_implicit_matrix_from_csv(
        self,
        filepath: Union[str, Path],
        watching_status_df: Optional[pd.DataFrame] = None,
        chunk_size: int = 500_000,
        min_implicit_anime_interactions: int = 50,
    ) -> csr_matrix:
        """
        Build the implicit matrix directly from a filtered animelist CSV.

        Khác với bản gốc: anime_set KHÔNG còn bị ràng buộc bởi explicit anime_to_idx.
        Anime có đủ implicit interactions (>= min_implicit_anime_interactions) sẽ được
        thêm vào implicit_anime_to_idx, kể cả khi chúng vắng trong explicit ratings
        (ví dụ: One Piece — đang phát sóng, ít explicit rating nhưng 130K+ watch entries).

        Sau khi build, self.implicit_anime_to_idx / implicit_idx_to_anime chứa mapping
        cho implicit matrix. self.anime_to_idx không bị thay đổi (vẫn dùng cho CF).
        """
        filepath = Path(filepath)
        logger.info("Building implicit matrix from CSV: %s", filepath)

        if not self.user_to_idx:
            raise ValueError(
                "Explicit rating mappings (user_to_idx) must be built before reading implicit CSV data."
            )

        user_set = set(self.user_to_idx.keys())

        # --- Pass 1: đếm implicit interactions per anime ---
        logger.info("  Pass 1: counting implicit interactions per anime...")
        implicit_anime_counts: Dict[int, int] = defaultdict(int)
        for chunk in pd.read_csv(
            filepath,
            chunksize=chunk_size,
            usecols=["user_id", "anime_id"],
            dtype={"user_id": "int32", "anime_id": "int32"},
        ):
            chunk = chunk[chunk["user_id"].isin(user_set)]
            if chunk.empty:
                continue
            for aid, cnt in chunk["anime_id"].value_counts(sort=False).items():
                implicit_anime_counts[int(aid)] += int(cnt)

        # Union: explicit anime + anime có đủ implicit interactions
        explicit_anime_set = set(self.anime_to_idx.keys())
        implicit_only = {
            aid
            for aid, cnt in implicit_anime_counts.items()
            if cnt >= min_implicit_anime_interactions and aid not in explicit_anime_set
        }
        logger.info(
            "  Explicit anime: %d | Implicit-only additions: %d (threshold=%d)",
            len(explicit_anime_set),
            len(implicit_only),
            min_implicit_anime_interactions,
        )

        # Build implicit-specific mappings (superset của explicit)
        all_implicit_anime = sorted(explicit_anime_set | implicit_only)
        self.implicit_anime_to_idx: Dict[int, int] = {aid: i for i, aid in enumerate(all_implicit_anime)}
        self.implicit_idx_to_anime: Dict[int, int] = {i: aid for aid, i in self.implicit_anime_to_idx.items()}
        n_implicit_items = len(self.implicit_anime_to_idx)
        logger.info("  Implicit matrix will have %d items", n_implicit_items)

        anime_set = set(self.implicit_anime_to_idx.keys())

        status_weights = {
            1: 0.8,
            2: 1.0,
            3: 0.5,
            4: 0.2,
            6: 0.1,
        }

        row_parts = []
        col_parts = []
        score_parts = []

        # --- Pass 2: materialize scores ---
        logger.info("  Pass 2: materializing implicit scores...")
        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                filepath,
                chunksize=chunk_size,
                usecols=["user_id", "anime_id", "watching_status", "watched_episodes"],
                dtype=_ANIMELIST_DTYPES,
            ),
            start=1,
        ):
            chunk = chunk.loc[
                chunk["user_id"].isin(user_set) & chunk["anime_id"].isin(anime_set)
            ]
            if chunk.empty:
                continue

            status_w = chunk["watching_status"].map(status_weights).fillna(0.1)
            episode_w = (
                np.log1p(chunk["watched_episodes"].fillna(0).astype(np.float32)) / 10
            ).clip(0, 1)
            scores = (status_w * 0.6 + episode_w * 0.4).astype(np.float32)

            row_parts.append(
                chunk["user_id"].map(self.user_to_idx).to_numpy(dtype=np.int32, copy=False)
            )
            col_parts.append(
                chunk["anime_id"].map(self.implicit_anime_to_idx).to_numpy(dtype=np.int32, copy=False)
            )
            score_parts.append(
                scores.to_numpy(dtype=np.float32, copy=False)
            )

            if chunk_idx % 20 == 0:
                logger.info("  Materialized implicit rows for %d chunks...", chunk_idx)

        if not score_parts:
            self.implicit_matrix = csr_matrix(
                (self.n_users, n_implicit_items),
                dtype=np.float32,
            )
            logger.info("Implicit matrix built: 0 entries")
            return self.implicit_matrix

        row_indices = np.concatenate(row_parts)
        col_indices = np.concatenate(col_parts)
        scores = np.concatenate(score_parts)

        self.implicit_matrix = csr_matrix(
            (scores, (row_indices, col_indices)),
            shape=(self.n_users, n_implicit_items),
        )

        logger.info(f"Implicit matrix built: {self.implicit_matrix.nnz:,} entries ({n_implicit_items} items)")
        return self.implicit_matrix

    # ------------------------------------------------------------------
    # Remaining methods — unchanged
    # ------------------------------------------------------------------

    def get_user_ratings(self, user_id: int) -> Dict[int, float]:
        if user_id not in self.user_to_idx:
            return {}
        user_idx = self.user_to_idx[user_id]
        user_row = self.user_item_matrix[user_idx].toarray().flatten()
        return {
            self.idx_to_anime[item_idx]: float(rating)
            for item_idx, rating in enumerate(user_row)
            if rating > 0
        }

    def get_item_ratings(self, anime_id: int) -> Dict[int, float]:
        if anime_id not in self.anime_to_idx:
            return {}
        item_idx = self.anime_to_idx[anime_id]
        item_col = self.item_user_matrix[item_idx].toarray().flatten()
        return {
            self.idx_to_user[user_idx]: float(rating)
            for user_idx, rating in enumerate(item_col)
            if rating > 0
        }

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[csr_matrix, csr_matrix]:
        """Split rating matrix into train and test sets."""
        if self.user_item_matrix is None:
            raise ValueError("Rating matrix not built. Call build_rating_matrix first.")

        np.random.seed(random_state)
        rows, cols = self.user_item_matrix.nonzero()
        data = self.user_item_matrix.data.copy()

        n_ratings = len(data)
        n_test = int(n_ratings * test_size)

        test_indices = np.random.choice(n_ratings, size=n_test, replace=False)
        train_mask = np.ones(n_ratings, dtype=bool)
        train_mask[test_indices] = False

        train_matrix = csr_matrix(
            (data[train_mask], (rows[train_mask], cols[train_mask])),
            shape=self.user_item_matrix.shape,
        )
        test_matrix = csr_matrix(
            (data[test_indices], (rows[test_indices], cols[test_indices])),
            shape=self.user_item_matrix.shape,
        )

        logger.info(f"Train: {train_matrix.nnz:,} ratings, Test: {test_matrix.nnz:,} ratings")
        return train_matrix, test_matrix

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.user_item_matrix is not None:
            save_npz(directory / "user_item_matrix.npz", self.user_item_matrix)
        if self.item_user_matrix is not None:
            save_npz(directory / "item_user_matrix.npz", self.item_user_matrix)
        if self.implicit_matrix is not None:
            save_npz(directory / "implicit_matrix.npz", self.implicit_matrix)

        mappings = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'anime_to_idx': self.anime_to_idx,
            'idx_to_anime': self.idx_to_anime,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_ratings': self.n_ratings,
        }
        with open(directory / "mappings.pkl", 'wb') as f:
            pickle.dump(mappings, f)
        logger.info(f"MatrixBuilder saved to {directory}")

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)

        if (directory / "user_item_matrix.npz").exists():
            self.user_item_matrix = load_npz(directory / "user_item_matrix.npz")
        if (directory / "item_user_matrix.npz").exists():
            self.item_user_matrix = load_npz(directory / "item_user_matrix.npz")
        if (directory / "implicit_matrix.npz").exists():
            self.implicit_matrix = load_npz(directory / "implicit_matrix.npz")

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
    from data_loader import DataLoader

    loader = DataLoader()
    loader.load_ratings(sample=True)
    loader.load_animelist(sample=True)

    builder = MatrixBuilder()
    builder.build_rating_matrix(loader.ratings_df)
    builder.build_implicit_matrix(loader.animelist_df)

    train, test = builder.get_train_test_split()
    builder.save(CACHE_DIR / "matrices")
    print("\nMatrix Builder Test Complete!")
