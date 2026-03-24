"""
Collaborative matrix-factorization models.

Supports legacy explicit-feedback baselines (SVD / ALS) and a BPR backend
powered by the `implicit` library for top-K ranking.
"""
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import model_config
from device_config import get_device, is_gpu_available, log_gpu_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PyTorch-based Matrix Factorization Model
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU acceleration disabled for MatrixFactorization.")


class TorchMatrixFactorization(nn.Module):
    """PyTorch-based Matrix Factorization for GPU training."""

    def __init__(self, n_users: int, n_items: int, n_factors: int):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        dot_product = (user_emb * item_emb).sum(dim=1)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()

        return self.global_bias + user_b + item_b + dot_product


class MatrixFactorization:
    """
    Matrix factorization wrapper used by the collaborative module.

    The class keeps the historical `MatrixFactorization` interface so the rest
    of the codebase can continue to call `fit`, `recommend_for_user`,
    `get_similar_items`, `save`, and `load` without knowing whether the backend
    is explicit-feedback MF or implicit-feedback BPR.
    """

    def __init__(
        self,
        n_factors: int = None,
        n_epochs: int = None,
        learning_rate: float = None,
        regularization: float = None,
        method: str = "bpr",
        rating_positive_threshold: float = None,
        verify_negative_samples: bool = None,
        use_implicit_signal: bool = None,
        warm_start_from_als: bool = None,
    ):
        self.method = method.lower()

        if self.method == "bpr":
            self.n_factors = model_config.bpr_factors if n_factors is None else n_factors
            self.n_epochs = model_config.bpr_iterations if n_epochs is None else n_epochs
            self.learning_rate = (
                model_config.bpr_learning_rate if learning_rate is None else learning_rate
            )
            self.regularization = (
                model_config.bpr_regularization if regularization is None else regularization
            )
        else:
            self.n_factors = model_config.svd_factors if n_factors is None else n_factors
            self.n_epochs = model_config.svd_epochs if n_epochs is None else n_epochs
            self.learning_rate = model_config.svd_lr if learning_rate is None else learning_rate
            self.regularization = model_config.svd_reg if regularization is None else regularization

        self.rating_positive_threshold = (
            model_config.bpr_positive_rating_threshold
            if rating_positive_threshold is None
            else rating_positive_threshold
        )
        self.verify_negative_samples = (
            model_config.bpr_verify_negative_samples
            if verify_negative_samples is None
            else verify_negative_samples
        )
        self.use_implicit_signal = (
            model_config.bpr_use_implicit_signal
            if use_implicit_signal is None
            else use_implicit_signal
        )
        self.warm_start_from_als = (
            model_config.bpr_warm_start_from_als
            if warm_start_from_als is None
            else warm_start_from_als
        )

        # Device configuration
        self.device = device or get_device()

        # Auto-detect whether to use PyTorch
        if use_torch is None:
            self.use_torch = TORCH_AVAILABLE and (method == "torch" or is_gpu_available())
        else:
            self.use_torch = use_torch and TORCH_AVAILABLE

        # PyTorch model
        self._torch_model: Optional[TorchMatrixFactorization] = None

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self._implicit_model = None
        self._train_interactions: Optional[csr_matrix] = None

        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}

        logger.info(f"MatrixFactorization initialized with device: {self.device}, use_torch: {self.use_torch}")

    def fit(
        self,
        user_item_matrix: csr_matrix,
        anime_to_idx: Dict[int, int],
        idx_to_anime: Dict[int, int],
        user_to_idx: Dict[int, int] = None,
        idx_to_user: Dict[int, int] = None,
        verbose: bool = True,
        implicit_matrix: Optional[csr_matrix] = None,
        implicit_model=None,
        use_gpu: bool = False,
    ) -> "MatrixFactorization":
        """
        Fit the matrix factorization model.

        Args:
            user_item_matrix: Sparse user-item rating matrix
            anime_to_idx: Anime ID to index mapping
            idx_to_anime: Index to anime ID mapping
            user_to_idx: User ID to index mapping
            idx_to_user: Index to user ID mapping
            verbose: Whether to print progress
            implicit_matrix: Optional implicit interaction matrix for BPR
            implicit_model: Optional trained ALS implicit model for warm-start
            use_gpu: Whether to use GPU acceleration when the backend supports it

        Returns:
            Self for chaining
        """
        self.anime_to_idx = anime_to_idx
        self.idx_to_anime = idx_to_anime
        self.user_to_idx = user_to_idx or {}
        self.idx_to_user = idx_to_user or {}

        n_users, n_items = user_item_matrix.shape
        logger.info(f"Fitting {self.method.upper()} with {self.n_factors} factors...")
        logger.info(f"Matrix shape: {n_users} users x {n_items} items")
        logger.info(f"Training on device: {self.device.upper()}")

        self.global_mean = float(user_item_matrix.data.mean()) if user_item_matrix.nnz else 0.0

        start_time = time.time()

        # Log GPU memory before training
        log_gpu_memory("Before MF training: ")

        if self.use_torch and TORCH_AVAILABLE:
            logger.info("Using PyTorch-based Matrix Factorization")
            self._fit_torch(user_item_matrix, n_users, n_items, verbose)
        elif self.method == "als":
            self._fit_als(user_item_matrix, n_users, n_items, verbose)
        elif self.method == "svd":
            self._fit_svd(user_item_matrix, n_users, n_items, verbose)
        elif self.method == "bpr":
            self._fit_bpr(
                user_item_matrix,
                implicit_matrix=implicit_matrix,
                implicit_model=implicit_model,
                verbose=verbose,
                use_gpu=use_gpu,
            )
        else:
            raise ValueError(f"Unsupported matrix factorization method: {self.method}")

        elapsed = time.time() - start_time
        logger.info(f"Matrix Factorization training completed in {elapsed:.2f}s")

        # Log GPU memory after training
        log_gpu_memory("After MF training: ")

        logger.info("Matrix Factorization fitted successfully")
        return self

    def _fit_torch(
        self,
        matrix: csr_matrix,
        n_users: int,
        n_items: int,
        verbose: bool
    ) -> None:
        """Fit using PyTorch with GPU support."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device)
        logger.info(f"PyTorch training on: {device}")

        # Create PyTorch model
        self._torch_model = TorchMatrixFactorization(n_users, n_items, self.n_factors)
        self._torch_model.to(device)
        self._torch_model.global_bias.data.fill_(self.global_mean)

        # Prepare training data
        rows, cols = matrix.nonzero()
        ratings = matrix.data.astype(np.float32)

        user_ids = torch.LongTensor(rows)
        item_ids = torch.LongTensor(cols)
        rating_values = torch.FloatTensor(ratings)

        dataset = TensorDataset(user_ids, item_ids, rating_values)
        batch_size = 4096 if self.device == "cuda" else 1024
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self._torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization
        )
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(self.n_epochs):
            total_loss = 0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_ratings = batch_ratings.to(device)

                optimizer.zero_grad()
                predictions = self._torch_model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            rmse = np.sqrt(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        # Extract factors to numpy arrays
        self._torch_model.eval()
        with torch.no_grad():
            self.user_factors = self._torch_model.user_embedding.weight.cpu().numpy()
            self.item_factors = self._torch_model.item_embedding.weight.cpu().numpy()
            self.user_bias = self._torch_model.user_bias.weight.cpu().numpy().flatten()
            self.item_bias = self._torch_model.item_bias.weight.cpu().numpy().flatten()
            self.global_mean = self._torch_model.global_bias.item()

    def _fit_svd(
        self,
        matrix: csr_matrix,
        n_users: int,
        n_items: int,
        verbose: bool,
    ) -> None:
        """Fit using SGD-based SVD."""
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        rows, cols = matrix.nonzero()
        ratings = matrix.data

        for epoch in range(self.n_epochs):
            indices = np.arange(len(ratings))
            np.random.shuffle(indices)

            total_loss = 0.0

            for idx in indices:
                u, i = rows[idx], cols[idx]
                r = ratings[idx]

                pred = (
                    self.global_mean
                    + self.user_bias[u]
                    + self.item_bias[i]
                    + np.dot(self.user_factors[u], self.item_factors[i])
                )

                error = r - pred
                total_loss += error ** 2

                self.user_bias[u] += self.learning_rate * (
                    error - self.regularization * self.user_bias[u]
                )
                self.item_bias[i] += self.learning_rate * (
                    error - self.regularization * self.item_bias[i]
                )

                user_factor = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (
                    error * self.item_factors[i] - self.regularization * self.user_factors[u]
                )
                self.item_factors[i] += self.learning_rate * (
                    error * user_factor - self.regularization * self.item_factors[i]
                )

            rmse = np.sqrt(total_loss / len(ratings))
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

    def _fit_als(
        self,
        matrix: csr_matrix,
        n_users: int,
        n_items: int,
        verbose: bool,
    ) -> None:
        """Fit using alternating least squares on explicit ratings."""
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        lambda_reg = self.regularization
        matrix_T = matrix.T.tocsr()

        for epoch in range(self.n_epochs):
            for u in range(n_users):
                item_indices = matrix[u].indices
                if len(item_indices) == 0:
                    continue

                item_matrix = self.item_factors[item_indices]
                ratings = matrix[u].data - self.global_mean - self.item_bias[item_indices]

                A = item_matrix.T @ item_matrix + lambda_reg * np.eye(self.n_factors)
                b = item_matrix.T @ ratings
                self.user_factors[u] = np.linalg.solve(A, b)

                pred = item_matrix @ self.user_factors[u]
                self.user_bias[u] = float((ratings - pred).mean())

            for i in range(n_items):
                user_indices = matrix_T[i].indices
                if len(user_indices) == 0:
                    continue

                user_matrix = self.user_factors[user_indices]
                ratings = matrix_T[i].data - self.global_mean - self.user_bias[user_indices]

                A = user_matrix.T @ user_matrix + lambda_reg * np.eye(self.n_factors)
                b = user_matrix.T @ ratings
                self.item_factors[i] = np.linalg.solve(A, b)

                pred = user_matrix @ self.item_factors[i]
                self.item_bias[i] = float((ratings - pred).mean())

            if verbose and (epoch + 1) % 5 == 0:
                rmse = self._calculate_rmse(matrix)
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

    def _fit_bpr(
        self,
        matrix: csr_matrix,
        implicit_matrix: Optional[csr_matrix],
        implicit_model,
        verbose: bool,
        use_gpu: bool,
    ) -> None:
        """Fit using BPR from the implicit library."""
        try:
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as exc:
            raise ImportError(
                "The implicit package is required for BPR training. "
                "Install project requirements inside the Python 3.11 environment first."
            ) from exc

        self._train_interactions = self._build_bpr_interactions(matrix, implicit_matrix)
        if self._train_interactions.nnz == 0:
            raise ValueError("BPR training matrix is empty after building positive interactions.")

        logger.info(
            "BPR positives: %s interactions across %s users x %s items",
            self._train_interactions.nnz,
            self._train_interactions.shape[0],
            self._train_interactions.shape[1],
        )

        self._implicit_model = BayesianPersonalizedRanking(
            factors=self.n_factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.n_epochs,
            use_gpu=use_gpu,
            verify_negative_samples=self.verify_negative_samples,
            random_state=42,
        )

        if self.warm_start_from_als and implicit_model is not None:
            self._apply_als_warm_start(implicit_model, self._train_interactions)

        self._implicit_model.fit(self._train_interactions, show_progress=verbose)

        self.user_factors = np.asarray(self._implicit_model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(self._implicit_model.item_factors, dtype=np.float32)
        self.user_bias = None
        self.item_bias = None

    def _build_bpr_interactions(
        self,
        user_item_matrix: csr_matrix,
        implicit_matrix: Optional[csr_matrix],
    ) -> csr_matrix:
        """Create a binary interaction matrix for BPR training."""
        explicit_positive = user_item_matrix.copy().tocsr().astype(np.float32)
        if explicit_positive.nnz:
            explicit_positive.data = (
                explicit_positive.data >= self.rating_positive_threshold
            ).astype(np.float32)
            explicit_positive.eliminate_zeros()

        implicit_binary = None
        if self.use_implicit_signal and implicit_matrix is not None:
            if implicit_matrix.shape != user_item_matrix.shape:
                raise ValueError(
                    "Implicit matrix shape must match the rating matrix shape for BPR training."
                )
            implicit_binary = implicit_matrix.copy().tocsr().astype(np.float32)
            if implicit_binary.nnz:
                implicit_binary.data = np.ones_like(implicit_binary.data, dtype=np.float32)
                implicit_binary.eliminate_zeros()

        combined = explicit_positive
        if implicit_binary is not None:
            combined = explicit_positive.maximum(implicit_binary)

        if combined.nnz == 0:
            logger.warning(
                "No explicit positives found at threshold %.2f. Falling back to all observed ratings.",
                self.rating_positive_threshold,
            )
            fallback = user_item_matrix.copy().tocsr().astype(np.float32)
            if fallback.nnz:
                fallback.data = np.ones_like(fallback.data, dtype=np.float32)
                fallback.eliminate_zeros()
            combined = fallback.maximum(implicit_binary) if implicit_binary is not None else fallback

        return combined

    def _apply_als_warm_start(self, implicit_model, train_interactions: csr_matrix) -> None:
        """Seed BPR factors from a trained ALS implicit model."""
        als_user_factors = getattr(implicit_model, "user_factors", None)
        als_item_factors = getattr(implicit_model, "item_factors", None)

        if als_user_factors is None or als_item_factors is None:
            logger.warning("Skipping ALS warm-start because the implicit model has no factors yet.")
            return

        n_users, n_items = train_interactions.shape
        als_user_factors = np.asarray(als_user_factors, dtype=np.float32)
        als_item_factors = np.asarray(als_item_factors, dtype=np.float32)

        if als_user_factors.shape[0] == n_items and als_item_factors.shape[0] == n_users:
            als_user_factors, als_item_factors = als_item_factors, als_user_factors

        if als_user_factors.shape[0] != n_users or als_item_factors.shape[0] != n_items:
            logger.warning(
                "Skipping ALS warm-start due to factor shape mismatch: "
                "ALS users=%s, ALS items=%s, expected users=%s, expected items=%s",
                als_user_factors.shape,
                als_item_factors.shape,
                n_users,
                n_items,
            )
            return

        latent_dim = min(self.n_factors, als_user_factors.shape[1], als_item_factors.shape[1])
        user_factors = np.zeros((n_users, self.n_factors + 1), dtype=np.float32)
        item_factors = np.zeros((n_items, self.n_factors + 1), dtype=np.float32)

        user_factors[:, :latent_dim] = als_user_factors[:, :latent_dim]
        item_factors[:, :latent_dim] = als_item_factors[:, :latent_dim]
        user_factors[:, -1] = 1.0

        item_popularity = np.asarray(train_interactions.sum(axis=0)).ravel().astype(np.float32)
        if item_popularity.size and item_popularity.max() > 0:
            item_bias = np.log1p(item_popularity)
            item_factors[:, -1] = item_bias / item_bias.max()

        self._implicit_model.user_factors = user_factors
        self._implicit_model.item_factors = item_factors
        self._implicit_model._user_norms = None
        self._implicit_model._item_norms = None

        logger.info("Seeded BPR factors from ALS implicit model (%s shared dimensions)", latent_dim)

    def _calculate_rmse(self, matrix: csr_matrix) -> float:
        """Calculate RMSE on training data for legacy explicit methods."""
        rows, cols = matrix.nonzero()
        predictions = [self.predict_rating_by_idx(u, i) for u, i in zip(rows, cols)]
        actuals = matrix.data
        return float(np.sqrt(np.mean((np.asarray(predictions) - actuals) ** 2)))

    def _score_to_rating(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map an unconstrained ranking score into the familiar 1-10 range."""
        clipped = np.clip(score, -20, 20)
        ratings = 1.0 + 9.0 / (1.0 + np.exp(-clipped))
        if np.isscalar(ratings):
            return float(ratings)
        return ratings.astype(np.float32)

    def predict_score_by_idx(self, user_idx: int, item_idx: int) -> float:
        """Predict the raw model score using matrix indices."""
        if self.method == "bpr":
            return float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

        return float(
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )

    def predict_rating_by_idx(self, user_idx: int, item_idx: int) -> float:
        """Predict a 1-10 rating using matrix indices."""
        if self.method == "bpr":
            return self._score_to_rating(self.predict_score_by_idx(user_idx, item_idx))

        pred = self.predict_score_by_idx(user_idx, item_idx)
        return float(np.clip(pred, 1, 10))

    def predict_rating(self, user_id: int, anime_id: int) -> float:
        """Predict rating for a user-item pair."""
        if anime_id not in self.anime_to_idx:
            return self.global_mean if self.global_mean else 5.5

        item_idx = self.anime_to_idx[anime_id]

        if user_id not in self.user_to_idx:
            if self.method == "bpr" and self.item_factors is not None:
                return self._score_to_rating(self.item_factors[item_idx, -1])
            return self.global_mean + (self.item_bias[item_idx] if self.item_bias is not None else 0)

        user_idx = self.user_to_idx[user_id]
        return self.predict_rating_by_idx(user_idx, item_idx)

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True,
        rated_items: set = None,
    ) -> List[Dict]:
        """Generate recommendations for a user."""
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return []

        if self.method == "bpr":
            return self._recommend_bpr(user_id, top_k, exclude_rated, rated_items)

        user_idx = self.user_to_idx[user_id]
        user_vec = self.user_factors[user_idx]
        predictions = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ user_vec
        )

        if exclude_rated and rated_items:
            exclude_indices = {
                self.anime_to_idx[aid] for aid in rated_items if aid in self.anime_to_idx
            }
            for idx in exclude_indices:
                predictions[idx] = -np.inf

        top_indices = predictions.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if predictions[idx] == -np.inf:
                continue
            results.append(
                {
                    "mal_id": self.idx_to_anime[idx],
                    "predicted_rating": float(np.clip(predictions[idx], 1, 10)),
                }
            )

        return results

    def _recommend_bpr(
        self,
        user_id: int,
        top_k: int,
        exclude_rated: bool,
        rated_items: Optional[set],
    ) -> List[Dict]:
        """Generate top-K recommendations using the implicit BPR backend."""
        user_idx = self.user_to_idx[user_id]
        rated_items = set(rated_items or [])

        filter_indices = sorted(
            {self.anime_to_idx[aid] for aid in rated_items if aid in self.anime_to_idx}
        )
        filter_items = np.asarray(filter_indices, dtype=np.int32) if filter_indices else None

        user_history = self._train_interactions[user_idx] if self._train_interactions is not None else None
        candidate_count = max(top_k * 3, top_k + len(filter_indices))
        if self.item_factors is not None:
            candidate_count = min(candidate_count, self.item_factors.shape[0])

        if self._implicit_model is not None and user_history is not None:
            item_ids, scores = self._implicit_model.recommend(
                user_idx,
                user_history,
                N=candidate_count,
                filter_already_liked_items=exclude_rated,
                filter_items=filter_items,
            )
        else:
            user_vec = self.user_factors[user_idx]
            scores = self.item_factors @ user_vec

            if exclude_rated and user_history is not None:
                scores[user_history.indices] = -np.inf
            if filter_items is not None:
                scores[filter_items] = -np.inf

            item_ids = np.argsort(scores)[::-1][:candidate_count]
            scores = scores[item_ids]

        results = []
        for idx, score in zip(item_ids, scores):
            idx = int(idx)
            if idx not in self.idx_to_anime:
                continue

            anime_id = self.idx_to_anime[idx]
            if anime_id in rated_items:
                continue

            results.append(
                {
                    "mal_id": anime_id,
                    "score": float(score),
                    "predicted_rating": self._score_to_rating(score),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def get_similar_items(self, anime_id: int, top_k: int = 10) -> List[Dict]:
        """Get similar items based on latent factors."""
        if anime_id not in self.anime_to_idx:
            return []

        item_idx = self.anime_to_idx[anime_id]

        if self.method == "bpr" and self._implicit_model is not None:
            similar_ids, scores = self._implicit_model.similar_items(item_idx, N=top_k + 1)

            results = []
            for idx, score in zip(similar_ids, scores):
                idx = int(idx)
                if idx == item_idx or idx not in self.idx_to_anime:
                    continue
                results.append({"mal_id": self.idx_to_anime[idx], "similarity": float(score)})
                if len(results) >= top_k:
                    break
            return results

        if self.method == "bpr":
            factors = self.item_factors[:, :-1]
        else:
            factors = self.item_factors

        item_vec = factors[item_idx]
        norms = np.linalg.norm(factors, axis=1)
        norms[norms == 0] = 1.0
        normalized = factors / norms[:, np.newaxis]

        item_norm = np.linalg.norm(item_vec)
        item_vec_norm = item_vec / item_norm if item_norm > 0 else item_vec

        similarities = normalized @ item_vec_norm
        similar_indices = similarities.argsort()[::-1]

        results = []
        for idx in similar_indices:
            if idx == item_idx:
                continue
            if len(results) >= top_k:
                break
            results.append({"mal_id": self.idx_to_anime[idx], "similarity": float(similarities[idx])})

        return results

    def _rebuild_bpr_backend(self) -> None:
        """Reconstruct the implicit BPR model from saved factors."""
        if self.method != "bpr" or self.user_factors is None or self.item_factors is None:
            return

        try:
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError:
            self._implicit_model = None
            return

        self._implicit_model = BayesianPersonalizedRanking(
            factors=self.n_factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.n_epochs,
            verify_negative_samples=self.verify_negative_samples,
            random_state=42,
        )
        self._implicit_model.user_factors = np.asarray(self.user_factors, dtype=np.float32)
        self._implicit_model.item_factors = np.asarray(self.item_factors, dtype=np.float32)
        self._implicit_model._user_norms = None
        self._implicit_model._item_norms = None

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "method": self.method,
            "rating_positive_threshold": self.rating_positive_threshold,
            "verify_negative_samples": self.verify_negative_samples,
            "use_implicit_signal": self.use_implicit_signal,
            "warm_start_from_als": self.warm_start_from_als,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_bias": self.user_bias,
            "item_bias": self.item_bias,
            "global_mean": self.global_mean,
            "train_interactions": self._train_interactions,
            "anime_to_idx": self.anime_to_idx,
            "idx_to_anime": self.idx_to_anime,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"MatrixFactorization saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "MatrixFactorization":
        """Load model from file."""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.n_factors = state["n_factors"]
        self.n_epochs = state.get("n_epochs", self.n_epochs)
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.regularization = state.get("regularization", self.regularization)
        self.method = state["method"]
        self.rating_positive_threshold = state.get(
            "rating_positive_threshold", self.rating_positive_threshold
        )
        self.verify_negative_samples = state.get(
            "verify_negative_samples", self.verify_negative_samples
        )
        self.use_implicit_signal = state.get("use_implicit_signal", self.use_implicit_signal)
        self.warm_start_from_als = state.get("warm_start_from_als", self.warm_start_from_als)
        self.user_factors = state["user_factors"]
        self.item_factors = state["item_factors"]
        self.user_bias = state.get("user_bias")
        self.item_bias = state.get("item_bias")
        self.global_mean = state.get("global_mean", 0.0)
        self._train_interactions = state.get("train_interactions")
        self.anime_to_idx = state["anime_to_idx"]
        self.idx_to_anime = state["idx_to_anime"]
        self.user_to_idx = state["user_to_idx"]
        self.idx_to_user = state["idx_to_user"]

        if self.method == "bpr":
            self._rebuild_bpr_backend()

        logger.info(f"MatrixFactorization loaded from {filepath}")
        return self


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader, MatrixBuilder

    loader = DataLoader()
    loader.load_ratings(sample=True)

    builder = MatrixBuilder()
    builder.build_rating_matrix(loader.ratings_df)

    mf = MatrixFactorization(n_factors=50, n_epochs=5, method="bpr")
    mf.fit(
        builder.user_item_matrix,
        builder.anime_to_idx,
        builder.idx_to_anime,
        builder.user_to_idx,
        builder.idx_to_user,
    )

    test_user = list(builder.user_to_idx.keys())[0]
    test_anime = list(builder.anime_to_idx.keys())[0]
    pred = mf.predict_rating(test_user, test_anime)
    print(f"\nPredicted score proxy for user {test_user}, anime {test_anime}: {pred:.2f}")

    recs = mf.recommend_for_user(test_user, top_k=5)
    print(f"\nRecommendations for user {test_user}:")
    for rec in recs:
        print(
            f"  Anime {rec['mal_id']}: "
            f"predicted={rec['predicted_rating']:.2f}, score={rec.get('score', 0):.4f}"
        )
