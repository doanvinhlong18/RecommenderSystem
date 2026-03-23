"""
Matrix Factorization models (SVD, ALS, PyTorch-based).

GPU-accelerated version with PyTorch support.
"""
import numpy as np
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
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
    Matrix Factorization using SVD, ALS, or PyTorch.

    GPU-accelerated with PyTorch support.
    Decomposes user-item matrix into latent factors.

    Attributes:
        user_factors: User latent factor matrix
        item_factors: Item latent factor matrix
        user_bias: User bias terms
        item_bias: Item bias terms
        global_mean: Global mean rating
        device: Device for computation
    """

    def __init__(
        self,
        n_factors: int = None,
        n_epochs: int = None,
        learning_rate: float = None,
        regularization: float = None,
        method: str = "svd",
        device: str = None,
        use_torch: bool = None
    ):
        """
        Initialize MatrixFactorization.

        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            learning_rate: Learning rate for SGD
            regularization: Regularization parameter
            method: "svd", "als", or "torch"
            device: Device for computation ("cuda" or "cpu"). Auto-detected if None.
            use_torch: Whether to use PyTorch. Auto-detected if None.
        """
        self.n_factors = n_factors or model_config.svd_factors
        self.n_epochs = n_epochs or model_config.svd_epochs
        self.learning_rate = learning_rate or model_config.svd_lr
        self.regularization = regularization or model_config.svd_reg
        self.method = method

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

        # Mappings
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
        verbose: bool = True
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

        # Calculate global mean
        self.global_mean = user_item_matrix.data.mean()

        start_time = time.time()

        # Log GPU memory before training
        log_gpu_memory("Before MF training: ")

        if self.use_torch and TORCH_AVAILABLE:
            logger.info("Using PyTorch-based Matrix Factorization")
            self._fit_torch(user_item_matrix, n_users, n_items, verbose)
        elif self.method == "als":
            self._fit_als(user_item_matrix, n_users, n_items, verbose)
        else:
            self._fit_svd(user_item_matrix, n_users, n_items, verbose)

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
        verbose: bool
    ) -> None:
        """Fit using SGD-based SVD."""
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Get non-zero entries
        rows, cols = matrix.nonzero()
        ratings = matrix.data

        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.arange(len(ratings))
            np.random.shuffle(indices)

            total_loss = 0

            for idx in indices:
                u, i = rows[idx], cols[idx]
                r = ratings[idx]

                # Predict
                pred = (
                    self.global_mean +
                    self.user_bias[u] +
                    self.item_bias[i] +
                    np.dot(self.user_factors[u], self.item_factors[i])
                )

                # Error
                error = r - pred
                total_loss += error ** 2

                # Update biases
                self.user_bias[u] += self.learning_rate * (
                    error - self.regularization * self.user_bias[u]
                )
                self.item_bias[i] += self.learning_rate * (
                    error - self.regularization * self.item_bias[i]
                )

                # Update factors
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
        verbose: bool
    ) -> None:
        """Fit using Alternating Least Squares."""
        # Initialize factors
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        lambda_reg = self.regularization

        for epoch in range(self.n_epochs):
            # Fix items, update users
            for u in range(n_users):
                # Get items rated by user
                item_indices = matrix[u].indices
                if len(item_indices) == 0:
                    continue

                item_matrix = self.item_factors[item_indices]
                ratings = matrix[u].data - self.global_mean - self.item_bias[item_indices]

                # Solve least squares
                A = item_matrix.T @ item_matrix + lambda_reg * np.eye(self.n_factors)
                b = item_matrix.T @ ratings
                self.user_factors[u] = np.linalg.solve(A, b)

                # Update user bias
                pred = item_matrix @ self.user_factors[u]
                self.user_bias[u] = (ratings - pred).mean()

            # Fix users, update items
            matrix_T = matrix.T.tocsr()
            for i in range(n_items):
                # Get users who rated this item
                user_indices = matrix_T[i].indices
                if len(user_indices) == 0:
                    continue

                user_matrix = self.user_factors[user_indices]
                ratings = matrix_T[i].data - self.global_mean - self.user_bias[user_indices]

                # Solve least squares
                A = user_matrix.T @ user_matrix + lambda_reg * np.eye(self.n_factors)
                b = user_matrix.T @ ratings
                self.item_factors[i] = np.linalg.solve(A, b)

                # Update item bias
                pred = user_matrix @ self.item_factors[i]
                self.item_bias[i] = (ratings - pred).mean()

            if verbose and (epoch + 1) % 5 == 0:
                # Calculate RMSE
                rmse = self._calculate_rmse(matrix)
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

    def _calculate_rmse(self, matrix: csr_matrix) -> float:
        """Calculate RMSE on the training data."""
        rows, cols = matrix.nonzero()
        predictions = []

        for u, i in zip(rows, cols):
            pred = self.predict_rating_by_idx(u, i)
            predictions.append(pred)

        predictions = np.array(predictions)
        actuals = matrix.data

        return np.sqrt(np.mean((predictions - actuals) ** 2))

    def predict_rating_by_idx(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using matrix indices."""
        pred = (
            self.global_mean +
            self.user_bias[user_idx] +
            self.item_bias[item_idx] +
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return float(np.clip(pred, 1, 10))

    def predict_rating(self, user_id: int, anime_id: int) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User ID
            anime_id: Anime ID

        Returns:
            Predicted rating
        """
        if user_id not in self.user_to_idx:
            return self.global_mean + (
                self.item_bias[self.anime_to_idx[anime_id]]
                if anime_id in self.anime_to_idx else 0
            )

        if anime_id not in self.anime_to_idx:
            return self.global_mean + self.user_bias[self.user_to_idx[user_id]]

        user_idx = self.user_to_idx[user_id]
        item_idx = self.anime_to_idx[anime_id]

        return self.predict_rating_by_idx(user_idx, item_idx)

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True,
        rated_items: set = None
    ) -> List[Dict]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            rated_items: Set of already rated anime IDs

        Returns:
            List of recommendation dictionaries
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return []

        user_idx = self.user_to_idx[user_id]

        # Compute all predictions
        user_vec = self.user_factors[user_idx]
        predictions = (
            self.global_mean +
            self.user_bias[user_idx] +
            self.item_bias +
            self.item_factors @ user_vec
        )

        # Get top items
        if exclude_rated and rated_items:
            exclude_indices = {
                self.anime_to_idx[aid]
                for aid in rated_items
                if aid in self.anime_to_idx
            }

            # Mask excluded items
            for idx in exclude_indices:
                predictions[idx] = -np.inf

        top_indices = predictions.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if predictions[idx] == -np.inf:
                continue
            results.append({
                'mal_id': self.idx_to_anime[idx],
                'predicted_rating': float(np.clip(predictions[idx], 1, 10))
            })

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
        item_vec = self.item_factors[item_idx]

        # Compute cosine similarity
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1
        normalized = self.item_factors / norms[:, np.newaxis]

        item_norm = np.linalg.norm(item_vec)
        if item_norm > 0:
            item_vec_norm = item_vec / item_norm
        else:
            item_vec_norm = item_vec

        similarities = normalized @ item_vec_norm

        # Get top similar (excluding self)
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
            'method': self.method,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'global_mean': self.global_mean,
            'anime_to_idx': self.anime_to_idx,
            'idx_to_anime': self.idx_to_anime,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"MatrixFactorization saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "MatrixFactorization":
        """Load model from file."""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.n_factors = state['n_factors']
        self.method = state['method']
        self.user_factors = state['user_factors']
        self.item_factors = state['item_factors']
        self.user_bias = state['user_bias']
        self.item_bias = state['item_bias']
        self.global_mean = state['global_mean']
        self.anime_to_idx = state['anime_to_idx']
        self.idx_to_anime = state['idx_to_anime']
        self.user_to_idx = state['user_to_idx']
        self.idx_to_user = state['idx_to_user']

        logger.info(f"MatrixFactorization loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Test Matrix Factorization
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader, MatrixBuilder

    loader = DataLoader()
    loader.load_ratings(sample=True)

    builder = MatrixBuilder()
    builder.build_rating_matrix(loader.ratings_df)

    # Test SVD
    mf = MatrixFactorization(n_factors=50, n_epochs=10, method="svd")
    mf.fit(
        builder.user_item_matrix,
        builder.anime_to_idx,
        builder.idx_to_anime,
        builder.user_to_idx,
        builder.idx_to_user
    )

    # Test prediction
    test_user = list(builder.user_to_idx.keys())[0]
    test_anime = list(builder.anime_to_idx.keys())[0]
    pred = mf.predict_rating(test_user, test_anime)
    print(f"\nPredicted rating for user {test_user}, anime {test_anime}: {pred:.2f}")

    # Test recommendations
    recs = mf.recommend_for_user(test_user, top_k=5)
    print(f"\nRecommendations for user {test_user}:")
    for rec in recs:
        print(f"  Anime {rec['mal_id']}: predicted={rec['predicted_rating']:.2f}")
