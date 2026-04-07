"""
ALS Implicit Feedback Model.

GPU-accelerated version with implicit library GPU support.
Improvements over baseline:
  - Explicit negative sampling (dropped/abandoned anime penalized)
  - Temporal weighting (time-decay: recent interactions weighted higher)
  - Re-ranking with MMR diversity (reduces filter bubble in top-K)
  - Evaluation pipeline (Precision@K, Recall@K, NDCG@K, Coverage)
"""
import numpy as np
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from scipy.sparse import csr_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import model_config
from device_config import get_device, is_gpu_available, get_implicit_als_class, log_gpu_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ALSImplicit:
    """
    Alternating Least Squares for Implicit Feedback.

    GPU-accelerated using implicit library GPU support.
    Uses watching behavior (episodes watched, status) as implicit feedback.
    Based on "Collaborative Filtering for Implicit Feedback Datasets" (Hu et al., 2008)

    Attributes:
        user_factors: User latent factor matrix
        item_factors: Item latent factor matrix
        device: Device for computation
    """

    def __init__(
        self,
        n_factors: int = None,
        n_iterations: int = None,
        regularization: float = None,
        alpha: float = 40.0,
        device: str = None,
        # --- Negative sampling ---
        negative_weight: float = 0.5,
        # --- Temporal weighting ---
        use_temporal_weighting: bool = False,
        temporal_decay_days: float = 365.0,
        # --- Re-ranking diversity ---
        diversity_lambda: float = 0.3,
    ):
        """
        Initialize ALS Implicit model.

        Args:
            n_factors: Number of latent factors
            n_iterations: Number of ALS iterations
            regularization: Regularization parameter
            alpha: Confidence scaling factor
            device: Device for computation ("cuda" or "cpu"). Auto-detected if None.
            negative_weight: Confidence penalty for explicit negatives (dropped items).
                Range [0, 1]. 0 = ignore, 1 = same weight as positives.
                Recommended: 0.3-0.7.
            use_temporal_weighting: Apply exponential time-decay to confidence scores.
            temporal_decay_days: Half-life in days. An interaction (decay_days) ago
                gets 50% the confidence of a today interaction.
            diversity_lambda: MMR trade-off [0, 1].
                0 = pure relevance, 1 = pure diversity. Recommended: 0.2-0.4.
        """
        self.n_factors = n_factors or model_config.implicit_factors
        self.n_iterations = n_iterations or model_config.implicit_iterations
        self.regularization = regularization or model_config.implicit_regularization
        self.alpha = alpha
        self.negative_weight = negative_weight
        self.use_temporal_weighting = use_temporal_weighting
        self.temporal_decay_days = temporal_decay_days
        self.diversity_lambda = diversity_lambda

        # Device configuration
        self.device = device or get_device()
        self._use_gpu = is_gpu_available() and self.device == "cuda"

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        # External library model
        self._implicit_model = None
        self._is_gpu_model = False

        # Mappings
        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}

        logger.info(
            f"ALSImplicit initialized | device={self.device} | "
            f"neg_weight={negative_weight} | temporal={use_temporal_weighting} | "
            f"diversity_lambda={diversity_lambda}"
        )

    # ------------------------------------------------------------------
    # PUBLIC: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        implicit_matrix: csr_matrix,
        anime_to_idx: Dict[int, int],
        idx_to_anime: Dict[int, int],
        user_to_idx: Dict[int, int] = None,
        idx_to_user: Dict[int, int] = None,
        use_gpu: bool = None,
        negative_matrix: csr_matrix = None,
        interaction_dates: np.ndarray = None,
    ) -> "ALSImplicit":
        """
        Fit the ALS model on implicit feedback data.

        Args:
            implicit_matrix: User-item matrix with implicit feedback scores [n_users x n_items].
            anime_to_idx: Anime ID to index mapping.
            idx_to_anime: Index to anime ID mapping.
            user_to_idx: User ID to index mapping.
            idx_to_user: Index to user ID mapping.
            use_gpu: Whether to use GPU acceleration. Auto-detected if None.
            negative_matrix: Optional user-item matrix marking explicit negatives
                (e.g. dropped/abandoned anime). Non-zero = negative signal.
                Same shape as implicit_matrix.
            interaction_dates: Optional array of days-since-interaction for each
                non-zero entry in implicit_matrix.data (same length as .data).
                Used when use_temporal_weighting=True.

        Returns:
            Self for chaining.
        """
        self.anime_to_idx = anime_to_idx
        self.idx_to_anime = idx_to_anime
        self.user_to_idx = user_to_idx or {}
        self.idx_to_user = idx_to_user or {}

        n_users, n_items = implicit_matrix.shape
        logger.info(f"Fitting ALS Implicit on {n_users} users x {n_items} items...")

        if use_gpu is None:
            use_gpu = self._use_gpu

        start_time = time.time()
        log_gpu_memory("Before ALS training: ")

        # Step 1: Apply temporal weighting before building confidence
        matrix = self._apply_temporal_weighting(implicit_matrix, interaction_dates)

        # Step 2: Apply explicit negative sampling
        matrix = self._apply_negative_sampling(matrix, negative_matrix)

        try:
            self._fit_with_implicit_library(matrix, use_gpu)
        except ImportError:
            logger.warning("implicit library not found. Using custom implementation.")
            self._fit_custom(matrix, n_users, n_items)

        elapsed = time.time() - start_time
        logger.info(f"ALS training completed in {elapsed:.2f}s")
        log_gpu_memory("After ALS training: ")

        logger.info("ALS Implicit fitted successfully")
        return self

    # ------------------------------------------------------------------
    # IMPROVEMENT 1: Temporal Weighting
    # ------------------------------------------------------------------

    def _apply_temporal_weighting(
        self,
        matrix: csr_matrix,
        interaction_dates: Optional[np.ndarray],
    ) -> csr_matrix:
        """
        Apply exponential time-decay to implicit feedback scores.

        Each interaction's confidence is scaled by:
            decay = exp(-ln(2) * days_ago / half_life)

        So an interaction `temporal_decay_days` ago gets 50% weight compared
        to a today interaction. decay is clipped to [0.1, 1.0] to ensure
        very old interactions are not zeroed out entirely.

        Args:
            matrix: Original implicit feedback matrix.
            interaction_dates: Days-since-interaction per non-zero entry
                               (same length as matrix.data).
                               If None or use_temporal_weighting=False,
                               returns matrix unchanged.

        Returns:
            Matrix with time-decayed confidence scores.
        """
        if not self.use_temporal_weighting or interaction_dates is None:
            return matrix

        if len(interaction_dates) != len(matrix.data):
            logger.warning(
                f"interaction_dates length {len(interaction_dates)} != "
                f"matrix.data length {len(matrix.data)}. Skipping temporal weighting."
            )
            return matrix

        matrix = matrix.copy().astype(np.float32)
        decay = np.exp(-np.log(2) * interaction_dates / self.temporal_decay_days)
        decay = np.clip(decay, 0.1, 1.0).astype(np.float32)
        matrix.data *= decay

        logger.info(
            f"Temporal weighting applied | half-life={self.temporal_decay_days}d | "
            f"mean_decay={decay.mean():.3f} | min_decay={decay.min():.3f}"
        )
        return matrix

    # ------------------------------------------------------------------
    # IMPROVEMENT 2: Explicit Negative Sampling
    # ------------------------------------------------------------------

    def _apply_negative_sampling(
        self,
        matrix: csr_matrix,
        negative_matrix: Optional[csr_matrix],
    ) -> csr_matrix:
        """
        Inject explicit negatives into the confidence matrix.

        For dropped/abandoned items confidence is reduced by negative_weight.
        Only positions without positive signal are penalized — items the user
        dropped after watching many episodes may still carry partial positive
        signal and are left unchanged.

        Args:
            matrix: Positive implicit feedback matrix.
            negative_matrix: Binary or weighted matrix where non-zero = explicit
                             negative (e.g. status=dropped). Same shape as matrix.

        Returns:
            Matrix with negative signals incorporated.
        """
        if negative_matrix is None or self.negative_weight == 0:
            return matrix

        if matrix.shape != negative_matrix.shape:
            logger.warning(
                f"negative_matrix shape {negative_matrix.shape} != "
                f"matrix shape {matrix.shape}. Skipping negative sampling."
            )
            return matrix

        matrix = matrix.copy().astype(np.float32)
        neg = negative_matrix.astype(np.float32)
        neg_scaled = neg.multiply(self.negative_weight)

        # Only subtract where there is no positive signal
        positive_mask = (matrix > 0).astype(np.float32)
        neg_only = neg_scaled - neg_scaled.multiply(positive_mask)

        matrix = matrix - neg_only
        # Ensure values stay above a small epsilon (library expects >= 0)
        matrix.data = np.maximum(matrix.data, 1e-4)

        n_negatives = (neg_only > 0).nnz
        logger.info(
            f"Negative sampling applied | negatives={n_negatives:,} | "
            f"weight={self.negative_weight}"
        )
        return matrix

    # ------------------------------------------------------------------
    # Internal fit helpers
    # ------------------------------------------------------------------

    def _fit_with_implicit_library(
        self,
        matrix: csr_matrix,
        use_gpu: bool
    ) -> None:
        """Fit using the implicit library with auto GPU detection."""
        ALSClass, is_gpu = get_implicit_als_class()

        if use_gpu and is_gpu:
            logger.info("ALS GPU enabled - using implicit GPU implementation")
            self._is_gpu_model = True
            self._implicit_model = ALSClass(
                factors=self.n_factors,
                iterations=self.n_iterations,
                regularization=self.regularization
            )
        else:
            logger.info("Using implicit CPU ALS implementation")
            self._is_gpu_model = False
            self._implicit_model = ALSClass(
                factors=self.n_factors,
                iterations=self.n_iterations,
                regularization=self.regularization
            )

        # implicit expects item-user matrix
        item_user_matrix = matrix.T.tocsr()

        # Scale by confidence
        item_user_matrix.data = 1 + self.alpha * item_user_matrix.data

        self._implicit_model.fit(item_user_matrix, show_progress=True)

        self.user_factors = self._implicit_model.user_factors
        self.item_factors = self._implicit_model.item_factors

        # Handle GPU arrays if necessary
        if hasattr(self.user_factors, 'to_numpy'):
            self.user_factors = self.user_factors.to_numpy()
        if hasattr(self.item_factors, 'to_numpy'):
            self.item_factors = self.item_factors.to_numpy()

    def _fit_custom(
        self,
        matrix: csr_matrix,
        n_users: int,
        n_items: int
    ) -> None:
        """Custom ALS implementation (fallback — use only without implicit library)."""
        logger.info("Using custom ALS implementation...")

        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        lambda_eye = self.regularization * np.eye(self.n_factors)

        for iteration in range(self.n_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}")

            item_factors_T = self.item_factors.T
            YtY = item_factors_T @ self.item_factors

            for u in range(n_users):
                item_indices = matrix[u].indices
                confidences = 1 + self.alpha * matrix[u].data
                Y_u = self.item_factors[item_indices]
                Cu_diag = np.diag(confidences - 1)
                A = YtY + Y_u.T @ Cu_diag @ Y_u + lambda_eye
                p_u = np.ones(len(item_indices))
                b = Y_u.T @ (confidences * p_u)
                self.user_factors[u] = np.linalg.solve(A, b)

            matrix_T = matrix.T.tocsr()
            XtX = self.user_factors.T @ self.user_factors

            for i in range(n_items):
                user_indices = matrix_T[i].indices
                if len(user_indices) == 0:
                    continue
                confidences = 1 + self.alpha * matrix_T[i].data
                X_i = self.user_factors[user_indices]
                Ci_diag = np.diag(confidences - 1)
                A = XtX + X_i.T @ Ci_diag @ X_i + lambda_eye
                p_i = np.ones(len(user_indices))
                b = X_i.T @ (confidences * p_i)
                self.item_factors[i] = np.linalg.solve(A, b)

    # ------------------------------------------------------------------
    # Helper: resolve factor orientation
    # ------------------------------------------------------------------

    def _resolve_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (user_factors, item_factors) with correct orientation.

        The implicit library sometimes returns swapped factors depending on
        how the item-user matrix was passed. This detects and corrects the swap.
        """
        n_users = len(self.user_to_idx)
        n_items = len(self.anime_to_idx)

        if (self.user_factors.shape[0] == n_items
                and self.item_factors.shape[0] == n_users):
            return self.item_factors, self.user_factors

        return self.user_factors, self.item_factors

    # ------------------------------------------------------------------
    # PUBLIC: recommend_for_user
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_known: bool = True,
        known_items: set = None,
        use_diversity: bool = True,
    ) -> List[Dict]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID.
            top_k: Number of final recommendations.
            exclude_known: Whether to exclude already-watched anime.
            known_items: Set of known anime IDs.
            use_diversity: Apply MMR re-ranking for diversity. When True,
                a larger candidate pool (top_k * 5) is scored before re-ranking.

        Returns:
            List of dicts: [{'mal_id': int, 'score': float}, ...]
        """
        if user_id not in self.user_to_idx:
            logger.debug(f"User {user_id} not in training data — cold-start fallback")
            return self._get_popular_items(top_k, exclude=known_items)

        user_idx = self.user_to_idx[user_id]
        actual_user_factors, actual_item_factors = self._resolve_factors()

        if user_idx >= actual_user_factors.shape[0]:
            logger.debug(f"User index {user_idx} out of bounds")
            return self._get_popular_items(top_k, exclude=known_items)

        # Fetch a larger candidate pool so MMR has enough items to pick from
        candidate_k = top_k * 5 if use_diversity else top_k * 2

        if self._implicit_model is not None:
            try:
                item_ids, scores = self._implicit_model.recommend(
                    user_idx,
                    None,
                    N=candidate_k,
                    filter_already_liked_items=False
                )
                candidates = []
                for idx, score in zip(item_ids, scores):
                    if idx not in self.idx_to_anime:
                        continue
                    anime_id = self.idx_to_anime[idx]
                    if exclude_known and known_items and anime_id in known_items:
                        continue
                    candidates.append({
                        'mal_id': anime_id,
                        'score': float(score),
                        '_idx': int(idx),
                    })
            except Exception as e:
                logger.debug(f"implicit recommend failed: {e}, falling back to custom")
                candidates = self._score_all_items(
                    user_idx, actual_user_factors, actual_item_factors,
                    candidate_k, exclude_known, known_items
                )
        else:
            candidates = self._score_all_items(
                user_idx, actual_user_factors, actual_item_factors,
                candidate_k, exclude_known, known_items
            )

        if not candidates:
            return []

        # Apply MMR diversity re-ranking
        if use_diversity and len(candidates) > top_k:
            candidates = self._mmr_rerank(candidates, top_k, actual_item_factors)
        else:
            candidates = candidates[:top_k]

        for c in candidates:
            c.pop('_idx', None)

        return candidates

    def _score_all_items(
        self,
        user_idx: int,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        top_n: int,
        exclude_known: bool,
        known_items: set,
    ) -> List[Dict]:
        """Score all items for a user and return top_n candidates."""
        user_vec = user_factors[user_idx]
        scores = item_factors @ user_vec

        if exclude_known and known_items:
            for aid in known_items:
                if aid in self.anime_to_idx:
                    scores[self.anime_to_idx[aid]] = -np.inf

        top_indices = scores.argsort()[::-1][:top_n * 2]
        results = []
        for idx in top_indices:
            if scores[idx] == -np.inf or idx not in self.idx_to_anime:
                continue
            anime_id = self.idx_to_anime[idx]
            if exclude_known and known_items and anime_id in known_items:
                continue
            results.append({
                'mal_id': anime_id,
                'score': float(scores[idx]),
                '_idx': int(idx),
            })
            if len(results) >= top_n:
                break
        return results

    def _get_popular_items(self, top_k: int, exclude: set = None) -> List[Dict]:
        """
        Cold-start fallback: rank items by L2 norm of item factors.
        Items with larger norm tend to appear in many users' histories.
        """
        if self.item_factors is None:
            return []
        _, actual_item_factors = self._resolve_factors()
        popularity = np.linalg.norm(actual_item_factors, axis=1)
        top_indices = popularity.argsort()[::-1]
        results = []
        for idx in top_indices:
            anime_id = self.idx_to_anime.get(int(idx))
            if anime_id is None:
                continue
            if exclude and anime_id in exclude:
                continue
            results.append({'mal_id': anime_id, 'score': float(popularity[idx])})
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # IMPROVEMENT 3: MMR Re-ranking for Diversity
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        candidates: List[Dict],
        top_k: int,
        item_factors: np.ndarray,
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance (MMR) re-ranking.

        Iteratively selects items that balance relevance (ALS score) with
        dissimilarity to already-selected items:

            MMR(i) = (1 - lambda) * relevance(i)
                     - lambda * max_{j in selected} cosine_sim(i, j)

        A higher diversity_lambda pushes results toward more varied genres/types.

        Args:
            candidates: Scored candidates list (must include '_idx' field).
            top_k: Number of items to select.
            item_factors: Item latent factor matrix for similarity computation.

        Returns:
            Re-ranked list of top_k diverse recommendations.
        """
        if not candidates:
            return []

        # Normalize ALS scores to [0, 1]
        scores = np.array([c['score'] for c in candidates])
        score_range = scores.max() - scores.min()
        norm_scores = (scores - scores.min()) / score_range if score_range > 0 else np.ones(len(scores))

        # Build normalized item vectors for cosine similarity
        indices = []
        for c in candidates:
            idx = c.get('_idx')
            if idx is None and c['mal_id'] in self.anime_to_idx:
                idx = self.anime_to_idx[c['mal_id']]
            indices.append(idx)

        vecs = np.zeros((len(candidates), item_factors.shape[1]))
        for i, idx in enumerate(indices):
            if idx is not None and idx < item_factors.shape[0]:
                vecs[i] = item_factors[idx]

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs_norm = vecs / norms

        selected = []
        remaining = list(range(len(candidates)))
        lam = self.diversity_lambda

        while len(selected) < top_k and remaining:
            if not selected:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: norm_scores[i])
            else:
                selected_vecs = np.array([vecs_norm[i] for i in selected])
                # Shape: [len(remaining), len(selected)]
                sim_to_selected = vecs_norm[remaining] @ selected_vecs.T
                max_sim = sim_to_selected.max(axis=1)

                mmr_scores = (1 - lam) * norm_scores[remaining] - lam * max_sim
                best = remaining[int(np.argmax(mmr_scores))]

            selected.append(best)
            remaining.remove(best)

        logger.debug(
            f"MMR re-ranking: {len(candidates)} candidates -> {len(selected)} selected "
            f"(lambda={lam})"
        )
        return [candidates[i] for i in selected]

    # ------------------------------------------------------------------
    # PUBLIC: get_similar_items
    # ------------------------------------------------------------------

    def get_similar_items(
        self,
        anime_id: int,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get similar items based on latent factors.

        Args:
            anime_id: Anime ID.
            top_k: Number of similar items.

        Returns:
            List of similar anime dicts: [{'mal_id': int, 'similarity': float}, ...]
        """
        if anime_id not in self.anime_to_idx:
            return []

        item_idx = self.anime_to_idx[anime_id]

        if self._implicit_model is not None:
            try:
                similar_items = self._implicit_model.similar_items(item_idx, N=top_k + 1)
                # Handle new implicit lib format: returns (ids_array, scores_array)
                if (isinstance(similar_items, tuple) and len(similar_items) == 2
                        and hasattr(similar_items[0], '__len__')):
                    similar_items = list(zip(similar_items[0], similar_items[1]))
                results = []
                for idx, score in similar_items:
                    idx = int(idx)
                    if idx == item_idx:
                        continue
                    if idx not in self.idx_to_anime:
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

        _, item_factors = self._resolve_factors()
        item_vec = item_factors[item_idx]

        norms = np.linalg.norm(item_factors, axis=1)
        norms[norms == 0] = 1
        normalized = item_factors / norms[:, np.newaxis]

        item_norm = np.linalg.norm(item_vec)
        item_vec_norm = item_vec / item_norm if item_norm > 0 else item_vec

        similarities = normalized @ item_vec_norm
        similar_indices = similarities.argsort()[::-1]

        results = []
        for idx in similar_indices:
            idx = int(idx)
            if idx == item_idx:
                continue
            if idx not in self.idx_to_anime:
                continue
            if len(results) >= top_k:
                break
            results.append({
                'mal_id': self.idx_to_anime[idx],
                'similarity': float(similarities[idx])
            })

        return results

    # ------------------------------------------------------------------
    # IMPROVEMENT 4: Evaluation Pipeline
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_interactions: Dict[int, List[int]],
        k_values: List[int] = None,
        exclude_known: bool = True,
        train_interactions: Dict[int, List[int]] = None,
        sample_users: int = None,
    ) -> Dict:
        """
        Offline evaluation with Precision@K, Recall@K, NDCG@K, and Coverage.

        Recommended split strategy: train on older interactions, test on
        the most recent ones (temporal split), not random split — this
        better simulates real recommendation scenarios.

        Args:
            test_interactions: Dict {user_id: [held_out_anime_ids]}.
                Only users present in training data are evaluated.
            k_values: List of K values to evaluate. Default: [5, 10, 20].
            exclude_known: Exclude train items when generating recommendations.
            train_interactions: Dict {user_id: [train_anime_ids]} used to
                build known_items for exclusion. If None, exclusion is skipped.
            sample_users: Evaluate on a random subset of users for speed.
                None = all users.

        Returns:
            Dict keyed by K with sub-dicts of metrics, e.g.:
            {
                10: {
                    'precision': 0.12,
                    'recall': 0.08,
                    'ndcg': 0.15,
                    'coverage': 0.34,
                    'n_users_evaluated': 1200,
                },
                ...
                'eval_time_s': 42.1,
                'n_users': 1200,
            }
        """
        if k_values is None:
            k_values = [5, 10, 20]

        max_k = max(k_values)
        all_anime_ids = set(self.anime_to_idx.keys())

        eval_users = [
            uid for uid in test_interactions
            if uid in self.user_to_idx and len(test_interactions[uid]) > 0
        ]

        if not eval_users:
            logger.warning("No evaluable users found (no overlap between test set and training users).")
            return {}

        if sample_users and sample_users < len(eval_users):
            rng = np.random.default_rng(42)
            eval_users = rng.choice(eval_users, size=sample_users, replace=False).tolist()

        logger.info(f"Evaluating on {len(eval_users)} users | K={k_values}")

        metrics: Dict[int, Dict[str, List[float]]] = {
            k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values
        }
        recommended_items_all: set = set()

        start = time.time()
        for i, user_id in enumerate(eval_users):
            if i % 500 == 0:
                logger.info(f"  [{i}/{len(eval_users)}] evaluating users...")

            ground_truth = set(test_interactions[user_id])
            known = set(train_interactions[user_id]) if train_interactions else None

            # Fetch max_k recs once; slice per K value below
            recs = self.recommend_for_user(
                user_id,
                top_k=max_k,
                exclude_known=exclude_known,
                known_items=known,
                use_diversity=False,   # Evaluate base model scores, not diversity variant
            )
            rec_ids = [r['mal_id'] for r in recs]
            recommended_items_all.update(rec_ids)

            for k in k_values:
                rec_at_k = rec_ids[:k]
                hits = [1 if r in ground_truth else 0 for r in rec_at_k]

                precision = sum(hits) / k
                recall = sum(hits) / len(ground_truth) if ground_truth else 0.0
                ndcg = self._ndcg(hits, k)

                metrics[k]['precision'].append(precision)
                metrics[k]['recall'].append(recall)
                metrics[k]['ndcg'].append(ndcg)

        elapsed = time.time() - start
        coverage = len(recommended_items_all) / len(all_anime_ids) if all_anime_ids else 0.0

        results: Dict = {}
        for k in k_values:
            results[k] = {
                'precision': float(np.mean(metrics[k]['precision'])),
                'recall': float(np.mean(metrics[k]['recall'])),
                'ndcg': float(np.mean(metrics[k]['ndcg'])),
                'coverage': coverage,
                'n_users_evaluated': len(eval_users),
            }
            logger.info(
                f"@{k:2d} | P={results[k]['precision']:.4f} | "
                f"R={results[k]['recall']:.4f} | "
                f"NDCG={results[k]['ndcg']:.4f} | "
                f"Coverage={coverage:.4f}"
            )

        results['eval_time_s'] = elapsed
        results['n_users'] = len(eval_users)
        logger.info(f"Evaluation done in {elapsed:.1f}s")
        return results

    @staticmethod
    def _ndcg(hits: List[int], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain @ K.

            DCG  = sum_i  hits[i] / log2(i + 2)
            IDCG = DCG of ideal ranking (all relevant items at top)
            NDCG = DCG / IDCG
        """
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        n_relevant = sum(hits)
        if n_relevant == 0:
            return 0.0
        ideal_hits = [1] * n_relevant + [0] * (k - n_relevant)
        idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    # ------------------------------------------------------------------
    # PUBLIC: save / load
    # ------------------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'n_factors': self.n_factors,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'negative_weight': self.negative_weight,
            'use_temporal_weighting': self.use_temporal_weighting,
            'temporal_decay_days': self.temporal_decay_days,
            'diversity_lambda': self.diversity_lambda,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'anime_to_idx': self.anime_to_idx,
            'idx_to_anime': self.idx_to_anime,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
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
        self.negative_weight = state.get('negative_weight', 0.5)
        self.use_temporal_weighting = state.get('use_temporal_weighting', False)
        self.temporal_decay_days = state.get('temporal_decay_days', 365.0)
        self.diversity_lambda = state.get('diversity_lambda', 0.3)
        self.user_factors = state['user_factors']
        self.item_factors = state['item_factors']
        self.anime_to_idx = state['anime_to_idx']
        self.idx_to_anime = state['idx_to_anime']
        self.user_to_idx = state['user_to_idx']
        self.idx_to_user = state['idx_to_user']

        logger.info(f"ALSImplicit loaded from {filepath}")
        return self


# ----------------------------------------------------------------------
# Quick smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader, MatrixBuilder

    loader = DataLoader()
    loader.load_ratings(sample=False)  # Load full ratings for evaluation
    loader.load_animelist(sample=False)

    builder = MatrixBuilder()
    builder.build_rating_matrix(loader.ratings_df)
    builder.build_implicit_matrix(loader.animelist_df)

    # Build negative matrix from dropped anime (status=6) if available
    neg_matrix = getattr(builder, 'dropped_matrix', None)

    als = ALSImplicit(
        n_factors=30,
        n_iterations=10,
        negative_weight=0.4,
        use_temporal_weighting=False,   # set True + pass interaction_dates when available
        diversity_lambda=0.3,
    )
    als.fit(
        builder.implicit_matrix,
        builder.anime_to_idx,
        builder.idx_to_anime,
        builder.user_to_idx,
        builder.idx_to_user,
        negative_matrix=neg_matrix,
    )

    # Recommendations with MMR diversity
    test_user = list(builder.user_to_idx.keys())[0]
    print(f"\nRecommendations (MMR diversity) for user {test_user}:")
    for r in als.recommend_for_user(test_user, top_k=5, use_diversity=True):
        print(f"  Anime {r['mal_id']}: score={r['score']:.4f}")

    # Similar items
    test_anime = list(builder.anime_to_idx.keys())[0]
    print(f"\nSimilar items to anime {test_anime}:")
    for item in als.get_similar_items(test_anime, top_k=5):
        print(f"  Anime {item['mal_id']}: similarity={item['similarity']:.4f}")

    # Evaluation: temporal split — train on first half, test on second half
    all_users = list(builder.user_to_idx.keys())
    train_interactions: Dict[int, List[int]] = {}
    test_interactions: Dict[int, List[int]] = {}

    for uid in all_users:
        uidx = builder.user_to_idx[uid]
        row = builder.implicit_matrix[uidx]
        item_ids = [builder.idx_to_anime[i] for i in row.indices]
        if len(item_ids) >= 4:
            split = len(item_ids) // 2
            train_interactions[uid] = item_ids[:split]
            test_interactions[uid] = item_ids[split:]
        else:
            train_interactions[uid] = item_ids

    if test_interactions:
        print("\nRunning evaluation (sample=200 users)...")
        results = als.evaluate(
            test_interactions=test_interactions,
            k_values=[5, 10, 20],
            train_interactions=train_interactions,
            sample_users=1000,
        )
        for k in [5, 10, 20]:
            if k in results:
                m = results[k]
                print(f"  @{k:2d}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
                      f"NDCG={m['ndcg']:.4f}  Coverage={m['coverage']:.4f}")