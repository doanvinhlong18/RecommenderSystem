"""
Item-Based Collaborative Filtering.

Fix so với bản gốc:
  BUG OOM: _build_faiss_index() gọi item_user_matrix.toarray()
  → 17K items × 320K users × 4 bytes = 21.8 GB RAM → crash

  FIX: ItemBasedCF KHÔNG dùng FAISS trên user-space vectors.
  Lý do:
    - item_user_matrix.T có dimension = 320K → FAISS với dim=320K vô nghĩa
    - ALS implicit đã có FAISS built-in qua thư viện implicit
    - Với 17K items, sklearn cosine_similarity on-demand đủ nhanh (~50ms)
    - Nếu muốn FAISS cho CF item similarity: cần reduced vectors (PCA hoặc ALS factors)
      → xem phần _build_faiss_index_from_factors() bên dưới

  Thêm: _build_faiss_index_from_factors() — optional, dùng khi có ALS item factors
  Nếu truyền item_factors (50-dim từ ALS) thì build FAISS trên đó thay vì user-space
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
    Item-Based Collaborative Filtering.

    FAISS usage:
      - Không dùng FAISS trên user-space vectors (dim=320K → OOM)
      - Optional: _build_faiss_index_from_factors(item_factors) với 50-dim ALS factors
        Chỉ hữu ích nếu cần item similarity nhanh và không muốn dùng implicit library
    """

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors
        self.item_similarity: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[csr_matrix] = None
        self.mean_ratings: Optional[np.ndarray] = None

        self.anime_to_idx: Dict[int, int] = {}
        self.idx_to_anime: Dict[int, int] = {}
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}

        # FAISS index — chỉ được build nếu gọi _build_faiss_index_from_factors()
        # KHÔNG build từ user-space vectors (OOM với 320K users)
        self._faiss_index = None
        self._faiss_item_factors: Optional[np.ndarray] = None

    def fit(
        self,
        user_item_matrix: csr_matrix,
        anime_to_idx: Dict[int, int],
        idx_to_anime: Dict[int, int],
        user_to_idx: Dict[int, int] = None,
        idx_to_user: Dict[int, int] = None,
        compute_full_similarity: bool = False,
        item_factors: Optional[np.ndarray] = None,  # ALS factors nếu có
    ) -> "ItemBasedCF":
        """
        Fit Item-Based CF.

        Parameters
        ----------
        user_item_matrix : csr_matrix (n_users, n_items)
        compute_full_similarity : tính full item×item matrix (memory intensive, ~2.3GB với 17K items)
        item_factors : optional np.ndarray (n_items, n_factors) từ ALS
                       Nếu có, build FAISS index trên 50-dim factors thay vì user-space

        NOTE về FAISS:
          KHÔNG gọi .toarray() trên item_user_matrix để build FAISS
          item_user_matrix.T.toarray() = (17K, 320K) = 21.8 GB → OOM crash
          Thay vào đó:
            - Nếu compute_full_similarity=True: dùng sklearn cosine_similarity (chunk-wise)
            - Nếu False: dùng on-demand cosine similarity khi query
            - Nếu item_factors (ALS 50-dim): build FAISS IndexFlatIP trên đó (~3MB)
        """
        self.user_item_matrix = user_item_matrix
        self.anime_to_idx = anime_to_idx
        self.idx_to_anime = idx_to_anime
        self.user_to_idx = user_to_idx or {}
        self.idx_to_user = idx_to_user or {}

        n_users, n_items = user_item_matrix.shape
        logger.info(f"Fitting Item-Based CF: {n_users} users × {n_items} items")

        # Compute mean ratings per item
        item_sums = np.array(user_item_matrix.sum(axis=0)).flatten()
        item_counts = np.array((user_item_matrix > 0).sum(axis=0)).flatten()
        item_counts[item_counts == 0] = 1
        self.mean_ratings = item_sums / item_counts

        item_user_matrix = user_item_matrix.T.tocsr()

        if compute_full_similarity:
            # Full similarity matrix — dùng sklearn (chunk-wise, không load toàn bộ dense)
            # Chỉ nên dùng khi n_items nhỏ (<5K)
            logger.warning(
                "compute_full_similarity=True với %d items có thể tốn RAM (~%.1f GB). "
                "Dùng False và on-demand similarity để an toàn hơn.",
                n_items,
                n_items * n_items * 4 / 1e9,
            )
            logger.info(
                "Computing full item-item similarity matrix (sklearn, sparse-aware)..."
            )
            self.item_similarity = cosine_similarity(
                item_user_matrix, dense_output=True
            )
            logger.info(f"Item similarity matrix shape: {self.item_similarity.shape}")
        else:
            logger.info("Dùng on-demand cosine similarity (không tính full matrix)")
            self.item_similarity = None

            # Nếu có ALS item factors → build FAISS index nhỏ (50-dim)
            if item_factors is not None:
                self._build_faiss_index_from_factors(item_factors)

        logger.info("Item-Based CF fitted successfully")
        return self

    def _build_faiss_index_from_factors(self, item_factors: np.ndarray) -> None:
        """
        Build FAISS IndexFlatIP từ ALS item factors (50-dim).

        Đây là cách đúng để dùng FAISS cho ItemBasedCF:
          - item_factors: (17K, 50-dim) thay vì (17K, 320K-dim)
          - IndexFlatIP: 17K × 50 × 4 bytes = 3.4 MB (trivial)
          - Exact search, nhanh hơn cosine_similarity on-demand

        Parameters
        ----------
        item_factors : np.ndarray (n_items, n_factors) — từ ALS model sau khi train
        """
        try:
            import faiss
        except ImportError:
            logger.info("faiss chưa cài — bỏ qua FAISS cho ItemBasedCF")
            return

        from sklearn.preprocessing import normalize as sk_normalize

        vecs = sk_normalize(item_factors.astype(np.float32))
        n, d = vecs.shape

        # 17K items, 50-dim → IndexFlatIP là exact và đủ nhanh (~0.1ms)
        index = faiss.IndexFlatIP(d)
        index.add(vecs)

        self._faiss_index = index
        self._faiss_item_factors = vecs
        logger.info(f"FAISS IndexFlatIP built từ item factors: {n} items, dim={d}")

    def get_similar_items(
        self,
        anime_id: int,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Lấy top_k anime tương tự theo CF (user co-occurrence).

        Ưu tiên:
          1. Full similarity matrix (nếu đã tính trong fit)
          2. FAISS trên ALS item factors (nếu có)
          3. On-demand cosine similarity (fallback an toàn)

        Parameters
        ----------
        anime_id : anime ID
        top_k    : số kết quả

        Returns
        -------
        List[Dict]: mal_id, similarity
        """
        if anime_id not in self.anime_to_idx:
            logger.warning(f"Anime {anime_id} không có trong training data")
            return []

        idx = self.anime_to_idx[anime_id]

        if self.item_similarity is not None:
            # Option 1: Precomputed full similarity
            similarities = self.item_similarity[idx]
            similar_indices = similarities.argsort()[::-1][1 : top_k + 1]
            similar_scores = similarities[similar_indices]

        elif self._faiss_index is not None:
            # Option 2: FAISS trên ALS item factors (50-dim)
            query = self._faiss_item_factors[idx].reshape(1, -1)
            scores, indices = self._faiss_index.search(query, top_k + 1)
            mask = indices[0] != idx
            similar_indices = indices[0][mask][:top_k]
            similar_scores = scores[0][mask][:top_k]

        else:
            # Option 3: On-demand cosine similarity (an toàn, không OOM)
            item_user_matrix = self.user_item_matrix.T.tocsr()
            query = item_user_matrix[idx]
            # cosine_similarity với sparse matrix — không toarray() toàn bộ
            similarities = cosine_similarity(query, item_user_matrix).flatten()
            similar_indices = similarities.argsort()[::-1][1 : top_k + 1]
            similar_scores = similarities[similar_indices]

        results = []
        for sim_idx, score in zip(similar_indices, similar_scores):
            sim_idx = int(sim_idx)
            if sim_idx not in self.idx_to_anime:
                continue
            results.append(
                {
                    "mal_id": self.idx_to_anime[sim_idx],
                    "similarity": float(score),
                }
            )

        return results

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True,
        rated_items: set = None,
    ) -> List[Dict]:
        """
        Gợi ý cho user dựa trên item-based CF.

        Predict score = mean-centered weighted sum của similar items.
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} không có trong training data")
            return []

        user_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        rated_indices = user_ratings.indices
        rated_values = user_ratings.data

        if len(rated_indices) == 0:
            return []

        # Predict scores for all unrated items
        scores = {}

        for item_idx, rating in zip(rated_indices, rated_values):
            sim_recs = self.get_similar_items(
                self.idx_to_anime[item_idx], top_k=self.k_neighbors
            )
            for rec in sim_recs:
                mal_id = rec["mal_id"]
                target_idx = self.anime_to_idx[mal_id]
                if target_idx in rated_indices:
                    continue
                sim = rec["similarity"]
                center_rating = rating - self.mean_ratings[item_idx]
                if mal_id not in scores:
                    scores[mal_id] = {"num": 0.0, "denom": 0.0, "idx": target_idx}
                scores[mal_id]["num"] += sim * center_rating
                scores[mal_id]["denom"] += abs(sim)

        # Build results
        results = []
        exclude_set = rated_items or set()
        for mal_id, s in scores.items():
            if exclude_rated and mal_id in exclude_set:
                continue
            if s["denom"] == 0:
                continue
            pred = self.mean_ratings[s["idx"]] + s["num"] / s["denom"]
            results.append(
                {
                    "mal_id": mal_id,
                    "predicted_rating": float(np.clip(pred, 1, 10)),
                    "similarity": float(s["num"] / s["denom"]),
                }
            )

        results.sort(key=lambda x: -x["predicted_rating"])
        return results[:top_k]

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Không lưu _faiss_index (không serializable)
        state = {k: v for k, v in self.__dict__.items() if k not in ("_faiss_index",)}
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"ItemBasedCF saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "ItemBasedCF":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self._faiss_index = None
        # Nếu có _faiss_item_factors đã lưu → rebuild FAISS
        if self._faiss_item_factors is not None:
            self._build_faiss_index_from_factors(self._faiss_item_factors)
        logger.info(f"ItemBasedCF loaded from {filepath}")
        return self
