import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional
from sklearn.preprocessing import normalize


class ContentBasedRecommender:

    def __init__(self):
        self.anime_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None

        self._id_to_idx: Dict[int, int] = {}
        self._idx_to_id: Dict[int, int] = {}
        self._name_to_idx: Dict[str, int] = {}

        # Cache MAL scores đã được parse để tránh parse lại nhiều lần
        self._score_cache: Dict[int, float] = {}
        # Cache cho user profile vectors để tăng tốc các lần gọi lặp
        self._user_vector_cache: Dict[int, np.ndarray] = {}

    # =====================================================
    # FIT
    # =====================================================
    def fit(self, anime_df: pd.DataFrame, embeddings: np.ndarray):

        if len(anime_df) != len(embeddings):
            raise ValueError("Anime dataframe and embedding size mismatch")

        self.anime_df = anime_df.reset_index(drop=True)

        # Normalize once here — dot product = cosine similarity
        self.embeddings = normalize(embeddings.astype(np.float32))

        ids = self.anime_df["MAL_ID"].values
        names = self.anime_df["Name"].astype(str).str.lower().values

        self._id_to_idx = {aid: i for i, aid in enumerate(ids)}
        self._idx_to_id = {i: aid for i, aid in enumerate(ids)}
        self._name_to_idx = {name: i for i, name in enumerate(names)}

        if "English name" in self.anime_df.columns:
            eng = (
                self.anime_df["English name"].fillna("").astype(str).str.lower().values
            )
            for i, name in enumerate(eng):
                if name:
                    self._name_to_idx[name] = i

        # Pre-build score cache khi fit — parse "Unknown"/"N/A" một lần duy nhất
        # FIX: tránh float() crash khi Score là string không hợp lệ
        self._score_cache = {}
        for i, row in self.anime_df.iterrows():
            self._score_cache[i] = self._parse_score(row.get("Score", 0))

        # Clear any cached user vectors when model changes
        self._user_vector_cache.clear()

        return self

    # =====================================================
    # SIMILAR ITEM
    # =====================================================
    def get_similar_anime(self, identifier, top_k=10):

        idx = self._get_idx(identifier)
        if idx is None:
            return []

        query = self.embeddings[idx]

        sims = self.embeddings @ query
        sims = np.maximum(sims, 0)
        sims[idx] = -1

        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = []
        for i in top_indices:
            row = self.anime_df.iloc[i]
            results.append(
                {
                    "mal_id": int(row["MAL_ID"]),
                    "name": row["Name"],
                    "similarity": float(
                        sims[i]
                    ),  # key là "similarity" — dùng nhất quán
                    "score": self._score_cache.get(i, 0.0),
                }
            )

        return results

    # =====================================================
    # USER PROFILE RECOMMENDATION
    # =====================================================
    def build_user_vector(
        self,
        user_ratings: Dict[int, float],
        all_ratings: Dict[int, float] = None,
        positive_percentile: float = 75.0,
        negative_percentile: float = 25.0,
        negative_weight: float = 0.4,
    ) -> np.ndarray:
        """
        Build normalized user vector từ anime ratings.

        Ngưỡng positive/negative được tính theo percentile của chính user,
        thay vì hardcode — adaptive với từng user có rating pattern khác nhau.

        Ví dụ user chỉ rate 6-8:
          positive_percentile=75 → ngưỡng ~7.5  (top 25% ratings của user)
          negative_percentile=25 → ngưỡng ~6.5  (bottom 25% ratings của user)

        Positive weighting:  rating - pos_threshold  (càng cao hơn ngưỡng → weight lớn)
        Negative weighting:  neg_threshold - rating  (càng thấp hơn ngưỡng → weight lớn)
        → Cả hai đều dùng khoảng cách từ ngưỡng, symmetric.

        Args:
            user_ratings:        {item_id: rating} dùng để build positive vector
                                 (thường là train items trong evaluation)
            all_ratings:         {item_id: rating} toàn bộ ratings của user,
                                 dùng để tính percentile và lấy negative signal.
                                 Nếu None thì fallback về user_ratings.
            positive_percentile: Percentile để tính pos_threshold (default 75.0)
                                 Items có rating >= percentile này → liked
            negative_percentile: Percentile để tính neg_threshold (default 25.0)
                                 Items có rating <= percentile này → disliked
            negative_weight:     Mức độ trừ negative vector (default 0.4)

        Returns:
            Normalized user vector (np.float32), hoặc None nếu không có liked items.
        """
        source_ratings = all_ratings if all_ratings is not None else user_ratings

        # ── Tính ngưỡng theo percentile của user ─────────────────────────────
        all_rating_vals = np.array(list(source_ratings.values()), dtype=np.float32)

        pos_threshold = float(np.percentile(all_rating_vals, positive_percentile))
        neg_threshold = float(np.percentile(all_rating_vals, negative_percentile))

        # Edge case: nếu toàn bộ rating bằng nhau thì không có signal
        # → dùng vector trung bình đơn giản
        if pos_threshold == neg_threshold:
            indices = [
                self._id_to_idx[i]
                for i in user_ratings
                if self._id_to_idx.get(i) is not None
            ]
            if not indices:
                return None
            vec = self.embeddings[indices].mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0 else vec

        # ── Positive vector ───────────────────────────────────────────────────
        # Dùng user_ratings (train items) để tránh leak test item
        pos_indices = []
        pos_weights = []

        for item_id, rating in user_ratings.items():
            if rating < pos_threshold:
                continue
            idx = self._id_to_idx.get(item_id)
            if idx is None:
                continue
            pos_indices.append(idx)
            # Khoảng cách trên ngưỡng → weight (đối xứng với negative)
            pos_weights.append(rating - pos_threshold)

        if not pos_indices:
            return None

        pos_vecs = self.embeddings[pos_indices]
        pos_w = np.array(pos_weights, dtype=np.float32)
        pos_w = np.maximum(pos_w, 1e-6)  # tránh zero weight khi đúng bằng ngưỡng
        pos_vector = np.average(pos_vecs, axis=0, weights=pos_w)
        pos_norm = np.linalg.norm(pos_vector)
        if pos_norm > 0:
            pos_vector /= pos_norm

        # ── Negative vector ───────────────────────────────────────────────────
        # Dùng source_ratings (all_ratings) để lấy đủ dislike signal
        # Không gây data leak: disliked items (bottom percentile) rất khó trùng
        # test item (top percentile) khi pos_threshold != neg_threshold
        neg_indices = []
        neg_weights = []

        for item_id, rating in source_ratings.items():
            if rating > neg_threshold:
                continue
            idx = self._id_to_idx.get(item_id)
            if idx is None:
                continue
            neg_indices.append(idx)
            # Khoảng cách dưới ngưỡng → weight (đối xứng với positive)
            neg_weights.append(neg_threshold - rating)

        user_vector = pos_vector  # default nếu không có negative signal

        if neg_indices:
            neg_vecs = self.embeddings[neg_indices]
            neg_w = np.array(neg_weights, dtype=np.float32)
            neg_w = np.maximum(neg_w, 1e-6)
            neg_vector = np.average(neg_vecs, axis=0, weights=neg_w)
            neg_norm = np.linalg.norm(neg_vector)
            if neg_norm > 0:
                neg_vector /= neg_norm

            user_vector = pos_vector - negative_weight * neg_vector
            u_norm = np.linalg.norm(user_vector)
            if u_norm > 0:
                user_vector /= u_norm

        return user_vector.astype(np.float32)

    # =====================================================
    # UTIL
    # =====================================================
    def _get_idx(self, identifier):
        if isinstance(identifier, int):
            return self._id_to_idx.get(identifier)
        name = str(identifier).lower()
        return self._name_to_idx.get(name)

    # FIX: Method tên nhất quán — hybrid.py gọi _get_idx, không phải _get_anime_idx
    # Thêm alias để backward compatible nếu có code cũ dùng tên cũ
    def _get_anime_idx(self, identifier):
        return self._get_idx(identifier)

    @staticmethod
    def _parse_score(raw) -> float:
        """
        FIX: Parse MAL score an toàn — handle "Unknown", "N/A", None, NaN.

        Args:
            raw: Raw score value từ DataFrame

        Returns:
            float score, 0.0 nếu không parse được
        """
        if raw is None:
            return 0.0
        if isinstance(raw, (int, float)):
            return (
                0.0
                if (np.isnan(raw) if isinstance(raw, float) else False)
                else float(raw)
            )
        try:
            return float(str(raw).strip())
        except (ValueError, TypeError):
            return 0.0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
