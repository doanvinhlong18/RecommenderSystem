"""
Learned Hybrid Engine — thay thế hoàn toàn HybridEngine.

Interface giống hệt HybridEngine cũ → chỉ đổi import, không sửa gì khác.

THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
  1. collect_training_data: lọc user có >= min_ratings_to_train (default 20)
     và ưu tiên top user theo số ratings nhiều nhất
  2. recommend_similar_anime: cascade thực sự
     Stage 1 — Content: top-50 candidates (FAISS)
     Stage 2 — ALS + Collab: expand candidates, boost overlap
     Stage 3 — Weighted score combination (meta-model KHÔNG áp dụng ở đây
               vì anime-to-anime không có user context)
     Stage 4 — Genre MMR diversity
  3. recommend_for_user: cascade + meta-model
     Stage 1 — Retrieval: ALS top-100 + BPR top-50 + Content top-30
     Stage 2 — Meta-model re-rank (LightGBM predict_proba)
     Stage 3 — Genre MMR diversity
  4. _content_scores: batch matmul thay vì per-item loop (~10x faster)
  5. _collab/_implicit_scores: cap top_k hợp lý tránh request quá lớn
  6. _meta_score fallback: tách helper riêng, không trùng lặp code

CÁCH DÙNG:
  Gọi từ cuối train.py (xem train_learned_hybrid ở cuối file).
  Tất cả sub-models được save cùng vào 1 thư mục.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RATING_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "rating": "int8",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — meta-model
# ─────────────────────────────────────────────────────────────────────────────


def _try_import_lgbm():
    try:
        import lightgbm as lgb

        return lgb
    except ImportError:
        return None


def _build_sklearn_meta():
    """Fallback khi không có LightGBM."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    return Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — genre diversity
# ─────────────────────────────────────────────────────────────────────────────


def _parse_genres(genres_str: str) -> Set[str]:
    if not genres_str:
        return set()
    return {g.strip().lower() for g in str(genres_str).split(",") if g.strip()}


def _genre_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _genre_mmr(
    candidates: List[Dict],
    top_k: int,
    lambda_: float,
    anime_info: Dict[int, Dict],
) -> List[Dict]:
    """
    Genre-aware Maximal Marginal Relevance.

    MMR(i) = (1-lambda)*relevance(i) - lambda*max_j(genre_jaccard(i, selected_j))
    lambda_=0 → pure relevance, lambda_=0.3 → balanced, lambda_=1 → pure diversity
    """
    if not candidates or top_k <= 0 or lambda_ == 0 or not anime_info:
        return candidates[:top_k]

    genre_sets = [
        _parse_genres(anime_info.get(c["mal_id"], {}).get("genres", ""))
        for c in candidates
    ]
    scores = np.array([c.get("hybrid_score", 0.0) for c in candidates], dtype=float)
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones(len(scores))

    selected, sel_genres, remaining = [], [], list(range(len(candidates)))
    for _ in range(min(top_k, len(candidates))):
        best_i, best_score = None, -np.inf
        for i in remaining:
            rel = (1 - lambda_) * norm[i]
            max_sim = max(
                (_genre_jaccard(genre_sets[i], sg) for sg in sel_genres), default=0.0
            )
            mmr = rel - lambda_ * max_sim
            if mmr > best_score:
                best_score, best_i = mmr, i
        if best_i is None:
            break
        selected.append(best_i)
        sel_genres.append(genre_sets[best_i])
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "content_score",
    "collaborative_score",
    "implicit_score",
    "popularity_score",
    "user_activity_bucket",  # 0-3 / 3.0
    "item_avg_score_norm",  # MAL score / 10
    "item_num_ratings_log",  # log1p / 15
]


def _activity_bucket(n: int) -> int:
    if n == 0:
        return 0
    if n < 20:
        return 1
    if n < 100:
        return 2
    return 3


def _make_feature(
    content: float,
    collab: float,
    implicit: float,
    pop: float,
    user_n_ratings: int,
    item_avg_score: float,
    item_num_ratings: int,
) -> np.ndarray:
    return np.array(
        [
            float(np.clip(content, 0, 1)),
            float(np.clip(collab, 0, 1)),
            float(np.clip(implicit, 0, 1)),
            float(np.clip(pop, 0, 1)),
            _activity_bucket(user_n_ratings) / 3.0,
            float(np.clip(item_avg_score / 10.0, 0, 1)),
            float(np.log1p(item_num_ratings) / 15.0),
        ],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LearnedHybridEngine
# ─────────────────────────────────────────────────────────────────────────────


class LearnedHybridEngine:
    """
    Learned Hybrid Engine — thay thế HybridEngine với interface giống hệt.

    Thay vì:
        final_score = w1*content + w2*collab + w3*implicit + w4*pop

    Dùng:
        final_score = LightGBM.predict_proba([7 features])[1]   (sau khi train)
        hoặc fallback weighted sum nếu meta-model chưa train.

    recommend_similar_anime (anime → anime):
        Stage 1 Content:  FAISS top-50 theo embedding similarity
        Stage 2 ALS+CF:   thêm ALS-only và Collab-only candidates
        Stage 3 Scoring:  weighted combination của item similarities
                          (meta-model KHÔNG áp dụng — cần user context)
        Stage 4 MMR:      genre diversity

    recommend_for_user (user → anime):
        Stage 1 Retrieval: ALS top-100 + BPR top-50 + Content top-30
        Stage 2 Rerank:    meta-model predict_proba (fallback weighted sum)
        Stage 3 MMR:       genre diversity
    """

    def __init__(
        self,
        content_model=None,
        collaborative_model=None,
        implicit_model=None,
        popularity_model=None,
        relevance_threshold: float = 7.0,
        n_negatives_per_positive: int = 4,
        use_lgbm: bool = True,
        diversity_lambda: float = 0.3,
        fallback_weights: Optional[Dict[str, float]] = None,
    ):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.implicit_model = implicit_model
        self.popularity_model = popularity_model

        self.relevance_threshold = relevance_threshold
        self.n_negatives_per_positive = n_negatives_per_positive
        self.use_lgbm = use_lgbm
        self.diversity_lambda = diversity_lambda

        # Fallback khi meta-model chưa train — dựa trên eval results
        self.fallback_weights = fallback_weights or {
            "content": 0.20,
            "collaborative": 0.25,
            "implicit": 0.50,
            "popularity": 0.05,
        }

        self._meta_model = None
        self._meta_trained = False
        self._feature_importance: Optional[Dict[str, float]] = None

        self._anime_info: Dict[int, Dict] = {}
        self._user_ratings: Dict[int, Dict[int, float]] = {}
        self._user_watched: Dict[int, Set[int]] = {}
        self._item_avg_score: Dict[int, float] = {}
        self._item_num_ratings: Dict[int, int] = {}

    # ─────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────

    def set_anime_info(self, anime_df: pd.DataFrame) -> None:
        for _, row in anime_df.iterrows():
            aid = int(row["MAL_ID"])
            try:
                score = (
                    float(row.get("Score", 0)) if pd.notna(row.get("Score")) else 0.0
                )
            except (ValueError, TypeError):
                score = 0.0
            self._anime_info[aid] = {
                "mal_id": aid,
                "name": row["Name"],
                "english_name": row.get("English name", row["Name"]),
                "genres": row.get("Genres", ""),
                "score": score,
                "type": row.get("Type", "Unknown"),
                "synopsis": row.get("synopsis", ""),
            }
            self._item_avg_score[aid] = score

    def set_user_history(
        self,
        user_id: int,
        ratings: Dict[int, float] = None,
        watched: Set[int] = None,
    ) -> None:
        if ratings:
            self._user_ratings[user_id] = ratings
        if watched:
            self._user_watched[user_id] = watched

    def set_item_stats(self, ratings_df: pd.DataFrame) -> None:
        """Tính item stats từ ratings_df. Gọi trước khi train meta-model."""
        stats = (
            ratings_df.groupby("anime_id")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg", "count": "cnt"})
        )
        for aid, row in stats.iterrows():
            self._item_avg_score[int(aid)] = float(row["avg"])
            self._item_num_ratings[int(aid)] = int(row["cnt"])
        logger.info(f"Item stats loaded: {len(stats)} anime")

    def set_item_stats_from_csv(
        self,
        ratings_csv_path: Union[str, Path],
        chunk_size: int = 500_000,
    ) -> None:
        """Streaming version of item stat aggregation for large ratings files."""
        rating_sums: Dict[int, float] = {}
        rating_counts: Dict[int, int] = {}

        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                ratings_csv_path,
                chunksize=chunk_size,
                usecols=["anime_id", "rating"],
                dtype={"anime_id": "int32", "rating": "int8"},
            ),
            start=1,
        ):
            chunk = chunk.loc[chunk["rating"] > 0, ["anime_id", "rating"]]
            if chunk.empty:
                continue

            grouped = chunk.groupby("anime_id")["rating"].agg(["sum", "count"])
            for anime_id, row in grouped.iterrows():
                anime_id = int(anime_id)
                rating_sums[anime_id] = rating_sums.get(anime_id, 0.0) + float(row["sum"])
                rating_counts[anime_id] = rating_counts.get(anime_id, 0) + int(row["count"])

            if chunk_idx % 20 == 0:
                logger.info("  Aggregated item stats for %d chunks...", chunk_idx)

        for anime_id, count in rating_counts.items():
            self._item_num_ratings[anime_id] = int(count)
            self._item_avg_score[anime_id] = float(rating_sums[anime_id] / count) if count else 0.0

        logger.info(f"Item stats loaded from CSV: {len(rating_counts)} anime")

    # ─────────────────────────────────────────────────────────────────
    # Score extraction (per sub-model)
    # ─────────────────────────────────────────────────────────────────

    def _content_scores(self, user_id: int, aids: List[int]) -> Dict[int, float]:
        """
        Content similarity scores — batch matmul, O(len(aids)) thay vì loop.
        Trả về dict {anime_id: score [0,1]}.
        """
        out = {a: 0.0 for a in aids}
        if not self.content_model:
            return out
        user_ratings = self._user_ratings.get(user_id, {})
        if not user_ratings:
            return out
        try:
            vec = self.content_model.build_user_vector(user_ratings=user_ratings)
            if vec is None:
                return out

            # Vectorized: lấy tất cả indices một lần, batch matmul
            emb = self.content_model.embeddings
            id2idx = self.content_model._id_to_idx

            valid_aids = []
            valid_idxs = []
            for a in aids:
                idx = id2idx.get(a)
                if idx is not None:
                    valid_aids.append(a)
                    valid_idxs.append(idx)

            if valid_aids:
                # Batch dot product: (len(valid), D) @ (D,) → (len(valid),)
                batch_emb = emb[valid_idxs]  # shape (K, D)
                raw_scores = batch_emb @ vec  # shape (K,)
                mapped = np.clip((raw_scores + 1.0) / 2.0, 0.0, 1.0)
                for a, s in zip(valid_aids, mapped):
                    out[a] = float(s)
        except Exception as e:
            logger.debug(f"content_scores error: {e}")
        return out

    def _collab_scores(self, user_id: int, aids: List[int]) -> Dict[int, float]:
        """
        Collaborative scores — gọi recommend_for_user một lần, map lên aids.
        Cap top_k ở max(200, len(aids)) để tránh request quá lớn.
        """
        out = {a: 0.0 for a in aids}
        if not self.collaborative_model:
            return out
        try:
            fetch_k = max(200, len(aids))
            recs = self.collaborative_model.recommend_for_user(
                user_id, top_k=fetch_k, exclude_rated=False
            )
            for r in recs:
                raw = r.get("score", r.get("predicted_rating", 5.0))
                norm = float(1.0 / (1.0 + np.exp(-np.clip(float(raw), -20, 20))))
                out[r["mal_id"]] = norm
        except Exception as e:
            logger.debug(f"collab_scores error: {e}")
        return out

    def _implicit_scores(self, user_id: int, aids: List[int]) -> Dict[int, float]:
        """
        ALS implicit scores — gọi recommend_for_user một lần, normalize, map lên aids.
        """
        out = {a: 0.0 for a in aids}
        if not self.implicit_model:
            return out
        try:
            fetch_k = max(200, len(aids))
            recs = self.implicit_model.recommend_for_user(
                user_id, top_k=fetch_k, exclude_known=False
            )
            rec_map = {r["mal_id"]: float(r.get("score", 0.0)) for r in recs}
            max_s = max(rec_map.values(), default=1.0)
            if max_s > 0:
                rec_map = {k: v / max_s for k, v in rec_map.items()}
            for a in aids:
                out[a] = rec_map.get(a, 0.0)
        except Exception as e:
            logger.debug(f"implicit_scores error: {e}")
        return out

    def _pop_scores(self, aids: List[int]) -> Dict[int, float]:
        out = {a: 0.0 for a in aids}
        if not self.popularity_model:
            return out
        try:
            pop = self.popularity_model.get_top_rated(top_k=200)
            n = len(pop)
            for rank, r in enumerate(pop):
                if r["mal_id"] in out:
                    out[r["mal_id"]] = (n - rank) / n
        except Exception as e:
            logger.debug(f"pop_scores error: {e}")
        return out

    def _weighted_sum(self, user_id: int, aids: List[int]) -> Dict[int, float]:
        """
        Weighted sum fallback — dùng fallback_weights.
        Tách thành helper riêng để tránh trùng lặp code.
        """
        w = self.fallback_weights
        cs = self._content_scores(user_id, aids)
        co = self._collab_scores(user_id, aids)
        im = self._implicit_scores(user_id, aids)
        po = self._pop_scores(aids)
        return {
            a: w["content"] * cs[a]
            + w["collaborative"] * co[a]
            + w["implicit"] * im[a]
            + w["popularity"] * po[a]
            for a in aids
        }

    def _build_X(self, user_id: int, aids: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Build feature matrix (len(aids), 7) cho meta-model."""
        cs = self._content_scores(user_id, aids)
        co = self._collab_scores(user_id, aids)
        im = self._implicit_scores(user_id, aids)
        po = self._pop_scores(aids)
        n_rat = len(self._user_ratings.get(user_id, {}))
        rows = [
            _make_feature(
                cs.get(a, 0),
                co.get(a, 0),
                im.get(a, 0),
                po.get(a, 0),
                n_rat,
                self._item_avg_score.get(a, 0),
                self._item_num_ratings.get(a, 0),
            )
            for a in aids
        ]
        return np.stack(rows), aids

    # ─────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────

    def collect_training_data(
        self,
        train_user_ids: List[int],
        test_interactions: Dict[int, List[int]],
        all_anime_ids: List[int],
        min_ratings_to_train: int = 20,
        top_users: int = 3000,
        rng_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thu thập (X, y) để train meta-model.

        Lọc user:
          - Chỉ lấy user có >= min_ratings_to_train ratings
          - Sort giảm dần theo số ratings → user active nhất lên đầu
          - Lấy tối đa top_users user
        """
        rng = np.random.default_rng(rng_seed)
        all_anime_set = set(all_anime_ids)

        eligible = []
        for uid in train_user_ids:
            n_rat = len(self._user_ratings.get(uid, {}))
            if n_rat >= min_ratings_to_train and uid in test_interactions:
                eligible.append((uid, n_rat))

        eligible.sort(key=lambda x: -x[1])
        users_to_use = [uid for uid, _ in eligible[:top_users]]

        logger.info(
            f"Training data: {len(eligible)} eligible users "
            f"(>= {min_ratings_to_train} ratings), dùng top {len(users_to_use)}"
        )

        X_list, y_list = [], []
        t0 = time.time()

        for i, uid in enumerate(users_to_use):
            positives = [
                a for a in test_interactions.get(uid, []) if a in self._anime_info
            ]
            if not positives:
                continue

            known = self._user_watched.get(uid, set()) | set(
                self._user_ratings.get(uid, {}).keys()
            )
            neg_pool = list(all_anime_set - known - set(positives))
            n_neg = min(len(positives) * self.n_negatives_per_positive, len(neg_pool))
            if n_neg == 0:
                continue

            negatives = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
            candidates = positives + negatives

            try:
                X_u, valid_ids = self._build_X(uid, candidates)
                pos_set = set(positives)
                y_u = np.array([1 if a in pos_set else 0 for a in valid_ids])
                X_list.append(X_u)
                y_list.append(y_u)
            except Exception as e:
                logger.debug(f"Skip user {uid}: {e}")
                continue

            if (i + 1) % 200 == 0:
                logger.info(
                    f"  {i+1}/{len(users_to_use)} users ({time.time()-t0:.0f}s)"
                )

        if not X_list:
            raise ValueError(
                "Không thu thập được training data. "
                "Kiểm tra user history, model đã train chưa, và min_ratings_to_train."
            )

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        logger.info(
            f"Dataset: {len(X)} samples, {int(y.sum())} positives "
            f"({y.mean():.1%}), features={X.shape[1]}"
        )
        return X, y

    def fit_meta_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train meta-model (LightGBM hoặc fallback LogisticRegression)."""
        lgb = _try_import_lgbm() if self.use_lgbm else None

        if lgb is not None:
            logger.info("Training LightGBM meta-model...")
            self._meta_model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                verbose=-1,
                random_state=42,
            )
        else:
            logger.info("LightGBM không có — dùng LogisticRegression fallback")
            self._meta_model = _build_sklearn_meta()

        self._meta_model.fit(self._prepare_meta_input(X), y)
        self._meta_trained = True
        self._log_feature_importance()
        logger.info("Meta-model training done.")

    def _log_feature_importance(self) -> None:
        lgb = _try_import_lgbm()
        try:
            if lgb and isinstance(self._meta_model, lgb.LGBMClassifier):
                imps = self._meta_model.feature_importances_
                total = imps.sum() or 1.0
                self._feature_importance = {
                    n: float(v / total) for n, v in zip(FEATURE_NAMES, imps)
                }
                logger.info("Feature importance:")
                for n, v in sorted(
                    self._feature_importance.items(), key=lambda x: -x[1]
                ):
                    logger.info(f"  {n:<30s} {v:.4f}  {'█' * int(v * 40)}")
        except Exception:
            pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return self._feature_importance

    def train(
        self,
        train_user_ids: List[int],
        test_interactions: Dict[int, List[int]],
        all_anime_ids: List[int],
        min_ratings_to_train: int = 20,
        top_users: int = 3000,
    ) -> None:
        """Pipeline đầy đủ: collect data + fit meta-model."""
        X, y = self.collect_training_data(
            train_user_ids,
            test_interactions,
            all_anime_ids,
            min_ratings_to_train=min_ratings_to_train,
            top_users=top_users,
        )
        self.fit_meta_model(X, y)

    # ─────────────────────────────────────────────────────────────────
    # Inference — meta-model scoring
    # ─────────────────────────────────────────────────────────────────

    def _meta_score(self, user_id: int, aids: List[int]) -> Dict[int, float]:
        """
        Score candidates bằng meta-model.
        Fallback về weighted sum nếu meta-model chưa train hoặc predict thất bại.
        """
        if not self._meta_trained or self._meta_model is None:
            return self._weighted_sum(user_id, aids)

        X, valid = self._build_X(user_id, aids)
        try:
            proba = self._meta_model.predict_proba(self._prepare_meta_input(X))[:, 1]
            return {a: float(p) for a, p in zip(valid, proba)}
        except Exception as e:
            logger.warning(f"Meta predict error: {e} — fallback to weighted sum")
            return self._weighted_sum(user_id, aids)

    def _prepare_meta_input(self, X: np.ndarray):
        """Keep LightGBM inference consistent with the feature names seen at fit time."""
        lgb = _try_import_lgbm()
        if lgb and isinstance(self._meta_model, lgb.LGBMClassifier):
            return pd.DataFrame(X, columns=FEATURE_NAMES)
        return X

    # ─────────────────────────────────────────────────────────────────
    # recommend_similar_anime — cascade
    # ─────────────────────────────────────────────────────────────────

    def recommend_similar_anime(
        self,
        anime_identifier: Union[int, str],
        top_k: int = 10,
        method: str = "hybrid",
        use_diversity: bool = True,
    ) -> List[Dict]:
        """
        Anime-to-anime recommendation — cascade 4 stages.

        Stage 1 — Content retrieval (FAISS, ~0.4ms):
            top_k*5 anime có embedding gần nhất với query anime.

        Stage 2 — ALS + Collab expand (~1ms):
            Lấy ALS item similarity và CF item similarity với query anime.
            Union tất cả → ~80-120 candidates.
            Anime xuất hiện ở nhiều model được boost score.

        Stage 3 — Weighted score combination:
            Dùng fallback_weights để kết hợp content/ALS/collab similarity.
            Meta-model KHÔNG áp dụng ở đây vì cần user context.

        Stage 4 — Genre MMR:
            Đảm bảo kết quả đa dạng thể loại.
        """
        # ── Stage 1: Content FAISS retrieval ─────────────────────────
        content_candidates: Dict[int, float] = {}
        if self.content_model and method in ("content", "hybrid"):
            try:
                recs = self.content_model.get_similar_anime(
                    anime_identifier, top_k=top_k * 5
                )
                for r in recs:
                    content_candidates[r["mal_id"]] = r["similarity"]
            except Exception as e:
                logger.warning(f"Stage1 content error: {e}")

        # ── Stage 2: ALS + Collab item similarity ────────────────────
        als_candidates: Dict[int, float] = {}
        collab_candidates: Dict[int, float] = {}
        query_anime_id = self._resolve_anime_id(anime_identifier)

        if self.implicit_model and method == "hybrid" and query_anime_id:
            try:
                recs = self.implicit_model.get_similar_items(
                    query_anime_id, top_k=top_k * 3
                )
                for r in recs:
                    als_candidates[r["mal_id"]] = r.get("similarity", 0.0)
            except Exception as e:
                logger.warning(f"Stage2 ALS error: {e}")

        if (
            self.collaborative_model
            and method in ("collaborative", "hybrid")
            and query_anime_id
        ):
            try:
                recs = self.collaborative_model.get_similar_items(
                    query_anime_id, top_k=top_k * 3
                )
                for r in recs:
                    collab_candidates[r["mal_id"]] = r.get("similarity", 0.0)
            except Exception as e:
                logger.warning(f"Stage2 Collab error: {e}")

        # Union candidates, ghi nhận nguồn gốc
        all_candidates: Dict[int, Dict[str, float]] = {}
        for aid, s in content_candidates.items():
            all_candidates.setdefault(aid, {})["content"] = s
        for aid, s in als_candidates.items():
            all_candidates.setdefault(aid, {})["implicit"] = s
        for aid, s in collab_candidates.items():
            all_candidates.setdefault(aid, {})["collaborative"] = s

        if not all_candidates:
            return []

        # ── Stage 3: Weighted score combination ──────────────────────
        w = self.fallback_weights
        aggregated = []
        for aid, sources in all_candidates.items():
            score = (
                w["content"] * sources.get("content", 0.0)
                + w["collaborative"] * sources.get("collaborative", 0.0)
                + w["implicit"] * sources.get("implicit", 0.0)
            )
            # Confidence boost: xuất hiện ở nhiều model → score cao hơn
            n_src = len(sources)
            if n_src >= 2:
                score *= 1.0 + 0.15 * (n_src - 1)
            aggregated.append(
                {
                    "mal_id": aid,
                    "hybrid_score": score,
                    "sources": list(sources.keys()),
                }
            )
        aggregated.sort(key=lambda x: -x["hybrid_score"])

        # ── Stage 4: Enrich + Genre MMR ──────────────────────────────
        enriched = []
        for item in aggregated[: top_k * 4]:
            rec = self._get_anime_info(item["mal_id"])
            rec["hybrid_score"] = item["hybrid_score"]
            rec["sources"] = item["sources"]
            enriched.append(rec)

        if use_diversity and self.diversity_lambda > 0 and len(enriched) > top_k:
            return _genre_mmr(enriched, top_k, self.diversity_lambda, self._anime_info)
        return enriched[:top_k]

    # ─────────────────────────────────────────────────────────────────
    # recommend_for_user — cascade + meta-model
    # ─────────────────────────────────────────────────────────────────

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_watched: bool = True,
        strategy: str = "auto",
        use_diversity: bool = True,
        diversity_lambda: float = None,
    ) -> List[Dict]:
        """
        Gợi ý cho user — cascade: Retrieval → Meta-model → MMR.
        Interface giống HybridEngine cũ.
        """
        is_new = self._is_new_user(user_id)
        if strategy == "auto":
            strategy = "new_user" if is_new else "existing_user"
        if strategy == "new_user":
            return self._recommend_new(user_id, top_k, use_diversity, diversity_lambda)
        return self._recommend_existing(
            user_id, top_k, exclude_watched, use_diversity, diversity_lambda
        )

    def _recommend_existing(
        self, user_id, top_k, exclude_watched, use_diversity, diversity_lambda
    ) -> List[Dict]:
        lambda_ = (
            diversity_lambda if diversity_lambda is not None else self.diversity_lambda
        )
        exclude_set: Set[int] = set()
        if exclude_watched:
            exclude_set = self._user_watched.get(user_id, set()) | set(
                self._user_ratings.get(user_id, {}).keys()
            )

        retrieval_k = max(top_k * 8, 100)
        candidates: Set[int] = set()

        # ALS Implicit (model tốt nhất — lấy nhiều nhất)
        if self.implicit_model:
            try:
                recs = self.implicit_model.recommend_for_user(
                    user_id,
                    top_k=retrieval_k,
                    exclude_known=exclude_watched,
                    known_items=exclude_set,
                    use_diversity=False,
                )
                candidates.update(r["mal_id"] for r in recs)
            except Exception as e:
                logger.warning(f"ALS retrieval: {e}")

        # BPR/Collab
        if self.collaborative_model:
            try:
                recs = self.collaborative_model.recommend_for_user(
                    user_id,
                    top_k=retrieval_k // 2,
                    exclude_rated=exclude_watched,
                    rated_items=exclude_set,
                )
                candidates.update(r["mal_id"] for r in recs)
            except Exception as e:
                logger.warning(f"Collab retrieval: {e}")

        # Content
        if self.content_model:
            user_ratings = self._user_ratings.get(user_id, {})
            if user_ratings:
                try:
                    recs = self.content_model.recommend_for_user(
                        user_id=user_id,
                        user_ratings=user_ratings,
                        top_k=retrieval_k // 4,
                        exclude_ids=exclude_set,
                    )
                    candidates.update(r["mal_id"] for r in recs)
                except Exception as e:
                    logger.warning(f"Content retrieval: {e}")

        # Popularity
        if self.popularity_model:
            try:
                pop = self.popularity_model.get_top_rated(top_k=50)
                candidates.update(
                    r["mal_id"] for r in pop if r["mal_id"] not in exclude_set
                )
            except Exception:
                pass

        candidates -= exclude_set
        if not candidates:
            return []

        # Meta-model scoring (fallback weighted sum nếu chưa train)
        aid_list = list(candidates)
        scores = self._meta_score(user_id, aid_list)
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        pool = []
        for aid, score in ranked[: top_k * 5]:
            rec = self._get_anime_info(aid)
            rec["hybrid_score"] = score
            rec["strategy"] = "learned" if self._meta_trained else "fallback"
            pool.append(rec)

        if use_diversity and lambda_ > 0 and len(pool) > top_k:
            return _genre_mmr(pool, top_k, lambda_, self._anime_info)
        return pool[:top_k]

    def _recommend_new(
        self, user_id, top_k, use_diversity, diversity_lambda
    ) -> List[Dict]:
        """Cold-start: Popularity + Content (nếu user có >= 3 ratings)."""
        lambda_ = (
            diversity_lambda if diversity_lambda is not None else self.diversity_lambda
        )
        user_ratings = self._user_ratings.get(user_id, {})
        exclude_set = self._user_watched.get(user_id, set()) | set(user_ratings.keys())
        scores: Dict[int, float] = {}

        if self.popularity_model:
            try:
                pop = self.popularity_model.get_recommendations_for_new_user(
                    top_k=top_k * 3,
                    preferred_genres=self._preferred_genres(user_ratings),
                )
                for r in pop:
                    aid = r["mal_id"]
                    if aid not in exclude_set:
                        scores[aid] = (
                            0.6 * r.get("popularity_score", r.get("score", 0)) / 100
                        )
            except Exception:
                pass

        if self.content_model and len(user_ratings) >= 3:
            try:
                recs = self.content_model.recommend_for_user(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    top_k=top_k * 2,
                    exclude_ids=exclude_set,
                )
                for r in recs:
                    aid = r["mal_id"]
                    scores[aid] = scores.get(aid, 0) + 0.4 * r.get("similarity", 0)
            except Exception:
                pass

        pool = []
        for aid, score in sorted(scores.items(), key=lambda x: -x[1])[: top_k * 3]:
            rec = self._get_anime_info(aid)
            rec["hybrid_score"] = score
            rec["strategy"] = "new_user"
            pool.append(rec)

        if use_diversity and lambda_ > 0 and len(pool) > top_k:
            return _genre_mmr(pool, top_k, lambda_, self._anime_info)
        return pool[:top_k]

    # ─────────────────────────────────────────────────────────────────
    # Diversity evaluation
    # ─────────────────────────────────────────────────────────────────

    def evaluate_diversity(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Tính ILD, genre coverage, entropy cho kết quả gợi ý.
        Dùng để tune diversity_lambda: target ILD ∈ [0.60, 0.75].
        """
        if not recommendations:
            return {"ILD": 0.0, "coverage": 0.0, "entropy": 0.0, "n_unique_genres": 0}

        genre_sets = []
        gcounts: Dict[str, int] = {}
        for r in recommendations:
            gs = _parse_genres(
                self._anime_info.get(r.get("mal_id", 0), {}).get("genres", "")
            )
            genre_sets.append(gs)
            for g in gs:
                gcounts[g] = gcounts.get(g, 0) + 1

        n = len(genre_sets)
        if n > 1:
            pairs = [
                _genre_jaccard(genre_sets[i], genre_sets[j])
                for i in range(n)
                for j in range(i + 1, n)
            ]
            ild = 1.0 - float(np.mean(pairs))
        else:
            ild = 0.0

        all_rec = set().union(*genre_sets) if genre_sets else set()
        all_corpus: Set[str] = set()
        for info in self._anime_info.values():
            all_corpus |= _parse_genres(info.get("genres", ""))
        coverage = len(all_rec) / len(all_corpus) if all_corpus else 0.0

        total = sum(gcounts.values())
        if total > 0:
            probs = np.array([c / total for c in gcounts.values()])
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        else:
            entropy = 0.0

        return {
            "ILD": round(ild, 4),
            "coverage": round(coverage, 4),
            "entropy": round(entropy, 4),
            "n_unique_genres": len(all_rec),
        }

    # ─────────────────────────────────────────────────────────────────
    # Compatibility API — routes.py cần
    # ─────────────────────────────────────────────────────────────────

    @property
    def weights(self) -> Dict[str, float]:
        """Expose fallback_weights dưới tên 'weights' — backward compatible."""
        return self.fallback_weights

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Cập nhật fallback_weights (dùng khi meta-model fail hoặc chưa train)."""
        self.fallback_weights.update(weights)
        total = sum(self.fallback_weights.values())
        if total > 0:
            for k in self.fallback_weights:
                self.fallback_weights[k] /= total
        logger.info(f"Fallback weights updated: {self.fallback_weights}")

    def get_explanation(self, user_id: int, anime_id: int) -> Dict:
        """Giải thích tại sao anime được gợi ý — routes.py endpoint /explain."""
        explanation = {
            "anime_id": anime_id,
            "anime_info": self._get_anime_info(anime_id),
            "reasons": [],
        }
        user_ratings = self._user_ratings.get(user_id, {})

        # Content reason
        if self.content_model:
            for rated_id, rating in sorted(user_ratings.items(), key=lambda x: -x[1])[
                :3
            ]:
                try:
                    for s in self.content_model.get_similar_anime(rated_id, top_k=20):
                        if s["mal_id"] == anime_id:
                            rated_name = self._get_anime_info(rated_id)["name"]
                            explanation["reasons"].append(
                                {
                                    "type": "content_similarity",
                                    "message": f"Similar to '{rated_name}' (you rated {rating:.0f}/10)",
                                    "similarity": s["similarity"],
                                }
                            )
                            break
                except Exception:
                    pass

        # Genre match
        user_genres = self._preferred_genres(user_ratings)
        anime_genres_raw = self._get_anime_info(anime_id).get("genres", "").lower()
        matching = [g for g in user_genres if g in anime_genres_raw]
        if matching:
            explanation["reasons"].append(
                {
                    "type": "genre_match",
                    "message": f"Matches your favorite genres: {', '.join(matching)}",
                }
            )

        # Meta-model confidence (nếu đã train)
        if self._meta_trained:
            try:
                X, _ = self._build_X(user_id, [anime_id])
                proba = float(
                    self._meta_model.predict_proba(self._prepare_meta_input(X))[0, 1]
                )
                explanation["reasons"].append(
                    {
                        "type": "meta_model",
                        "message": f"Meta-model confidence: {proba:.1%}",
                        "confidence": proba,
                    }
                )
            except Exception:
                pass

        # Popularity
        if self.popularity_model:
            try:
                for rank, p in enumerate(self.popularity_model.get_top_rated(top_k=50)):
                    if p["mal_id"] == anime_id:
                        explanation["reasons"].append(
                            {
                                "type": "popularity",
                                "message": f"#{rank + 1} top rated overall",
                            }
                        )
                        break
            except Exception:
                pass

        return explanation

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def _resolve_anime_id(self, identifier: Union[int, str]) -> Optional[int]:
        if isinstance(identifier, int):
            return identifier
        if self.content_model:
            idx = self.content_model._get_anime_idx(identifier)
            if idx is not None:
                return self.content_model._idx_to_id.get(idx)
        return None

    def _is_new_user(self, user_id: int) -> bool:
        if self.collaborative_model and hasattr(
            self.collaborative_model, "user_to_idx"
        ):
            if user_id in self.collaborative_model.user_to_idx:
                return False
        if self.implicit_model and hasattr(self.implicit_model, "user_to_idx"):
            if user_id in self.implicit_model.user_to_idx:
                return False
        return True

    def _preferred_genres(self, user_ratings: Dict[int, float]) -> List[str]:
        if not user_ratings:
            return []
        gs: Dict[str, float] = {}
        for aid, rating in user_ratings.items():
            for g in _parse_genres(self._anime_info.get(aid, {}).get("genres", "")):
                gs[g] = gs.get(g, 0) + rating
        return [g for g, _ in sorted(gs.items(), key=lambda x: -x[1])[:5]]

    def _get_anime_info(self, anime_id: int) -> Dict:
        if anime_id in self._anime_info:
            return self._anime_info[anime_id].copy()
        if self.content_model and hasattr(self.content_model, "anime_df"):
            df = self.content_model.anime_df
            row = df[df["MAL_ID"] == anime_id]
            if not row.empty:
                r = row.iloc[0]
                try:
                    score = (
                        float(r.get("Score", 0)) if pd.notna(r.get("Score")) else 0.0
                    )
                except Exception:
                    score = 0.0
                return {
                    "mal_id": int(anime_id),
                    "name": r.get("Name", "Unknown"),
                    "english_name": r.get("English name", r.get("Name", "Unknown")),
                    "genres": r.get("Genres", ""),
                    "score": score,
                    "type": r.get("Type", "Unknown"),
                }
        return {
            "mal_id": anime_id,
            "name": f"Anime {anime_id}",
            "english_name": f"Anime {anime_id}",
            "genres": "",
            "score": 0,
            "type": "Unknown",
        }

    # ─────────────────────────────────────────────────────────────────
    # Save / Load — một thư mục duy nhất
    # ─────────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """
        Lưu toàn bộ engine (sub-models + meta-model) vào 1 thư mục.
        Thư mục mặc định: saved_models/learned_hybrid/
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.content_model:
            self.content_model.save(directory / "content_model.pkl")
        if self.collaborative_model:
            self.collaborative_model.save(directory / "collaborative_model.pkl")
        if self.implicit_model:
            self.implicit_model.save(directory / "implicit_model.pkl")
        if self.popularity_model:
            self.popularity_model.save(directory / "popularity_model.pkl")

        state = {
            "meta_model": self._meta_model,
            "meta_trained": self._meta_trained,
            "feature_importance": self._feature_importance,
            "fallback_weights": self.fallback_weights,
            "relevance_threshold": self.relevance_threshold,
            "diversity_lambda": self.diversity_lambda,
            "anime_info": self._anime_info,
            "item_avg_score": self._item_avg_score,
            "item_num_ratings": self._item_num_ratings,
        }
        with open(directory / "learned_hybrid.pkl", "wb") as f:
            pickle.dump(state, f)
        logger.info(f"LearnedHybridEngine saved to {directory}")

    def load(self, directory: Union[str, Path]) -> "LearnedHybridEngine":
        """
        Load engine từ thư mục.
        Nếu thư mục có learned_hybrid.pkl → load meta-model và state.
        Sub-models được load tự động từ cùng thư mục.
        """
        directory = Path(directory)

        from models.content import ContentBasedRecommender
        from models.collaborative import ItemBasedCF, MatrixFactorization
        from models.implicit import ALSImplicit
        from models.popularity import PopularityModel

        if (directory / "content_model.pkl").exists():
            self.content_model = ContentBasedRecommender()
            self.content_model.load(directory / "content_model.pkl")

        if (directory / "collaborative_model.pkl").exists():
            try:
                self.collaborative_model = MatrixFactorization()
                self.collaborative_model.load(directory / "collaborative_model.pkl")
            except Exception:
                self.collaborative_model = ItemBasedCF()
                self.collaborative_model.load(directory / "collaborative_model.pkl")

        if (directory / "implicit_model.pkl").exists():
            self.implicit_model = ALSImplicit()
            self.implicit_model.load(directory / "implicit_model.pkl")

        if (directory / "popularity_model.pkl").exists():
            self.popularity_model = PopularityModel()
            self.popularity_model.load(directory / "popularity_model.pkl")

        if (directory / "learned_hybrid.pkl").exists():
            with open(directory / "learned_hybrid.pkl", "rb") as f:
                state = pickle.load(f)
            self._meta_model = state.get("meta_model")
            self._meta_trained = state.get("meta_trained", False)
            self._feature_importance = state.get("feature_importance")
            self.fallback_weights = state.get("fallback_weights", self.fallback_weights)
            self.relevance_threshold = state.get("relevance_threshold", 7.0)
            self.diversity_lambda = state.get("diversity_lambda", 0.3)
            self._anime_info = state.get("anime_info", {})
            self._item_avg_score = state.get("item_avg_score", {})
            self._item_num_ratings = state.get("item_num_ratings", {})

        logger.info(
            f"LearnedHybridEngine loaded from {directory} (meta_trained={self._meta_trained})"
        )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function — gọi từ train.py
# ─────────────────────────────────────────────────────────────────────────────


def train_learned_hybrid(
    content_model,
    collaborative_model,
    implicit_model,
    popularity_model,
    anime_df: pd.DataFrame,
    train_ratings_df: pd.DataFrame,
    test_ratings_df: pd.DataFrame,
    all_ratings_df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    min_ratings_to_train: int = 20,
    top_users: int = 3000,
    relevance_threshold: float = 7.0,
) -> "LearnedHybridEngine":
    """
    Khởi tạo + train LearnedHybridEngine trong một bước.

    Parameters
    ----------
    content_model, collaborative_model, implicit_model, popularity_model
        Sub-models đã được fit.
    anime_df           : DataFrame anime metadata (set_anime_info)
    train_ratings_df   : ratings trong train split (user history)
    test_ratings_df    : ratings trong test split (held-out positives)
    all_ratings_df     : toàn bộ ratings để tính item stats
                         Truyền vào ratings đã load sẵn — KHÔNG load lại
    save_dir           : lưu engine ra đây sau khi train
    min_ratings_to_train : chỉ train trên user có >= N ratings
    top_users          : lấy tối đa N user active nhất
    relevance_threshold: ngưỡng rating để coi là positive

    Returns
    -------
    LearnedHybridEngine đã train xong
    """
    engine = LearnedHybridEngine(
        content_model=content_model,
        collaborative_model=collaborative_model,
        implicit_model=implicit_model,
        popularity_model=popularity_model,
        relevance_threshold=relevance_threshold,
    )

    engine.set_anime_info(anime_df)
    engine.set_item_stats(all_ratings_df)

    logger.info("Loading user history...")
    for uid, grp in train_ratings_df.groupby("user_id"):
        engine.set_user_history(
            int(uid),
            ratings=dict(zip(grp["anime_id"].astype(int), grp["rating"].astype(float))),
        )

    # Held-out positives
    test_interactions: Dict[int, List[int]] = {}
    for uid, grp in test_ratings_df[
        test_ratings_df["rating"] >= relevance_threshold
    ].groupby("user_id"):
        test_interactions[int(uid)] = grp["anime_id"].astype(int).tolist()

    all_anime_ids = anime_df["MAL_ID"].astype(int).tolist()
    train_user_ids = list(test_interactions.keys())

    engine.train(
        train_user_ids=train_user_ids,
        test_interactions=test_interactions,
        all_anime_ids=all_anime_ids,
        min_ratings_to_train=min_ratings_to_train,
        top_users=top_users,
    )

    if save_dir:
        engine.save(Path(save_dir))

    return engine


def _stream_test_interactions_from_csv(
    test_ratings_csv_path: Union[str, Path],
    relevance_threshold: float,
    chunk_size: int = 500_000,
) -> Dict[int, List[int]]:
    """Load held-out positive test items without materializing the whole CSV."""
    test_interactions: Dict[int, List[int]] = {}

    for chunk in pd.read_csv(
        test_ratings_csv_path,
        chunksize=chunk_size,
        usecols=["user_id", "anime_id", "rating"],
        dtype=_RATING_DTYPES,
    ):
        chunk = chunk.loc[
            chunk["rating"] >= relevance_threshold, ["user_id", "anime_id"]
        ]
        if chunk.empty:
            continue

        for user_id, group in chunk.groupby("user_id", sort=False):
            user_id = int(user_id)
            items = group["anime_id"].astype(np.int64).tolist()
            if user_id in test_interactions:
                test_interactions[user_id].extend(items)
            else:
                test_interactions[user_id] = items

    return test_interactions


def _stream_user_rating_counts_from_csv(
    train_ratings_csv_path: Union[str, Path],
    chunk_size: int = 500_000,
) -> Dict[int, int]:
    """Count ratings per user from the train split CSV."""
    user_counts: Dict[int, int] = {}

    for chunk in pd.read_csv(
        train_ratings_csv_path,
        chunksize=chunk_size,
        usecols=["user_id", "rating"],
        dtype={"user_id": "int32", "rating": "int8"},
    ):
        counts = chunk["user_id"].value_counts(sort=False)
        for user_id, count in counts.items():
            user_id = int(user_id)
            user_counts[user_id] = user_counts.get(user_id, 0) + int(count)

    return user_counts


def _stream_selected_user_histories_from_csv(
    train_ratings_csv_path: Union[str, Path],
    selected_users: Set[int],
    chunk_size: int = 500_000,
) -> Dict[int, Dict[int, float]]:
    """Load train histories only for the users selected for meta-model training."""
    user_histories: Dict[int, Dict[int, float]] = {int(uid): {} for uid in selected_users}

    if not user_histories:
        return user_histories

    for chunk in pd.read_csv(
        train_ratings_csv_path,
        chunksize=chunk_size,
        usecols=["user_id", "anime_id", "rating"],
        dtype=_RATING_DTYPES,
    ):
        chunk = chunk.loc[chunk["user_id"].isin(selected_users)]
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            user_histories[int(row.user_id)][int(row.anime_id)] = float(row.rating)

    return user_histories


def train_learned_hybrid_from_csv(
    content_model,
    collaborative_model,
    implicit_model,
    popularity_model,
    anime_df: pd.DataFrame,
    train_ratings_csv_path: Union[str, Path],
    test_ratings_csv_path: Union[str, Path],
    all_ratings_csv_path: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None,
    min_ratings_to_train: int = 20,
    top_users: int = 3000,
    relevance_threshold: float = 7.0,
    chunk_size: int = 500_000,
) -> "LearnedHybridEngine":
    """
    Streaming-friendly learned-hybrid training for full-dataset runs.
    """
    engine = LearnedHybridEngine(
        content_model=content_model,
        collaborative_model=collaborative_model,
        implicit_model=implicit_model,
        popularity_model=popularity_model,
        relevance_threshold=relevance_threshold,
    )

    engine.set_anime_info(anime_df)
    engine.set_item_stats_from_csv(all_ratings_csv_path, chunk_size=chunk_size)

    logger.info("Loading held-out positives from %s...", test_ratings_csv_path)
    test_interactions = _stream_test_interactions_from_csv(
        test_ratings_csv_path,
        relevance_threshold=relevance_threshold,
        chunk_size=chunk_size,
    )

    logger.info("Counting train interactions per user from %s...", train_ratings_csv_path)
    user_counts = _stream_user_rating_counts_from_csv(
        train_ratings_csv_path,
        chunk_size=chunk_size,
    )

    eligible = [
        (user_id, user_counts.get(user_id, 0))
        for user_id in test_interactions.keys()
        if user_counts.get(user_id, 0) >= min_ratings_to_train
    ]
    eligible.sort(key=lambda item: -item[1])
    selected_users = {user_id for user_id, _ in eligible[:top_users]}

    logger.info(
        "Learned hybrid selected %d/%d eligible users (min_ratings=%d, top_users=%d)",
        len(selected_users),
        len(eligible),
        min_ratings_to_train,
        top_users,
    )

    user_histories = _stream_selected_user_histories_from_csv(
        train_ratings_csv_path,
        selected_users,
        chunk_size=chunk_size,
    )
    for user_id, ratings in user_histories.items():
        engine.set_user_history(user_id, ratings=ratings)

    filtered_test_interactions = {
        user_id: items
        for user_id, items in test_interactions.items()
        if user_id in selected_users
    }
    all_anime_ids = anime_df["MAL_ID"].astype(int).tolist()

    engine.train(
        train_user_ids=list(selected_users),
        test_interactions=filtered_test_interactions,
        all_anime_ids=all_anime_ids,
        min_ratings_to_train=min_ratings_to_train,
        top_users=top_users,
    )

    if save_dir:
        engine.save(Path(save_dir))

    return engine
