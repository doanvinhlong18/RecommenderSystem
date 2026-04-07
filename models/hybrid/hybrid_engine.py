"""
Hybrid Recommendation Engine v2.

Thay đổi so với bản gốc:

1. WEIGHTS CẢI TIẾN dựa trên kết quả evaluation thực tế:
   Bản gốc: content=0.25, collaborative=0.30, implicit=0.35, popularity=0.10
   Bản mới: content=0.20, collaborative=0.25, implicit=0.50, popularity=0.05
   Lý do: Implicit ALS đạt Precision@10=0.265, nhưng chỉ được 35% weight
           → Hybrid (0.173) tệ hơn Implicit đơn lẻ — cần tăng weight lên 50%

2. CASCADE PIPELINE thay cho weighted sum đơn giản:
   Stage 1 Retrieval: mỗi model lấy top-N candidates → Union
   Stage 2 Scoring:   normalize + weighted combination
   Stage 3 Diversity: Genre-aware MMR (xem bên dưới)
   Stage 4 Result:    trả về kèm source_scores

3. GENRE-AWARE MMR thay cho MMR dùng latent factor cosine:
   Dùng Jaccard overlap của genre sets để đo similarity giữa items
   → Đa dạng genre thực sự, không chỉ latent-factor diversity
   lambda_=0 (no diversity) ... 0.3 (recommended) ... 1.0 (pure diversity)

4. DIVERSITY METRICS (ILD, Coverage, Entropy):
   evaluate_diversity(recommendations) → dict
   Dùng để tune diversity_lambda:
     ILD < 0.5 → tăng lambda
     ILD > 0.8 → giảm lambda
     Target: ILD ∈ [0.60, 0.75]

5. ANIME-TO-ANIME: Union content + ALS item sim → aggregate → Genre MMR

6. recommend_for_user() dùng content_model.recommend_for_user() mới
   (giao diện thống nhất với FAISS)
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Set

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import model_config, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Genre diversity utilities
# ─────────────────────────────────────────────────────────────────────────────


def _parse_genres(genres_str: str) -> Set[str]:
    """'Action, Comedy, Drama' → {'action', 'comedy', 'drama'}"""
    if not genres_str:
        return set()
    return {g.strip().lower() for g in str(genres_str).split(",") if g.strip()}


def _genre_jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity của 2 genre sets. Range [0,1]."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def genre_aware_mmr(
    candidates: List[Dict],
    top_k: int,
    lambda_: float = 0.3,
    anime_info: Dict[int, Dict] = None,
) -> List[Dict]:
    """
    Maximal Marginal Relevance với genre diversity.

    Iteratively chọn items theo:
        MMR(i) = (1 - lambda_) * relevance(i)
                 - lambda_ * max_{j in selected} genre_jaccard(i, j)

    lambda_=0   → pure relevance (giống sort thông thường)
    lambda_=0.3 → balance (recommended cho anime rec)
    lambda_=1.0 → pure diversity

    Parameters
    ----------
    candidates  : list of dicts có 'mal_id' và 'hybrid_score'
    top_k       : số items muốn chọn
    lambda_     : diversity trade-off
    anime_info  : {anime_id: {"genres": str, ...}}

    Returns
    -------
    List[Dict] top_k items đã re-rank
    """
    if not candidates or top_k <= 0:
        return candidates[:top_k]
    if lambda_ == 0 or anime_info is None:
        return candidates[:top_k]

    # Parse genres một lần
    genre_sets = [
        _parse_genres(anime_info.get(c["mal_id"], {}).get("genres", ""))
        for c in candidates
    ]

    # Normalize scores về [0, 1]
    scores = np.array([c.get("hybrid_score", 0.0) for c in candidates], dtype=float)
    s_min, s_max = scores.min(), scores.max()
    norm_scores = (
        (scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones(len(scores))
    )

    selected_idx: List[int] = []
    selected_genres: List[Set[str]] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_i, best_score = None, -np.inf
        for i in remaining:
            relevance = (1 - lambda_) * norm_scores[i]
            max_sim = max(
                (_genre_jaccard(genre_sets[i], sg) for sg in selected_genres),
                default=0.0,
            )
            mmr = relevance - lambda_ * max_sim
            if mmr > best_score:
                best_score, best_i = mmr, i
        if best_i is None:
            break
        selected_idx.append(best_i)
        selected_genres.append(genre_sets[best_i])
        remaining.remove(best_i)

    return [candidates[i] for i in selected_idx]


def compute_diversity_metrics(
    recommendations: List[Dict],
    anime_info: Dict[int, Dict],
) -> Dict[str, float]:
    """
    Tính diversity metrics cho một list gợi ý.

    Returns
    -------
    {
      "ILD"          : Intra-List Diversity = 1 - mean(pairwise genre Jaccard)
                       Range [0,1], target [0.60, 0.75]
      "coverage"     : % unique genres trong recs / total genres in corpus
      "entropy"      : Shannon entropy của genre distribution trong recs
      "n_unique_genres" : số genres unique trong recs
    }
    """
    if not recommendations:
        return {"ILD": 0.0, "coverage": 0.0, "entropy": 0.0, "n_unique_genres": 0}

    genre_sets = []
    genre_counts: Dict[str, int] = {}
    for rec in recommendations:
        gs = _parse_genres(anime_info.get(rec.get("mal_id", 0), {}).get("genres", ""))
        genre_sets.append(gs)
        for g in gs:
            genre_counts[g] = genre_counts.get(g, 0) + 1

    n = len(genre_sets)
    if n > 1:
        pairwise = [
            _genre_jaccard(genre_sets[i], genre_sets[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        ild = 1.0 - float(np.mean(pairwise))
    else:
        ild = 0.0

    all_rec_genres = set().union(*genre_sets) if genre_sets else set()
    all_corpus_genres: Set[str] = set()
    for info in anime_info.values():
        all_corpus_genres |= _parse_genres(info.get("genres", ""))
    coverage = (
        len(all_rec_genres) / len(all_corpus_genres) if all_corpus_genres else 0.0
    )

    total = sum(genre_counts.values())
    if total > 0:
        probs = np.array([c / total for c in genre_counts.values()])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        entropy = 0.0

    return {
        "ILD": round(ild, 4),
        "coverage": round(coverage, 4),
        "entropy": round(entropy, 4),
        "n_unique_genres": len(all_rec_genres),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HybridEngine v2
# ─────────────────────────────────────────────────────────────────────────────


class HybridEngine:
    """
    Hybrid Recommendation Engine v2 — Cascade + Genre Diversity.

    Weights mặc định (dựa trên Precision@10 evaluation):
        implicit:      0.50  (Precision@10 = 0.265 — tốt nhất, ~3x collab)
        collaborative: 0.25  (Precision@10 = 0.093)
        content:       0.20  (Precision@10 = 0.073)
        popularity:    0.05  (chỉ dùng như diversity boost)

    Bản gốc weights (implicit=0.35) khiến Hybrid (0.173) tệ hơn Implicit đơn lẻ (0.265).
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        content_model=None,
        collaborative_model=None,
        implicit_model=None,
        popularity_model=None,
        diversity_lambda: float = 0.3,
    ):
        self.weights = weights or {
            "content": 0.20,
            "collaborative": 0.25,
            "implicit": 0.50,
            "popularity": 0.05,
        }
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.implicit_model = implicit_model
        self.popularity_model = popularity_model
        self.diversity_lambda = diversity_lambda

        self._anime_info: Dict[int, Dict] = {}
        self._user_ratings: Dict[int, Dict[int, float]] = {}
        self._user_watched: Dict[int, Set[int]] = {}

    # ─────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights.update(weights)
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total
        logger.info(f"Weights updated: {self.weights}")

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

    # ─────────────────────────────────────────────────────────────
    # ANIME-TO-ANIME
    # ─────────────────────────────────────────────────────────────

    def recommend_similar_anime(
        self,
        anime_identifier: Union[int, str],
        top_k: int = 10,
        method: str = "hybrid",
        use_diversity: bool = True,
    ) -> List[Dict]:
        """
        Gợi ý anime tương tự — cascade pipeline.

        Stage 1: Content similarity (top-50) + ALS item sim (top-30)
                 + Collaborative item sim (top-30)
        Stage 2: Union candidates, aggregate weighted scores
        Stage 3: Genre-aware MMR nếu use_diversity=True
        """
        scores: Dict[int, Dict[str, float]] = {}

        # Content
        if self.content_model and method in ("content", "hybrid"):
            try:
                recs = self.content_model.get_similar_anime(
                    anime_identifier, top_k=top_k * 4
                )
                w = self.weights["content"] if method == "hybrid" else 1.0
                for rec in recs:
                    scores.setdefault(rec["mal_id"], {})["content"] = (
                        w * rec["similarity"]
                    )
            except Exception as e:
                logger.warning(f"Content similarity error: {e}")

        # Collaborative (BPR/SVD item factors)
        if self.collaborative_model and method in ("collaborative", "hybrid"):
            try:
                anime_id = self._resolve_anime_id(anime_identifier)
                if anime_id:
                    recs = self.collaborative_model.get_similar_items(
                        anime_id, top_k=top_k * 3
                    )
                    w = self.weights["collaborative"] if method == "hybrid" else 1.0
                    for rec in recs:
                        scores.setdefault(rec["mal_id"], {})["collaborative"] = (
                            w * rec["similarity"]
                        )
            except Exception as e:
                logger.warning(f"Collaborative similarity error: {e}")

        # Implicit (ALS item factors — dùng implicit library's FAISS internally)
        if self.implicit_model and method in ("implicit", "hybrid"):
            try:
                anime_id = self._resolve_anime_id(anime_identifier)
                if anime_id:
                    recs = self.implicit_model.get_similar_items(
                        anime_id, top_k=top_k * 3
                    )
                    w = self.weights["implicit"] if method == "hybrid" else 1.0
                    for rec in recs:
                        scores.setdefault(rec["mal_id"], {})["implicit"] = w * rec.get(
                            "similarity", 0
                        )
            except Exception as e:
                logger.warning(f"Implicit similarity error: {e}")

        # Aggregate + sort
        aggregated = sorted(
            [
                {
                    "mal_id": aid,
                    "hybrid_score": sum(s.values()),
                    "sources": list(s.keys()),
                }
                for aid, s in scores.items()
            ],
            key=lambda x: -x["hybrid_score"],
        )

        # Enrich với anime info
        pool_size = min(top_k * 4, len(aggregated))
        enriched = []
        for item in aggregated[:pool_size]:
            rec = self._get_anime_info(item["mal_id"])
            rec["hybrid_score"] = item["hybrid_score"]
            rec["sources"] = item["sources"]
            enriched.append(rec)

        # Genre-aware MMR
        if use_diversity and self.diversity_lambda > 0 and len(enriched) > top_k:
            return genre_aware_mmr(
                enriched, top_k, self.diversity_lambda, self._anime_info
            )
        return enriched[:top_k]

    # ─────────────────────────────────────────────────────────────
    # USER RECOMMENDATION
    # ─────────────────────────────────────────────────────────────

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
        Gợi ý cho user — cascade pipeline.

        Parameters
        ----------
        user_id         : user ID
        top_k           : số kết quả cuối
        exclude_watched : loại bỏ anime đã xem/rated
        strategy        : "auto", "new_user", "existing_user"
        use_diversity   : dùng genre-aware MMR
        diversity_lambda: override self.diversity_lambda (dùng để A/B test)
        """
        is_new = self._is_new_user(user_id)
        if strategy == "auto":
            strategy = "new_user" if is_new else "existing_user"

        if strategy == "new_user":
            return self._recommend_new_user(
                user_id, top_k, use_diversity, diversity_lambda
            )
        return self._recommend_existing_user(
            user_id, top_k, exclude_watched, use_diversity, diversity_lambda
        )

    def _recommend_existing_user(
        self,
        user_id: int,
        top_k: int,
        exclude_watched: bool,
        use_diversity: bool,
        diversity_lambda: float = None,
    ) -> List[Dict]:
        """Full cascade pipeline cho existing users."""
        lambda_ = (
            diversity_lambda if diversity_lambda is not None else self.diversity_lambda
        )

        exclude_set: Set[int] = set()
        if exclude_watched:
            exclude_set = self._user_watched.get(user_id, set()) | set(
                self._user_ratings.get(user_id, {}).keys()
            )

        # Retrieval pool: lấy nhiều để diversity có đủ candidates
        retrieval_k = max(top_k * 8, 100)
        candidates: Dict[int, Dict[str, float]] = {}

        # ── ALS Implicit (model tốt nhất, lấy nhiều nhất) ───────────
        if self.implicit_model:
            try:
                recs = self.implicit_model.recommend_for_user(
                    user_id,
                    top_k=retrieval_k,
                    exclude_known=exclude_watched,
                    known_items=exclude_set,
                    use_diversity=False,  # diversity xử lý ở Stage 3
                )
                w = self.weights["implicit"]
                for rec in recs:
                    aid = rec["mal_id"]
                    # ALS scores: clip về [0, inf) rồi normalize theo max
                    score = float(np.clip(rec.get("score", 0), 0, None))
                    candidates.setdefault(aid, {})["implicit"] = w * score
            except Exception as e:
                logger.warning(f"Implicit retrieval: {e}")

        # ── Collaborative (BPR/SVD) ──────────────────────────────────
        if self.collaborative_model:
            try:
                recs = self.collaborative_model.recommend_for_user(
                    user_id,
                    top_k=retrieval_k // 2,
                    exclude_rated=exclude_watched,
                    rated_items=exclude_set,
                )
                w = self.weights["collaborative"]
                for rec in recs:
                    aid = rec["mal_id"]
                    raw = rec.get("score", rec.get("predicted_rating", 5.0))
                    # BPR scores unbounded → sigmoid normalize
                    norm = float(1.0 / (1.0 + np.exp(-np.clip(float(raw), -20, 20))))
                    candidates.setdefault(aid, {})["collaborative"] = w * norm
            except Exception as e:
                logger.warning(f"Collaborative retrieval: {e}")

        # ── Content-Based ────────────────────────────────────────────
        if self.content_model:
            try:
                user_ratings = self._user_ratings.get(user_id, {})
                if user_ratings:
                    # Dùng recommend_for_user mới (tích hợp FAISS)
                    recs = self.content_model.recommend_for_user(
                        user_id=user_id,
                        user_ratings=user_ratings,
                        top_k=retrieval_k // 4,
                        exclude_ids=exclude_set,
                    )
                    w = self.weights["content"]
                    for rec in recs:
                        candidates.setdefault(rec["mal_id"], {})["content"] = (
                            w * rec.get("similarity", 0)
                        )
            except Exception as e:
                logger.warning(f"Content retrieval: {e}")

        # ── Popularity (diversity boost) ─────────────────────────────
        if self.popularity_model:
            try:
                pop_recs = self.popularity_model.get_top_rated(top_k=100)
                w = self.weights["popularity"]
                n_pop = len(pop_recs)
                for rank, rec in enumerate(pop_recs):
                    aid = rec["mal_id"]
                    if aid not in exclude_set:
                        candidates.setdefault(aid, {})["popularity"] = (
                            w * (n_pop - rank) / n_pop
                        )
            except Exception as e:
                logger.warning(f"Popularity retrieval: {e}")

        # ── Aggregate scores ─────────────────────────────────────────
        aggregated = sorted(
            [
                {"mal_id": aid, "hybrid_score": sum(s.values()), "source_scores": s}
                for aid, s in candidates.items()
                if aid not in exclude_set
            ],
            key=lambda x: -x["hybrid_score"],
        )

        # ── Enrich với anime info ────────────────────────────────────
        pool_size = min(top_k * 5, len(aggregated))
        enriched = []
        for item in aggregated[:pool_size]:
            rec = self._get_anime_info(item["mal_id"])
            rec["hybrid_score"] = item["hybrid_score"]
            rec["source_scores"] = item["source_scores"]
            rec["strategy"] = "existing_user"
            enriched.append(rec)

        # ── Genre-aware MMR ──────────────────────────────────────────
        if use_diversity and lambda_ > 0 and len(enriched) > top_k:
            return genre_aware_mmr(enriched, top_k, lambda_, self._anime_info)
        return enriched[:top_k]

    def _recommend_new_user(
        self,
        user_id: int,
        top_k: int,
        use_diversity: bool,
        diversity_lambda: float = None,
    ) -> List[Dict]:
        """Cold-start: Popularity + Content (nếu user có ít nhất vài ratings)."""
        lambda_ = (
            diversity_lambda if diversity_lambda is not None else self.diversity_lambda
        )
        user_ratings = self._user_ratings.get(user_id, {})
        watched = self._user_watched.get(user_id, set())
        exclude_set = watched | set(user_ratings.keys())
        scores: Dict[int, float] = {}

        if self.popularity_model:
            try:
                pop_recs = self.popularity_model.get_recommendations_for_new_user(
                    top_k=top_k * 3,
                    preferred_genres=self._extract_preferred_genres(user_ratings),
                )
                for rec in pop_recs:
                    aid = rec["mal_id"]
                    if aid not in exclude_set:
                        scores[aid] = (
                            0.6 * rec.get("popularity_score", rec.get("score", 0)) / 100
                        )
            except Exception as e:
                logger.warning(f"Popularity new_user: {e}")

        if self.content_model and len(user_ratings) >= 3:
            try:
                recs = self.content_model.recommend_for_user(
                    user_id=user_id,
                    user_ratings=user_ratings,
                    top_k=top_k * 2,
                    exclude_ids=exclude_set,
                )
                for rec in recs:
                    aid = rec["mal_id"]
                    scores[aid] = scores.get(aid, 0) + 0.4 * rec.get("similarity", 0)
            except Exception as e:
                logger.warning(f"Content new_user: {e}")

        enriched = []
        for aid, score in sorted(scores.items(), key=lambda x: -x[1])[: top_k * 3]:
            rec = self._get_anime_info(aid)
            rec["hybrid_score"] = score
            rec["strategy"] = "new_user"
            enriched.append(rec)

        if use_diversity and lambda_ > 0 and len(enriched) > top_k:
            return genre_aware_mmr(enriched, top_k, lambda_, self._anime_info)
        return enriched[:top_k]

    # ─────────────────────────────────────────────────────────────
    # DIVERSITY EVALUATION
    # ─────────────────────────────────────────────────────────────

    def evaluate_diversity(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Tính diversity metrics cho kết quả gợi ý.

        Dùng để tune diversity_lambda:
            ILD < 0.5 → tăng lambda
            ILD > 0.8 → giảm lambda (có thể ảnh hưởng relevance)
            Target: ILD ∈ [0.60, 0.75]

        Returns
        -------
        {"ILD": float, "coverage": float, "entropy": float, "n_unique_genres": int}
        """
        return compute_diversity_metrics(recommendations, self._anime_info)

    # ─────────────────────────────────────────────────────────────
    # EXPLANATION
    # ─────────────────────────────────────────────────────────────

    def get_explanation(self, user_id: int, anime_id: int) -> Dict:
        """Giải thích tại sao anime được gợi ý cho user."""
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
        user_genres = self._extract_preferred_genres(user_ratings)
        anime_genres_raw = self._get_anime_info(anime_id).get("genres", "").lower()
        matching = [g for g in user_genres if g in anime_genres_raw]
        if matching:
            explanation["reasons"].append(
                {
                    "type": "genre_match",
                    "message": f"Matches your favorite genres: {', '.join(matching)}",
                }
            )

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

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────

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

    def _extract_preferred_genres(self, user_ratings: Dict[int, float]) -> List[str]:
        if not user_ratings:
            return []
        genre_scores: Dict[str, float] = {}
        for aid, rating in user_ratings.items():
            for g in _parse_genres(self._anime_info.get(aid, {}).get("genres", "")):
                genre_scores[g] = genre_scores.get(g, 0) + rating
        return [g for g, _ in sorted(genre_scores.items(), key=lambda x: -x[1])[:5]]

    def _get_anime_info(self, anime_id: int) -> Dict:
        if anime_id in self._anime_info:
            return self._anime_info[anime_id].copy()
        # Fallback: try content model
        if self.content_model and hasattr(self.content_model, "anime_df"):
            df = self.content_model.anime_df
            row = df[df["MAL_ID"] == anime_id]
            if not row.empty:
                r = row.iloc[0]
                try:
                    score = (
                        float(r.get("Score", 0)) if pd.notna(r.get("Score")) else 0.0
                    )
                except (ValueError, TypeError):
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

    # ─────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
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
            "weights": self.weights,
            "diversity_lambda": self.diversity_lambda,
            "anime_info": self._anime_info,
        }
        with open(directory / "hybrid_engine.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info(f"HybridEngine v2 saved to {directory}")

    def load(self, directory: Union[str, Path]) -> "HybridEngine":
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

        if (directory / "hybrid_engine.pkl").exists():
            with open(directory / "hybrid_engine.pkl", "rb") as f:
                state = pickle.load(f)
            self.weights = state.get("weights", self.weights)
            self.diversity_lambda = state.get("diversity_lambda", self.diversity_lambda)
            self._anime_info = state.get("anime_info", {})

        logger.info(f"HybridEngine v2 loaded from {directory}")
        return self
