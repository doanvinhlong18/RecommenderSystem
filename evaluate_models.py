"""
evaluate_models.py — Đánh giá toàn bộ mô hình recommendation.

Các model được đánh giá:
  - Content-Based (SBERT + Structured Features)
  - Collaborative Filtering (MatrixFactorization / BPR)
  - Implicit Feedback (ALS)
  - Popularity (baseline)
  - LearnedHybridEngine (meta-model, dùng fallback weighted sum nếu chưa train)

Tất cả models được load từ saved_models/learned_hybrid/ (1 thư mục duy nhất).
Tất cả models dùng chung 1 split artifact tạo trong train.py → so sánh công bằng.

Metrics:
  - Precision@K, Recall@K, F1@K
  - Hit Rate@K, MRR, NDCG@K, MAP@K
  - Diversity: ILD, Genre Coverage, Genre Entropy

Usage:
    python evaluate_models.py
    python evaluate_models.py --sample-users 500 --k 5 10 20
    python evaluate_models.py --skip-content --skip-collaborative
    python evaluate_models.py --only-learned-hybrid
    python evaluate_models.py --no-diversity --sample-users 200  # nhanh hơn
    python evaluate_models.py --output results.json
"""

import argparse
import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import sys
from scipy.sparse import csr_matrix, load_npz

sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, SPLITS_DIR, eval_config
from preprocessing import load_ratings_disk_split, load_ratings_user_split
from models.content import ContentBasedRecommender
from models.collaborative import MatrixFactorization
from models.implicit import ALSImplicit
from models.popularity import PopularityModel
from models.hybrid.learned_hybrid import LearnedHybridEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_RATING_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "rating": "int8",
}


# =============================================================================
# Metric helpers
# =============================================================================


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant or not recommended:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def f1_at_k(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant or not recommended:
        return 0.0
    return 1.0 if set(recommended[:k]) & relevant else 0.0


def mrr_score(recommended: List[int], relevant: Set[int]) -> float:
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant or not recommended:
        return 0.0
    rec_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant or not recommended:
        return 0.0
    hits, sum_prec = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant), k)


def all_metrics(recommended: List[int], relevant: Set[int], k: int) -> Dict[str, float]:
    prec = precision_at_k(recommended, relevant, k)
    rec = recall_at_k(recommended, relevant, k)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1_at_k(prec, rec),
        "hit_rate": hit_rate_at_k(recommended, relevant, k),
        "mrr": mrr_score(recommended[:k], relevant),
        "ndcg": ndcg_at_k(recommended, relevant, k),
        "map": map_at_k(recommended, relevant, k),
    }


# =============================================================================
# Diversity helpers
# =============================================================================


def _parse_genres(genres_str: str) -> Set[str]:
    if not genres_str:
        return set()
    return {g.strip().lower() for g in str(genres_str).split(",") if g.strip()}


def diversity_metrics(
    rec_ids: List[int],
    anime_info: Dict[int, Dict],
    corpus_genres: Set[str],
) -> Dict[str, float]:
    """ILD, genre coverage, entropy cho một list gợi ý."""
    if not rec_ids:
        return {"ILD": 0.0, "genre_coverage": 0.0, "genre_entropy": 0.0}

    genre_sets = []
    gcounts: Dict[str, int] = {}
    for aid in rec_ids:
        gs = _parse_genres(anime_info.get(aid, {}).get("genres", ""))
        genre_sets.append(gs)
        for g in gs:
            gcounts[g] = gcounts.get(g, 0) + 1

    n = len(genre_sets)
    if n > 1:
        pairs = [
            (
                len(genre_sets[i] & genre_sets[j]) / len(genre_sets[i] | genre_sets[j])
                if (genre_sets[i] or genre_sets[j])
                else 0.0
            )
            for i in range(n)
            for j in range(i + 1, n)
        ]
        ild = 1.0 - float(np.mean(pairs)) if pairs else 0.0
    else:
        ild = 0.0

    all_rec_genres = set().union(*genre_sets) if genre_sets else set()
    coverage = len(all_rec_genres) / len(corpus_genres) if corpus_genres else 0.0

    total = sum(gcounts.values())
    if total > 0:
        probs = np.array([c / total for c in gcounts.values()])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        entropy = 0.0

    return {
        "ILD": round(ild, 4),
        "genre_coverage": round(coverage, 4),
        "genre_entropy": round(entropy, 4),
    }


def build_corpus_genres(anime_info: Dict[int, Dict]) -> Set[str]:
    genres: Set[str] = set()
    for info in anime_info.values():
        genres |= _parse_genres(info.get("genres", ""))
    return genres


# =============================================================================
# Aggregate helper
# =============================================================================


def aggregate(
    results_per_k: Dict[int, Dict[str, list]],
    n_evaluated: int,
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    summary = {}
    for k in k_values:
        summary[f"K={k}"] = {
            metric: float(np.mean(vals)) if vals else 0.0
            for metric, vals in results_per_k[k].items()
        }
        summary[f"K={k}"]["evaluated_users"] = n_evaluated
    return summary


@dataclass
class EvaluationData:
    """Unified evaluation access for both legacy pickle splits and on-disk CSV splits."""

    eval_users: List[int]
    metadata: Dict
    split_format: str
    user_train: Optional[Dict[int, Dict[int, float]]] = None
    user_test: Optional[Dict[int, Set[int]]] = None
    train_matrix: Optional[csr_matrix] = None
    test_matrix: Optional[csr_matrix] = None
    user_to_idx: Optional[Dict[int, int]] = None
    anime_to_idx: Optional[Dict[int, int]] = None
    idx_to_anime: Optional[Dict[int, int]] = None

    def get_train_items(self, user_id: int) -> Dict[int, float]:
        if self.user_train is not None:
            return self.user_train.get(user_id, {})

        if (
            self.train_matrix is None
            or self.user_to_idx is None
            or self.idx_to_anime is None
        ):
            return {}

        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return {}

        row = self.train_matrix.getrow(user_idx)
        return {
            int(self.idx_to_anime[int(item_idx)]): float(rating)
            for item_idx, rating in zip(row.indices, row.data)
        }

    def get_train_item_ids(self, user_id: int) -> Set[int]:
        if self.user_train is not None:
            return set(self.user_train.get(user_id, {}).keys())
        return set(self.get_train_items(user_id).keys())

    def get_test_items(self, user_id: int) -> Set[int]:
        if self.user_test is not None:
            return self.user_test.get(user_id, set())

        if (
            self.test_matrix is None
            or self.user_to_idx is None
            or self.idx_to_anime is None
        ):
            return set()

        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return set()

        row = self.test_matrix.getrow(user_idx)
        return {int(self.idx_to_anime[int(item_idx)]) for item_idx in row.indices}


def _load_saved_train_matrix_context(matrix_dir: Path) -> Tuple[csr_matrix, Dict, Dict, Dict]:
    """Load the saved explicit train matrix and mappings from train.py output."""
    matrix_dir = Path(matrix_dir)
    matrix_path = matrix_dir / "user_item_matrix.npz"
    mappings_path = matrix_dir / "mappings.pkl"

    if not matrix_path.exists() or not mappings_path.exists():
        raise FileNotFoundError(
            f"Saved train matrices not found in {matrix_dir}. Run train.py first."
        )

    train_matrix = load_npz(matrix_path)
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    return (
        train_matrix,
        mappings["user_to_idx"],
        mappings["anime_to_idx"],
        mappings["idx_to_anime"],
    )


def _load_eval_users_from_test_csv(
    test_ratings_path: Path,
    chunk_size: int = 500_000,
) -> List[int]:
    """Collect the unique evaluation users from the on-disk held-out CSV."""
    eval_users: Set[int] = set()

    for chunk in pd.read_csv(
        test_ratings_path,
        chunksize=chunk_size,
        usecols=["user_id"],
        dtype={"user_id": "int32"},
    ):
        eval_users.update(chunk["user_id"].astype(np.int64).tolist())

    return sorted(int(user_id) for user_id in eval_users)


def _build_test_matrix_from_csv(
    test_ratings_path: Path,
    user_to_idx: Dict[int, int],
    anime_to_idx: Dict[int, int],
    selected_users: Set[int],
    chunk_size: int = 500_000,
) -> csr_matrix:
    """Build a sparse held-out test matrix aligned to the saved train mappings."""
    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            test_ratings_path,
            chunksize=chunk_size,
            usecols=["user_id", "anime_id"],
            dtype={"user_id": "int32", "anime_id": "int32"},
        ),
        start=1,
    ):
        chunk = chunk.loc[
            chunk["user_id"].isin(selected_users)
            & chunk["anime_id"].isin(anime_to_idx)
        ]
        if chunk.empty:
            continue

        mapped_users = chunk["user_id"].map(user_to_idx)
        mapped_items = chunk["anime_id"].map(anime_to_idx)
        valid_mask = mapped_users.notna() & mapped_items.notna()
        if not valid_mask.any():
            continue
        chunk = chunk.loc[valid_mask]
        mapped_users = mapped_users.loc[valid_mask]
        mapped_items = mapped_items.loc[valid_mask]

        row_parts.append(
            mapped_users.to_numpy(dtype=np.int32, copy=False)
        )
        col_parts.append(
            mapped_items.to_numpy(dtype=np.int32, copy=False)
        )

        if chunk_idx % 20 == 0:
            logger.info("  Built test matrix rows for %d chunks...", chunk_idx)

    if not row_parts:
        return csr_matrix((len(user_to_idx), len(anime_to_idx)), dtype=np.float32)

    row_indices = np.concatenate(row_parts)
    col_indices = np.concatenate(col_parts)
    data = np.ones(len(row_indices), dtype=np.float32)

    return csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_to_idx), len(anime_to_idx)),
    )


def load_split_for_evaluation(
    split_path: Path,
    sample_users: int,
    chunk_size: int = 500_000,
    matrix_dir: Path = MODELS_DIR / "matrices",
) -> EvaluationData:
    """Load either the legacy pickle split or the newer on-disk streaming split."""
    legacy_error = None

    try:
        split_artifact = load_ratings_user_split(split_path)
        eval_users = list(split_artifact.eval_users)
        if len(eval_users) > sample_users:
            rng = np.random.default_rng(
                int(
                    split_artifact.metadata.get(
                        "random_state", eval_config.random_state
                    )
                )
            )
            eval_users = list(rng.choice(eval_users, sample_users, replace=False))

        return EvaluationData(
            eval_users=eval_users,
            metadata=split_artifact.metadata,
            split_format="legacy_pickle",
            user_train=split_artifact.user_train,
            user_test=split_artifact.user_test,
        )
    except Exception as exc:
        legacy_error = exc

    try:
        disk_split = load_ratings_disk_split(split_path)
    except Exception as disk_error:
        raise RuntimeError(
            f"Could not load split artifact as legacy pickle or on-disk manifest.\n"
            f"Legacy error: {legacy_error}\n"
            f"Disk error: {disk_error}"
        ) from disk_error

    all_eval_users = _load_eval_users_from_test_csv(
        disk_split.test_ratings_path,
        chunk_size=chunk_size,
    )
    eval_users = all_eval_users
    if len(eval_users) > sample_users:
        rng = np.random.default_rng(
            int(disk_split.metadata.get("random_state", eval_config.random_state))
        )
        eval_users = list(rng.choice(eval_users, sample_users, replace=False))

    train_matrix, user_to_idx, anime_to_idx, idx_to_anime = _load_saved_train_matrix_context(
        matrix_dir
    )
    test_matrix = _build_test_matrix_from_csv(
        disk_split.test_ratings_path,
        user_to_idx=user_to_idx,
        anime_to_idx=anime_to_idx,
        selected_users=set(int(user_id) for user_id in eval_users),
        chunk_size=chunk_size,
    )
    if eval_users and test_matrix.nnz == 0:
        logger.warning(
            "The on-disk split loaded successfully, but none of the sampled users "
            "mapped into the saved train matrix. Make sure the split artifact and "
            "saved_models/matrices come from the same train run."
        )

    return EvaluationData(
        eval_users=list(int(user_id) for user_id in eval_users),
        metadata=disk_split.metadata,
        split_format="disk_manifest",
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        user_to_idx=user_to_idx,
        anime_to_idx=anime_to_idx,
        idx_to_anime=idx_to_anime,
    )


# =============================================================================
# 1. Content-Based
# =============================================================================


def evaluate_content_model(
    content_model: ContentBasedRecommender,
    eval_data: EvaluationData,
    k_values: List[int],
    max_users: int = 500,
    anime_info: Optional[Dict] = None,
    corpus_genres: Optional[Set[str]] = None,
) -> Dict:
    logger.info("Evaluating Content-Based model...")

    if content_model is None:
        logger.warning("Content model not available — skip")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0

    eval_users = eval_data.eval_users

    for i, user_id in enumerate(eval_users[:max_users]):
        train_items = eval_data.get_train_items(user_id)
        test_items = eval_data.get_test_items(user_id)
        if not train_items or not test_items:
            continue

        try:
            recs = content_model.recommend_for_user(
                user_id=user_id,
                user_ratings=train_items,
                top_k=max(k_values) * 3,
                exclude_ids=set(train_items.keys()),
            )
            rec_ids = [r["mal_id"] for r in recs if r["mal_id"] not in train_items]
            if not rec_ids:
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    for metric, val in diversity_metrics(
                        rec_ids[:k], anime_info, corpus_genres
                    ).items():
                        results[k][metric].append(val)
            evaluated += 1

        except Exception as e:
            logger.debug(f"Content user {user_id}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Content: {i+1}/{min(max_users, len(eval_users))}, ok={evaluated}"
            )

    logger.info(f"  Content done: {evaluated} users")
    return aggregate(results, evaluated, k_values)


# =============================================================================
# 2. Collaborative Filtering
# =============================================================================


def evaluate_collaborative_model(
    collab_model: MatrixFactorization,
    eval_data: EvaluationData,
    k_values: List[int],
    max_users: int = 500,
    anime_info: Optional[Dict] = None,
    corpus_genres: Optional[Set[str]] = None,
) -> Dict:
    logger.info("Evaluating Collaborative model...")

    if collab_model is None:
        logger.warning("Collaborative model not available — skip")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_not_in_model = 0

    eval_users = eval_data.eval_users

    for i, user_id in enumerate(eval_users[:max_users]):
        train_items = eval_data.get_train_items(user_id)
        test_items = eval_data.get_test_items(user_id)
        if not test_items:
            continue

        if (
            hasattr(collab_model, "user_to_idx")
            and user_id not in collab_model.user_to_idx
        ):
            skipped_not_in_model += 1
            continue

        try:
            recs = collab_model.recommend_for_user(
                user_id,
                top_k=max(k_values) * 3,
                exclude_rated=True,
                rated_items=set(train_items.keys()),
            )
            rec_ids = [r["mal_id"] for r in recs if r["mal_id"] not in train_items]
            if not rec_ids:
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    for metric, val in diversity_metrics(
                        rec_ids[:k], anime_info, corpus_genres
                    ).items():
                        results[k][metric].append(val)
            evaluated += 1

        except Exception as e:
            logger.debug(f"Collab user {user_id}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Collab: {i+1}/{min(max_users, len(eval_users))}, ok={evaluated}"
            )

    logger.info(
        f"  Collab done: {evaluated} users, not_in_model={skipped_not_in_model}"
    )
    return aggregate(results, evaluated, k_values)


# =============================================================================
# 3. Implicit ALS
# =============================================================================


def evaluate_implicit_model(
    implicit_model: ALSImplicit,
    eval_data: EvaluationData,
    k_values: List[int],
    max_users: int = 500,
    anime_info: Optional[Dict] = None,
    corpus_genres: Optional[Set[str]] = None,
) -> Dict:
    logger.info("Evaluating ALS Implicit model...")

    if implicit_model is None:
        logger.warning("Implicit model not available — skip")
        return {}

    if not implicit_model.user_to_idx or implicit_model.user_factors is None:
        logger.error("Implicit model không hợp lệ")
        return {}

    n_users = len(implicit_model.user_to_idx)
    uf_rows = implicit_model.user_factors.shape[0]
    # Detect swapped factors
    actual_n_users = (
        uf_rows if uf_rows == n_users else implicit_model.item_factors.shape[0]
    )

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skip_oob = 0

    eval_users = eval_data.eval_users

    for i, user_id in enumerate(eval_users[:max_users]):
        train_items = eval_data.get_train_item_ids(user_id)
        test_items = eval_data.get_test_items(user_id)
        if not test_items:
            continue
        if user_id not in implicit_model.user_to_idx:
            continue
        if implicit_model.user_to_idx[user_id] >= actual_n_users:
            skip_oob += 1
            continue

        try:
            recs = implicit_model.recommend_for_user(
                user_id,
                top_k=max(k_values) * 3,
                exclude_known=True,
                known_items=train_items,
                use_diversity=False,
            )
            rec_ids = [r["mal_id"] for r in recs if r["mal_id"] not in train_items]
            if not rec_ids:
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    for metric, val in diversity_metrics(
                        rec_ids[:k], anime_info, corpus_genres
                    ).items():
                        results[k][metric].append(val)
            evaluated += 1

        except Exception as e:
            logger.debug(f"Implicit user {user_id}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Implicit: {i+1}/{min(max_users, len(eval_users))}, ok={evaluated}"
            )

    logger.info(f"  Implicit done: {evaluated} users, oob={skip_oob}")
    return aggregate(results, evaluated, k_values)


# =============================================================================
# 4. Popularity
# =============================================================================


def evaluate_popularity_model(
    popularity_model: PopularityModel,
    eval_data: EvaluationData,
    k_values: List[int],
    max_users: int = 500,
    anime_info: Optional[Dict] = None,
    corpus_genres: Optional[Set[str]] = None,
) -> Dict:
    logger.info("Evaluating Popularity model (baseline)...")

    if popularity_model is None:
        logger.warning("Popularity model not available — skip")
        return {}

    try:
        popular_recs = popularity_model.get_popular(
            top_k=max(k_values) * 5, popularity_type="top_rated"
        )
        all_popular_ids = [r["mal_id"] for r in popular_recs]
    except Exception as e:
        logger.error(f"Không lấy được popular recs: {e}")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0

    eval_users = eval_data.eval_users

    for i, user_id in enumerate(eval_users[:max_users]):
        train_items = eval_data.get_train_items(user_id)
        test_items = eval_data.get_test_items(user_id)
        if not test_items:
            continue

        rec_ids = [pid for pid in all_popular_ids if pid not in train_items]
        if not rec_ids:
            continue

        for k in k_values:
            m = all_metrics(rec_ids, test_items, k)
            for metric, val in m.items():
                results[k][metric].append(val)
            if anime_info and corpus_genres:
                for metric, val in diversity_metrics(
                    rec_ids[:k], anime_info, corpus_genres
                ).items():
                    results[k][metric].append(val)
        evaluated += 1

    logger.info(f"  Popularity done: {evaluated} users")
    return aggregate(results, evaluated, k_values)


# =============================================================================
# 5. LearnedHybrid
# =============================================================================


def evaluate_learned_hybrid(
    engine: LearnedHybridEngine,
    eval_data: EvaluationData,
    k_values: List[int],
    max_users: int = 500,
    anime_info: Optional[Dict] = None,
    corpus_genres: Optional[Set[str]] = None,
) -> Dict:
    """
    Đánh giá LearnedHybridEngine.
    Với mỗi user:
      1. set_user_history(train_ratings) — engine chỉ thấy train data
      2. recommend_for_user(exclude_watched=True)
      3. So sánh với test items
    """
    mode = "learned" if engine._meta_trained else "fallback_weighted_sum"
    logger.info(f"Evaluating LearnedHybrid ({mode})...")

    if engine is None:
        logger.warning("Engine not available — skip")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skip_cold = 0
    skip_norecs = 0

    eval_users = eval_data.eval_users

    for i, user_id in enumerate(eval_users[:max_users]):
        train_items = eval_data.get_train_items(user_id)
        test_items = eval_data.get_test_items(user_id)
        if not test_items:
            continue

        try:
            # Quan trọng: chỉ cấp train data — không để lộ test
            engine.set_user_history(
                user_id,
                ratings=train_items,
                watched=set(train_items.keys()),
            )

            recs = engine.recommend_for_user(
                user_id,
                top_k=max(k_values) * 3,
                exclude_watched=True,
            )

            if not recs:
                recs = engine.recommend_for_user(
                    user_id,
                    top_k=max(k_values) * 2,
                    exclude_watched=False,
                )
                if not recs:
                    skip_cold += 1
                    continue

            rec_ids = [r.get("mal_id") or r.get("anime_id") for r in recs]
            rec_ids = [aid for aid in rec_ids if aid and aid not in train_items]

            if not rec_ids:
                skip_norecs += 1
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    for metric, val in diversity_metrics(
                        rec_ids[:k], anime_info, corpus_genres
                    ).items():
                        results[k][metric].append(val)
            evaluated += 1

        except Exception as e:
            logger.debug(f"LearnedHybrid user {user_id}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(
                f"  LearnedHybrid: {i+1}/{min(max_users, len(eval_users))}, ok={evaluated}"
            )

    logger.info(
        f"  LearnedHybrid done: {evaluated} users, "
        f"cold={skip_cold}, no_recs={skip_norecs}"
    )
    return aggregate(results, evaluated, k_values)


# =============================================================================
# Reporting
# =============================================================================

METRIC_ORDER = [
    "precision",
    "recall",
    "f1",
    "hit_rate",
    "mrr",
    "ndcg",
    "map",
    "ILD",
    "genre_coverage",
    "genre_entropy",
]
MODEL_ORDER = ["Content", "Collaborative", "Implicit", "Popularity", "LearnedHybrid"]


def print_model_results(model_name: str, results: Dict) -> None:
    if not results:
        print(f"\n{model_name}: No results")
        return
    print(f"\n{'=' * 65}")
    print(f"  {model_name}")
    print(f"{'=' * 65}")
    for k_key in sorted(results):
        metrics = results[k_key]
        n = metrics.get("evaluated_users", 0)
        print(f"\n  {k_key}  (n={n})")
        print(f"  {'Metric':<20} {'Value':>10}")
        print(f"  {'-' * 32}")
        for m in METRIC_ORDER:
            if m in metrics:
                print(f"  {m:<20} {metrics[m]:>10.4f}")


def print_comparison_table(
    all_results: Dict, k: int, show_diversity: bool = True
) -> None:
    k_key = f"K={k}"
    metrics = ["precision", "recall", "f1", "hit_rate", "mrr", "ndcg", "map"]
    if show_diversity:
        metrics += ["ILD", "genre_coverage"]

    col_w = 10
    name_w = 18

    print(f"\n{'=' * (name_w + col_w * len(metrics) + 4)}")
    print(f"  Comparison @ K={k}")
    print(f"{'=' * (name_w + col_w * len(metrics) + 4)}")

    header = f"  {'Model':<{name_w}}"
    for m in metrics:
        header += f"{m[:col_w]:>{col_w}}"
    print(header)
    print(f"  {'-' * (name_w + col_w * len(metrics))}")

    for model_name in MODEL_ORDER:
        if model_name not in all_results:
            continue
        row_data = all_results[model_name].get(k_key, {})
        row = f"  {model_name:<{name_w}}"
        for m in metrics:
            row += f"{row_data.get(m, 0.0):>{col_w}.4f}"
        print(row)

    print(f"{'=' * (name_w + col_w * len(metrics) + 4)}")


def find_best_model(all_results: Dict, k: int, metric: str = "ndcg") -> Tuple:
    k_key = f"K={k}"
    best_model, best_val = None, -1.0
    for model_name, results in all_results.items():
        val = results.get(k_key, {}).get(metric, 0.0)
        if val > best_val:
            best_val = val
            best_model = model_name
    return best_model, best_val


# =============================================================================
# Main
# =============================================================================


def main():
    default_split_path = SPLITS_DIR / "ratings_user_split.pkl"
    default_full_split_path = SPLITS_DIR / "full_train_split.json"

    parser = argparse.ArgumentParser(description="Evaluate all recommendation models")
    parser.add_argument("--sample-users", type=int, default=500)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output", type=str, default="model_evaluation_results.json")
    parser.add_argument("--split-path", type=str, default=str(default_split_path))
    parser.add_argument(
        "--split-chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for reading on-disk split CSV artifacts",
    )

    parser.add_argument("--skip-content", action="store_true")
    parser.add_argument("--skip-collaborative", action="store_true")
    parser.add_argument("--skip-implicit", action="store_true")
    parser.add_argument("--skip-popularity", action="store_true")
    parser.add_argument("--skip-learned-hybrid", action="store_true")
    parser.add_argument(
        "--only-learned-hybrid", action="store_true", help="Chỉ evaluate LearnedHybrid"
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Bỏ qua diversity metrics (nhanh hơn)",
    )

    args = parser.parse_args()

    if args.only_learned_hybrid:
        args.skip_content = args.skip_collaborative = True
        args.skip_implicit = args.skip_popularity = True
        args.skip_learned_hybrid = False

    total_start = time.time()

    print("=" * 70)
    print("  ANIME RECOMMENDATION SYSTEM — EVALUATION")
    print("=" * 70)
    print(f"  Sample users : {args.sample_users}")
    print(f"  K values     : {args.k}")
    print(f"  Diversity    : {'OFF' if args.no_diversity else 'ON'}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # Load split
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading split artifact...")
    split_path = Path(args.split_path)
    if (
        args.split_path == str(default_split_path)
        and not split_path.exists()
        and default_full_split_path.exists()
    ):
        split_path = default_full_split_path
        print(f"  Using full-data split manifest: {split_path}")

    if not split_path.exists():
        print(f"  ERROR: {split_path} not found — run train.py first")
        return

    eval_data = load_split_for_evaluation(
        split_path,
        sample_users=args.sample_users,
        chunk_size=args.split_chunk_size,
    )
    eval_users = eval_data.eval_users

    print(
        f"  Split format       : {eval_data.split_format}"
    )
    print(
        f"  Train interactions : {eval_data.metadata.get('train_interactions', 'N/A')}"
    )
    print(f"  Test interactions  : {eval_data.metadata.get('test_interactions', 'N/A')}")
    print(f"  Eval users (used)  : {len(eval_users)}")

    # ─────────────────────────────────────────────────────────────────────
    # Load models — tất cả từ saved_models/learned_hybrid/
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2/3] Loading models...")

    model_path = MODELS_DIR / "learned_hybrid"

    content_model = None
    collaborative_model = None
    implicit_model = None
    popularity_model = None
    learned_engine = None

    if not args.skip_content:
        try:
            content_model = ContentBasedRecommender()
            content_model.load(model_path / "content_model.pkl")
            print("  [OK] Content-Based model")
        except Exception as e:
            print(f"  [SKIP] Content-Based: {e}")

    if not args.skip_collaborative:
        try:
            collaborative_model = MatrixFactorization()
            collaborative_model.load(model_path / "collaborative_model.pkl")
            print("  [OK] Collaborative model")
        except Exception as e:
            print(f"  [SKIP] Collaborative: {e}")

    if not args.skip_implicit:
        try:
            implicit_model = ALSImplicit()
            implicit_model.load(model_path / "implicit_model.pkl")
            print("  [OK] Implicit ALS model")
        except Exception as e:
            print(f"  [SKIP] Implicit: {e}")

    if not args.skip_popularity:
        try:
            popularity_model = PopularityModel()
            popularity_model.load(model_path / "popularity_model.pkl")
            print("  [OK] Popularity model")
        except Exception as e:
            print(f"  [SKIP] Popularity: {e}")

    if not args.skip_learned_hybrid:
        try:
            learned_engine = LearnedHybridEngine()
            if model_path.exists():
                learned_engine.load(model_path)
                mode = (
                    "learned"
                    if learned_engine._meta_trained
                    else "fallback_weighted_sum"
                )
                print(f"  [OK] LearnedHybridEngine (mode={mode})")
                if not learned_engine._meta_trained:
                    print(
                        "       ⚠️  Meta-model chưa train — dùng fallback weighted sum"
                    )
            else:
                print(
                    f"  [SKIP] LearnedHybrid: {model_path} not found — run train.py first"
                )
                learned_engine = None
        except Exception as e:
            print(f"  [SKIP] LearnedHybrid: {e}")
            learned_engine = None

    # Build anime_info và corpus genres cho diversity metrics
    anime_info: Dict[int, Dict] = {}
    corpus_genres: Set[str] = set()

    if not args.no_diversity and learned_engine is not None:
        anime_info = learned_engine._anime_info
        corpus_genres = build_corpus_genres(anime_info)
        if anime_info:
            print(
                f"  Anime info: {len(anime_info)} anime, {len(corpus_genres)} genres (diversity ON)"
            )
        else:
            print("  ⚠️  anime_info trống — diversity metrics sẽ bị bỏ qua")

    ai = anime_info if (not args.no_diversity and anime_info) else None
    cg = corpus_genres if (not args.no_diversity and corpus_genres) else None

    # ─────────────────────────────────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating models...")
    print("-" * 40)

    all_results: Dict[str, Dict] = {}

    if content_model is not None:
        t0 = time.time()
        all_results["Content"] = evaluate_content_model(
            content_model,
            eval_data,
            args.k,
            args.sample_users,
            ai,
            cg,
        )
        print(f"  Content done: {time.time()-t0:.1f}s")

    if collaborative_model is not None:
        t0 = time.time()
        all_results["Collaborative"] = evaluate_collaborative_model(
            collaborative_model,
            eval_data,
            args.k,
            args.sample_users,
            ai,
            cg,
        )
        print(f"  Collaborative done: {time.time()-t0:.1f}s")

    if implicit_model is not None:
        t0 = time.time()
        all_results["Implicit"] = evaluate_implicit_model(
            implicit_model,
            eval_data,
            args.k,
            args.sample_users,
            ai,
            cg,
        )
        print(f"  Implicit done: {time.time()-t0:.1f}s")

    if popularity_model is not None:
        t0 = time.time()
        all_results["Popularity"] = evaluate_popularity_model(
            popularity_model,
            eval_data,
            args.k,
            args.sample_users,
            ai,
            cg,
        )
        print(f"  Popularity done: {time.time()-t0:.1f}s")

    if learned_engine is not None:
        t0 = time.time()
        all_results["LearnedHybrid"] = evaluate_learned_hybrid(
            learned_engine,
            eval_data,
            args.k,
            args.sample_users,
            ai,
            cg,
        )
        print(f"  LearnedHybrid done: {time.time()-t0:.1f}s")

    if not all_results:
        print("\nKhông có model nào được evaluate.")
        return {}

    # ─────────────────────────────────────────────────────────────────────
    # Print results
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS — Detail")
    print("=" * 70)
    for model_name, results in all_results.items():
        print_model_results(model_name, results)

    for k in args.k:
        print_comparison_table(all_results, k, show_diversity=not args.no_diversity)

    # Best model
    print("\n" + "=" * 70)
    print("  BEST MODEL PER METRIC")
    print("=" * 70)
    for k in args.k:
        print(f"\n  @ K={k}:")
        for metric in ["precision", "recall", "ndcg", "hit_rate", "mrr", "map"]:
            best_model, best_val = find_best_model(all_results, k, metric)
            if best_model:
                print(f"    {metric:<12}  →  {best_model}  ({best_val:.4f})")

    # ─────────────────────────────────────────────────────────────────────
    # Save JSON
    # ─────────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "sample_users": args.sample_users,
            "k_values": args.k,
            "split_path": str(split_path),
            "split_format": eval_data.split_format,
            "diversity": not args.no_diversity,
        },
        "results": all_results,
        "total_time": f"{total_time:.2f}s",
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  Done!  {total_time:.1f}s  →  {output_path}")
    print(f"{'=' * 70}\n")

    return all_results


if __name__ == "__main__":
    results = main()
