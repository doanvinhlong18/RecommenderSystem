"""
evaluate_models.py — Đánh giá toàn bộ mô hình recommendation.

Các model được đánh giá:
  - Content-Based (SBERT + Structured Features)
  - Collaborative Filtering (MatrixFactorization / BPR)
  - Implicit Feedback (ALS)
  - Popularity (baseline)
  - Hybrid (hard-coded weights, load từ saved_models/hybrid/)
  - LearnedHybrid (meta-model LightGBM, load từ saved_models/learned_hybrid/)

Tất cả model dùng chung 1 split artifact được tạo trong train.py
→ so sánh công bằng trên cùng tập test.

Metrics đánh giá:
  - Precision@K, Recall@K, F1@K
  - Hit Rate@K (có tìm được item liên quan trong top K không)
  - MRR (Mean Reciprocal Rank)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MAP@K (Mean Average Precision)
  - Diversity: ILD, Genre Coverage, Genre Entropy

Usage:
    python evaluate_models.py
    python evaluate_models.py --sample-users 500 --k 5 10 20
    python evaluate_models.py --skip-content --skip-collaborative
    python evaluate_models.py --only-learned-hybrid  # chỉ eval LearnedHybrid
    python evaluate_models.py --output my_results.json
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import sys

sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, SPLITS_DIR, eval_config
from preprocessing import load_ratings_user_split
from models.content import ContentBasedRecommender
from models.collaborative import MatrixFactorization
from models.implicit import ALSImplicit
from models.popularity import PopularityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Metric helpers
# =============================================================================


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant or not recommended:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def f1_at_k(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


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
    all_corpus_genres: Set[str],
) -> Dict[str, float]:
    """
    Tính diversity metrics cho một list gợi ý.

    ILD   = 1 - mean(pairwise genre Jaccard) — cao = đa dạng
    cov   = % genres unique trong kết quả / tổng genres corpus
    entropy = Shannon entropy phân bố genre
    """
    if not rec_ids:
        return {"ILD": 0.0, "genre_coverage": 0.0, "genre_entropy": 0.0}

    genre_sets = []
    gcounts: Dict[str, int] = {}
    for aid in rec_ids:
        gs = _parse_genres(anime_info.get(aid, {}).get("genres", ""))
        genre_sets.append(gs)
        for g in gs:
            gcounts[g] = gcounts.get(g, 0) + 1

    # ILD
    n = len(genre_sets)
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = genre_sets[i], genre_sets[j]
                if a or b:
                    inter = len(a & b)
                    union = len(a | b)
                    pairs.append(inter / union if union > 0 else 0.0)
        ild = 1.0 - float(np.mean(pairs)) if pairs else 0.0
    else:
        ild = 0.0

    # Coverage
    all_rec_genres = set().union(*genre_sets) if genre_sets else set()
    coverage = (
        len(all_rec_genres) / len(all_corpus_genres) if all_corpus_genres else 0.0
    )

    # Entropy
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
# Aggregate results helper
# =============================================================================


def aggregate_results(
    results_per_k: Dict[int, Dict[str, list]],
    n_evaluated: int,
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """Tính mean của mỗi metric, trả về dict {K=5: {metric: value}}."""
    summary = {}
    for k in k_values:
        kd = results_per_k[k]
        summary[f"K={k}"] = {
            metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in kd.items()
        }
        summary[f"K={k}"]["evaluated_users"] = n_evaluated
    return summary


# =============================================================================
# 1. Content-Based Evaluation
# =============================================================================


def evaluate_content_model(
    content_model: ContentBasedRecommender,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500,
    anime_info: Dict[int, Dict] = None,
    corpus_genres: Set[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá Content-Based model.

    Dùng recommend_for_user() của model mới (tích hợp FAISS + user vector cache).
    User vector được build từ toàn bộ ratings trong train set.
    """
    logger.info("Evaluating Content-Based model...")

    if content_model is None:
        logger.warning("Content model not available — skip")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    div_results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_no_vec = 0
    skipped_no_recs = 0

    sample_users = eval_users[:max_users]

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not train_items or not test_items:
            continue

        try:
            # recommend_for_user dùng FAISS nếu có, fallback numpy
            recs = content_model.recommend_for_user(
                user_id=user_id,
                user_ratings=train_items,
                top_k=max(k_values) * 3,
                exclude_ids=set(train_items.keys()),
            )

            if not recs:
                skipped_no_recs += 1
                continue

            rec_ids = [r["mal_id"] for r in recs if r["mal_id"] not in train_items]

            if not rec_ids:
                skipped_no_recs += 1
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    dm = diversity_metrics(rec_ids[:k], anime_info, corpus_genres)
                    for metric, val in dm.items():
                        div_results[k][metric].append(val)

            evaluated += 1

        except Exception as e:
            logger.debug(f"Content user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Content: {i+1}/{len(sample_users)}, evaluated={evaluated}")

    logger.info(
        f"  Content done: {evaluated} users, skipped={skipped_no_vec+skipped_no_recs}"
    )

    # Merge diversity into results
    for k in k_values:
        for metric, vals in div_results[k].items():
            results[k][metric] = vals

    return aggregate_results(results, evaluated, k_values)


# =============================================================================
# 2. Collaborative Filtering Evaluation
# =============================================================================


def evaluate_collaborative_model(
    collab_model: MatrixFactorization,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500,
    anime_info: Dict[int, Dict] = None,
    corpus_genres: Set[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Đánh giá Collaborative Filtering (BPR / SVD / ALS explicit)."""
    logger.info("Evaluating Collaborative Filtering model...")

    if collab_model is None:
        logger.warning("Collaborative model not available — skip")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    div_results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_not_in_model = 0

    sample_users = eval_users[:max_users]

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        # User phải có trong model
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

            if not recs:
                continue

            rec_ids = [r["mal_id"] for r in recs if r["mal_id"] not in train_items]

            if not rec_ids:
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    dm = diversity_metrics(rec_ids[:k], anime_info, corpus_genres)
                    for metric, val in dm.items():
                        div_results[k][metric].append(val)

            evaluated += 1

        except Exception as e:
            logger.debug(f"Collaborative user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Collaborative: {i+1}/{len(sample_users)}, evaluated={evaluated}"
            )

    logger.info(
        f"  Collaborative done: {evaluated} users, not_in_model={skipped_not_in_model}"
    )

    for k in k_values:
        for metric, vals in div_results[k].items():
            results[k][metric] = vals

    return aggregate_results(results, evaluated, k_values)


# =============================================================================
# 3. Implicit ALS Evaluation
# =============================================================================


def evaluate_implicit_model(
    implicit_model: ALSImplicit,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500,
    anime_info: Dict[int, Dict] = None,
    corpus_genres: Set[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá ALS Implicit model.

    ALS dùng implicit feedback (hành vi xem) → không cần rating threshold
    để xác định "relevant". Test items là items user có rating > 0
    (đã tương tác) nhưng model chưa thấy.
    """
    logger.info("Evaluating ALS Implicit model...")

    if implicit_model is None:
        logger.warning("Implicit model not available — skip")
        return {}

    if not implicit_model.user_to_idx or not implicit_model.anime_to_idx:
        logger.error("Implicit model thiếu mappings")
        return {}

    if implicit_model.user_factors is None or implicit_model.item_factors is None:
        logger.error("Implicit model factors chưa được train")
        return {}

    # Xác định kích thước factor thực sự (tránh swapped factors)
    n_model_users = len(implicit_model.user_to_idx)
    uf_rows = implicit_model.user_factors.shape[0]
    actual_n_users = (
        uf_rows if uf_rows == n_model_users else implicit_model.item_factors.shape[0]
    )

    logger.info(
        f"  ALS: {n_model_users} users, {len(implicit_model.anime_to_idx)} items"
    )

    results = {k: defaultdict(list) for k in k_values}
    div_results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skip_oob = 0
    skip_norec = 0

    sample_users = eval_users[:max_users]

    for i, user_id in enumerate(sample_users):
        train_items = set(user_train.get(user_id, {}).keys())
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        if user_id not in implicit_model.user_to_idx:
            continue

        user_idx = implicit_model.user_to_idx[user_id]
        if user_idx >= actual_n_users:
            skip_oob += 1
            continue

        try:
            recs = implicit_model.recommend_for_user(
                user_id,
                top_k=max(k_values) * 3,
                exclude_known=True,
                known_items=train_items,
                use_diversity=False,  # diversity đánh giá riêng bên dưới
            )

            if not recs:
                skip_norec += 1
                continue

            rec_ids = [r["mal_id"] for r in recs if r["mal_id"] not in train_items]

            if not rec_ids:
                skip_norec += 1
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    dm = diversity_metrics(rec_ids[:k], anime_info, corpus_genres)
                    for metric, val in dm.items():
                        div_results[k][metric].append(val)

            evaluated += 1

        except Exception as e:
            logger.debug(f"Implicit user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Implicit: {i+1}/{len(sample_users)}, evaluated={evaluated}")

    logger.info(
        f"  Implicit done: {evaluated} users, " f"oob={skip_oob}, no_rec={skip_norec}"
    )

    for k in k_values:
        for metric, vals in div_results[k].items():
            results[k][metric] = vals

    return aggregate_results(results, evaluated, k_values)


# =============================================================================
# 4. Popularity Evaluation
# =============================================================================


def evaluate_popularity_model(
    popularity_model: PopularityModel,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500,
    anime_info: Dict[int, Dict] = None,
    corpus_genres: Set[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá Popularity baseline.

    Top anime phổ biến giống nhau với mọi user — chỉ exclude items user đã xem.
    """
    logger.info("Evaluating Popularity model (baseline)...")

    if popularity_model is None:
        logger.warning("Popularity model not available — skip")
        return {}

    # Lấy popular items một lần
    try:
        popular_recs = popularity_model.get_popular(
            top_k=max(k_values) * 5,
            popularity_type="top_rated",
        )
        all_popular_ids = [r["mal_id"] for r in popular_recs]
    except Exception as e:
        logger.error(f"Không lấy được popular recs: {e}")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    div_results = {k: defaultdict(list) for k in k_values}
    evaluated = 0

    sample_users = eval_users[:max_users]

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

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
                dm = diversity_metrics(rec_ids[:k], anime_info, corpus_genres)
                for metric, val in dm.items():
                    div_results[k][metric].append(val)

        evaluated += 1

    logger.info(f"  Popularity done: {evaluated} users")

    for k in k_values:
        for metric, vals in div_results[k].items():
            results[k][metric] = vals

    return aggregate_results(results, evaluated, k_values)


# =============================================================================
# 5. Hybrid / LearnedHybrid Evaluation (dùng chung 1 hàm)
# =============================================================================


def evaluate_hybrid_model(
    hybrid_engine,
    user_train: Dict[int, Dict[int, float]],
    user_test: Dict[int, Set[int]],
    eval_users: List[int],
    k_values: List[int],
    max_users: int = 500,
    model_label: str = "Hybrid",
    anime_info: Dict[int, Dict] = None,
    corpus_genres: Set[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá HybridEngine hoặc LearnedHybridEngine (interface giống nhau).

    Với mỗi user:
      1. set_user_history(train ratings) → engine chỉ thấy train data
      2. recommend_for_user(exclude_watched=True)
      3. So sánh với test items
    """
    logger.info(f"Evaluating {model_label}...")

    if hybrid_engine is None:
        logger.warning(f"{model_label} not available — skip")
        return {}

    results = {k: defaultdict(list) for k in k_values}
    div_results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_no_recs = 0
    skipped_cold = 0

    sample_users = eval_users[:max_users]

    for i, user_id in enumerate(sample_users):
        train_items = user_train.get(user_id, {})
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        try:
            # Quan trọng: chỉ cấp train data cho engine, không để lộ test
            hybrid_engine.set_user_history(
                user_id,
                ratings=train_items,
                watched=set(train_items.keys()),
            )

            recs = hybrid_engine.recommend_for_user(
                user_id,
                top_k=max(k_values) * 3,
                exclude_watched=True,
            )

            # Cold-start fallback
            if not recs:
                recs = hybrid_engine.recommend_for_user(
                    user_id,
                    top_k=max(k_values) * 2,
                    exclude_watched=False,
                )
                if not recs:
                    skipped_cold += 1
                    continue

            rec_ids = [r.get("mal_id") or r.get("anime_id") for r in recs]
            rec_ids = [aid for aid in rec_ids if aid and aid not in train_items]

            if not rec_ids:
                skipped_no_recs += 1
                continue

            for k in k_values:
                m = all_metrics(rec_ids, test_items, k)
                for metric, val in m.items():
                    results[k][metric].append(val)
                if anime_info and corpus_genres:
                    dm = diversity_metrics(rec_ids[:k], anime_info, corpus_genres)
                    for metric, val in dm.items():
                        div_results[k][metric].append(val)

            evaluated += 1

        except Exception as e:
            logger.debug(f"{model_label} user {user_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(
                f"  {model_label}: {i+1}/{len(sample_users)}, evaluated={evaluated}"
            )

    logger.info(
        f"  {model_label} done: {evaluated} users, "
        f"cold={skipped_cold}, no_recs={skipped_no_recs}"
    )

    for k in k_values:
        for metric, vals in div_results[k].items():
            results[k][metric] = vals

    return aggregate_results(results, evaluated, k_values)


# =============================================================================
# Reporting
# =============================================================================

METRIC_DISPLAY_ORDER = [
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

MODEL_DISPLAY_ORDER = [
    "Content",
    "Collaborative",
    "Implicit",
    "Popularity",
    "Hybrid",
    "LearnedHybrid",
]


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def print_model_results(model_name: str, results: Dict[str, Dict[str, float]]) -> None:
    if not results:
        print(f"\n{model_name}: No results")
        return
    print(f"\n{'=' * 65}")
    print(f"  {model_name}")
    print(f"{'=' * 65}")
    for k_key, metrics in sorted(results.items()):
        n = metrics.get("evaluated_users", 0)
        print(f"\n  {k_key}  (evaluated_users={n})")
        print(f"  {'Metric':<18} {'Value':>10}")
        print(f"  {'-' * 30}")
        for metric in METRIC_DISPLAY_ORDER:
            if metric in metrics:
                print(f"  {metric:<18} {_fmt(metrics[metric]):>10}")


def print_comparison_table(
    all_results: Dict[str, Dict],
    k: int,
    metrics: List[str] = None,
) -> None:
    if metrics is None:
        metrics = ["precision", "recall", "f1", "hit_rate", "mrr", "ndcg", "map"]

    k_key = f"K={k}"
    col_w = 10
    name_w = 16

    print(f"\n{'=' * (name_w + col_w * len(metrics) + 8)}")
    print(f"  Comparison @ K={k}")
    print(f"{'=' * (name_w + col_w * len(metrics) + 8)}")

    header = f"  {'Model':<{name_w}}"
    for m in metrics:
        header += f"{m:>{col_w}}"
    print(header)
    print(f"  {'-' * (name_w + col_w * len(metrics))}")

    for model_name in MODEL_DISPLAY_ORDER:
        if model_name not in all_results:
            continue
        row_data = all_results[model_name].get(k_key, {})
        row = f"  {model_name:<{name_w}}"
        for m in metrics:
            val = row_data.get(m, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)

    print(f"{'=' * (name_w + col_w * len(metrics) + 8)}")


def print_diversity_table(all_results: Dict[str, Dict], k: int) -> None:
    k_key = f"K={k}"
    div_metrics = ["ILD", "genre_coverage", "genre_entropy"]
    col_w = 16
    name_w = 16

    # Check if any model has diversity metrics
    has_div = any(
        div_metrics[0] in all_results.get(m, {}).get(k_key, {})
        for m in MODEL_DISPLAY_ORDER
    )
    if not has_div:
        return

    print(f"\n  Diversity @ K={k}")
    print(f"  {'-' * (name_w + col_w * len(div_metrics))}")
    header = f"  {'Model':<{name_w}}"
    for m in div_metrics:
        header += f"{m:>{col_w}}"
    print(header)
    print(f"  {'-' * (name_w + col_w * len(div_metrics))}")

    for model_name in MODEL_DISPLAY_ORDER:
        if model_name not in all_results:
            continue
        row_data = all_results[model_name].get(k_key, {})
        if not any(m in row_data for m in div_metrics):
            continue
        row = f"  {model_name:<{name_w}}"
        for m in div_metrics:
            val = row_data.get(m, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)


def find_best_model(
    all_results: Dict[str, Dict], k: int, metric: str = "ndcg"
) -> Tuple[Optional[str], float]:
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
    parser = argparse.ArgumentParser(description="Evaluate all recommendation models")
    parser.add_argument(
        "--sample-users",
        type=int,
        default=500,
        help="Số user để evaluate (default: 500)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Các giá trị K (default: 5 10 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_evaluation_results.json",
        help="File lưu kết quả JSON",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=str(SPLITS_DIR / "ratings_user_split.pkl"),
        help="Path tới split artifact",
    )

    # Skip flags
    parser.add_argument("--skip-content", action="store_true")
    parser.add_argument("--skip-collaborative", action="store_true")
    parser.add_argument("--skip-implicit", action="store_true")
    parser.add_argument("--skip-popularity", action="store_true")
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip HybridEngine (hard-coded weights)",
    )
    parser.add_argument(
        "--skip-learned-hybrid",
        action="store_true",
        help="Skip LearnedHybridEngine (meta-model)",
    )
    parser.add_argument(
        "--only-learned-hybrid",
        action="store_true",
        help="Chỉ evaluate LearnedHybrid (bỏ qua tất cả model khác)",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Bỏ qua tính diversity metrics (nhanh hơn)",
    )

    args = parser.parse_args()

    # --only-learned-hybrid shortcut
    if args.only_learned_hybrid:
        args.skip_content = args.skip_collaborative = args.skip_implicit = True
        args.skip_popularity = args.skip_hybrid = True
        args.skip_learned_hybrid = False

    total_start = time.time()

    print("=" * 70)
    print("  ANIME RECOMMENDATION SYSTEM — FULL EVALUATION")
    print("=" * 70)
    print(f"  Sample users : {args.sample_users}")
    print(f"  K values     : {args.k}")
    print(f"  Split path   : {args.split_path}")
    print(f"  Diversity    : {'OFF' if args.no_diversity else 'ON'}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # Load split artifact
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading split artifact...")
    split_path = Path(args.split_path)
    if not split_path.exists():
        print(f"  ERROR: Split not found at {split_path}")
        print("  Run train.py first to create the split.")
        return

    split_artifact = load_ratings_user_split(split_path)
    user_train = split_artifact.user_train
    user_test = split_artifact.user_test
    eval_users = list(split_artifact.eval_users)

    # Sample users deterministcially
    if len(eval_users) > args.sample_users:
        rng = np.random.default_rng(
            int(split_artifact.metadata.get("random_state", eval_config.random_state))
        )
        eval_users = list(rng.choice(eval_users, args.sample_users, replace=False))

    print(
        f"  Train interactions : {split_artifact.metadata.get('train_interactions', 'N/A')}"
    )
    print(
        f"  Test interactions  : {split_artifact.metadata.get('test_interactions', 'N/A')}"
    )
    print(f"  Eval users (total) : {len(split_artifact.eval_users)}")
    print(f"  Eval users (used)  : {len(eval_users)}")

    # ─────────────────────────────────────────────────────────────────────
    # Load models
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2/3] Loading models...")

    # Sub-models lưu trong saved_models/hybrid/
    model_path = MODELS_DIR / "hybrid"
    # LearnedHybrid lưu trong saved_models/learned_hybrid/
    learned_path = MODELS_DIR / "learned_hybrid"

    content_model = None
    collaborative_model = None
    implicit_model = None
    popularity_model = None
    hybrid_engine = None
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
            print("  [OK] Collaborative (MatrixFactorization/BPR) model")
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

    if not args.skip_hybrid:
        try:
            # __init__.py mới export LearnedHybridEngine as HybridEngine
            # Nếu vẫn muốn load HybridEngine gốc, dùng import trực tiếp
            from models.hybrid.hybrid_engine import HybridEngine

            hybrid_engine = HybridEngine()
            hybrid_engine.load(model_path)
            print("  [OK] HybridEngine (hard-coded weights)")
        except Exception as e:
            print(f"  [SKIP] HybridEngine: {e}")

    if not args.skip_learned_hybrid:
        try:
            from models.hybrid.learned_hybrid import LearnedHybridEngine

            learned_engine = LearnedHybridEngine()
            if learned_path.exists():
                learned_engine.load(learned_path)
                meta_trained = learned_engine._meta_trained
                print(f"  [OK] LearnedHybridEngine (meta_trained={meta_trained})")
                if not meta_trained:
                    print(
                        "       ⚠️  Meta-model chưa train — sẽ dùng fallback weighted sum"
                    )
            else:
                print(f"  [SKIP] LearnedHybrid: {learned_path} không tồn tại")
                print("         Chạy train.py trước để train LearnedHybrid")
                learned_engine = None
        except Exception as e:
            print(f"  [SKIP] LearnedHybrid: {e}")
            learned_engine = None

    # ─────────────────────────────────────────────────────────────────────
    # Build anime_info và corpus genres cho diversity metrics
    # ─────────────────────────────────────────────────────────────────────
    anime_info: Dict[int, Dict] = {}
    corpus_genres: Set[str] = set()

    if not args.no_diversity:
        # Lấy anime_info từ engine nào có sẵn
        for eng in [learned_engine, hybrid_engine]:
            if eng is not None and hasattr(eng, "_anime_info") and eng._anime_info:
                anime_info = eng._anime_info
                corpus_genres = build_corpus_genres(anime_info)
                print(
                    f"  Anime info loaded: {len(anime_info)} anime, "
                    f"{len(corpus_genres)} genres (for diversity)"
                )
                break
        if not anime_info:
            print("  ⚠️  Không load được anime_info — diversity metrics sẽ bị bỏ qua")

    # ─────────────────────────────────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating models...")
    print("-" * 40)

    all_results: Dict[str, Dict] = {}
    eval_anime_info = anime_info if not args.no_diversity else None
    eval_corpus = corpus_genres if not args.no_diversity else None

    if content_model is not None:
        t0 = time.time()
        all_results["Content"] = evaluate_content_model(
            content_model,
            user_train,
            user_test,
            eval_users,
            args.k,
            args.sample_users,
            anime_info=eval_anime_info,
            corpus_genres=eval_corpus,
        )
        print(f"  Content done: {time.time()-t0:.1f}s")

    if collaborative_model is not None:
        t0 = time.time()
        all_results["Collaborative"] = evaluate_collaborative_model(
            collaborative_model,
            user_train,
            user_test,
            eval_users,
            args.k,
            args.sample_users,
            anime_info=eval_anime_info,
            corpus_genres=eval_corpus,
        )
        print(f"  Collaborative done: {time.time()-t0:.1f}s")

    if implicit_model is not None:
        t0 = time.time()
        all_results["Implicit"] = evaluate_implicit_model(
            implicit_model,
            user_train,
            user_test,
            eval_users,
            args.k,
            args.sample_users,
            anime_info=eval_anime_info,
            corpus_genres=eval_corpus,
        )
        print(f"  Implicit done: {time.time()-t0:.1f}s")

    if popularity_model is not None:
        t0 = time.time()
        all_results["Popularity"] = evaluate_popularity_model(
            popularity_model,
            user_train,
            user_test,
            eval_users,
            args.k,
            args.sample_users,
            anime_info=eval_anime_info,
            corpus_genres=eval_corpus,
        )
        print(f"  Popularity done: {time.time()-t0:.1f}s")

    if hybrid_engine is not None:
        t0 = time.time()
        all_results["Hybrid"] = evaluate_hybrid_model(
            hybrid_engine,
            user_train,
            user_test,
            eval_users,
            args.k,
            args.sample_users,
            model_label="Hybrid",
            anime_info=eval_anime_info,
            corpus_genres=eval_corpus,
        )
        print(f"  Hybrid done: {time.time()-t0:.1f}s")

    if learned_engine is not None:
        t0 = time.time()
        all_results["LearnedHybrid"] = evaluate_hybrid_model(
            learned_engine,
            user_train,
            user_test,
            eval_users,
            args.k,
            args.sample_users,
            model_label="LearnedHybrid",
            anime_info=eval_anime_info,
            corpus_genres=eval_corpus,
        )
        print(f"  LearnedHybrid done: {time.time()-t0:.1f}s")

    if not all_results:
        print("\nKhông có model nào được evaluate. Kiểm tra lại path và --skip flags.")
        return {}

    # ─────────────────────────────────────────────────────────────────────
    # Print results
    # ─────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  EVALUATION RESULTS — Detail")
    print("=" * 70)

    for model_name, results in all_results.items():
        print_model_results(model_name, results)

    # Comparison tables per K
    for k in args.k:
        print_comparison_table(all_results, k)
        if not args.no_diversity and anime_info:
            print_diversity_table(all_results, k)

    # Best model summary
    print("\n" + "=" * 70)
    print("  BEST MODEL PER METRIC")
    print("=" * 70)
    for k in args.k:
        print(f"\n  @ K={k}:")
        for metric in ["precision", "recall", "ndcg", "hit_rate", "mrr", "map"]:
            best_model, best_val = find_best_model(all_results, k, metric)
            if best_model:
                print(f"    {metric:<12}  →  {best_model}  ({best_val:.4f})")

    # LearnedHybrid improvement over Hybrid
    if "Hybrid" in all_results and "LearnedHybrid" in all_results:
        print("\n" + "=" * 70)
        print("  LearnedHybrid vs Hybrid (hard-coded) — improvement")
        print("=" * 70)
        for k in args.k:
            k_key = f"K={k}"
            print(f"\n  @ K={k}:")
            for metric in ["precision", "recall", "ndcg", "hit_rate", "mrr"]:
                h_val = all_results["Hybrid"].get(k_key, {}).get(metric, 0.0)
                l_val = all_results["LearnedHybrid"].get(k_key, {}).get(metric, 0.0)
                if h_val > 0:
                    diff = l_val - h_val
                    pct = diff / h_val * 100
                    sign = "+" if diff >= 0 else ""
                    print(
                        f"    {metric:<12}  {h_val:.4f} → {l_val:.4f}  "
                        f"({sign}{diff:.4f}, {sign}{pct:.1f}%)"
                    )

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
            "split_meta": split_artifact.metadata,
            "diversity": not args.no_diversity,
        },
        "results": all_results,
        "total_time": f"{total_time:.2f}s",
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  Evaluation complete!  Total time: {total_time:.1f}s")
    print(f"  Results saved → {output_path}")
    print(f"{'=' * 70}\n")

    return all_results


if __name__ == "__main__":
    results = main()
