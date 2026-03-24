"""
Fast vectorized evaluation for Content-Based Recommender
Protocol: Leave-One-Out + Negative Sampling Pool (100 candidates per user)

Thay đổi so với version cũ:
- Leave-one-out thay vì 80/20 split → test set luôn đúng 1 item per user
- Negative sampling pool: 1 positive + 99 negatives → rank trong 100 candidates
  thay vì toàn catalog → metrics phản ánh thực tế hơn
- Chỉ dùng items đã rate >= threshold làm positive candidates
- Train = tất cả positives trừ 1 item test
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import logging
import time
import argparse

from config import MODELS_DIR
from preprocessing.data_loader import DataLoader
from models.content import ContentBasedRecommender
from evaluation import RecommenderMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cb_eval_loo")


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────


def load_data():
    loader = DataLoader()
    logger.info("Loading anime metadata and ratings")

    anime_df = loader.load_anime()
    ratings_df = loader.load_ratings()

    # Bỏ implicit negative (rating == 0)
    ratings_df = ratings_df[ratings_df["rating"] != 0]

    logger.info(f"{len(anime_df)} anime | {len(ratings_df)} ratings loaded")
    return anime_df, ratings_df


# ─────────────────────────────────────────
# FILTER & SAMPLE USERS
# ─────────────────────────────────────────


def filter_users(ratings_df, min_ratings: int = 10):
    counts = ratings_df["user_id"].value_counts()
    valid = counts[counts >= min_ratings].index
    df = ratings_df[ratings_df["user_id"].isin(valid)]
    logger.info(f"{len(valid)} users after min_ratings={min_ratings} filter")
    return df


def sample_users(df, n_users: int = 1000, seed: int = 42):
    rng = np.random.RandomState(seed)
    users = df["user_id"].unique()
    if len(users) > n_users:
        users = rng.choice(users, n_users, replace=False)
    df = df[df["user_id"].isin(users)]
    logger.info(f"{len(users)} sampled users")
    return df, users


# ─────────────────────────────────────────
# LEAVE-ONE-OUT SPLIT
# ─────────────────────────────────────────


def build_loo_split(
    df,
    users,
    positive_percentile: float = 50.0,
    seed: int = 42,
):
    """
    Leave-One-Out split nhất quán với percentile-based user vector.

    Test item = 1 item được sample ngẫu nhiên từ top-percentile của user
    (rating >= np.percentile(user_ratings, positive_percentile)).

    Train = toàn bộ ratings còn lại (kể cả rating thấp) —
    build_user_vector tự lo filter theo percentile, không cần lọc ở đây.

    User bị loại nếu:
    - Không có item nào trên ngưỡng percentile (không có test candidate)
    - Chỉ có đúng 1 item trên ngưỡng (giữ lại làm train, không còn để test)
    """
    rng = np.random.RandomState(seed)
    grouped = df.groupby("user_id")

    user_train: Dict[int, List[int]] = {}
    user_test: Dict[int, int] = {}
    rating_map: Dict[int, Dict[int, float]] = {}

    for uid in users:
        if uid not in grouped.groups:
            continue

        g = grouped.get_group(uid)
        item_rating = dict(zip(g["anime_id"].values, g["rating"].values))

        all_ratings_arr = np.array(list(item_rating.values()), dtype=np.float32)

        # Ngưỡng tính theo chính user này — nhất quán với build_user_vector
        pos_threshold = float(np.percentile(all_ratings_arr, positive_percentile))

        # Candidates cho test: phải >= ngưỡng percentile
        pos_candidates = [i for i, r in item_rating.items() if r >= pos_threshold]

        # Cần >= 2 để có thể giữ lại ít nhất 1 item train sau khi bỏ 1 test
        if len(pos_candidates) < 2:
            continue

        rng.shuffle(pos_candidates)

        test_item = pos_candidates[-1]
        train_items = [i for i in item_rating if i != test_item]  # toàn bộ trừ test

        user_test[uid] = test_item
        user_train[uid] = train_items
        rating_map[uid] = item_rating

    logger.info(
        f"{len(user_train)} users kept after LOO split "
        f"(positive_percentile={positive_percentile})"
    )
    return user_train, user_test, rating_map


# ─────────────────────────────────────────
# NEGATIVE SAMPLING POOL
# ─────────────────────────────────────────


def build_eval_pools(
    user_test: Dict[int, int],
    user_train: Dict[int, List[int]],
    all_item_ids: np.ndarray,
    n_neg: int = 99,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Với mỗi user tạo candidate pool = [test_item] + n_neg negatives.

    Negative = item chưa xuất hiện trong train hoặc test của user.
    Trả về dict uid -> list of item_ids (length = n_neg + 1).
    """
    rng = np.random.RandomState(seed)
    all_items_set = set(all_item_ids.tolist())
    pools: Dict[int, List[int]] = {}

    for uid, test_item in user_test.items():
        seen = set(user_train[uid]) | {test_item}
        candidates = np.array(list(all_items_set - seen))

        if len(candidates) < n_neg:
            # Không đủ negatives — dùng hết
            neg_sample = candidates.tolist()
        else:
            neg_sample = rng.choice(candidates, n_neg, replace=False).tolist()

        pools[uid] = [test_item] + neg_sample  # test item luôn ở index 0

    logger.info(
        f"Eval pools built: {n_neg} negatives + 1 positive = "
        f"{n_neg + 1} candidates per user"
    )
    return pools


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────


def load_model():
    model_path = MODELS_DIR / "hybrid" / "content_model.pkl"
    model = ContentBasedRecommender()
    model.load(model_path)

    item_matrix = model.embeddings  # shape (n_items, d)
    id_to_idx = model._id_to_idx  # item_id -> row index in item_matrix
    idx_to_id = model._idx_to_id  # row index -> item_id

    logger.info(f"Item matrix: {item_matrix.shape}")
    return model, item_matrix, id_to_idx, idx_to_id


# ─────────────────────────────────────────
# BUILD USER MATRIX
# ─────────────────────────────────────────


def build_user_matrix(
    model,
    user_train: Dict[int, List[int]],
    rating_map: Dict[int, Dict[int, float]],
    positive_percentile: float = 50.0,
    negative_percentile: float = 25.0,
    negative_weight: float = 0.4,
):
    """
    Build user matrix bằng cách gọi model.build_user_vector cho mỗi user.

    - user_ratings = train items only  → positive vector (không leak test item)
    - all_ratings  = toàn bộ ratings   → tính percentile threshold + negative signal
    """
    rows = []
    user_ids = []

    for uid, train_items in user_train.items():
        train_ratings = {i: rating_map[uid][i] for i in train_items}
        all_ratings = rating_map[uid]

        vec = model.build_user_vector(
            user_ratings=train_ratings,
            all_ratings=all_ratings,
            positive_percentile=positive_percentile,
            negative_percentile=negative_percentile,
            negative_weight=negative_weight,
        )
        if vec is None:
            continue
        rows.append(vec)
        user_ids.append(uid)

    user_matrix = np.vstack(rows)
    logger.info(f"User matrix: {user_matrix.shape}")
    return user_matrix, user_ids


# ─────────────────────────────────────────
# SCORE & RANK TRONG POOL
# ─────────────────────────────────────────


def score_and_rank_pools(
    user_matrix: np.ndarray,
    user_ids: List[int],
    item_matrix: np.ndarray,
    id_to_idx: Dict[int, int],
    idx_to_id: Dict[int, int],
    pools: Dict[int, List[int]],
    k_values: List[int],
) -> Dict[int, List[int]]:
    """
    Với mỗi user:
      1. Lấy embedding của các items trong pool
      2. Tính score = user_vec @ item_vecs.T
      3. Rank pool theo score giảm dần
      4. Trả về ranked list (item_ids)

    Không mask train items vì pool đã không chứa train items.
    """
    uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
    recommendations: Dict[int, List[int]] = {}

    max_k = max(k_values)

    for uid, pool_items in pools.items():
        if uid not in uid_to_row:
            continue

        row = uid_to_row[uid]
        user_vec = user_matrix[row]  # (d,)

        # Lấy indices trong item_matrix
        valid_pool = [
            (item_id, id_to_idx[item_id])
            for item_id in pool_items
            if item_id in id_to_idx
        ]

        if not valid_pool:
            continue

        pool_item_ids, pool_indices = zip(*valid_pool)
        pool_item_ids = list(pool_item_ids)
        pool_vecs = item_matrix[list(pool_indices)]  # (pool_size, d)

        scores = user_vec @ pool_vecs.T  # (pool_size,)

        # Rank giảm dần, lấy tối đa max_k
        top_k = min(max_k, len(scores))
        order = np.argsort(-scores)[:top_k]

        recommendations[uid] = [pool_item_ids[i] for i in order]

    logger.info(f"Ranked pools for {len(recommendations)} users")
    return recommendations


# ─────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────


def compute_metrics(
    recommendations: Dict[int, List[int]],
    user_test: Dict[int, int],
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Tính ranking metrics với test set = {1 item}.

    HitRate@K là metric chính cho LOO:
      - = 1 nếu test item nằm trong top-K của pool
      - = 0 nếu không

    NDCG, MRR cũng hợp lý vì chỉ có 1 relevant item.
    Precision/Recall ít ý nghĩa hơn nhưng vẫn giữ để tiện so sánh.
    """
    results = {}

    for k in k_values:
        lists: Dict[str, List[float]] = {
            m: []
            for m in ["precision", "recall", "f1", "hit_rate", "mrr", "ndcg", "map"]
        }

        for uid, rec_list in recommendations.items():
            if uid not in user_test:
                continue
            relevant = {user_test[uid]}
            m = RecommenderMetric.calculate_all_metrics(rec_list, relevant, k)
            for key in lists:
                lists[key].append(m[key])

        results[f"K={k}"] = {
            metric: float(np.mean(vals)) for metric, vals in lists.items()
        }
        results[f"K={k}"]["n_users"] = len(lists["hit_rate"])

    return results


# ─────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────


def print_results(results: Dict, k_values: List[int], elapsed: float):
    print("\n" + "=" * 52)
    print("  Content-Based Evaluation  │  LOO + Pool(100)")
    print(f"  Time: {elapsed:.2f}s")
    print("=" * 52)

    for k in k_values:
        r = results[f"K={k}"]
        n = int(r["n_users"])
        print(f"\n  K={k}  (n_users={n})")
        print(f"  {'HitRate':<12} {r['hit_rate']:.4f}   ← metric chính LOO")
        print(f"  {'MRR':<12} {r['mrr']:.4f}")
        print(f"  {'NDCG':<12} {r['ndcg']:.4f}")
        print(f"  {'Precision':<12} {r['precision']:.4f}")
        print(f"  {'Recall':<12} {r['recall']:.4f}")
        print(f"  {'MAP':<12} {r['map']:.4f}")

    print("=" * 52)


# ─────────────────────────────────────────
# HYPERPARAMETER GRID
# ─────────────────────────────────────────

# Định nghĩa từng trục — grid = tích cartesian của 3 mảng
POSITIVE_PERCENTILES = [50, 60.0, 70.0, 80.0]
NEGATIVE_PERCENTILES = [25.0, 30.0, 35.0]
NEGATIVE_WEIGHTS = [0.5, 0.4, 0.3]

# Tổng số combos = len(POSITIVE_PERCENTILES) * len(NEGATIVE_PERCENTILES) * len(NEGATIVE_WEIGHTS)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────


def run_one(
    model,
    item_matrix,
    id_to_idx,
    idx_to_id,
    user_train,
    user_test,
    rating_map,
    pools,
    k_values: List[int],
    positive_percentile: float,
    negative_percentile: float,
    negative_weight: float,
) -> Dict:
    """Chạy một combo hyperparameter, trả về metrics dict."""
    user_matrix, user_ids = build_user_matrix(
        model,
        user_train,
        rating_map,
        positive_percentile=positive_percentile,
        negative_percentile=negative_percentile,
        negative_weight=negative_weight,
    )
    recommendations = score_and_rank_pools(
        user_matrix,
        user_ids,
        item_matrix,
        id_to_idx,
        idx_to_id,
        pools,
        k_values,
    )
    return compute_metrics(recommendations, user_test, k_values)


def print_grid_results(
    grid_results: List[Tuple],
    k_values: List[int],
    elapsed: float,
):
    """
    In bảng so sánh tất cả combo hyperparameter.
    grid_results: list of (pos_pct, neg_pct, neg_w, metrics_dict)
    """
    primary_k = k_values[0]  # dùng K đầu tiên làm cột sort

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Hyperparameter Grid Search  |  LOO + Pool(100)  |  time={elapsed:.1f}s")
    print(f"  Sort by HitRate@{primary_k}")
    print(sep)

    header = f"  {'pos_pct':>7} {'neg_pct':>7} {'neg_w':>6}"
    for k in k_values:
        header += f"  {'HR@'+str(k):>7} {'MRR@'+str(k):>7} {'NDCG@'+str(k):>8}"
    print(header)
    print(f"  {'-'*7} {'-'*7} {'-'*6}" + (f"  {'-'*7} {'-'*7} {'-'*8}" * len(k_values)))

    # Sort theo HitRate@primary_k giảm dần
    sorted_results = sorted(
        grid_results,
        key=lambda x: x[3][f"K={primary_k}"]["hit_rate"],
        reverse=True,
    )

    for pos_pct, neg_pct, neg_w, metrics in sorted_results:
        row = f"  {pos_pct:>7.1f} {neg_pct:>7.1f} {neg_w:>6.2f}"
        for k in k_values:
            r = metrics[f"K={k}"]
            row += f"  {r['hit_rate']:>7.4f} {r['mrr']:>7.4f} {r['ndcg']:>8.4f}"
        print(row)

    print("=" * 80 + "\n")


def main(
    k_values: List[int] = [5, 10, 20],
    sample_users_n: int = 1000,
    n_neg: int = 99,
    seed: int = 40,
    positive_percentiles: List[float] = None,  # None = dùng POSITIVE_PERCENTILES
    negative_percentiles: List[float] = None,  # None = dùng NEGATIVE_PERCENTILES
    negative_weights: List[float] = None,  # None = dùng NEGATIVE_WEIGHTS
):
    """
    Grid search với 3 vòng lặp lồng nhau:
      for pos_pct in positive_percentiles:
        for neg_pct in negative_percentiles:
          for neg_w in negative_weights:

    LOO split chỉ phụ thuộc pos_pct → được cache, không recompute
    khi chỉ neg_pct hoặc neg_w thay đổi.
    """
    start = time.time()

    # 1. Load data
    anime_df, ratings_df = load_data()
    anime_id_col = "MAL_ID" if "MAL_ID" in anime_df.columns else "anime_id"
    all_item_ids = anime_df[anime_id_col].values

    # 2. Filter & sample — làm 1 lần, dùng cho tất cả combos
    ratings_df = filter_users(ratings_df, min_ratings=10)
    sampled_df, users = sample_users(ratings_df, sample_users_n, seed)

    # 3. Load model — làm 1 lần
    model, item_matrix, id_to_idx, idx_to_id = load_model()

    # 4. Grid search — 3 vòng lặp lồng nhau, cache LOO split theo pos_pct
    pos_list = (
        positive_percentiles
        if positive_percentiles is not None
        else POSITIVE_PERCENTILES
    )
    neg_list = (
        negative_percentiles
        if negative_percentiles is not None
        else NEGATIVE_PERCENTILES
    )
    w_list = negative_weights if negative_weights is not None else NEGATIVE_WEIGHTS

    n_total = len(pos_list) * len(neg_list) * len(w_list)
    logger.info(
        f"Grid: {len(pos_list)} pos x {len(neg_list)} neg x {len(w_list)} weights = {n_total} combos"
    )

    grid_results = []
    loo_cache: Dict[float, tuple] = {}

    for pos_pct in pos_list:

        # LOO split chỉ phụ thuộc pos_pct → cache lại dùng cho neg_pct/neg_w bên trong
        if pos_pct not in loo_cache:
            user_train, user_test, rating_map = build_loo_split(
                sampled_df,
                users,
                positive_percentile=pos_pct,
                seed=seed,
            )
            pools = build_eval_pools(user_test, user_train, all_item_ids, n_neg, seed)
            loo_cache[pos_pct] = (user_train, user_test, rating_map, pools)
        else:
            user_train, user_test, rating_map, pools = loo_cache[pos_pct]

        for neg_pct in neg_list:
            for neg_w in w_list:
                logger.info(f"  pos_pct={pos_pct} neg_pct={neg_pct} neg_w={neg_w}")

                metrics = run_one(
                    model,
                    item_matrix,
                    id_to_idx,
                    idx_to_id,
                    user_train,
                    user_test,
                    rating_map,
                    pools,
                    k_values,
                    positive_percentile=pos_pct,
                    negative_percentile=neg_pct,
                    negative_weight=neg_w,
                )
                grid_results.append((pos_pct, neg_pct, neg_w, metrics))

    # 5. Print comparison table
    print_grid_results(grid_results, k_values, time.time() - start)

    return grid_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Content-Based Evaluation — LOO + Pool + Hyperparameter Grid"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values (default: 5 10 20)",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=10_000,
        help="Number of users to sample (default: 10000)",
    )
    parser.add_argument(
        "--n-neg",
        type=int,
        default=99,
        help="Negatives per user in eval pool (default: 99)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--pos-pct",
        type=float,
        nargs="+",
        default=None,
        help="Positive percentile list, e.g. --pos-pct 70 75 80 90",
    )
    parser.add_argument(
        "--neg-pct",
        type=float,
        nargs="+",
        default=None,
        help="Negative percentile list, e.g. --neg-pct 10 20 25 30",
    )
    parser.add_argument(
        "--neg-w",
        type=float,
        nargs="+",
        default=None,
        help="Negative weight list, e.g. --neg-w 0.0 0.2 0.4 0.6 0.8",
    )
    args = parser.parse_args()

    main(
        k_values=args.k,
        sample_users_n=args.sample_users,
        n_neg=args.n_neg,
        seed=args.seed,
        positive_percentiles=args.pos_pct,
        negative_percentiles=args.neg_pct,
        negative_weights=args.neg_w,
    )
