"""
Full-dataset collaborative benchmark runner.

This script is designed for the large Anime Recommendation dataset where the
legacy train/eval pipeline becomes memory-heavy because it persists user_train
as nested Python dictionaries. Instead, this benchmark:

1. Creates a single train/test split as CSV files on disk
2. Builds rating and implicit matrices in streaming passes
3. Trains the collaborative model (BPR/SVD/ALS)
4. Evaluates collaborative-only metrics on the held-out test split

The output JSON intentionally mirrors the compact benchmark format already used
for the 200k experiments so the results are easy to compare.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import DATASET_PATH, MODELS_DIR, data_config, model_config
from evaluate_models import calculate_all_metrics
from models.collaborative import MatrixFactorization
from models.implicit import ALSImplicit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


RATING_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "rating": "int8",
}

ANIMELIST_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "watching_status": "int8",
    "watched_episodes": "float32",
}


def _write_frame(df: pd.DataFrame, filepath: Path, header: bool = False) -> None:
    """Append a dataframe to a CSV file."""
    if df.empty:
        return
    df.to_csv(filepath, mode="a", header=header, index=False)


def _process_split_group(
    group: pd.DataFrame,
    rng: np.random.Generator,
    relevance_threshold: float,
    min_train_items: int,
    min_test_items: int,
    leave_one_out: bool,
    test_ratio: float,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
    """Split one user's rating history into train/test dataframes."""
    group = group.drop_duplicates(subset=["anime_id"], keep="last")

    if len(group) < min_train_items:
        return None, None, False

    anime_ids = group["anime_id"].to_numpy(dtype=np.int32, copy=True)
    ratings = group["rating"].to_numpy(dtype=np.float32, copy=True)
    relevant_items = anime_ids[ratings >= relevance_threshold].copy()

    required_relevant = 2 if leave_one_out else (min_test_items + 1)
    if len(relevant_items) < required_relevant:
        return None, None, False

    rng.shuffle(relevant_items)

    if leave_one_out:
        test_items = relevant_items[:1]
    else:
        n_test = max(min_test_items, int(len(relevant_items) * test_ratio))
        n_test = min(n_test, len(relevant_items) - 1)
        test_items = relevant_items[:n_test]

    test_mask = np.isin(anime_ids, test_items)
    train_mask = ~test_mask

    if int(train_mask.sum()) < min_train_items:
        return None, None, False
    if int(test_mask.sum()) < (1 if leave_one_out else min_test_items):
        return None, None, False

    user_ids = group["user_id"].to_numpy(dtype=np.int32, copy=False)

    train_df = pd.DataFrame(
        {
            "user_id": user_ids[train_mask],
            "anime_id": anime_ids[train_mask],
            "rating": ratings[train_mask].astype(np.float32, copy=False),
        }
    )
    test_df = pd.DataFrame(
        {
            "user_id": user_ids[test_mask],
            "anime_id": anime_ids[test_mask],
            "rating": ratings[test_mask].astype(np.float32, copy=False),
        }
    )

    return train_df, test_df, True


def create_split_files(
    ratings_file: Path,
    split_dir: Path,
    relevance_threshold: float,
    min_train_items: int,
    min_test_items: int,
    leave_one_out: bool,
    test_ratio: float,
    random_state: int,
    chunksize: int,
    max_rows: Optional[int] = None,
) -> Dict[str, object]:
    """Create persisted train/test split CSVs from the ratings file."""
    split_dir.mkdir(parents=True, exist_ok=True)
    train_csv = split_dir / "train_ratings.csv"
    test_csv = split_dir / "test_ratings.csv"
    metadata_path = split_dir / "metadata.json"

    if train_csv.exists():
        train_csv.unlink()
    if test_csv.exists():
        test_csv.unlink()

    header_df = pd.DataFrame(columns=["user_id", "anime_id", "rating"])
    header_df.to_csv(train_csv, index=False)
    header_df.to_csv(test_csv, index=False)

    rng = np.random.default_rng(random_state)
    carryover = pd.DataFrame(columns=["user_id", "anime_id", "rating"])

    source_rows = 0
    eval_users = 0
    train_interactions = 0
    test_interactions = 0

    for chunk in pd.read_csv(
        ratings_file,
        usecols=["user_id", "anime_id", "rating"],
        dtype=RATING_DTYPES,
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["rating"] > 0]
        if max_rows is not None and source_rows >= max_rows:
            break
        if max_rows is not None and source_rows + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - source_rows]

        source_rows += len(chunk)
        if not carryover.empty:
            chunk = pd.concat([carryover, chunk], ignore_index=True)

        if chunk.empty:
            continue

        last_user = int(chunk.iloc[-1]["user_id"])
        carryover = chunk[chunk["user_id"] == last_user].copy()
        complete = chunk[chunk["user_id"] != last_user]

        if complete.empty:
            continue

        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []

        for _, group in complete.groupby("user_id", sort=False):
            train_df, test_df, accepted = _process_split_group(
                group,
                rng=rng,
                relevance_threshold=relevance_threshold,
                min_train_items=min_train_items,
                min_test_items=min_test_items,
                leave_one_out=leave_one_out,
                test_ratio=test_ratio,
            )
            if not accepted:
                continue
            train_parts.append(train_df)
            test_parts.append(test_df)
            eval_users += 1
            train_interactions += len(train_df)
            test_interactions += len(test_df)

        if train_parts:
            _write_frame(pd.concat(train_parts, ignore_index=True), train_csv)
        if test_parts:
            _write_frame(pd.concat(test_parts, ignore_index=True), test_csv)

        logger.info(
            "Split progress: source_rows=%s eval_users=%s train=%s test=%s",
            source_rows,
            eval_users,
            train_interactions,
            test_interactions,
        )

    if not carryover.empty:
        train_df, test_df, accepted = _process_split_group(
            carryover,
            rng=rng,
            relevance_threshold=relevance_threshold,
            min_train_items=min_train_items,
            min_test_items=min_test_items,
            leave_one_out=leave_one_out,
            test_ratio=test_ratio,
        )
        if accepted:
            _write_frame(train_df, train_csv)
            _write_frame(test_df, test_csv)
            eval_users += 1
            train_interactions += len(train_df)
            test_interactions += len(test_df)

    metadata = {
        "version": 2,
        "format": "csv_split",
        "random_state": random_state,
        "test_ratio": test_ratio,
        "relevance_threshold": relevance_threshold,
        "leave_one_out": leave_one_out,
        "min_train_items": min_train_items,
        "min_test_items": min_test_items,
        "source_rows": int(source_rows),
        "eval_users": int(eval_users),
        "train_interactions": int(train_interactions),
        "test_interactions": int(test_interactions),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_or_create_split_files(
    ratings_file: Path,
    split_dir: Path,
    relevance_threshold: float,
    min_train_items: int,
    min_test_items: int,
    leave_one_out: bool,
    test_ratio: float,
    random_state: int,
    chunksize: int,
    force_resplit: bool,
    max_rows: Optional[int] = None,
) -> Dict[str, object]:
    """Reuse an existing split or create it on demand."""
    metadata_path = split_dir / "metadata.json"
    train_csv = split_dir / "train_ratings.csv"
    test_csv = split_dir / "test_ratings.csv"

    if not force_resplit and metadata_path.exists() and train_csv.exists() and test_csv.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    return create_split_files(
        ratings_file=ratings_file,
        split_dir=split_dir,
        relevance_threshold=relevance_threshold,
        min_train_items=min_train_items,
        min_test_items=min_test_items,
        leave_one_out=leave_one_out,
        test_ratio=test_ratio,
        random_state=random_state,
        chunksize=chunksize,
        max_rows=max_rows,
    )


def compute_valid_ids(
    train_csv: Path,
    min_user_ratings: int,
    min_anime_ratings: int,
    chunksize: int,
) -> Tuple[Set[int], Set[int]]:
    """Compute valid users/items for the rating matrix filters."""
    user_counts: Dict[int, int] = defaultdict(int)
    anime_counts: Dict[int, int] = defaultdict(int)

    for chunk in pd.read_csv(train_csv, dtype={"user_id": "int32", "anime_id": "int32", "rating": "float32"}, chunksize=chunksize):
        user_vc = chunk["user_id"].value_counts(sort=False)
        anime_vc = chunk["anime_id"].value_counts(sort=False)

        for user_id, count in user_vc.items():
            user_counts[int(user_id)] += int(count)
        for anime_id, count in anime_vc.items():
            anime_counts[int(anime_id)] += int(count)

    valid_users = {
        int(user_id)
        for user_id, count in user_counts.items()
        if count >= min_user_ratings
    }
    valid_anime = {
        int(anime_id)
        for anime_id, count in anime_counts.items()
        if count >= min_anime_ratings
    }

    return valid_users, valid_anime


def build_rating_matrix_from_split(
    train_csv: Path,
    min_user_ratings: int,
    min_anime_ratings: int,
    chunksize: int,
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], int]:
    """Build the collaborative rating matrix from the train split CSV."""
    valid_users, valid_anime = compute_valid_ids(
        train_csv=train_csv,
        min_user_ratings=min_user_ratings,
        min_anime_ratings=min_anime_ratings,
        chunksize=chunksize,
    )

    user_to_idx: Dict[int, int] = {}
    idx_to_user: Dict[int, int] = {}
    anime_to_idx: Dict[int, int] = {}
    idx_to_anime: Dict[int, int] = {}

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []

    nnz = 0

    for chunk in pd.read_csv(
        train_csv,
        dtype={"user_id": "int32", "anime_id": "int32", "rating": "float32"},
        chunksize=chunksize,
    ):
        filtered = chunk[
            chunk["user_id"].isin(valid_users) & chunk["anime_id"].isin(valid_anime)
        ]
        if filtered.empty:
            continue

        for user_id in filtered["user_id"].unique():
            user_id = int(user_id)
            if user_id not in user_to_idx:
                idx = len(user_to_idx)
                user_to_idx[user_id] = idx
                idx_to_user[idx] = user_id

        for anime_id in filtered["anime_id"].unique():
            anime_id = int(anime_id)
            if anime_id not in anime_to_idx:
                idx = len(anime_to_idx)
                anime_to_idx[anime_id] = idx
                idx_to_anime[idx] = anime_id

        rows.append(filtered["user_id"].map(user_to_idx).to_numpy(dtype=np.int32, copy=False))
        cols.append(filtered["anime_id"].map(anime_to_idx).to_numpy(dtype=np.int32, copy=False))
        data.append(filtered["rating"].to_numpy(dtype=np.float32, copy=False))
        nnz += len(filtered)

    if not rows:
        raise ValueError("No training ratings survived the matrix filters.")

    row_indices = np.concatenate(rows)
    col_indices = np.concatenate(cols)
    values = np.concatenate(data)

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(user_to_idx), len(anime_to_idx)),
    )

    return matrix, anime_to_idx, idx_to_anime, user_to_idx, idx_to_user, nnz


def load_test_items_map(test_csv: Path, chunksize: int) -> Dict[int, Set[int]]:
    """Load held-out test items keyed by user."""
    user_test: Dict[int, Set[int]] = defaultdict(set)

    for chunk in pd.read_csv(
        test_csv,
        dtype={"user_id": "int32", "anime_id": "int32", "rating": "float32"},
        chunksize=chunksize,
    ):
        for user_id, group in chunk.groupby("user_id", sort=False):
            user_test[int(user_id)].update(int(anime_id) for anime_id in group["anime_id"].tolist())

    return dict(user_test)


def _finish_animelist_chunk(
    chunk: pd.DataFrame,
    carryover: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split an animelist chunk into complete users and carryover."""
    if not carryover.empty:
        chunk = pd.concat([carryover, chunk], ignore_index=True)

    if chunk.empty:
        return chunk, pd.DataFrame(columns=chunk.columns)

    last_user = int(chunk.iloc[-1]["user_id"])
    new_carryover = chunk[chunk["user_id"] == last_user].copy()
    complete = chunk[chunk["user_id"] != last_user]
    return complete, new_carryover


def build_implicit_matrix_from_animelist(
    animelist_file: Path,
    user_to_idx: Dict[int, int],
    anime_to_idx: Dict[int, int],
    heldout_test: Dict[int, Set[int]],
    chunksize: int,
    max_rows: Optional[int] = None,
) -> Tuple[csr_matrix, int]:
    """Stream animelist.csv and build the implicit matrix."""
    status_weights = {
        1: 0.8,
        2: 1.0,
        3: 0.5,
        4: 0.2,
        6: 0.1,
    }

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    kept_rows = 0
    consumed_rows = 0

    carryover = pd.DataFrame(columns=["user_id", "anime_id", "watching_status", "watched_episodes"])
    valid_anime_ids = set(anime_to_idx.keys())

    for chunk in pd.read_csv(
        animelist_file,
        usecols=["user_id", "anime_id", "watching_status", "watched_episodes"],
        dtype=ANIMELIST_DTYPES,
        chunksize=chunksize,
    ):
        if max_rows is not None and consumed_rows >= max_rows:
            break
        if max_rows is not None and consumed_rows + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - consumed_rows]

        consumed_rows += len(chunk)
        complete, carryover = _finish_animelist_chunk(chunk, carryover)
        if complete.empty:
            continue

        for user_id, group in complete.groupby("user_id", sort=False):
            user_id = int(user_id)
            user_idx = user_to_idx.get(user_id)
            if user_idx is None:
                continue

            filtered = group[group["anime_id"].isin(valid_anime_ids)].copy()
            if filtered.empty:
                continue

            heldout = heldout_test.get(user_id)
            if heldout:
                filtered = filtered[~filtered["anime_id"].isin(heldout)]
                if filtered.empty:
                    continue

            filtered["status_weight"] = filtered["watching_status"].map(status_weights).fillna(0.1)
            filtered["episode_weight"] = np.log1p(filtered["watched_episodes"].fillna(0)) / 10
            filtered["episode_weight"] = filtered["episode_weight"].clip(0, 1)
            filtered["implicit_score"] = (
                filtered["status_weight"] * 0.6 + filtered["episode_weight"] * 0.4
            ).astype(np.float32)

            rows.append(np.full(len(filtered), user_idx, dtype=np.int32))
            cols.append(filtered["anime_id"].map(anime_to_idx).to_numpy(dtype=np.int32, copy=False))
            data.append(filtered["implicit_score"].to_numpy(dtype=np.float32, copy=False))
            kept_rows += len(filtered)

        logger.info("Implicit stream progress: consumed_rows=%s kept_rows=%s", consumed_rows, kept_rows)

    if not carryover.empty:
        complete = carryover
        for user_id, group in complete.groupby("user_id", sort=False):
            user_id = int(user_id)
            user_idx = user_to_idx.get(user_id)
            if user_idx is None:
                continue

            filtered = group[group["anime_id"].isin(valid_anime_ids)].copy()
            if filtered.empty:
                continue

            heldout = heldout_test.get(user_id)
            if heldout:
                filtered = filtered[~filtered["anime_id"].isin(heldout)]
                if filtered.empty:
                    continue

            filtered["status_weight"] = filtered["watching_status"].map(status_weights).fillna(0.1)
            filtered["episode_weight"] = np.log1p(filtered["watched_episodes"].fillna(0)) / 10
            filtered["episode_weight"] = filtered["episode_weight"].clip(0, 1)
            filtered["implicit_score"] = (
                filtered["status_weight"] * 0.6 + filtered["episode_weight"] * 0.4
            ).astype(np.float32)

            rows.append(np.full(len(filtered), user_idx, dtype=np.int32))
            cols.append(filtered["anime_id"].map(anime_to_idx).to_numpy(dtype=np.int32, copy=False))
            data.append(filtered["implicit_score"].to_numpy(dtype=np.float32, copy=False))
            kept_rows += len(filtered)

    if not rows:
        raise ValueError("Implicit matrix is empty after filtering the full animelist.")

    row_indices = np.concatenate(rows)
    col_indices = np.concatenate(cols)
    values = np.concatenate(data)

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(user_to_idx), len(anime_to_idx)),
    )
    return matrix, kept_rows


def evaluate_collaborative_full(
    model: MatrixFactorization,
    rating_matrix: csr_matrix,
    idx_to_anime: Dict[int, int],
    user_test: Dict[int, Set[int]],
    k_values: List[int],
    max_users: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a collaborative model using the full split mapping."""
    results = {k: defaultdict(list) for k in k_values}
    evaluated = 0
    skipped_not_in_model = 0

    eval_users = list(user_test.keys())
    if max_users is not None and len(eval_users) > max_users:
        rng = np.random.default_rng(42)
        eval_users = list(rng.choice(eval_users, max_users, replace=False))

    max_k = max(k_values)

    for index, user_id in enumerate(eval_users, start=1):
        if user_id not in model.user_to_idx:
            skipped_not_in_model += 1
            continue

        user_idx = model.user_to_idx[user_id]
        train_indices = rating_matrix[user_idx].indices
        rated_items = {idx_to_anime[int(item_idx)] for item_idx in train_indices}
        test_items = user_test.get(user_id, set())

        if not test_items:
            continue

        recommendations = model.recommend_for_user(
            user_id,
            top_k=max_k + len(rated_items),
            exclude_rated=True,
            rated_items=rated_items,
        )
        rec_ids = [rec["mal_id"] for rec in recommendations if rec["mal_id"] not in rated_items]
        rec_ids = rec_ids[:max_k]
        if not rec_ids:
            continue

        for k in k_values:
            metrics = calculate_all_metrics(rec_ids, test_items, k)
            for metric, value in metrics.items():
                results[k][metric].append(value)

        evaluated += 1
        if index % 10000 == 0:
            logger.info("Evaluation progress: processed=%s evaluated=%s", index, evaluated)

    summary = {}
    for k in k_values:
        summary[f"K={k}"] = {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in results[k].items()
        }
        summary[f"K={k}"]["evaluated_users"] = evaluated

    summary["skipped_not_in_model"] = skipped_not_in_model
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark collaborative filtering on the full dataset")
    parser.add_argument("--cf-method", type=str, default="bpr", choices=["bpr", "svd", "als"])
    parser.add_argument("--output", type=str, default="full_dataset_collab_results.json")
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(MODELS_DIR / "full_collab_split"),
        help="Directory for persisted train/test split CSV files",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODELS_DIR / "full_collab_model"),
        help="Directory where trained collaborative artifacts will be saved",
    )
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--relevance-threshold", type=float, default=7.0)
    parser.add_argument("--min-train-items", type=int, default=10)
    parser.add_argument("--min-test-items", type=int, default=3)
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--split-chunksize", type=int, default=1_000_000)
    parser.add_argument("--matrix-chunksize", type=int, default=1_000_000)
    parser.add_argument("--animelist-chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--matrix-min-user-ratings",
        type=int,
        default=data_config.min_user_ratings,
    )
    parser.add_argument(
        "--matrix-min-anime-ratings",
        type=int,
        default=data_config.min_anime_ratings,
    )
    parser.add_argument(
        "--max-rating-rows",
        type=int,
        default=None,
        help="Optional cap for development smoke tests",
    )
    parser.add_argument(
        "--max-animelist-rows",
        type=int,
        default=None,
        help="Optional cap for development smoke tests",
    )
    return parser.parse_args()


def main() -> Dict[str, object]:
    args = parse_args()
    total_start = time.time()

    ratings_file = DATASET_PATH / "rating_complete.csv"
    animelist_file = DATASET_PATH / "animelist.csv"
    split_dir = Path(args.split_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing or loading the full-data split...")
    split_start = time.time()
    split_metadata = load_or_create_split_files(
        ratings_file=ratings_file,
        split_dir=split_dir,
        relevance_threshold=args.relevance_threshold,
        min_train_items=args.min_train_items,
        min_test_items=args.min_test_items,
        leave_one_out=args.leave_one_out,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        chunksize=args.split_chunksize,
        force_resplit=args.force_resplit,
        max_rows=args.max_rating_rows,
    )
    split_time = time.time() - split_start

    train_csv = Path(split_metadata["train_csv"])
    test_csv = Path(split_metadata["test_csv"])

    logger.info("Loading held-out test items...")
    test_map_start = time.time()
    user_test = load_test_items_map(test_csv, chunksize=args.matrix_chunksize)
    test_map_time = time.time() - test_map_start

    logger.info("Building rating matrix from the persisted split...")
    rating_matrix_start = time.time()
    (
        rating_matrix,
        anime_to_idx,
        idx_to_anime,
        user_to_idx,
        idx_to_user,
        rating_nnz,
    ) = build_rating_matrix_from_split(
        train_csv=train_csv,
        min_user_ratings=args.matrix_min_user_ratings,
        min_anime_ratings=args.matrix_min_anime_ratings,
        chunksize=args.matrix_chunksize,
    )
    rating_matrix_time = time.time() - rating_matrix_start

    implicit_model = None
    implicit_matrix = None
    implicit_rows = 0
    implicit_train_time = 0.0

    if args.cf_method == "bpr":
        logger.info("Building implicit matrix from animelist.csv...")
        implicit_matrix_start = time.time()
        implicit_matrix, implicit_rows = build_implicit_matrix_from_animelist(
            animelist_file=animelist_file,
            user_to_idx=user_to_idx,
            anime_to_idx=anime_to_idx,
            heldout_test=user_test,
            chunksize=args.animelist_chunksize,
            max_rows=args.max_animelist_rows,
        )
        implicit_matrix_time = time.time() - implicit_matrix_start

        logger.info("Training ALS implicit warm-start model...")
        implicit_train_start = time.time()
        implicit_model = ALSImplicit(
            n_factors=model_config.implicit_factors,
            n_iterations=model_config.implicit_iterations,
            regularization=model_config.implicit_regularization,
        )
        implicit_model.fit(
            implicit_matrix,
            anime_to_idx,
            idx_to_anime,
            user_to_idx,
            idx_to_user,
        )
        implicit_train_time = time.time() - implicit_train_start
    else:
        implicit_matrix_time = 0.0

    logger.info("Training collaborative model (%s)...", args.cf_method.upper())
    collab_train_start = time.time()
    if args.cf_method == "bpr":
        collab_model = MatrixFactorization(
            n_factors=model_config.bpr_factors,
            n_epochs=model_config.bpr_iterations,
            learning_rate=model_config.bpr_learning_rate,
            regularization=model_config.bpr_regularization,
            method="bpr",
            rating_positive_threshold=args.relevance_threshold,
            verify_negative_samples=model_config.bpr_verify_negative_samples,
            use_implicit_signal=model_config.bpr_use_implicit_signal,
            warm_start_from_als=model_config.bpr_warm_start_from_als,
        )
    else:
        collab_model = MatrixFactorization(
            n_factors=model_config.svd_factors,
            n_epochs=model_config.svd_epochs,
            learning_rate=model_config.svd_lr,
            regularization=model_config.svd_reg,
            method=args.cf_method,
        )

    collab_model.fit(
        rating_matrix,
        anime_to_idx,
        idx_to_anime,
        user_to_idx,
        idx_to_user,
        verbose=True,
        implicit_matrix=implicit_matrix,
        implicit_model=implicit_model,
    )
    collab_train_time = time.time() - collab_train_start

    model_path = model_dir / "collaborative_model.pkl"
    collab_model.save(model_path)
    if implicit_model is not None:
        implicit_model.save(model_dir / "implicit_model.pkl")

    logger.info("Evaluating collaborative model...")
    eval_start = time.time()
    results = evaluate_collaborative_full(
        model=collab_model,
        rating_matrix=rating_matrix,
        idx_to_anime=idx_to_anime,
        user_test=user_test,
        k_values=args.k,
        max_users=args.max_eval_users,
    )
    eval_time = time.time() - eval_start

    total_time = time.time() - total_start
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_dir": str(split_dir),
        "model_dir": str(model_dir),
        "model": {
            "method": args.cf_method,
            "n_factors": collab_model.n_factors,
            "n_epochs": collab_model.n_epochs,
            "learning_rate": collab_model.learning_rate,
            "regularization": collab_model.regularization,
            "rating_positive_threshold": args.relevance_threshold,
            "use_implicit_signal": getattr(collab_model, "use_implicit_signal", False),
            "warm_start_from_als": getattr(collab_model, "warm_start_from_als", False),
        },
        "split_metadata": split_metadata,
        "data": {
            "train_users_after_filters": len(user_to_idx),
            "train_items_after_filters": len(anime_to_idx),
            "train_ratings_after_filters": int(rating_nnz),
            "heldout_eval_users": int(len(user_test)),
            "heldout_test_interactions": int(sum(len(items) for items in user_test.values())),
            "implicit_interactions": int(implicit_rows),
        },
        "timings_seconds": {
            "split_creation_or_load": split_time,
            "test_map_load": test_map_time,
            "rating_matrix": rating_matrix_time,
            "implicit_matrix": implicit_matrix_time,
            "implicit_train": implicit_train_time,
            "collaborative_train": collab_train_time,
            "evaluation": eval_time,
            "total": total_time,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
