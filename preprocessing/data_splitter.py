"""
Utilities for creating and persisting a single train/evaluation split.
"""

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_RATING_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "rating": "int8",
}

_ANIMELIST_DTYPES = {
    "user_id": "int32",
    "anime_id": "int32",
    "watching_status": "int8",
    "watched_episodes": "int32",
}


@dataclass
class RatingsUserSplit:
    """Persisted user-level split used by both training and evaluation."""

    user_train: Dict[int, Dict[int, float]]
    user_test: Dict[int, Set[int]]
    eval_users: List[int]
    metadata: Dict[str, Union[int, float, bool, str]]


@dataclass
class RatingsDiskSplit:
    """On-disk split manifest for streaming training/evaluation."""

    manifest_path: Path
    train_ratings_path: Path
    test_ratings_path: Path
    train_animelist_path: Path
    metadata: Dict[str, Union[int, float, bool, str]] = field(default_factory=dict)


def get_ratings_disk_split(split_path: Union[str, Path]) -> RatingsDiskSplit:
    """
    Resolve all on-disk split artifact paths from a single base path.

    Example:
      split_path = saved_models/splits/full_train_split.pkl
      -> full_train_split.train_ratings.csv
      -> full_train_split.test_ratings.csv
      -> full_train_split.train_animelist.csv
    """
    manifest_path = Path(split_path)
    base_path = manifest_path.with_suffix("") if manifest_path.suffix else manifest_path

    return RatingsDiskSplit(
        manifest_path=manifest_path,
        train_ratings_path=base_path.parent / f"{base_path.name}.train_ratings.csv",
        test_ratings_path=base_path.parent / f"{base_path.name}.test_ratings.csv",
        train_animelist_path=base_path.parent / f"{base_path.name}.train_animelist.csv",
    )


def create_ratings_user_split(
    ratings_df: pd.DataFrame,
    min_train_items: int = 10,
    min_test_items: int = 3,
    test_ratio: float = 0.2,
    relevance_threshold: float = 7.0,
    leave_one_out: bool = False,
    random_state: int = 42,
) -> RatingsUserSplit:
    """
    Create a single user-level split for both training and evaluation.

    The resulting split satisfies:
    - training only sees `user_train`
    - evaluation only measures `user_test`
    - held-out interactions are removed from training
    """
    logger.info("Creating ratings split before training...")
    logger.info(
        "  test_ratio=%.3f, relevance_threshold=%.2f, leave_one_out=%s, random_state=%d",
        test_ratio,
        relevance_threshold,
        leave_one_out,
        random_state,
    )

    rng = np.random.default_rng(random_state)

    df = ratings_df[["user_id", "anime_id", "rating"]].copy()
    df = df[df["rating"] > 0]

    user_train: Dict[int, Dict[int, float]] = {}
    user_test: Dict[int, Set[int]] = {}
    eval_users: List[int] = []

    grouped = df.groupby("user_id", sort=True)

    for user_id, group in grouped:
        # Deduplicate user-item pairs deterministically if the source file contains repeats.
        group = group.drop_duplicates(subset=["anime_id"], keep="last")

        all_items = {
            int(anime_id): float(rating)
            for anime_id, rating in zip(group["anime_id"], group["rating"])
        }
        relevant_items = (
            group.loc[group["rating"] >= relevance_threshold, "anime_id"]
            .astype(np.int64)
            .to_numpy()
            .copy()
        )

        required_relevant = 2 if leave_one_out else (min_test_items + 1)
        if len(all_items) < min_train_items or len(relevant_items) < required_relevant:
            continue

        rng.shuffle(relevant_items)

        if leave_one_out:
            test_items = {int(relevant_items[0])}
        else:
            n_test = max(min_test_items, int(len(relevant_items) * test_ratio))
            n_test = min(n_test, len(relevant_items) - 1)
            test_items = {int(item_id) for item_id in relevant_items[:n_test]}

        train_items = {
            anime_id: rating
            for anime_id, rating in all_items.items()
            if anime_id not in test_items
        }

        if len(train_items) < min_train_items:
            continue
        if len(test_items) < (1 if leave_one_out else min_test_items):
            continue

        user_train[int(user_id)] = train_items
        user_test[int(user_id)] = test_items
        eval_users.append(int(user_id))

    train_interactions = sum(len(items) for items in user_train.values())
    test_interactions = sum(len(items) for items in user_test.values())

    metadata = {
        "version": 1,
        "random_state": random_state,
        "test_ratio": test_ratio,
        "relevance_threshold": relevance_threshold,
        "leave_one_out": leave_one_out,
        "min_train_items": min_train_items,
        "min_test_items": min_test_items,
        "source_rows": int(len(df)),
        "eval_users": int(len(eval_users)),
        "train_interactions": int(train_interactions),
        "test_interactions": int(test_interactions),
    }

    logger.info(
        "Created split with %d users, %d train interactions, %d test interactions",
        len(eval_users),
        train_interactions,
        test_interactions,
    )

    return RatingsUserSplit(
        user_train=user_train,
        user_test=user_test,
        eval_users=eval_users,
        metadata=metadata,
    )


def split_to_ratings_df(user_train: Dict[int, Dict[int, float]]) -> pd.DataFrame:
    """Convert the persisted train split back into a ratings dataframe."""
    user_ids: List[int] = []
    anime_ids: List[int] = []
    ratings: List[float] = []

    for user_id, items in user_train.items():
        for anime_id, rating in items.items():
            user_ids.append(int(user_id))
            anime_ids.append(int(anime_id))
            ratings.append(float(rating))

    train_df = pd.DataFrame(
        {
            "user_id": np.asarray(user_ids, dtype=np.int32),
            "anime_id": np.asarray(anime_ids, dtype=np.int32),
            "rating": np.asarray(ratings, dtype=np.float32),
        }
    )

    logger.info("Converted user_train split to dataframe with %d rows", len(train_df))
    return train_df


def filter_holdout_interactions(
    interactions_df: pd.DataFrame,
    user_test: Dict[int, Set[int]],
    user_col: str = "user_id",
    item_col: str = "anime_id",
) -> pd.DataFrame:
    """
    Remove held-out test interactions from another interaction table.

    This is used to prevent models trained from auxiliary tables (for example
    `animelist.csv`) from seeing the same user-item pairs held out for testing.
    """
    if interactions_df.empty or not user_test:
        return interactions_df.copy()

    holdout_lookup = {
        int(user_id): {int(anime_id) for anime_id in items}
        for user_id, items in user_test.items()
        if items
    }
    if not holdout_lookup:
        return interactions_df.copy()

    holdout_users = list(holdout_lookup.keys())
    candidate_mask = interactions_df[user_col].isin(holdout_users).to_numpy()
    if not candidate_mask.any():
        return interactions_df.copy()

    keep_mask = np.ones(len(interactions_df), dtype=bool)
    candidate_df = interactions_df.loc[candidate_mask, [user_col, item_col]].copy()
    candidate_df["_pos"] = np.flatnonzero(candidate_mask)

    for user_id, group in candidate_df.groupby(user_col, sort=False):
        held_out_items = holdout_lookup.get(int(user_id))
        if not held_out_items:
            continue

        drop_positions = group.loc[
            group[item_col].isin(held_out_items), "_pos"
        ].to_numpy(dtype=np.int64, copy=False)
        keep_mask[drop_positions] = False

    filtered_df = interactions_df.loc[keep_mask].copy()
    logger.info(
        "Filtered %d held-out rows from the interaction table",
        len(interactions_df) - len(filtered_df),
    )
    return filtered_df


def extract_holdout_ratings_df(
    ratings_df: pd.DataFrame,
    user_test: Dict[int, Set[int]],
    user_col: str = "user_id",
    item_col: str = "anime_id",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Recover the held-out ratings rows that correspond to `user_test`.

    This is primarily used by the learned hybrid meta-model, which needs the
    original rating values of the held-out positives instead of just the item IDs.
    """
    empty_cols = [user_col, item_col, rating_col]
    if ratings_df.empty or not user_test:
        return pd.DataFrame(columns=empty_cols)

    holdout_lookup = {
        int(user_id): {int(anime_id) for anime_id in items}
        for user_id, items in user_test.items()
        if items
    }
    if not holdout_lookup:
        return pd.DataFrame(columns=empty_cols)

    candidate_mask = ratings_df[user_col].isin(list(holdout_lookup.keys()))
    candidate_df = ratings_df.loc[candidate_mask, empty_cols].copy()
    if candidate_df.empty:
        return candidate_df

    matched_parts = []
    for user_id, group in candidate_df.groupby(user_col, sort=False):
        held_out_items = holdout_lookup.get(int(user_id))
        if not held_out_items:
            continue
        matched = group.loc[group[item_col].isin(held_out_items)]
        if not matched.empty:
            matched_parts.append(matched)

    if not matched_parts:
        return pd.DataFrame(columns=empty_cols)

    holdout_df = pd.concat(matched_parts, ignore_index=True)
    logger.info("Extracted %d held-out rating rows", len(holdout_df))
    return holdout_df


def save_ratings_user_split(split: RatingsUserSplit, filepath: Union[str, Path]) -> None:
    """Save the split artifact to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "user_train": split.user_train,
        "user_test": split.user_test,
        "eval_users": split.eval_users,
        "metadata": split.metadata,
    }

    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved ratings split to %s", filepath)


def load_ratings_user_split(filepath: Union[str, Path]) -> RatingsUserSplit:
    """Load a previously persisted split artifact."""
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        payload = pickle.load(f)

    split = RatingsUserSplit(
        user_train=payload["user_train"],
        user_test=payload["user_test"],
        eval_users=payload["eval_users"],
        metadata=payload.get("metadata", {}),
    )

    logger.info("Loaded ratings split from %s", filepath)
    return split


def save_ratings_disk_split(split: RatingsDiskSplit) -> None:
    """Persist the on-disk split manifest."""
    split.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": split.metadata,
        "train_ratings_path": str(split.train_ratings_path),
        "test_ratings_path": str(split.test_ratings_path),
        "train_animelist_path": str(split.train_animelist_path),
    }

    with open(split.manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved on-disk split manifest to %s", split.manifest_path)


def load_ratings_disk_split(split_path: Union[str, Path]) -> RatingsDiskSplit:
    """Load an existing on-disk split manifest."""
    split = get_ratings_disk_split(split_path)

    with open(split.manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    split.train_ratings_path = Path(payload["train_ratings_path"])
    split.test_ratings_path = Path(payload["test_ratings_path"])
    split.train_animelist_path = Path(payload["train_animelist_path"])
    split.metadata = payload.get("metadata", {})

    logger.info("Loaded on-disk split manifest from %s", split.manifest_path)
    return split


def create_ratings_disk_split(
    ratings_csv_path: Union[str, Path],
    split_path: Union[str, Path],
    min_train_items: int = 10,
    min_test_items: int = 3,
    test_ratio: float = 0.2,
    relevance_threshold: float = 7.0,
    leave_one_out: bool = False,
    random_state: int = 42,
    chunk_size: int = 500_000,
) -> RatingsDiskSplit:
    """
    Stream the ratings file once to plan the split, then write train/test CSVs.

    This avoids materializing the full ratings dataframe in memory.
    """
    split = get_ratings_disk_split(split_path)
    split.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for path in [split.train_ratings_path, split.test_ratings_path]:
        if path.exists():
            path.unlink()

    logger.info("Creating on-disk ratings split from %s", ratings_csv_path)
    logger.info(
        "  chunk_size=%d, test_ratio=%.3f, threshold=%.2f, leave_one_out=%s",
        chunk_size,
        test_ratio,
        relevance_threshold,
        leave_one_out,
    )

    ratings_csv_path = Path(ratings_csv_path)
    rng = np.random.default_rng(random_state)
    required_relevant = 2 if leave_one_out else (min_test_items + 1)

    user_total_counts: Dict[int, int] = defaultdict(int)
    user_relevant_items: Dict[int, List[int]] = defaultdict(list)
    source_rows = 0

    # Pass 1: gather per-user counts and relevant items only.
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            ratings_csv_path,
            chunksize=chunk_size,
            usecols=["user_id", "anime_id", "rating"],
            dtype=_RATING_DTYPES,
        ),
        start=1,
    ):
        chunk = chunk.loc[chunk["rating"] > 0, ["user_id", "anime_id", "rating"]]
        if chunk.empty:
            continue

        source_rows += len(chunk)

        user_counts = chunk["user_id"].value_counts(sort=False)
        for user_id, count in user_counts.items():
            user_total_counts[int(user_id)] += int(count)

        relevant_chunk = chunk.loc[
            chunk["rating"] >= relevance_threshold, ["user_id", "anime_id"]
        ]
        for user_id, group in relevant_chunk.groupby("user_id", sort=False):
            user_relevant_items[int(user_id)].extend(
                group["anime_id"].astype(np.int64).tolist()
            )

        if chunk_idx % 20 == 0:
            logger.info("  Planned split on %d chunks...", chunk_idx)

    user_test_lookup: Dict[int, Set[int]] = {}
    eval_users: List[int] = []
    train_interactions = 0
    test_interactions = 0

    for user_id, total_count in user_total_counts.items():
        relevant_items = user_relevant_items.get(user_id)
        if total_count < min_train_items or not relevant_items:
            continue

        relevant_array = np.asarray(relevant_items, dtype=np.int64)
        if len(relevant_array) < required_relevant:
            continue

        rng.shuffle(relevant_array)

        if leave_one_out:
            test_items = {int(relevant_array[0])}
        else:
            n_test = max(min_test_items, int(len(relevant_array) * test_ratio))
            n_test = min(n_test, len(relevant_array) - 1)
            test_items = {int(item_id) for item_id in relevant_array[:n_test]}

        if len(test_items) < (1 if leave_one_out else min_test_items):
            continue

        train_count = total_count - len(test_items)
        if train_count < min_train_items:
            continue

        user_test_lookup[int(user_id)] = test_items
        eval_users.append(int(user_id))
        train_interactions += int(train_count)
        test_interactions += int(len(test_items))

    eval_user_set = set(eval_users)
    logger.info(
        "Split plan ready: %d users, %d train interactions, %d test interactions",
        len(eval_users),
        train_interactions,
        test_interactions,
    )

    # Pass 2: write train/test CSVs for the selected users only.
    train_rows_written = 0
    test_rows_written = 0
    train_header_written = False
    test_header_written = False

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            ratings_csv_path,
            chunksize=chunk_size,
            usecols=["user_id", "anime_id", "rating"],
            dtype=_RATING_DTYPES,
        ),
        start=1,
    ):
        chunk = chunk.loc[chunk["rating"] > 0, ["user_id", "anime_id", "rating"]]
        if chunk.empty:
            continue

        chunk = chunk.loc[chunk["user_id"].isin(eval_user_set)]
        if chunk.empty:
            continue

        users = chunk["user_id"].astype(np.int64).to_numpy(copy=False)
        items = chunk["anime_id"].astype(np.int64).to_numpy(copy=False)

        is_test = np.fromiter(
            (
                int(item_id) in user_test_lookup.get(int(user_id), set())
                for user_id, item_id in zip(users, items)
            ),
            dtype=bool,
            count=len(chunk),
        )

        train_chunk = chunk.loc[~is_test]
        test_chunk = chunk.loc[is_test]

        if not train_chunk.empty:
            train_chunk.to_csv(
                split.train_ratings_path,
                mode="a",
                header=not train_header_written,
                index=False,
            )
            train_header_written = True
            train_rows_written += len(train_chunk)

        if not test_chunk.empty:
            test_chunk.to_csv(
                split.test_ratings_path,
                mode="a",
                header=not test_header_written,
                index=False,
            )
            test_header_written = True
            test_rows_written += len(test_chunk)

        if chunk_idx % 20 == 0:
            logger.info("  Wrote split data for %d chunks...", chunk_idx)

    split.metadata = {
        "version": 2,
        "format": "csv_streaming",
        "random_state": random_state,
        "test_ratio": test_ratio,
        "relevance_threshold": relevance_threshold,
        "leave_one_out": leave_one_out,
        "min_train_items": min_train_items,
        "min_test_items": min_test_items,
        "source_rows": int(source_rows),
        "eval_users": int(len(eval_users)),
        "train_interactions": int(train_rows_written),
        "test_interactions": int(test_rows_written),
        "chunk_size": int(chunk_size),
    }
    save_ratings_disk_split(split)

    logger.info(
        "Created on-disk split: train=%s rows, test=%s rows",
        train_rows_written,
        test_rows_written,
    )
    return split


def load_user_test_lookup_from_test_csv(
    test_ratings_path: Union[str, Path],
    chunk_size: int = 500_000,
) -> Dict[int, Set[int]]:
    """Load held-out user-item pairs from the persisted test CSV."""
    lookup: Dict[int, Set[int]] = defaultdict(set)

    for chunk in pd.read_csv(
        test_ratings_path,
        chunksize=chunk_size,
        usecols=["user_id", "anime_id"],
        dtype={"user_id": "int32", "anime_id": "int32"},
    ):
        for user_id, group in chunk.groupby("user_id", sort=False):
            lookup[int(user_id)].update(group["anime_id"].astype(np.int64).tolist())

    return dict(lookup)


def filter_animelist_to_disk(
    animelist_csv_path: Union[str, Path],
    test_ratings_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    chunk_size: int = 500_000,
) -> Path:
    """
    Stream-filter animelist rows so held-out user-item pairs never reach training.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if output_csv_path.exists():
        output_csv_path.unlink()

    holdout_lookup = load_user_test_lookup_from_test_csv(
        test_ratings_path, chunk_size=chunk_size
    )
    holdout_users = set(holdout_lookup.keys())

    header_written = False
    filtered_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            animelist_csv_path,
            chunksize=chunk_size,
            usecols=["user_id", "anime_id", "watching_status", "watched_episodes"],
            dtype=_ANIMELIST_DTYPES,
        ),
        start=1,
    ):
        if not holdout_users:
            filtered_chunk = chunk
        else:
            candidate_mask = chunk["user_id"].isin(holdout_users).to_numpy()
            if candidate_mask.any():
                keep_mask = np.ones(len(chunk), dtype=bool)
                candidate_df = chunk.loc[candidate_mask, ["user_id", "anime_id"]].copy()
                candidate_df["_pos"] = np.flatnonzero(candidate_mask)

                for user_id, group in candidate_df.groupby("user_id", sort=False):
                    held_out_items = holdout_lookup.get(int(user_id))
                    if not held_out_items:
                        continue
                    drop_positions = group.loc[
                        group["anime_id"].isin(held_out_items), "_pos"
                    ].to_numpy(dtype=np.int64, copy=False)
                    keep_mask[drop_positions] = False

                filtered_chunk = chunk.loc[keep_mask]
            else:
                filtered_chunk = chunk

        if not filtered_chunk.empty:
            filtered_chunk.to_csv(
                output_csv_path,
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True
            filtered_rows += len(filtered_chunk)

        if chunk_idx % 20 == 0:
            logger.info("  Filtered animelist chunk %d...", chunk_idx)

    logger.info(
        "Filtered animelist written to %s (%d rows kept)",
        output_csv_path,
        filtered_rows,
    )
    return output_csv_path
