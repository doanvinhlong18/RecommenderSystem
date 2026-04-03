"""
Utilities for creating and persisting a single train/evaluation split.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RatingsUserSplit:
    """Persisted user-level split used by both training and evaluation."""

    user_train: Dict[int, Dict[int, float]]
    user_test: Dict[int, Set[int]]
    eval_users: List[int]
    metadata: Dict[str, Union[int, float, bool, str]]


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
