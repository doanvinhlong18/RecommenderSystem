"""
Data loader module for loading and preprocessing anime datasets.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_PATH, CACHE_DIR, data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Explicit dtypes — avoids pandas defaulting everything to int64/object.
# int32 vs int64 halves memory for ID columns (100M rows × 2 cols = ~400MB saved).
# int8 is enough for rating (values -1..10) and watching_status (1..6).
# ---------------------------------------------------------------------------
_RATING_DTYPES = {
    'user_id':  'int32',
    'anime_id': 'int32',
    'rating':   'int8',
}

_ANIMELIST_DTYPES = {
    'user_id':          'int32',
    'anime_id':         'int32',
    'watching_status':  'int8',
    'watched_episodes': 'int32',
}


class DataLoader:
    """
    Handles loading and preprocessing of all anime datasets.

    Attributes:
        dataset_path: Path to the dataset directory
        anime_df: DataFrame containing anime metadata
        synopsis_df: DataFrame containing anime synopses
        ratings_df: DataFrame containing user ratings
        animelist_df: DataFrame containing user watch lists
        watching_status_df: DataFrame containing status mappings
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH
        self.anime_df: Optional[pd.DataFrame] = None
        self.synopsis_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.animelist_df: Optional[pd.DataFrame] = None
        self.watching_status_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # load_all
    # ------------------------------------------------------------------

    def load_all(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Load all datasets."""
        cache_file = CACHE_DIR / "loaded_data.pkl"

        if use_cache and cache_file.exists():
            logger.info("Loading data from cache...")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                self.anime_df = cached_data["anime"]
                self.synopsis_df = cached_data["synopsis"]
                self.ratings_df = cached_data["ratings"]
                self.animelist_df = cached_data["animelist"]
                self.watching_status_df = cached_data["watching_status"]
                return cached_data

        logger.info("Loading datasets from files...")
        self.load_anime()
        self.load_synopsis()
        self.load_ratings()
        self.load_animelist()
        self.load_watching_status()

        data_dict = {
            "anime": self.anime_df,
            "synopsis": self.synopsis_df,
            "ratings": self.ratings_df,
            "animelist": self.animelist_df,
            "watching_status": self.watching_status_df,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info(f"Data cached to {cache_file}")
        return data_dict

    # ------------------------------------------------------------------
    # load_anime / load_synopsis / load_watching_status  (unchanged)
    # ------------------------------------------------------------------

    def load_anime(self) -> pd.DataFrame:
        logger.info("Loading anime.csv...")
        file_path = self.dataset_path / data_config.anime_file
        self.anime_df = pd.read_csv(file_path)
        self.anime_df = self._clean_anime_data(self.anime_df)
        logger.info(f"Loaded {len(self.anime_df)} anime records")
        return self.anime_df

    def load_synopsis(self) -> pd.DataFrame:
        logger.info("Loading anime_with_synopsis.csv...")
        file_path = self.dataset_path / data_config.anime_synopsis_file
        self.synopsis_df = pd.read_csv(file_path)
        if 'sypnopsis' in self.synopsis_df.columns:
            self.synopsis_df = self.synopsis_df.rename(columns={'sypnopsis': 'synopsis'})
        self.synopsis_df['synopsis'] = self.synopsis_df['synopsis'].fillna('')
        self.synopsis_df['synopsis'] = self.synopsis_df['synopsis'].apply(self._clean_text)
        logger.info(f"Loaded {len(self.synopsis_df)} synopses")
        return self.synopsis_df

    def load_watching_status(self) -> pd.DataFrame:
        logger.info("Loading watching_status.csv...")
        file_path = self.dataset_path / data_config.watching_status_file
        self.watching_status_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.watching_status_df)} status mappings")
        return self.watching_status_df

    # ------------------------------------------------------------------
    # load_ratings  — fixed dtype + real sample-per-chunk
    # ------------------------------------------------------------------

    def load_ratings(self, sample: bool = None) -> pd.DataFrame:
        """
        Load user ratings.

        When sample=True AND data_config.rating_sample_size is set:
            Reads only enough chunks to reach (sample_size * 2) rows,
            then down-samples to sample_size.  No full-file scan.

        When sample=True but sample_size is None  (was broken before):
            Falls through to load all — same as sample=False.

        Memory optimizations vs original:
            - Explicit int32/int8 dtypes  (~50% less RAM per chunk)
            - chunksize 500_000 instead of 3_000_000 (smoother memory curve)
            - del chunks list after concat
        """
        logger.info("Loading rating_complete.csv...")
        file_path = self.dataset_path / data_config.rating_file

        use_sample = sample if sample is not None else data_config.sample_ratings
        sample_size = data_config.rating_sample_size

        if use_sample and sample_size is not None:
            logger.info(f"Sampling {sample_size:,} ratings (reading up to {sample_size * 2:,} rows)...")
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(
                file_path,
                chunksize=500_000,
                dtype=_RATING_DTYPES,
            ):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows >= sample_size * 2:
                    break

            self.ratings_df = pd.concat(chunks, ignore_index=True)
            del chunks
            if len(self.ratings_df) > sample_size:
                self.ratings_df = self.ratings_df.sample(n=sample_size, random_state=42)

        else:
            logger.info("Loading all ratings (chunked, dtype-optimized)...")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(
                file_path,
                chunksize=500_000,
                dtype=_RATING_DTYPES,
            ), start=1):
                chunks.append(chunk)
                logger.info(f"  Loaded chunk {i} ({len(chunk):,} records)")

            self.ratings_df = pd.concat(chunks, ignore_index=True)
            del chunks

        # Filter invalid ratings
        self.ratings_df = self.ratings_df[self.ratings_df['rating'] > 0]
        logger.info(f"Loaded {len(self.ratings_df):,} ratings")
        return self.ratings_df

    # ------------------------------------------------------------------
    # load_animelist  — fixed: sample actually works + dtype optimization
    # ------------------------------------------------------------------

    def load_animelist(self, sample: bool = None) -> pd.DataFrame:
        """
        Load user anime lists.

        Key fixes vs original:
            1. sample=True with sample_size=None used to silently load ALL 109M rows.
               Now when sample=True but sample_size is None, we default to 5M rows
               so 'sample=True' always means "give me a manageable subset".
            2. Explicit int32/int8 dtypes halve per-chunk RAM.
            3. chunksize 500_000 (vs 3M) keeps peak memory low during concat.
        """
        logger.info("Loading animelist.csv...")
        file_path = self.dataset_path / data_config.animelist_file

        use_sample = sample if sample is not None else data_config.sample_ratings
        sample_size = data_config.animelist_sample_size

        # --- Fix: provide a sensible default when sample=True but size unset ---
        if use_sample and sample_size is None:
            sample_size = 5_000_000
            logger.info(
                "animelist_sample_size not configured — defaulting to "
                f"{sample_size:,} rows for sample mode."
            )

        if use_sample and sample_size is not None:
            logger.info(f"Sampling {sample_size:,} animelist records...")
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(
                file_path,
                chunksize=500_000,
                dtype=_ANIMELIST_DTYPES,
            ):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows >= sample_size * 2:
                    break

            self.animelist_df = pd.concat(chunks, ignore_index=True)
            del chunks
            if len(self.animelist_df) > sample_size:
                self.animelist_df = self.animelist_df.sample(n=sample_size, random_state=42)

        else:
            logger.info("Loading all animelist data (chunked, dtype-optimized)...")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(
                file_path,
                chunksize=500_000,
                dtype=_ANIMELIST_DTYPES,
            ), start=1):
                chunks.append(chunk)
                logger.info(f"  Loaded chunk {i} ({len(chunk):,} records)")

            self.animelist_df = pd.concat(chunks, ignore_index=True)
            del chunks

        logger.info(f"Loaded {len(self.animelist_df):,} animelist records")
        return self.animelist_df

    # ------------------------------------------------------------------
    # Helpers (unchanged)
    # ------------------------------------------------------------------

    def get_merged_anime_data(self) -> pd.DataFrame:
        if self._merged_df is not None:
            return self._merged_df
        if self.anime_df is None:
            self.load_anime()
        if self.synopsis_df is None:
            self.load_synopsis()
        self._merged_df = self.anime_df.merge(
            self.synopsis_df[['MAL_ID', 'synopsis']],
            on='MAL_ID',
            how='left'
        )
        self._merged_df['synopsis'] = self._merged_df['synopsis'].fillna('')
        logger.info(f"Merged anime data: {len(self._merged_df)} records")
        return self._merged_df

    def get_user_item_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.ratings_df is None:
            self.load_ratings()
        if self.animelist_df is None:
            self.load_animelist()
        return self.ratings_df, self.animelist_df

    def _clean_anime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Name'] = df['Name'].fillna('Unknown')
        df['Genres'] = df['Genres'].fillna('')
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
        df['Genres'] = df['Genres'].apply(self._normalize_genres)
        if 'English name' in df.columns:
            df['English name'] = df['English name'].fillna(df['Name'])
        df = df.drop_duplicates(subset=['MAL_ID'], keep='first')
        return df

    def _normalize_genres(self, genres: str) -> str:
        if pd.isna(genres) or genres == '':
            return ''
        genre_list = [g.strip() for g in str(genres).split(',')]
        return ', '.join(sorted(set(genre_list)))

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ''
        return ' '.join(str(text).split())

    def get_anime_id_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.anime_df is None:
            self.load_anime()
        anime_ids = self.anime_df['MAL_ID'].unique()
        id_to_idx = {aid: idx for idx, aid in enumerate(anime_ids)}
        idx_to_id = {idx: aid for aid, idx in id_to_idx.items()}
        return id_to_idx, idx_to_id

    def get_user_id_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.ratings_df is None:
            self.load_ratings()
        user_ids = self.ratings_df['user_id'].unique()
        id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        idx_to_id = {idx: uid for uid, idx in id_to_idx.items()}
        return id_to_idx, idx_to_id


if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_all(use_cache=False)
    print("\n=== Dataset Summary ===")
    for name, df in data.items():
        print(f"{name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    merged = loader.get_merged_anime_data()
    print(f"\nMerged anime data: {merged.shape}")
    print(merged[['MAL_ID', 'Name', 'Genres', 'synopsis']].head())