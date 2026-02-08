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
        """
        Initialize DataLoader.

        Args:
            dataset_path: Path to dataset directory. Uses default if not provided.
        """
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH
        self.anime_df: Optional[pd.DataFrame] = None
        self.synopsis_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.animelist_df: Optional[pd.DataFrame] = None
        self.watching_status_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None

    def load_all(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary containing all loaded DataFrames
        """
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

        # Cache the loaded data
        data_dict = {
            "anime": self.anime_df,
            "synopsis": self.synopsis_df,
            "ratings": self.ratings_df,
            "animelist": self.animelist_df,
            "watching_status": self.watching_status_df
        }

        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info(f"Data cached to {cache_file}")

        return data_dict

    def load_anime(self) -> pd.DataFrame:
        """
        Load anime metadata.

        Returns:
            DataFrame with anime information
        """
        logger.info("Loading anime.csv...")
        file_path = self.dataset_path / data_config.anime_file
        self.anime_df = pd.read_csv(file_path)

        # Clean and preprocess
        self.anime_df = self._clean_anime_data(self.anime_df)
        logger.info(f"Loaded {len(self.anime_df)} anime records")
        return self.anime_df

    def load_synopsis(self) -> pd.DataFrame:
        """
        Load anime synopses.

        Returns:
            DataFrame with anime synopses
        """
        logger.info("Loading anime_with_synopsis.csv...")
        file_path = self.dataset_path / data_config.anime_synopsis_file
        self.synopsis_df = pd.read_csv(file_path)

        # Rename column if needed
        if 'sypnopsis' in self.synopsis_df.columns:
            self.synopsis_df = self.synopsis_df.rename(columns={'sypnopsis': 'synopsis'})

        # Clean synopsis text
        self.synopsis_df['synopsis'] = self.synopsis_df['synopsis'].fillna('')
        self.synopsis_df['synopsis'] = self.synopsis_df['synopsis'].apply(self._clean_text)

        logger.info(f"Loaded {len(self.synopsis_df)} synopses")
        return self.synopsis_df

    def load_ratings(self, sample: bool = None) -> pd.DataFrame:
        """
        Load user ratings with optional sampling for large datasets.

        Args:
            sample: Whether to sample the data (uses config default if None)

        Returns:
            DataFrame with user ratings
        """
        logger.info("Loading rating_complete.csv...")
        file_path = self.dataset_path / data_config.rating_file

        sample = sample if sample is not None else data_config.sample_ratings

        if sample:
            # Load in chunks and sample
            logger.info(f"Sampling {data_config.rating_sample_size:,} ratings...")
            chunks = []
            total_rows = 0

            for chunk in pd.read_csv(file_path, chunksize=1_000_000):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows >= data_config.rating_sample_size * 2:
                    break

            self.ratings_df = pd.concat(chunks, ignore_index=True)
            if len(self.ratings_df) > data_config.rating_sample_size:
                self.ratings_df = self.ratings_df.sample(
                    n=data_config.rating_sample_size,
                    random_state=42
                )
        else:
            self.ratings_df = pd.read_csv(file_path)

        # Filter out invalid ratings
        self.ratings_df = self.ratings_df[self.ratings_df['rating'] > 0]

        logger.info(f"Loaded {len(self.ratings_df):,} ratings")
        return self.ratings_df

    def load_animelist(self, sample: bool = None) -> pd.DataFrame:
        """
        Load user anime lists with optional sampling.

        Args:
            sample: Whether to sample the data

        Returns:
            DataFrame with user anime lists
        """
        logger.info("Loading animelist.csv...")
        file_path = self.dataset_path / data_config.animelist_file

        sample = sample if sample is not None else data_config.sample_ratings

        if sample:
            logger.info(f"Sampling {data_config.animelist_sample_size:,} records...")
            chunks = []
            total_rows = 0

            for chunk in pd.read_csv(file_path, chunksize=1_000_000):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows >= data_config.animelist_sample_size * 2:
                    break

            self.animelist_df = pd.concat(chunks, ignore_index=True)
            if len(self.animelist_df) > data_config.animelist_sample_size:
                self.animelist_df = self.animelist_df.sample(
                    n=data_config.animelist_sample_size,
                    random_state=42
                )
        else:
            self.animelist_df = pd.read_csv(file_path)

        logger.info(f"Loaded {len(self.animelist_df):,} animelist records")
        return self.animelist_df

    def load_watching_status(self) -> pd.DataFrame:
        """
        Load watching status mappings.

        Returns:
            DataFrame with status mappings
        """
        logger.info("Loading watching_status.csv...")
        file_path = self.dataset_path / data_config.watching_status_file
        self.watching_status_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.watching_status_df)} status mappings")
        return self.watching_status_df

    def get_merged_anime_data(self) -> pd.DataFrame:
        """
        Get merged anime data with synopses.

        Returns:
            DataFrame with merged anime and synopsis data
        """
        if self._merged_df is not None:
            return self._merged_df

        if self.anime_df is None:
            self.load_anime()
        if self.synopsis_df is None:
            self.load_synopsis()

        # Merge on MAL_ID
        self._merged_df = self.anime_df.merge(
            self.synopsis_df[['MAL_ID', 'synopsis']],
            on='MAL_ID',
            how='left'
        )
        self._merged_df['synopsis'] = self._merged_df['synopsis'].fillna('')

        logger.info(f"Merged anime data: {len(self._merged_df)} records")
        return self._merged_df

    def get_user_item_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get prepared user-item interaction data.

        Returns:
            Tuple of (ratings DataFrame, animelist DataFrame)
        """
        if self.ratings_df is None:
            self.load_ratings()
        if self.animelist_df is None:
            self.load_animelist()

        return self.ratings_df, self.animelist_df

    def _clean_anime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean anime DataFrame.

        Args:
            df: Raw anime DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Handle missing values
        df['Name'] = df['Name'].fillna('Unknown')
        df['Genres'] = df['Genres'].fillna('')
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)

        # Normalize genres
        df['Genres'] = df['Genres'].apply(self._normalize_genres)

        # Handle English name
        if 'English name' in df.columns:
            df['English name'] = df['English name'].fillna(df['Name'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['MAL_ID'], keep='first')

        return df

    def _normalize_genres(self, genres: str) -> str:
        """
        Normalize genre string.

        Args:
            genres: Raw genre string

        Returns:
            Normalized genre string
        """
        if pd.isna(genres) or genres == '':
            return ''

        # Split, strip, and rejoin
        genre_list = [g.strip() for g in str(genres).split(',')]
        return ', '.join(sorted(set(genre_list)))

    def _clean_text(self, text: str) -> str:
        """
        Clean text for NLP processing.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ''

        text = str(text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def get_anime_id_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Create mappings between anime IDs and indices.

        Returns:
            Tuple of (id_to_idx, idx_to_id) dictionaries
        """
        if self.anime_df is None:
            self.load_anime()

        anime_ids = self.anime_df['MAL_ID'].unique()
        id_to_idx = {aid: idx for idx, aid in enumerate(anime_ids)}
        idx_to_id = {idx: aid for aid, idx in id_to_idx.items()}

        return id_to_idx, idx_to_id

    def get_user_id_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Create mappings between user IDs and indices.

        Returns:
            Tuple of (id_to_idx, idx_to_id) dictionaries
        """
        if self.ratings_df is None:
            self.load_ratings()

        user_ids = self.ratings_df['user_id'].unique()
        id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        idx_to_id = {idx: uid for uid, idx in id_to_idx.items()}

        return id_to_idx, idx_to_id


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    data = loader.load_all(use_cache=False)

    print("\n=== Dataset Summary ===")
    for name, df in data.items():
        print(f"{name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Test merged data
    merged = loader.get_merged_anime_data()
    print(f"\nMerged anime data: {merged.shape}")
    print(merged[['MAL_ID', 'Name', 'Genres', 'synopsis']].head())
