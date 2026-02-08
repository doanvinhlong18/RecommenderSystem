"""
Popularity-Based Recommendation Model.
"""
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopularityModel:
    """
    Popularity-based recommender for cold start handling.

    Provides:
    - Top rated anime
    - Most watched anime
    - Trending anime (based on recent activity)

    Attributes:
        anime_df: DataFrame with anime metadata
        ratings_df: DataFrame with ratings
        animelist_df: DataFrame with watch data
    """

    def __init__(self):
        """Initialize PopularityModel."""
        self.anime_df: Optional[pd.DataFrame] = None
        self._top_rated: Optional[List[Dict]] = None
        self._most_watched: Optional[List[Dict]] = None
        self._most_members: Optional[List[Dict]] = None
        self._trending: Optional[List[Dict]] = None

        # ID to info mapping
        self._anime_info: Dict[int, Dict] = {}

    def fit(
        self,
        anime_df: pd.DataFrame,
        ratings_df: Optional[pd.DataFrame] = None,
        animelist_df: Optional[pd.DataFrame] = None,
        min_ratings: int = 100
    ) -> "PopularityModel":
        """
        Fit the popularity model.

        Args:
            anime_df: DataFrame with anime metadata
            ratings_df: DataFrame with user ratings
            animelist_df: DataFrame with watch data
            min_ratings: Minimum ratings for an anime to be considered

        Returns:
            Self for chaining
        """
        self.anime_df = anime_df.copy()

        # Build anime info mapping
        for _, row in self.anime_df.iterrows():
            self._anime_info[row['MAL_ID']] = {
                'mal_id': int(row['MAL_ID']),
                'name': row['Name'],
                'english_name': row.get('English name', row['Name']),
                'genres': row.get('Genres', ''),
                'score': float(row.get('Score', 0)) if pd.notna(row.get('Score')) else 0,
                'type': row.get('Type', 'Unknown'),
                'members': int(row.get('Members', 0)) if pd.notna(row.get('Members')) else 0
            }

        # Compute top rated
        self._compute_top_rated(min_ratings)

        # Compute most watched
        if ratings_df is not None:
            self._compute_most_watched(ratings_df)

        # Compute most members
        self._compute_most_members()

        # Compute trending (from animelist if available)
        if animelist_df is not None:
            self._compute_trending(animelist_df)

        logger.info("PopularityModel fitted successfully")
        return self

    def _compute_top_rated(self, min_ratings: int) -> None:
        """Compute top rated anime."""
        logger.info("Computing top rated anime...")

        # Use Score and Members columns if available
        df = self.anime_df.copy()

        # Filter by minimum number of ratings (use Members as proxy)
        if 'Members' in df.columns:
            df = df[df['Members'] >= min_ratings]

        # Sort by score
        df = df[pd.to_numeric(df['Score'], errors='coerce').notna()]
        df['Score'] = pd.to_numeric(df['Score'])
        df = df.sort_values('Score', ascending=False)

        self._top_rated = []
        for _, row in df.head(500).iterrows():
            self._top_rated.append({
                'mal_id': int(row['MAL_ID']),
                'name': row['Name'],
                'english_name': row.get('English name', row['Name']),
                'genres': row.get('Genres', ''),
                'score': float(row['Score']),
                'type': row.get('Type', 'Unknown'),
                'members': int(row.get('Members', 0)) if pd.notna(row.get('Members')) else 0
            })

        logger.info(f"Top rated: {len(self._top_rated)} anime")

    def _compute_most_watched(self, ratings_df: pd.DataFrame) -> None:
        """Compute most watched anime based on number of ratings."""
        logger.info("Computing most watched anime...")

        # Count ratings per anime
        rating_counts = ratings_df['anime_id'].value_counts()

        # Calculate average rating
        avg_ratings = ratings_df.groupby('anime_id')['rating'].mean()

        self._most_watched = []
        for anime_id in rating_counts.head(500).index:
            if anime_id in self._anime_info:
                info = self._anime_info[anime_id].copy()
                info['rating_count'] = int(rating_counts[anime_id])
                info['avg_rating'] = float(avg_ratings.get(anime_id, 0))
                self._most_watched.append(info)

        logger.info(f"Most watched: {len(self._most_watched)} anime")

    def _compute_most_members(self) -> None:
        """Compute most popular anime by member count."""
        logger.info("Computing most members anime...")

        if 'Members' not in self.anime_df.columns:
            self._most_members = self._top_rated
            return

        df = self.anime_df.copy()
        df['Members'] = pd.to_numeric(df['Members'], errors='coerce').fillna(0)
        df = df.sort_values('Members', ascending=False)

        self._most_members = []
        for _, row in df.head(500).iterrows():
            self._most_members.append({
                'mal_id': int(row['MAL_ID']),
                'name': row['Name'],
                'english_name': row.get('English name', row['Name']),
                'genres': row.get('Genres', ''),
                'score': float(row.get('Score', 0)) if pd.notna(row.get('Score')) else 0,
                'type': row.get('Type', 'Unknown'),
                'members': int(row['Members'])
            })

        logger.info(f"Most members: {len(self._most_members)} anime")

    def _compute_trending(self, animelist_df: pd.DataFrame) -> None:
        """Compute trending anime based on recent watch activity."""
        logger.info("Computing trending anime...")

        # Count by anime and watching status
        # Status 1 = Currently Watching (most recent/active)
        current_watching = animelist_df[animelist_df['watching_status'] == 1]
        trending_counts = current_watching['anime_id'].value_counts()

        self._trending = []
        for anime_id in trending_counts.head(500).index:
            if anime_id in self._anime_info:
                info = self._anime_info[anime_id].copy()
                info['watching_count'] = int(trending_counts[anime_id])
                self._trending.append(info)

        logger.info(f"Trending: {len(self._trending)} anime")

    def get_top_rated(self, top_k: int = 10, genre: str = None) -> List[Dict]:
        """
        Get top rated anime.

        Args:
            top_k: Number of results
            genre: Filter by genre (optional)

        Returns:
            List of anime dictionaries
        """
        if self._top_rated is None:
            return []

        results = self._top_rated

        if genre:
            genre_lower = genre.lower()
            results = [
                r for r in results
                if genre_lower in str(r.get('genres', '')).lower()
            ]

        return results[:top_k]

    def get_most_watched(self, top_k: int = 10, genre: str = None) -> List[Dict]:
        """
        Get most watched anime.

        Args:
            top_k: Number of results
            genre: Filter by genre (optional)

        Returns:
            List of anime dictionaries
        """
        results = self._most_watched or self._most_members or self._top_rated or []

        if genre:
            genre_lower = genre.lower()
            results = [
                r for r in results
                if genre_lower in str(r.get('genres', '')).lower()
            ]

        return results[:top_k]

    def get_trending(self, top_k: int = 10, genre: str = None) -> List[Dict]:
        """
        Get trending anime.

        Args:
            top_k: Number of results
            genre: Filter by genre (optional)

        Returns:
            List of anime dictionaries
        """
        results = self._trending or self._most_watched or self._top_rated or []

        if genre:
            genre_lower = genre.lower()
            results = [
                r for r in results
                if genre_lower in str(r.get('genres', '')).lower()
            ]

        return results[:top_k]

    def get_popular(
        self,
        top_k: int = 10,
        popularity_type: str = "top_rated",
        genre: str = None
    ) -> List[Dict]:
        """
        Get popular anime by specified type.

        Args:
            top_k: Number of results
            popularity_type: "top_rated", "most_watched", "trending", or "most_members"
            genre: Filter by genre (optional)

        Returns:
            List of anime dictionaries
        """
        if popularity_type == "top_rated":
            return self.get_top_rated(top_k, genre)
        elif popularity_type == "most_watched":
            return self.get_most_watched(top_k, genre)
        elif popularity_type == "trending":
            return self.get_trending(top_k, genre)
        elif popularity_type == "most_members":
            results = self._most_members or []
            if genre:
                genre_lower = genre.lower()
                results = [
                    r for r in results
                    if genre_lower in str(r.get('genres', '')).lower()
                ]
            return results[:top_k]
        else:
            return self.get_top_rated(top_k, genre)

    def get_recommendations_for_new_user(
        self,
        top_k: int = 10,
        preferred_genres: List[str] = None
    ) -> List[Dict]:
        """
        Get recommendations for a new user (cold start).

        Combines top rated and popular anime.

        Args:
            top_k: Number of recommendations
            preferred_genres: List of preferred genres

        Returns:
            List of recommendation dictionaries
        """
        # Combine different popularity metrics
        all_anime = {}

        # Add top rated with weight
        for idx, anime in enumerate(self._top_rated or []):
            aid = anime['mal_id']
            all_anime[aid] = all_anime.get(aid, 0) + (100 - idx) * 0.4

        # Add most watched with weight
        for idx, anime in enumerate(self._most_watched or []):
            aid = anime['mal_id']
            all_anime[aid] = all_anime.get(aid, 0) + (100 - idx) * 0.3

        # Add trending with weight
        for idx, anime in enumerate(self._trending or []):
            aid = anime['mal_id']
            all_anime[aid] = all_anime.get(aid, 0) + (100 - idx) * 0.3

        # Sort by combined score
        sorted_anime = sorted(all_anime.items(), key=lambda x: x[1], reverse=True)

        results = []
        for aid, score in sorted_anime:
            if aid not in self._anime_info:
                continue

            info = self._anime_info[aid].copy()

            # Filter by genre if specified
            if preferred_genres:
                anime_genres = str(info.get('genres', '')).lower()
                if not any(g.lower() in anime_genres for g in preferred_genres):
                    continue

            info['popularity_score'] = score
            results.append(info)

            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'top_rated': self._top_rated,
            'most_watched': self._most_watched,
            'most_members': self._most_members,
            'trending': self._trending,
            'anime_info': self._anime_info
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"PopularityModel saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "PopularityModel":
        """Load model from file."""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self._top_rated = state['top_rated']
        self._most_watched = state['most_watched']
        self._most_members = state['most_members']
        self._trending = state['trending']
        self._anime_info = state['anime_info']

        logger.info(f"PopularityModel loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Test Popularity Model
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader

    loader = DataLoader()
    loader.load_anime()
    loader.load_ratings(sample=True)
    loader.load_animelist(sample=True)

    model = PopularityModel()
    model.fit(
        loader.anime_df,
        loader.ratings_df,
        loader.animelist_df
    )

    print("\n=== Top Rated ===")
    for anime in model.get_top_rated(5):
        print(f"  {anime['name']} - Score: {anime['score']}")

    print("\n=== Most Watched ===")
    for anime in model.get_most_watched(5):
        print(f"  {anime['name']} - Ratings: {anime.get('rating_count', 'N/A')}")

    print("\n=== Trending ===")
    for anime in model.get_trending(5):
        print(f"  {anime['name']} - Watching: {anime.get('watching_count', 'N/A')}")

    print("\n=== New User Recommendations (Action) ===")
    for anime in model.get_recommendations_for_new_user(5, ['Action']):
        print(f"  {anime['name']} - Score: {anime['score']}")
