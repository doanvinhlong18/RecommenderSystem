"""
Hybrid Recommendation Engine.

Combines multiple recommendation techniques:
- Content-Based Filtering
- Collaborative Filtering (Item-Based, Matrix Factorization)
- Implicit Feedback
- Popularity-Based
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


class HybridEngine:
    """
    Hybrid Recommendation Engine combining multiple models.

    Final Score = w1 * ContentScore + w2 * CollaborativeScore +
                  w3 * ImplicitScore + w4 * PopularityScore

    Strategies:
    - New user: Content + Popularity (no collaborative data)
    - Existing user: Full Hybrid
    - Cold start item: Content + Popularity

    Attributes:
        content_model: Content-based recommender
        collaborative_model: Item-based or Matrix Factorization model
        implicit_model: ALS implicit model
        popularity_model: Popularity-based model
        weights: Dictionary of model weights
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        content_model=None,
        collaborative_model=None,
        implicit_model=None,
        popularity_model=None
    ):
        """
        Initialize HybridEngine.

        Args:
            weights: Dictionary with keys 'content', 'collaborative', 'implicit', 'popularity'
            content_model: Content-based recommender instance
            collaborative_model: Collaborative filtering model instance
            implicit_model: Implicit feedback model instance
            popularity_model: Popularity model instance
        """
        self.weights = weights or model_config.hybrid_weights.copy()

        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.implicit_model = implicit_model
        self.popularity_model = popularity_model

        # Anime info cache
        self._anime_info: Dict[int, Dict] = {}

        # User history cache
        self._user_ratings: Dict[int, Dict[int, float]] = {}
        self._user_watched: Dict[int, Set[int]] = {}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update model weights.

        Args:
            weights: New weights dictionary
        """
        self.weights.update(weights)
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total

        logger.info(f"Updated weights: {self.weights}")

    def set_anime_info(self, anime_df: pd.DataFrame) -> None:
        """
        Set anime information for enriching recommendations.

        Args:
            anime_df: DataFrame with anime metadata
        """
        for _, row in anime_df.iterrows():
            self._anime_info[row['MAL_ID']] = {
                'mal_id': int(row['MAL_ID']),
                'name': row['Name'],
                'english_name': row.get('English name', row['Name']),
                'genres': row.get('Genres', ''),
                'score': float(row.get('Score', 0)) if pd.notna(row.get('Score')) else 0,
                'type': row.get('Type', 'Unknown'),
                'synopsis': row.get('synopsis', '')
            }

    def set_user_history(
        self,
        user_id: int,
        ratings: Dict[int, float] = None,
        watched: Set[int] = None
    ) -> None:
        """
        Set user history for personalization.

        Args:
            user_id: User ID
            ratings: Dictionary of anime_id -> rating
            watched: Set of watched anime IDs
        """
        if ratings:
            self._user_ratings[user_id] = ratings
        if watched:
            self._user_watched[user_id] = watched

    def recommend_similar_anime(
        self,
        anime_identifier: Union[int, str],
        top_k: int = 10,
        method: str = "hybrid"
    ) -> List[Dict]:
        """
        Get similar anime recommendations (anime-to-anime).

        Args:
            anime_identifier: Anime ID or name
            top_k: Number of recommendations
            method: "content", "collaborative", "hybrid"

        Returns:
            List of recommendation dictionaries
        """
        scores = {}

        # Content-based similarity
        if self.content_model and method in ["content", "hybrid"]:
            try:
                content_recs = self.content_model.get_similar_anime(
                    anime_identifier, top_k=top_k * 2
                )
                weight = self.weights.get('content', 0.3) if method == "hybrid" else 1.0
                for rec in content_recs:
                    aid = rec['mal_id']
                    scores[aid] = scores.get(aid, 0) + weight * rec['similarity']
            except Exception as e:
                logger.warning(f"Content model error: {e}")

        # Collaborative similarity
        if self.collaborative_model and method in ["collaborative", "hybrid"]:
            try:
                # Get anime ID if string was provided
                anime_id = anime_identifier
                if isinstance(anime_identifier, str) and self.content_model:
                    idx = self.content_model._get_anime_idx(anime_identifier)
                    if idx is not None:
                        anime_id = self.content_model._idx_to_id.get(idx)

                if isinstance(anime_id, int):
                    collab_recs = self.collaborative_model.get_similar_items(
                        anime_id, top_k=top_k * 2
                    )
                    weight = self.weights.get('collaborative', 0.4) if method == "hybrid" else 1.0
                    for rec in collab_recs:
                        aid = rec['mal_id']
                        scores[aid] = scores.get(aid, 0) + weight * rec['similarity']
            except Exception as e:
                logger.warning(f"Collaborative model error: {e}")

        # Implicit similarity (if available)
        if self.implicit_model and method == "hybrid":
            try:
                anime_id = anime_identifier
                if isinstance(anime_identifier, str) and self.content_model:
                    idx = self.content_model._get_anime_idx(anime_identifier)
                    if idx is not None:
                        anime_id = self.content_model._idx_to_id.get(idx)

                if isinstance(anime_id, int):
                    implicit_recs = self.implicit_model.get_similar_items(
                        anime_id, top_k=top_k * 2
                    )
                    weight = self.weights.get('implicit', 0.2)
                    for rec in implicit_recs:
                        aid = rec['mal_id']
                        scores[aid] = scores.get(aid, 0) + weight * rec['similarity']
            except Exception as e:
                logger.warning(f"Implicit model error: {e}")

        # Sort by combined score
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build results with anime info
        results = []
        for aid, score in sorted_recs[:top_k]:
            rec = self._get_anime_info(aid)
            rec['hybrid_score'] = score
            results.append(rec)

        return results

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_watched: bool = True,
        strategy: str = "auto"
    ) -> List[Dict]:
        """
        Get personalized recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_watched: Whether to exclude already watched anime
            strategy: "auto", "new_user", "existing_user"

        Returns:
            List of recommendation dictionaries
        """
        # Determine user type
        is_new_user = self._is_new_user(user_id)

        if strategy == "auto":
            strategy = "new_user" if is_new_user else "existing_user"

        if strategy == "new_user":
            return self._recommend_for_new_user(user_id, top_k)
        else:
            return self._recommend_for_existing_user(user_id, top_k, exclude_watched)

    def _is_new_user(self, user_id: int) -> bool:
        """Check if user is new (no collaborative data)."""
        # Check if user exists in collaborative models
        if self.collaborative_model:
            if hasattr(self.collaborative_model, 'user_to_idx'):
                if user_id in self.collaborative_model.user_to_idx:
                    return False

        if self.implicit_model:
            if hasattr(self.implicit_model, 'user_to_idx'):
                if user_id in self.implicit_model.user_to_idx:
                    return False

        return True

    def _recommend_for_new_user(
        self,
        user_id: int,
        top_k: int
    ) -> List[Dict]:
        """Recommendations for new users using content + popularity."""
        scores = {}

        # Get user preferences if available
        user_ratings = self._user_ratings.get(user_id, {})
        preferred_genres = self._extract_preferred_genres(user_ratings)

        # Popularity recommendations
        if self.popularity_model:
            pop_recs = self.popularity_model.get_recommendations_for_new_user(
                top_k=top_k * 2,
                preferred_genres=preferred_genres
            )
            for rec in pop_recs:
                aid = rec['mal_id']
                scores[aid] = scores.get(aid, 0) + 0.6 * rec.get('popularity_score', rec.get('score', 0)) / 100

        # Content-based if user has some ratings
        if self.content_model and user_ratings:
            # Get similar to highly rated anime
            top_rated = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:5]

            for anime_id, rating in top_rated:
                try:
                    content_recs = self.content_model.get_similar_anime(anime_id, top_k=10)
                    for rec in content_recs:
                        aid = rec['mal_id']
                        if aid not in user_ratings:  # Exclude already rated
                            weight = (rating / 10) * 0.4  # Weight by user rating
                            scores[aid] = scores.get(aid, 0) + weight * rec['similarity']
                except Exception:
                    pass

        # Sort and build results
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        watched = self._user_watched.get(user_id, set())

        for aid, score in sorted_recs:
            if aid in watched or aid in user_ratings:
                continue
            rec = self._get_anime_info(aid)
            rec['hybrid_score'] = score
            rec['strategy'] = 'new_user'
            results.append(rec)
            if len(results) >= top_k:
                break

        return results

    def _recommend_for_existing_user(
        self,
        user_id: int,
        top_k: int,
        exclude_watched: bool
    ) -> List[Dict]:
        """Full hybrid recommendations for existing users."""
        scores = {}

        # Get user's watched/rated items
        watched = self._user_watched.get(user_id, set())
        rated = set(self._user_ratings.get(user_id, {}).keys())
        exclude_set = watched.union(rated) if exclude_watched else set()

        # Collaborative filtering recommendations
        if self.collaborative_model:
            try:
                collab_recs = self.collaborative_model.recommend_for_user(
                    user_id, top_k=top_k * 3, exclude_rated=exclude_watched
                )
                weight = self.weights.get('collaborative', 0.4)
                for rec in collab_recs:
                    aid = rec['mal_id']
                    # Normalize predicted rating to 0-1
                    norm_score = rec.get('predicted_rating', rec.get('similarity', 5)) / 10
                    scores[aid] = scores.get(aid, 0) + weight * norm_score
            except Exception as e:
                logger.warning(f"Collaborative recommendation error: {e}")

        # Implicit feedback recommendations
        if self.implicit_model:
            try:
                implicit_recs = self.implicit_model.recommend_for_user(
                    user_id, top_k=top_k * 3, exclude_known=exclude_watched
                )
                weight = self.weights.get('implicit', 0.2)
                for rec in implicit_recs:
                    aid = rec['mal_id']
                    scores[aid] = scores.get(aid, 0) + weight * rec.get('score', 0)
            except Exception as e:
                logger.warning(f"Implicit recommendation error: {e}")

        # Content-based (similar to user's top rated)
        if self.content_model:
            user_ratings = self._user_ratings.get(user_id, {})
            if user_ratings:
                top_rated = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
                weight = self.weights.get('content', 0.3)

                for anime_id, rating in top_rated:
                    try:
                        content_recs = self.content_model.get_similar_anime(anime_id, top_k=10)
                        for rec in content_recs:
                            aid = rec['mal_id']
                            if aid not in exclude_set:
                                user_weight = (rating / 10) * weight
                                scores[aid] = scores.get(aid, 0) + user_weight * rec['similarity']
                    except Exception:
                        pass

        # Popularity boost for diversity
        if self.popularity_model:
            pop_recs = self.popularity_model.get_top_rated(top_k=50)
            weight = self.weights.get('popularity', 0.1)
            for idx, rec in enumerate(pop_recs):
                aid = rec['mal_id']
                if aid not in exclude_set:
                    # Decreasing weight by rank
                    rank_weight = (50 - idx) / 50
                    scores[aid] = scores.get(aid, 0) + weight * rank_weight

        # Sort and build results
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for aid, score in sorted_recs:
            if aid in exclude_set:
                continue
            rec = self._get_anime_info(aid)
            rec['hybrid_score'] = score
            rec['strategy'] = 'existing_user'
            results.append(rec)
            if len(results) >= top_k:
                break

        return results

    def _extract_preferred_genres(self, user_ratings: Dict[int, float]) -> List[str]:
        """Extract user's preferred genres from their ratings."""
        if not user_ratings:
            return []

        genre_scores = {}

        for anime_id, rating in user_ratings.items():
            if anime_id in self._anime_info:
                genres = self._anime_info[anime_id].get('genres', '')
                for genre in str(genres).split(','):
                    genre = genre.strip()
                    if genre:
                        genre_scores[genre] = genre_scores.get(genre, 0) + rating

        # Return top genres
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return [g for g, _ in sorted_genres[:5]]

    def _get_anime_info(self, anime_id: int) -> Dict:
        """Get anime information by ID."""
        if anime_id in self._anime_info:
            return self._anime_info[anime_id].copy()

        # Try to get from content model
        if self.content_model and hasattr(self.content_model, 'anime_df'):
            df = self.content_model.anime_df
            row = df[df['MAL_ID'] == anime_id]
            if not row.empty:
                row = row.iloc[0]
                return {
                    'mal_id': int(anime_id),
                    'name': row.get('Name', 'Unknown'),
                    'english_name': row.get('English name', row.get('Name', 'Unknown')),
                    'genres': row.get('Genres', ''),
                    'score': float(row.get('Score', 0)) if pd.notna(row.get('Score')) else 0,
                    'type': row.get('Type', 'Unknown')
                }

        return {
            'mal_id': anime_id,
            'name': f'Anime {anime_id}',
            'english_name': f'Anime {anime_id}',
            'genres': '',
            'score': 0,
            'type': 'Unknown'
        }

    def get_explanation(
        self,
        user_id: int,
        anime_id: int
    ) -> Dict:
        """
        Get explanation for why an anime was recommended.

        Args:
            user_id: User ID
            anime_id: Anime ID

        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'anime_id': anime_id,
            'anime_info': self._get_anime_info(anime_id),
            'reasons': []
        }

        # Check content similarity
        if self.content_model:
            user_ratings = self._user_ratings.get(user_id, {})
            for rated_id, rating in sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:3]:
                try:
                    similar = self.content_model.get_similar_anime(rated_id, top_k=20)
                    for s in similar:
                        if s['mal_id'] == anime_id:
                            rated_info = self._get_anime_info(rated_id)
                            explanation['reasons'].append({
                                'type': 'content_similarity',
                                'message': f"Similar to '{rated_info['name']}' which you rated {rating}",
                                'similarity': s['similarity']
                            })
                            break
                except Exception:
                    pass

        # Check popularity
        if self.popularity_model:
            for pop_type in ['top_rated', 'trending']:
                pop_list = self.popularity_model.get_popular(top_k=50, popularity_type=pop_type)
                for idx, p in enumerate(pop_list):
                    if p['mal_id'] == anime_id:
                        explanation['reasons'].append({
                            'type': 'popularity',
                            'message': f"#{idx + 1} in {pop_type.replace('_', ' ')}"
                        })
                        break

        # Genre match
        user_genres = self._extract_preferred_genres(self._user_ratings.get(user_id, {}))
        anime_genres = self._get_anime_info(anime_id).get('genres', '')
        matching_genres = [g for g in user_genres if g.lower() in anime_genres.lower()]
        if matching_genres:
            explanation['reasons'].append({
                'type': 'genre_match',
                'message': f"Matches your preferred genres: {', '.join(matching_genres)}"
            })

        return explanation

    def save(self, directory: Union[str, Path]) -> None:
        """Save hybrid engine and all models."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save individual models
        if self.content_model:
            self.content_model.save(directory / "content_model.pkl")
        if self.collaborative_model:
            self.collaborative_model.save(directory / "collaborative_model.pkl")
        if self.implicit_model:
            self.implicit_model.save(directory / "implicit_model.pkl")
        if self.popularity_model:
            self.popularity_model.save(directory / "popularity_model.pkl")

        # Save engine state
        state = {
            'weights': self.weights,
            'anime_info': self._anime_info
        }
        with open(directory / "hybrid_engine.pkl", 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"HybridEngine saved to {directory}")

    def load(self, directory: Union[str, Path]) -> "HybridEngine":
        """Load hybrid engine and all models."""
        directory = Path(directory)

        # Import model classes
        from models.content import ContentBasedRecommender
        from models.collaborative import ItemBasedCF, MatrixFactorization
        from models.implicit import ALSImplicit
        from models.popularity import PopularityModel

        # Load individual models
        if (directory / "content_model.pkl").exists():
            self.content_model = ContentBasedRecommender()
            self.content_model.load(directory / "content_model.pkl")

        if (directory / "collaborative_model.pkl").exists():
            # Try MatrixFactorization first, then ItemBasedCF
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

        # Load engine state
        if (directory / "hybrid_engine.pkl").exists():
            with open(directory / "hybrid_engine.pkl", 'rb') as f:
                state = pickle.load(f)
            self.weights = state['weights']
            self._anime_info = state['anime_info']

        logger.info(f"HybridEngine loaded from {directory}")
        return self


if __name__ == "__main__":
    # Test Hybrid Engine
    print("HybridEngine module loaded successfully")
    print("Use train.py to train all models and create a HybridEngine instance")
