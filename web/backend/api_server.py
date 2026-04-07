"""
Enhanced FastAPI Backend for Anime Recommendation Web Demo.

This module provides a comprehensive REST API for the hybrid anime recommendation system.
It includes endpoints for search, recommendations, popular anime, and user personalization.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from config import MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global hybrid engine instance
hybrid_engine = None

# Cache for demo user selection (computed once, reused)
_demo_user_cache: dict = {}

# Cache for per-user history loaded from animelist.csv (avoid rescanning on repeated UI refresh)
_user_history_cache: dict = {}


# =============================================================================
# Pydantic Models
# =============================================================================


class AnimeInfo(BaseModel):
    """Full anime information model."""

    mal_id: int
    name: str
    english_name: Optional[str] = None
    genres: Optional[str] = None
    score: Optional[float] = None
    type: Optional[str] = None
    episodes: Optional[int] = None
    synopsis: Optional[str] = None
    image_url: Optional[str] = None
    similarity: Optional[float] = None
    predicted_rating: Optional[float] = None
    hybrid_score: Optional[float] = None


class SearchResult(BaseModel):
    """Search result model."""

    success: bool = True
    count: int
    query: str
    results: List[AnimeInfo]


class AnimeDetailResponse(BaseModel):
    """Anime detail response."""

    success: bool = True
    anime: AnimeInfo


class RecommendationResponse(BaseModel):
    """Recommendation response model."""

    success: bool = True
    count: int
    method: Optional[str] = None
    query_anime: Optional[AnimeInfo] = None
    recommendations: List[AnimeInfo]
    strategy: Optional[str] = None


class UserRecommendationResponse(BaseModel):
    """User recommendation response."""

    success: bool = True
    user_id: int
    is_cold_start: bool = False
    strategy: str
    count: int
    recommendations: List[AnimeInfo]


class PopularResponse(BaseModel):
    """Popular anime response."""

    success: bool = True
    count: int
    type: str
    anime: List[AnimeInfo]


class WeightsModel(BaseModel):
    """Hybrid weights model."""

    content: float = Field(0.3, ge=0, le=1)
    collaborative: float = Field(0.4, ge=0, le=1)
    implicit: float = Field(0.2, ge=0, le=1)
    popularity: float = Field(0.1, ge=0, le=1)


class AutocompleteResult(BaseModel):
    """Autocomplete result model."""

    suggestions: List[Dict]


# =============================================================================
# Application Factory
# =============================================================================


def create_demo_app() -> FastAPI:
    """
    Create the FastAPI application for web demo.

    Returns:
        Configured FastAPI application
    """
    global hybrid_engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load models on startup."""
        global hybrid_engine

        model_path = MODELS_DIR / "hybrid"
        if model_path.exists():
            try:
                from models.hybrid import HybridEngine

                hybrid_engine = HybridEngine()
                hybrid_engine.load(model_path)
                logger.info(f"HybridEngine loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load HybridEngine: {e}")
                hybrid_engine = None
        else:
            logger.warning(f"No saved model found at {model_path}")
            hybrid_engine = None

        yield

    app = FastAPI(
        title="Anime Recommendation Web Demo API",
        description="""
        ## Hybrid Anime Recommendation System
        
        This API powers the web demo for a hybrid anime recommendation system that combines:
        - **Content-Based Filtering**: Using TF-IDF and SBERT embeddings
        - **Collaborative Filtering**: User-item and item-item similarity
        - **Implicit Feedback**: ALS-based implicit interactions
        - **Popularity-Based**: Trending and top-rated anime
        
        ### Features
        - 🔍 Search anime with autocomplete
        - 🎬 Get anime details and metadata
        - ✨ Similar anime recommendations
        - 👤 Personalized user recommendations
        - 📈 Popular and trending anime
        - ⚙️ Adjustable recommendation weights
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for frontend
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    # =========================================================================
    # Root & Health Endpoints
    # =========================================================================

    @app.get("/")
    async def root():
        """Serve the main demo page."""
        index_path = Path(__file__).parent.parent / "frontend" / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Anime Recommendation Demo API", "docs": "/docs"}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": hybrid_engine is not None,
            "components": {
                "content_model": (
                    hybrid_engine.content_model is not None if hybrid_engine else False
                ),
                "collaborative_model": (
                    hybrid_engine.collaborative_model is not None
                    if hybrid_engine
                    else False
                ),
                "implicit_model": (
                    hybrid_engine.implicit_model is not None if hybrid_engine else False
                ),
                "popularity_model": (
                    hybrid_engine.popularity_model is not None
                    if hybrid_engine
                    else False
                ),
            },
        }

    @app.get("/api/status")
    async def api_status():
        """Get API and model status."""
        if hybrid_engine is None:
            return {"status": "error", "message": "Model not loaded"}

        anime_count = len(hybrid_engine._anime_info) if hybrid_engine._anime_info else 0
        return {
            "status": "ready",
            "anime_count": anime_count,
            "weights": hybrid_engine.weights,
            "models_available": {
                "content": hybrid_engine.content_model is not None,
                "collaborative": hybrid_engine.collaborative_model is not None,
                "implicit": hybrid_engine.implicit_model is not None,
                "popularity": hybrid_engine.popularity_model is not None,
            },
        }

    # =========================================================================
    # Search Endpoints
    # =========================================================================

    @app.get("/api/search", response_model=SearchResult)
    async def search_anime(
        q: str = Query(..., min_length=1, max_length=100, description="Search query"),
        top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
    ):
        """
        Search for anime by name.

        Supports partial matching and returns anime metadata.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if hybrid_engine.content_model is None:
            raise HTTPException(status_code=503, detail="Content model not available")

        try:
            results = hybrid_engine.content_model.search_anime(q, top_k=top_k)

            # Enrich with image URLs
            enriched_results = []
            for r in results:
                r["image_url"] = _get_anime_image_url(r["mal_id"])
                enriched_results.append(AnimeInfo(**r))

            return SearchResult(
                count=len(enriched_results), query=q, results=enriched_results
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/autocomplete", response_model=AutocompleteResult)
    async def autocomplete(
        q: str = Query(..., min_length=1, max_length=100),
        limit: int = Query(default=8, ge=1, le=20),
    ):
        """
        Get autocomplete suggestions for anime search.

        Returns quick suggestions for the search dropdown.
        """
        if hybrid_engine is None or hybrid_engine.content_model is None:
            return AutocompleteResult(suggestions=[])

        try:
            results = hybrid_engine.content_model.search_anime(q, top_k=limit)
            suggestions = [
                {
                    "mal_id": r["mal_id"],
                    "name": r["name"],
                    "english_name": r.get("english_name", r["name"]),
                    "score": r.get("score", 0),
                    "type": r.get("type", "Unknown"),
                    "image_url": _get_anime_image_url(r["mal_id"]),
                }
                for r in results
            ]
            return AutocompleteResult(suggestions=suggestions)
        except Exception as e:
            logger.error(f"Autocomplete error: {e}")
            return AutocompleteResult(suggestions=[])

    # =========================================================================
    # Anime Detail Endpoints
    # =========================================================================

    @app.get("/api/anime/{anime_id}", response_model=AnimeDetailResponse)
    async def get_anime_detail(anime_id: int):
        """
        Get detailed information about a specific anime.

        Returns full metadata including synopsis.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        anime_info = hybrid_engine._get_anime_info(anime_id)
        if not anime_info or anime_info.get("name", "").startswith(f"Anime {anime_id}"):
            raise HTTPException(
                status_code=404, detail=f"Anime with ID {anime_id} not found"
            )

        anime_info["image_url"] = _get_anime_image_url(anime_id)
        return AnimeDetailResponse(anime=AnimeInfo(**anime_info))

    @app.get("/api/anime/name/{anime_name}", response_model=AnimeDetailResponse)
    async def get_anime_by_name(anime_name: str):
        """
        Get anime information by name.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Search for the anime
        if hybrid_engine.content_model is None:
            raise HTTPException(status_code=503, detail="Content model not available")

        try:
            results = hybrid_engine.content_model.search_anime(anime_name, top_k=1)
            if not results:
                raise HTTPException(
                    status_code=404, detail=f"Anime '{anime_name}' not found"
                )

            anime_info = results[0]
            anime_info["image_url"] = _get_anime_image_url(anime_info["mal_id"])
            return AnimeDetailResponse(anime=AnimeInfo(**anime_info))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get anime by name error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Recommendation Endpoints
    # =========================================================================
    @app.get(
        "/api/recommend/anime/name/{anime_name}", response_model=RecommendationResponse
    )
    async def recommend_similar_by_name(
        anime_name: str,
        top_k: int = Query(default=10, ge=1, le=50),
        method: str = Query(default="hybrid", pattern="^(content|collaborative|implicit|hybrid)$"),
    ):
        """
        Get anime recommendations similar to the specified anime name.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            recommendations = hybrid_engine.recommend_similar_anime(
                anime_name, top_k=top_k, method=method
            )

            if not recommendations:
                raise HTTPException(
                    status_code=404, detail=f"Anime '{anime_name}' not found"
                )

            # Get query anime info
            query_anime = None
            if hybrid_engine.content_model:
                results = hybrid_engine.content_model.search_anime(anime_name, top_k=1)
                if results:
                    query_anime = results[0]
                    query_anime["image_url"] = _get_anime_image_url(
                        query_anime["mal_id"]
                    )

            # Enrich recommendations
            enriched_recs = []
            for rec in recommendations:
                rec["image_url"] = _get_anime_image_url(rec["mal_id"])
                enriched_recs.append(AnimeInfo(**rec))

            return RecommendationResponse(
                count=len(enriched_recs),
                method=method,
                query_anime=AnimeInfo(**query_anime) if query_anime else None,
                recommendations=enriched_recs,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Recommend by name error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/recommend/anime/{anime_id}", response_model=RecommendationResponse)
    async def recommend_similar_by_id(
        anime_id: int,
        top_k: int = Query(default=10, ge=1, le=50),
        method: str = Query(default="hybrid", pattern="^(content|collaborative|implicit|hybrid)$"),
    ):
        """
        Get anime recommendations similar to the specified anime ID.

        - **content**: Uses TF-IDF + SBERT similarity
        - **collaborative**: Uses item-item collaborative filtering
        - **hybrid**: Combines all methods with weighted scores
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Get query anime info (fallback to content model if not in _anime_info)
            query_anime = hybrid_engine._get_anime_info(anime_id)
            if not query_anime or (query_anime.get("name", "").startswith(f"Anime {anime_id}") and not hybrid_engine._anime_info.get(anime_id)):
                raise HTTPException(
                    status_code=404, detail=f"Anime ID {anime_id} not found"
                )

            query_anime["image_url"] = _get_anime_image_url(anime_id)

            # Get recommendations
            recommendations = hybrid_engine.recommend_similar_anime(
                anime_id, top_k=top_k, method=method
            )
            logger.info(f"recommend_similar_anime({anime_id}, method={method}) → {len(recommendations)} results")

            # Check content model index
            if len(recommendations) == 0 and method == "content":
                cm = hybrid_engine.content_model
                in_idx = cm is not None and anime_id in getattr(cm, "_id_to_idx", {})
                logger.warning(f"  content_model._id_to_idx has id={anime_id}: {in_idx}, n_total={len(getattr(cm, '_id_to_idx', {}))}")
                if cm is not None:
                    sample = list(getattr(cm, "_id_to_idx", {}).keys())[:5]
                    logger.warning(f"  sample _id_to_idx keys: {sample}")

            # Enrich with images
            enriched_recs = []
            for rec in recommendations:
                rec["image_url"] = _get_anime_image_url(rec["mal_id"])
                enriched_recs.append(AnimeInfo(**rec))

            return RecommendationResponse(
                count=len(enriched_recs),
                method=method,
                query_anime=AnimeInfo(**query_anime),
                recommendations=enriched_recs,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Recommend by ID error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/recommend/user/{user_id}", response_model=UserRecommendationResponse)
    async def recommend_for_user(
        user_id: int,
        top_k: int = Query(default=10, ge=1, le=50),
        exclude_watched: bool = Query(default=True),
        strategy: str = Query(default="auto", pattern="^(auto|new_user|existing_user)$"),
    ):
        """
        Get personalized recommendations for a user.

        Automatically handles cold-start users with popularity-based recommendations.

        - **auto**: System decides based on user history
        - **new_user**: Force cold-start strategy (popularity + content)
        - **existing_user**: Force collaborative strategy
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            is_cold_start = hybrid_engine._is_new_user(user_id)

            recommendations = hybrid_engine.recommend_for_user(
                user_id, top_k=top_k, exclude_watched=exclude_watched, strategy=strategy
            )

            actual_strategy = "new_user" if is_cold_start else "existing_user"
            if strategy != "auto":
                actual_strategy = strategy

            # Enrich recommendations
            enriched_recs = []
            for rec in recommendations:
                rec["image_url"] = _get_anime_image_url(rec["mal_id"])
                enriched_recs.append(AnimeInfo(**rec))

            return UserRecommendationResponse(
                user_id=user_id,
                is_cold_start=is_cold_start,
                strategy=actual_strategy,
                count=len(enriched_recs),
                recommendations=enriched_recs,
            )
        except Exception as e:
            logger.error(f"User recommendation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/user/{user_id}/history")
    async def get_user_history(
        user_id: int,
        top_k: int = Query(default=20, ge=1, le=100),
    ):
        """Get watch history for a user.

        Primary source: in-memory history (if present).
        Fallback: stream from `data/animelist.csv` so demo users (e.g., user 1) always show history.

        Note: We intentionally allow the fallback to work even if the model isn't loaded,
        so the UI can still display watch behavior.
        """
        lookup_id = user_id

        # IMPORTANT: The frontend can sometimes call `/api/user/0/history` while no demo user
        # has been selected yet. Scanning the CSV for a non-existent user can take long and
        # will hit the frontend timeout (20s). Treat non-positive IDs as "no history".
        try:
            if int(lookup_id) <= 0:
                return {
                    "user_id": user_id,
                    "history": [],
                    "is_known_user": False,
                    "count": 0,
                }
        except Exception:
            return {"user_id": user_id, "history": [], "is_known_user": False, "count": 0}

        # If the engine is ready, use its in-memory caches first.
        user_ratings = None
        user_watched = None
        if hybrid_engine is not None:
            user_ratings = hybrid_engine._user_ratings.get(lookup_id)
            if user_ratings is None:
                user_ratings = hybrid_engine._user_ratings.get(str(lookup_id))

            user_watched = hybrid_engine._user_watched.get(lookup_id)
            if user_watched is None:
                user_watched = hybrid_engine._user_watched.get(str(lookup_id))

        # Fallback: load from animelist.csv if in-memory history isn't available (or engine missing).
        if not user_ratings and not user_watched:
            csv_items = _read_animelist_history_csv(user_id=user_id, top_k=top_k)
            if not csv_items:
                return {"user_id": user_id, "history": [], "is_known_user": False, "count": 0}

            history = []
            for anime_id, rating, watching_status, watched_eps in csv_items:
                implicit_score = (float(rating) / 10.0) if rating and rating > 0 else 0.5

                # If model isn't loaded, return minimal info.
                if hybrid_engine is None:
                    anime_info = {
                        "mal_id": int(anime_id),
                        "name": f"Anime {int(anime_id)}",
                        "english_name": None,
                        "genres": "",
                        "score": None,
                        "type": None,
                        "synopsis": "",
                    }
                else:
                    anime_info = hybrid_engine._get_anime_info(int(anime_id))

                anime_info["implicit_score"] = round(float(implicit_score), 4)
                anime_info["status_label"] = _watching_status_label(int(watching_status))
                anime_info["watched_episodes"] = (
                    int(watched_eps) if watched_eps and watched_eps > 0 else 0
                )
                anime_info["image_url"] = _get_anime_image_url(int(anime_id))

                history.append(anime_info)

            return {
                "user_id": user_id,
                "is_known_user": True,
                "count": len(history),
                "history": history,
            }

        # If engine isn't loaded and there was no CSV fallback hit, report empty.
        if hybrid_engine is None:
            return {"user_id": user_id, "history": [], "is_known_user": False}

        try:
            def _status_label(s: float) -> str:
                if s >= 0.55: return "Completed"
                if s >= 0.35: return "Watching"
                if s >= 0.15: return "On-hold"
                return "Dropped"

            history = []

            items_to_process = list(user_ratings.items()) if user_ratings else []
            if user_watched:
                existing_aids = {aid for aid, _ in items_to_process}
                for aid in user_watched:
                    if aid not in existing_aids:
                        items_to_process.append((aid, 5.0))

            items_to_process.sort(key=lambda x: x[1], reverse=True)
            items_to_process = items_to_process[:top_k]

            for anime_id, rating in items_to_process:
                implicit_score = rating / 10.0 if rating > 0 else 0.5
                anime_info = hybrid_engine._get_anime_info(anime_id)
                anime_info["implicit_score"] = round(implicit_score, 4)
                anime_info["status_label"] = _status_label(implicit_score)
                anime_info["watched_episodes"] = round(implicit_score * 50)
                anime_info["image_url"] = _get_anime_image_url(anime_id)
                history.append(anime_info)

            return {
                "user_id": user_id,
                "is_known_user": True,
                "count": len(history),
                "history": history,
            }
        except Exception as e:
            logger.error(f"User history error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/demo/users")
    async def get_demo_users():
        """Return demo user IDs for the For You page.

        Important: this endpoint must be fast. Any expensive model scoring should be done
        asynchronously (background) and cached; otherwise the frontend can appear to
        "load forever".
        """
        global _demo_user_cache

        fallback = {
            "users": [
                {"user_id": 1, "label": "User 1", "description": "Demo user"},
                {"user_id": 100, "label": "User 100", "description": "Demo user"},
                {
                    "user_id": 999999,
                    "label": "New User",
                    "description": "Cold start — no history",
                },
            ]
        }

        if hybrid_engine is None or hybrid_engine.implicit_model is None:
            return fallback

        # If we already computed (or attempted) demo users, return immediately.
        if _demo_user_cache.get("users"):
            return {"users": _demo_user_cache["users"]}

        implicit_model = hybrid_engine.implicit_model

        # Fast path: pick a couple of deterministic existing users without scoring.
        # This keeps latency constant and avoids blocking the event loop.
        try:
            all_user_ids = list(getattr(implicit_model, "user_to_idx", {}).keys())
            if not all_user_ids:
                return fallback

            # Pick early/middle IDs to increase chance they exist in auxiliary maps.
            u1 = all_user_ids[0]
            u2 = all_user_ids[len(all_user_ids) // 2] if len(all_user_ids) > 1 else all_user_ids[0]

            quick_users = [
                {"user_id": int(u1), "label": f"Demo (#{int(u1)})", "description": "Existing user"},
                {"user_id": int(u2), "label": f"Demo (#{int(u2)})", "description": "Existing user"},
                {
                    "user_id": 999999,
                    "label": "New User",
                    "description": "Cold start — no history",
                },
            ]

            # Cache immediately so the UI doesn't wait; we may upgrade the cache later.
            _demo_user_cache["users"] = quick_users
            _demo_user_cache["status"] = "quick"
        except Exception:
            return fallback

        # Optional upgrade: attempt to compute more "interesting" users in the background.
        # If it fails or takes too long, we keep the quick cached result.
        if not _demo_user_cache.get("upgrade_task_started"):
            _demo_user_cache["upgrade_task_started"] = True

            import asyncio

            def _compute_better_demo_users():
                try:
                    import numpy as np
                    import time

                    uf = getattr(implicit_model, "user_factors", None)
                    itf = getattr(implicit_model, "item_factors", None)
                    user_to_idx = getattr(implicit_model, "user_to_idx", {})
                    anime_to_idx = getattr(implicit_model, "anime_to_idx", {})

                    n_users = len(user_to_idx)
                    n_items = len(anime_to_idx)

                    # Handle swapped factor matrices if present.
                    if (
                        uf is not None
                        and itf is not None
                        and uf.shape[0] == n_items
                        and itf.shape[0] == n_users
                    ):
                        uf, itf = itf, uf

                    if uf is None or itf is None or n_users == 0 or n_items == 0:
                        return None

                    # Hard caps to guarantee bounded runtime.
                    max_users = 200
                    top_n = 50
                    time_budget_s = 0.75

                    all_uids = list(user_to_idx.keys())
                    sample_ids = all_uids[: min(max_users, len(all_uids))]

                    best_mean = -1.0
                    best_var = -1.0
                    action_fan_id = sample_ids[0]
                    diverse_id = sample_ids[-1]

                    t0 = time.perf_counter()
                    for uid in sample_ids:
                        if time.perf_counter() - t0 > time_budget_s:
                            break

                        uidx = user_to_idx.get(uid)
                        if uidx is None or uidx >= uf.shape[0]:
                            continue

                        # Compute scores for all items; then use argpartition to avoid full sort.
                        scores = itf @ uf[uidx]
                        if scores.size <= top_n:
                            top_scores = np.sort(scores)[::-1]
                        else:
                            idx = np.argpartition(scores, -top_n)[-top_n:]
                            top_scores = np.sort(scores[idx])[::-1]

                        mean_top = float(top_scores.mean())
                        var_top = float(top_scores.std())

                        if mean_top > best_mean:
                            best_mean = mean_top
                            action_fan_id = uid
                        if var_top > best_var:
                            best_var = var_top
                            diverse_id = uid

                    users = [
                        {
                            "user_id": int(action_fan_id),
                            "label": f"Action Fan (#{int(action_fan_id)})",
                            "description": "Most active watcher (ALS proxy)",
                        },
                        {
                            "user_id": int(diverse_id),
                            "label": f"Diverse (#{int(diverse_id)})",
                            "description": "Diverse taste (ALS proxy)",
                        },
                        {
                            "user_id": 999999,
                            "label": "New User",
                            "description": "Cold start — no history",
                        },
                    ]
                    return users
                except Exception as e:
                    logger.warning(f"Demo users background compute failed: {e}")
                    return None

            async def _upgrade_cache_task():
                loop = asyncio.get_running_loop()
                users = await loop.run_in_executor(None, _compute_better_demo_users)
                if users:
                    _demo_user_cache["users"] = users
                    _demo_user_cache["status"] = "upgraded"

            try:
                asyncio.create_task(_upgrade_cache_task())
            except Exception:
                # If task can't be scheduled (rare), keep quick result.
                pass

        return {"users": _demo_user_cache["users"]}

    # =========================================================================
    # Popular/Trending Endpoints
    # =========================================================================

    @app.get("/api/popular", response_model=PopularResponse)
    async def get_popular(
        type: str = Query(
            default="top_rated",
            pattern="^(top_rated|most_watched|trending|most_members)$",
        ),
        top_k: int = Query(default=20, ge=1, le=100),
        genre: Optional[str] = Query(default=None),
    ):
        """
        Get popular anime by different metrics.

        - **top_rated**: Highest MAL scores
        - **most_watched**: Most watched count
        - **trending**: Currently trending
        - **most_members**: Most MAL members
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if hybrid_engine.popularity_model is None:
            raise HTTPException(
                status_code=503, detail="Popularity model not available"
            )

        try:
            popular = hybrid_engine.popularity_model.get_popular(
                top_k=top_k, popularity_type=type, genre=genre
            )

            enriched = []
            for p in popular:
                p["image_url"] = _get_anime_image_url(p["mal_id"])
                enriched.append(AnimeInfo(**p))

            return PopularResponse(count=len(enriched), type=type, anime=enriched)
        except Exception as e:
            logger.error(f"Popular anime error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/trending")
    async def get_trending(top_k: int = Query(default=10, ge=1, le=50)):
        """Get trending anime (alias for popular with type=trending)."""
        return await get_popular(type="trending", top_k=top_k)

    # =========================================================================
    # Weights & Configuration Endpoints
    # =========================================================================

    @app.get("/api/weights")
    async def get_weights():
        """Get current hybrid model weights."""
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {"success": True, "weights": hybrid_engine.weights}

    @app.put("/api/weights")
    async def update_weights(weights: WeightsModel):
        """
        Update hybrid model weights.

        Weights will be normalized to sum to 1.0.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            new_weights = {
                "content": weights.content,
                "collaborative": weights.collaborative,
                "implicit": weights.implicit,
                "popularity": weights.popularity,
            }

            hybrid_engine.set_weights(new_weights)

            return {"success": True, "weights": hybrid_engine.weights}
        except Exception as e:
            logger.error(f"Update weights error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/genres")
    async def get_genres():
        """Get list of all available genres."""
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            genres = set()
            for anime_id, info in hybrid_engine._anime_info.items():
                for genre in str(info.get("genres", "")).split(","):
                    genre = genre.strip()
                    if genre and genre != "nan":
                        genres.add(genre)

            return {
                "success": True,
                "count": len(genres),
                "genres": sorted(list(genres)),
            }
        except Exception as e:
            logger.error(f"Get genres error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Comparison & Analysis Endpoints
    # =========================================================================

    @app.get("/api/compare")
    async def compare_methods(
        anime_id: int, top_k: int = Query(default=5, ge=1, le=20)
    ):
        """
        Compare recommendations from different methods.

        Returns side-by-side results from content, collaborative, and hybrid methods.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            results = {}

            for method in ["content", "collaborative", "implicit", "hybrid"]:
                try:
                    recs = hybrid_engine.recommend_similar_anime(
                        anime_id, top_k=top_k, method=method
                    )

                    normalized = []
                    for rec in recs:
                        # Ensure mal_id is int for downstream usage
                        aid = int(rec.get("mal_id"))

                        # Some models may return only mal_id + similarity/score.
                        merged = dict(rec)
                        if not merged.get("name"):
                            merged = {**hybrid_engine._get_anime_info(aid), **merged}

                        # Compare UI prints similarity% match; fall back gracefully.
                        if merged.get("hybrid_score") is None:
                            if merged.get("similarity") is not None:
                                merged["hybrid_score"] = float(merged["similarity"])
                            elif merged.get("predicted_rating") is not None:
                                merged["hybrid_score"] = float(merged["predicted_rating"])
                            else:
                                merged["hybrid_score"] = 0.0

                        merged["image_url"] = _get_anime_image_url(aid)
                        normalized.append(merged)

                    results[method] = normalized
                    logger.info(
                        f"[compare] anime_id={anime_id} method={method} count={len(normalized)}"
                    )
                except Exception as e:
                    results[method] = []
                    logger.warning(
                        f"[compare] anime_id={anime_id} method={method} failed: {e}"
                    )

            return {"success": True, "anime_id": anime_id, "results": results}
        except Exception as e:
            logger.error(f"Compare methods error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# =============================================================================
# Application Instance
# =============================================================================

# Expose the ASGI app at module level for uvicorn/TestClient usage.
app = create_demo_app()


# =============================================================================
# Helper Functions
# =============================================================================


def _get_anime_image_url(mal_id: int) -> str:
    """
    Get anime image URL from MyAnimeList CDN.

    Args:
        mal_id: MyAnimeList ID

    Returns:
        Image URL or placeholder
    """
    # MAL CDN URL pattern
    return f"https://cdn.myanimelist.net/images/anime/{mal_id % 100}/{mal_id}.jpg"


def _watching_status_label(status: int) -> str:
    """Map MAL watching_status codes to UI labels."""
    # Common MAL export codes:
    # 1=Watching, 2=Completed, 3=On-Hold, 4=Dropped, 6=Plan to Watch
    return {
        1: "Watching",
        2: "Completed",
        3: "On-hold",
        4: "Dropped",
        6: "Plan to Watch",
    }.get(int(status), "Unknown")


def _read_animelist_history_csv(user_id: int, top_k: int) -> list:
    """Stream-read `data/animelist.csv` and return up to `top_k` records for one user.

    Expected row format (no header in some dumps):
      user_id, anime_id, rating, watching_status, watched_episodes

    If a header exists, we still support it by mapping columns.

    Returns a list of tuples:
      (anime_id:int, rating:float, watching_status:int, watched_episodes:int)

    IMPORTANT: This function must be bounded in runtime. Unknown user IDs (like 0) should
    return immediately and we also enforce row/time budgets so the UI doesn't time out.
    """
    from pathlib import Path

    # Fast reject invalid IDs to avoid scanning the whole CSV.
    try:
        if int(user_id) <= 0:
            return []
    except Exception:
        return []

    csv_path = Path(ROOT_DIR) / "data" / "animelist.csv"
    if not csv_path.exists():
        return []

    # Keep work bounded even for extremely active users.
    max_collect = max(top_k * 10, 500)

    # Prefer cached user history if present (cache is keyed by (user_id, top_k)).
    cache_key = (int(user_id), int(top_k))
    cached = _user_history_cache.get(cache_key)
    if cached is not None:
        return cached

    results = []
    try:
        import csv
        import time

        # Hard budgets: guard against scanning gigantic files when the user isn't present.
        max_rows = 300_000
        time_budget_s = 1.0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                return []

            # Detect header row vs data row
            try:
                int(first_row[0])
                has_header = False
                initial_row = first_row
            except (ValueError, IndexError):
                has_header = True
                initial_row = None

            uid_col, aid_col, rat_col, wst_col, wep_col = 0, 1, 2, 3, 4

            if has_header:
                col_map = {v.strip().lower(): i for i, v in enumerate(first_row)}
                uid_col = col_map.get("user_id", 0)
                aid_col = col_map.get("anime_id", 1)
                rat_col = col_map.get("rating", 2)
                wst_col = col_map.get("watching_status", 3)
                wep_col = col_map.get("watched_episodes", 4)

            def _parse_row(row):
                try:
                    if int(row[uid_col]) != int(user_id):
                        return None
                    anime_id = int(row[aid_col])
                    rating = float(row[rat_col]) if row[rat_col] else 0.0
                    watching_status = int(float(row[wst_col])) if row[wst_col] else 0
                    watched_eps = int(float(row[wep_col])) if row[wep_col] else 0
                    return (anime_id, rating, watching_status, watched_eps)
                except (ValueError, IndexError):
                    return None

            t0 = time.perf_counter()

            if initial_row is not None:
                item = _parse_row(initial_row)
                if item is not None:
                    results.append(item)

            n_rows = 0
            for row in reader:
                n_rows += 1
                if n_rows >= max_rows:
                    break
                if (time.perf_counter() - t0) > time_budget_s:
                    break
                if len(results) >= max_collect:
                    break

                item = _parse_row(row)
                if item is not None:
                    results.append(item)
    except Exception as e:
        logger.warning(f"_read_animelist_history_csv error: {e}")
        return []

    results.sort(key=lambda x: x[1], reverse=True)
    final = results[:top_k]
    _user_history_cache[cache_key] = final
    return final


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
