"""
FastAPI REST API for Anime Recommendation System.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import api_config, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global hybrid engine instance
hybrid_engine = None


# Pydantic models for API
class AnimeRecommendation(BaseModel):
    """Anime recommendation response model."""
    mal_id: int
    name: str
    english_name: Optional[str] = None
    genres: Optional[str] = None
    score: Optional[float] = None
    type: Optional[str] = None
    similarity: Optional[float] = None
    predicted_rating: Optional[float] = None
    hybrid_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """API response for recommendations."""
    success: bool = True
    count: int
    recommendations: List[AnimeRecommendation]
    strategy: Optional[str] = None


class PopularResponse(BaseModel):
    """API response for popular anime."""
    success: bool = True
    count: int
    type: str
    anime: List[AnimeRecommendation]


class SearchResponse(BaseModel):
    """API response for search results."""
    success: bool = True
    count: int
    query: str
    results: List[AnimeRecommendation]


class WeightsUpdate(BaseModel):
    """Request model for updating hybrid weights."""
    content: Optional[float] = Field(None, ge=0, le=1)
    collaborative: Optional[float] = Field(None, ge=0, le=1)
    implicit: Optional[float] = Field(None, ge=0, le=1)
    popularity: Optional[float] = Field(None, ge=0, le=1)


class ExplanationResponse(BaseModel):
    """API response for recommendation explanation."""
    success: bool = True
    anime_id: int
    anime_info: Dict
    reasons: List[Dict]


def create_app(engine=None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        engine: HybridEngine instance (optional, will load from disk if not provided)

    Returns:
        FastAPI application
    """
    global hybrid_engine

    app = FastAPI(
        title="Anime Recommendation API",
        description="Hybrid Anime Recommendation System using Content-Based, Collaborative Filtering, and Implicit Feedback",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def startup_event():
        """Load models on startup."""
        global hybrid_engine

        if engine is not None:
            hybrid_engine = engine
            logger.info("Using provided HybridEngine instance")
        else:
            # Try to load from disk
            model_path = MODELS_DIR / "hybrid"
            if model_path.exists():
                try:
                    from models.hybrid import HybridEngine
                    hybrid_engine = HybridEngine()
                    hybrid_engine.load(model_path)
                    logger.info(f"Loaded HybridEngine from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load HybridEngine: {e}")
                    hybrid_engine = None
            else:
                logger.warning(f"No saved model found at {model_path}")
                hybrid_engine = None

    @app.get("/", response_class=FileResponse)
    async def root():
        """Serve the main page."""
        index_path = Path(__file__).parent.parent / "static" / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Anime Recommendation API", "docs": "/docs"}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": hybrid_engine is not None
        }

    @app.get("/recommend/anime/{anime_name}", response_model=RecommendationResponse)
    async def recommend_similar_anime(
        anime_name: str,
        top_k: int = Query(default=10, ge=1, le=100),
        method: str = Query(default="hybrid", regex="^(content|collaborative|hybrid)$")
    ):
        """
        Get anime recommendations similar to the specified anime.

        - **anime_name**: Name of the anime to find similar ones
        - **top_k**: Number of recommendations (1-100)
        - **method**: Recommendation method (content, collaborative, hybrid)
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            recommendations = hybrid_engine.recommend_similar_anime(
                anime_name, top_k=top_k, method=method
            )

            if not recommendations:
                raise HTTPException(status_code=404, detail=f"Anime '{anime_name}' not found")

            return RecommendationResponse(
                count=len(recommendations),
                recommendations=[AnimeRecommendation(**rec) for rec in recommendations]
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in recommend_similar_anime: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/recommend/anime/id/{anime_id}", response_model=RecommendationResponse)
    async def recommend_similar_anime_by_id(
        anime_id: int,
        top_k: int = Query(default=10, ge=1, le=100),
        method: str = Query(default="hybrid", regex="^(content|collaborative|hybrid)$")
    ):
        """
        Get anime recommendations similar to the specified anime ID.

        - **anime_id**: MAL ID of the anime
        - **top_k**: Number of recommendations
        - **method**: Recommendation method
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            recommendations = hybrid_engine.recommend_similar_anime(
                anime_id, top_k=top_k, method=method
            )

            if not recommendations:
                raise HTTPException(status_code=404, detail=f"Anime ID {anime_id} not found")

            return RecommendationResponse(
                count=len(recommendations),
                recommendations=[AnimeRecommendation(**rec) for rec in recommendations]
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in recommend_similar_anime_by_id: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/recommend/user/{user_id}", response_model=RecommendationResponse)
    async def recommend_for_user(
        user_id: int,
        top_k: int = Query(default=10, ge=1, le=100),
        exclude_watched: bool = Query(default=True),
        strategy: str = Query(default="auto", regex="^(auto|new_user|existing_user)$")
    ):
        """
        Get personalized recommendations for a user.

        - **user_id**: User ID
        - **top_k**: Number of recommendations
        - **exclude_watched**: Exclude already watched anime
        - **strategy**: Recommendation strategy (auto, new_user, existing_user)
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            recommendations = hybrid_engine.recommend_for_user(
                user_id,
                top_k=top_k,
                exclude_watched=exclude_watched,
                strategy=strategy
            )

            strategy_used = recommendations[0].get('strategy', strategy) if recommendations else strategy

            return RecommendationResponse(
                count=len(recommendations),
                recommendations=[AnimeRecommendation(**rec) for rec in recommendations],
                strategy=strategy_used
            )
        except Exception as e:
            logger.error(f"Error in recommend_for_user: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/popular", response_model=PopularResponse)
    async def get_popular(
        type: str = Query(default="top_rated", regex="^(top_rated|most_watched|trending|most_members)$"),
        top_k: int = Query(default=10, ge=1, le=100),
        genre: Optional[str] = Query(default=None)
    ):
        """
        Get popular anime.

        - **type**: Popularity type (top_rated, most_watched, trending, most_members)
        - **top_k**: Number of results
        - **genre**: Filter by genre (optional)
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if hybrid_engine.popularity_model is None:
            raise HTTPException(status_code=503, detail="Popularity model not available")

        try:
            popular = hybrid_engine.popularity_model.get_popular(
                top_k=top_k,
                popularity_type=type,
                genre=genre
            )

            return PopularResponse(
                count=len(popular),
                type=type,
                anime=[AnimeRecommendation(**p) for p in popular]
            )
        except Exception as e:
            logger.error(f"Error in get_popular: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/search", response_model=SearchResponse)
    async def search_anime(
        q: str = Query(..., min_length=1, max_length=100),
        top_k: int = Query(default=10, ge=1, le=100)
    ):
        """
        Search for anime by name.

        - **q**: Search query
        - **top_k**: Number of results
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if hybrid_engine.content_model is None:
            raise HTTPException(status_code=503, detail="Content model not available")

        try:
            results = hybrid_engine.content_model.search_anime(q, top_k=top_k)

            return SearchResponse(
                count=len(results),
                query=q,
                results=[AnimeRecommendation(**r) for r in results]
            )
        except Exception as e:
            logger.error(f"Error in search_anime: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/explain/{user_id}/{anime_id}", response_model=ExplanationResponse)
    async def explain_recommendation(
        user_id: int,
        anime_id: int
    ):
        """
        Get explanation for why an anime was recommended.

        - **user_id**: User ID
        - **anime_id**: Anime ID
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            explanation = hybrid_engine.get_explanation(user_id, anime_id)
            return ExplanationResponse(**explanation)
        except Exception as e:
            logger.error(f"Error in explain_recommendation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/weights")
    async def update_weights(weights: WeightsUpdate):
        """
        Update hybrid model weights.

        Weights will be normalized to sum to 1.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            new_weights = {}
            if weights.content is not None:
                new_weights['content'] = weights.content
            if weights.collaborative is not None:
                new_weights['collaborative'] = weights.collaborative
            if weights.implicit is not None:
                new_weights['implicit'] = weights.implicit
            if weights.popularity is not None:
                new_weights['popularity'] = weights.popularity

            if new_weights:
                hybrid_engine.set_weights(new_weights)

            return {
                "success": True,
                "weights": hybrid_engine.weights
            }
        except Exception as e:
            logger.error(f"Error in update_weights: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/weights")
    async def get_weights():
        """Get current hybrid model weights."""
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "success": True,
            "weights": hybrid_engine.weights
        }

    @app.get("/genres")
    async def get_genres():
        """Get list of all genres."""
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            genres = set()
            for anime_id, info in hybrid_engine._anime_info.items():
                for genre in str(info.get('genres', '')).split(','):
                    genre = genre.strip()
                    if genre:
                        genres.add(genre)

            return {
                "success": True,
                "count": len(genres),
                "genres": sorted(list(genres))
            }
        except Exception as e:
            logger.error(f"Error in get_genres: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "routes:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug
    )
