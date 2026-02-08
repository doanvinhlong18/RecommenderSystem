"""
Enhanced FastAPI Backend for Anime Recommendation Web Demo.

This module provides a comprehensive REST API for the hybrid anime recommendation system.
It includes endpoints for search, recommendations, popular anime, and user personalization.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import api_config, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global hybrid engine instance
hybrid_engine = None


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
        redoc_url="/redoc"
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
    
    @app.on_event("startup")
    async def startup_event():
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
                "content_model": hybrid_engine.content_model is not None if hybrid_engine else False,
                "collaborative_model": hybrid_engine.collaborative_model is not None if hybrid_engine else False,
                "implicit_model": hybrid_engine.implicit_model is not None if hybrid_engine else False,
                "popularity_model": hybrid_engine.popularity_model is not None if hybrid_engine else False,
            }
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
            }
        }
    
    # =========================================================================
    # Search Endpoints
    # =========================================================================
    
    @app.get("/api/search", response_model=SearchResult)
    async def search_anime(
        q: str = Query(..., min_length=1, max_length=100, description="Search query"),
        top_k: int = Query(default=10, ge=1, le=50, description="Number of results")
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
                r['image_url'] = _get_anime_image_url(r['mal_id'])
                enriched_results.append(AnimeInfo(**r))
            
            return SearchResult(
                count=len(enriched_results),
                query=q,
                results=enriched_results
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/autocomplete", response_model=AutocompleteResult)
    async def autocomplete(
        q: str = Query(..., min_length=1, max_length=100),
        limit: int = Query(default=8, ge=1, le=20)
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
                    "mal_id": r['mal_id'],
                    "name": r['name'],
                    "english_name": r.get('english_name', r['name']),
                    "score": r.get('score', 0),
                    "type": r.get('type', 'Unknown'),
                    "image_url": _get_anime_image_url(r['mal_id'])
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
        
        anime_info = hybrid_engine._anime_info.get(anime_id)
        if anime_info is None:
            raise HTTPException(status_code=404, detail=f"Anime with ID {anime_id} not found")
        
        anime_info['image_url'] = _get_anime_image_url(anime_id)
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
                raise HTTPException(status_code=404, detail=f"Anime '{anime_name}' not found")
            
            anime_info = results[0]
            anime_info['image_url'] = _get_anime_image_url(anime_info['mal_id'])
            return AnimeDetailResponse(anime=AnimeInfo(**anime_info))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get anime by name error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Recommendation Endpoints
    # =========================================================================
    
    @app.get("/api/recommend/anime/{anime_id}", response_model=RecommendationResponse)
    async def recommend_similar_by_id(
        anime_id: int,
        top_k: int = Query(default=10, ge=1, le=50),
        method: str = Query(default="hybrid", regex="^(content|collaborative|hybrid)$")
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
            # Get query anime info
            query_anime = hybrid_engine._anime_info.get(anime_id)
            if query_anime is None:
                raise HTTPException(status_code=404, detail=f"Anime ID {anime_id} not found")
            
            query_anime['image_url'] = _get_anime_image_url(anime_id)
            
            # Get recommendations
            recommendations = hybrid_engine.recommend_similar_anime(
                anime_id, top_k=top_k, method=method
            )
            
            # Enrich with images
            enriched_recs = []
            for rec in recommendations:
                rec['image_url'] = _get_anime_image_url(rec['mal_id'])
                enriched_recs.append(AnimeInfo(**rec))
            
            return RecommendationResponse(
                count=len(enriched_recs),
                method=method,
                query_anime=AnimeInfo(**query_anime),
                recommendations=enriched_recs
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Recommend by ID error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/recommend/anime/name/{anime_name}", response_model=RecommendationResponse)
    async def recommend_similar_by_name(
        anime_name: str,
        top_k: int = Query(default=10, ge=1, le=50),
        method: str = Query(default="hybrid", regex="^(content|collaborative|hybrid)$")
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
                raise HTTPException(status_code=404, detail=f"Anime '{anime_name}' not found")
            
            # Get query anime info
            query_anime = None
            if hybrid_engine.content_model:
                results = hybrid_engine.content_model.search_anime(anime_name, top_k=1)
                if results:
                    query_anime = results[0]
                    query_anime['image_url'] = _get_anime_image_url(query_anime['mal_id'])
            
            # Enrich recommendations
            enriched_recs = []
            for rec in recommendations:
                rec['image_url'] = _get_anime_image_url(rec['mal_id'])
                enriched_recs.append(AnimeInfo(**rec))
            
            return RecommendationResponse(
                count=len(enriched_recs),
                method=method,
                query_anime=AnimeInfo(**query_anime) if query_anime else None,
                recommendations=enriched_recs
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Recommend by name error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/recommend/user/{user_id}", response_model=UserRecommendationResponse)
    async def recommend_for_user(
        user_id: int,
        top_k: int = Query(default=10, ge=1, le=50),
        exclude_watched: bool = Query(default=True),
        strategy: str = Query(default="auto", regex="^(auto|new_user|existing_user)$")
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
                user_id,
                top_k=top_k,
                exclude_watched=exclude_watched,
                strategy=strategy
            )
            
            actual_strategy = "new_user" if is_cold_start else "existing_user"
            if strategy != "auto":
                actual_strategy = strategy
            
            # Enrich recommendations
            enriched_recs = []
            for rec in recommendations:
                rec['image_url'] = _get_anime_image_url(rec['mal_id'])
                enriched_recs.append(AnimeInfo(**rec))
            
            return UserRecommendationResponse(
                user_id=user_id,
                is_cold_start=is_cold_start,
                strategy=actual_strategy,
                count=len(enriched_recs),
                recommendations=enriched_recs
            )
        except Exception as e:
            logger.error(f"User recommendation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Popular/Trending Endpoints
    # =========================================================================
    
    @app.get("/api/popular", response_model=PopularResponse)
    async def get_popular(
        type: str = Query(
            default="top_rated",
            regex="^(top_rated|most_watched|trending|most_members)$"
        ),
        top_k: int = Query(default=20, ge=1, le=100),
        genre: Optional[str] = Query(default=None)
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
            raise HTTPException(status_code=503, detail="Popularity model not available")
        
        try:
            popular = hybrid_engine.popularity_model.get_popular(
                top_k=top_k,
                popularity_type=type,
                genre=genre
            )
            
            enriched = []
            for p in popular:
                p['image_url'] = _get_anime_image_url(p['mal_id'])
                enriched.append(AnimeInfo(**p))
            
            return PopularResponse(
                count=len(enriched),
                type=type,
                anime=enriched
            )
        except Exception as e:
            logger.error(f"Popular anime error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/trending")
    async def get_trending(
        top_k: int = Query(default=10, ge=1, le=50)
    ):
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
        
        return {
            "success": True,
            "weights": hybrid_engine.weights
        }
    
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
                'content': weights.content,
                'collaborative': weights.collaborative,
                'implicit': weights.implicit,
                'popularity': weights.popularity
            }
            
            hybrid_engine.set_weights(new_weights)
            
            return {
                "success": True,
                "weights": hybrid_engine.weights
            }
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
                for genre in str(info.get('genres', '')).split(','):
                    genre = genre.strip()
                    if genre and genre != 'nan':
                        genres.add(genre)
            
            return {
                "success": True,
                "count": len(genres),
                "genres": sorted(list(genres))
            }
        except Exception as e:
            logger.error(f"Get genres error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # =========================================================================
    # Comparison & Analysis Endpoints
    # =========================================================================
    
    @app.get("/api/compare")
    async def compare_methods(
        anime_id: int,
        top_k: int = Query(default=5, ge=1, le=20)
    ):
        """
        Compare recommendations from different methods.
        
        Returns side-by-side results from content, collaborative, and hybrid methods.
        """
        if hybrid_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            results = {}
            
            for method in ["content", "collaborative", "hybrid"]:
                try:
                    recs = hybrid_engine.recommend_similar_anime(
                        anime_id, top_k=top_k, method=method
                    )
                    results[method] = [
                        {
                            **rec,
                            "image_url": _get_anime_image_url(rec['mal_id'])
                        }
                        for rec in recs
                    ]
                except Exception as e:
                    results[method] = []
                    logger.warning(f"Method {method} failed: {e}")
            
            return {
                "success": True,
                "anime_id": anime_id,
                "results": results
            }
        except Exception as e:
            logger.error(f"Compare methods error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


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


# =============================================================================
# Application Instance
# =============================================================================

app = create_demo_app()


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
