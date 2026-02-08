"""
FastAPI Backend for Anime Hybrid Recommendation System
=======================================================
Provides REST API endpoints for the recommendation web app.

Endpoints:
- GET  /search?query=... - Search anime by title
- POST /recommend - Get hybrid recommendations
- GET  /metrics - Get evaluation metrics
- GET  /anime/{name} - Get anime details

Author: Full-Stack AI Engineer
Date: February 2026
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd

# Import the recommender
from Main import AdvancedHybridRecommender

# =============================================================================
# Initialize FastAPI App
# =============================================================================

app = FastAPI(
    title="Anime Hybrid Recommender API",
    description="API for the Advanced Hybrid Anime Recommendation System",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Model Instance (loaded once at startup)
# =============================================================================

recommender: Optional[AdvancedHybridRecommender] = None
MODEL_PATH = "advanced_recommender_model.pkl"

# Cached evaluation metrics
EVALUATION_METRICS = {
    "precision_at_5": 0.4720,
    "recall_at_5": 0.0228,
    "ndcg_at_5": 0.5085,
    "best_config": {
        "name": "Semantic Focus",
        "alpha": 0.3,
        "beta": 0.5,
        "gamma": 0.2,
        "ndcg": 0.6594
    },
    "weight_comparison": [
        {"name": "TF-IDF Heavy", "alpha": 0.8, "beta": 0.1, "gamma": 0.1, "ndcg": 0.6427},
        {"name": "SBERT Heavy", "alpha": 0.1, "beta": 0.8, "gamma": 0.1, "ndcg": 0.5095},
        {"name": "Balanced", "alpha": 0.4, "beta": 0.4, "gamma": 0.2, "ndcg": 0.5135},
        {"name": "Semantic Focus", "alpha": 0.3, "beta": 0.5, "gamma": 0.2, "ndcg": 0.6594},
        {"name": "Lexical Focus", "alpha": 0.5, "beta": 0.3, "gamma": 0.2, "ndcg": 0.5996},
    ]
}

# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class RecommendRequest(BaseModel):
    """Request body for /recommend endpoint"""
    title: str = Field(..., description="Anime title to get recommendations for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations")
    alpha: float = Field(default=0.4, ge=0, le=1, description="TF-IDF weight")
    beta: float = Field(default=0.4, ge=0, le=1, description="SBERT weight")
    gamma: float = Field(default=0.2, ge=0, le=1, description="Score weight")


class AnimeInfo(BaseModel):
    """Anime information"""
    name: str
    score: float
    type: Optional[str] = None
    themes: Optional[str] = None
    hybrid_score: Optional[float] = None
    tfidf_sim: Optional[float] = None
    sbert_sim: Optional[float] = None


class RecommendResponse(BaseModel):
    """Response for /recommend endpoint"""
    query_anime: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    weights: Dict[str, float]
    total_results: int


class MetricsResponse(BaseModel):
    """Response for /metrics endpoint"""
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    best_config: Dict[str, Any]
    weight_comparison: List[Dict[str, Any]]


# =============================================================================
# Startup Event - Load Model
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load the recommender model at startup."""
    global recommender

    print("=" * 60)
    print("🚀 Starting Anime Recommender API Server...")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model file '{MODEL_PATH}' not found!")
        print("   Please run demo.py first to train and save the model.")
        return

    print(f"📂 Loading model from '{MODEL_PATH}'...")
    recommender = AdvancedHybridRecommender.from_saved_model(MODEL_PATH)

    if recommender is not None:
        print(f"✅ Model loaded successfully! {len(recommender.df)} anime available")
    else:
        print("❌ Failed to load model!")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if recommender is not None else "model_not_loaded",
        "model_loaded": recommender is not None,
        "anime_count": len(recommender.df) if recommender else 0
    }


@app.get("/api/search")
async def search_anime(query: str = Query(..., min_length=1, description="Search query")):
    """
    Search for anime by title.

    Returns list of matching anime names for autocomplete.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(query) < 1:
        return {"results": []}

    # Search using the recommender's search function
    matches = recommender.search_anime(query, limit=15)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "name": row['name'],
            "score": float(row['score']),
            "type": row['type'] if pd.notna(row['type']) else "Unknown",
            "image_url": row.get('image_url', '') if pd.notna(row.get('image_url', '')) else '',
            "anime_url": row.get('anime_url', '') if pd.notna(row.get('anime_url', '')) else ''
        })

    return {"results": results, "count": len(results)}


@app.post("/api/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Get hybrid recommendations for an anime.

    Uses the 3-component hybrid model:
    - TF-IDF (lexical similarity)
    - SBERT (semantic similarity)
    - Score (quality weighting)
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate weights sum
    weight_sum = request.alpha + request.beta + request.gamma
    if abs(weight_sum - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 1.0 (current sum: {weight_sum:.2f})"
        )

    # Get recommendations
    recs_df = recommender.recommend(
        title=request.title,
        top_k=request.top_k,
        alpha=request.alpha,
        beta=request.beta,
        gamma=request.gamma,
        verbose=False
    )

    if recs_df.empty:
        raise HTTPException(status_code=404, detail=f"Anime '{request.title}' not found")

    # Get query anime info
    anime_idx = recommender._get_anime_index(request.title)
    query_anime_row = recommender.df.iloc[anime_idx]

    query_anime = {
        "name": query_anime_row['name'],
        "score": float(query_anime_row['score']),
        "type": query_anime_row.get('type', 'Unknown'),
        "themes": query_anime_row.get('themes', 'Unknown'),
        "normalized_score": float(query_anime_row.get('normalized_score', 0)),
        "image_url": query_anime_row.get('image_url', '') if pd.notna(query_anime_row.get('image_url', '')) else '',
        "anime_url": query_anime_row.get('anime_url', '') if pd.notna(query_anime_row.get('anime_url', '')) else ''
    }

    # Convert recommendations to list of dicts
    recommendations = []
    for idx, row in recs_df.iterrows():
        rec = {
            "rank": idx + 1,
            "name": row['name'],
            "score": float(row['score']),
            "type": row.get('type', 'Unknown'),
            "themes": row.get('themes', 'Unknown'),
            "hybrid_score": float(row['hybrid_score']),
            "tfidf_sim": float(row['tfidf_sim']),
            "sbert_sim": float(row['sbert_sim']),
            "image_url": row.get('image_url', '') if pd.notna(row.get('image_url', '')) else '',
            "anime_url": row.get('anime_url', '') if pd.notna(row.get('anime_url', '')) else ''
        }
        recommendations.append(rec)

    return RecommendResponse(
        query_anime=query_anime,
        recommendations=recommendations,
        weights={
            "alpha": request.alpha,
            "beta": request.beta,
            "gamma": request.gamma
        },
        total_results=len(recommendations)
    )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get stored evaluation metrics.

    Returns pre-computed metrics from the evaluation phase.
    """
    return MetricsResponse(**EVALUATION_METRICS)


@app.get("/api/anime/{anime_name}")
async def get_anime_details(anime_name: str):
    """
    Get detailed information about a specific anime.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Find the anime
    anime_idx = recommender._get_anime_index(anime_name)

    if anime_idx is None:
        raise HTTPException(status_code=404, detail=f"Anime '{anime_name}' not found")

    row = recommender.df.iloc[anime_idx]

    return {
        "name": row['name'],
        "score": float(row['score']),
        "type": row.get('type', 'Unknown'),
        "themes": row.get('themes', 'Unknown'),
        "demographics": row.get('demographics', 'Unknown'),
        "normalized_score": float(row.get('normalized_score', 0)),
        "image_url": row.get('image_url', '') if pd.notna(row.get('image_url', '')) else '',
        "anime_url": row.get('anime_url', '') if pd.notna(row.get('anime_url', '')) else ''
    }


@app.get("/api/random")
async def get_random_anime(count: int = Query(default=5, ge=1, le=20)):
    """Get random anime for suggestions."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get random sample
    sample = recommender.df.sample(n=min(count, len(recommender.df)))

    results = []
    for _, row in sample.iterrows():
        results.append({
            "name": row['name'],
            "score": float(row['score']),
            "type": row.get('type', 'Unknown')
        })

    return {"results": results}


# =============================================================================
# Static Files
# =============================================================================

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎌 ANIME HYBRID RECOMMENDER - WEB SERVER")
    print("=" * 60)
    print("\n📌 Open http://localhost:8000 in your browser\n")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
