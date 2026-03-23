"""
Configuration settings for the Anime Recommendation System.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
CACHE_DIR = BASE_DIR / "cache"

# Dataset paths
DATASET_PATH = Path(r"C:\Users\ACER\.cache\kagglehub\datasets\hernan4444\anime-recommendation-database-2020\versions\7")

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Data configuration settings."""
    anime_file: str = "anime.csv"
    anime_synopsis_file: str = "anime_with_synopsis.csv"
    animelist_file: str = "animelist.csv"
    rating_file: str = "rating_complete.csv"
    watching_status_file: str = "watching_status.csv"

    # Sampling settings for large datasets
    sample_ratings: bool = False
    rating_sample_size: int = None  # 5M samples for development
    animelist_sample_size: int = None  # 10M samples

    # Minimum thresholds
    min_user_ratings: int = 5
    min_anime_ratings: int = 10


@dataclass
class ModelConfig:
    """Model configuration settings."""
    # Content-based settings
    tfidf_max_features: int = 5000
    sbert_model_name: str = "all-MiniLM-L6-v2"
    content_weight: float = 0.3

    # Collaborative filtering settings
    svd_factors: int = 100
    svd_epochs: int = 20
    svd_lr: float = 0.005
    svd_reg: float = 0.02

    # ALS settings
    als_factors: int = 50
    als_iterations: int = 15
    als_regularization: float = 0.01

    # Implicit ALS settings
    implicit_factors: int = 50
    implicit_iterations: int = 15
    implicit_regularization: float = 0.01

    # FAISS settings
    faiss_nlist: int = 100  # Number of clusters for IVF
    faiss_nprobe: int = 10  # Number of clusters to search

    # Hybrid weights
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        "content": 0.3,
        "collaborative": 0.4,
        "implicit": 0.2,
        "popularity": 0.1
    })


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    default_top_k: int = 10
    max_top_k: int = 100


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    test_size: float = 0.2
    k_values: tuple = (5, 10, 20)
    random_state: int = 42


# Global config instances
data_config = DataConfig()
model_config = ModelConfig()
api_config = APIConfig()
eval_config = EvaluationConfig()
