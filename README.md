# Hybrid Anime Recommendation System

A production-ready hybrid recommendation system for anime using multiple techniques:
- **Content-Based Filtering** (TF-IDF + Sentence-BERT)
- **Collaborative Filtering** (Item-Based CF + Matrix Factorization/SVD)
- **Implicit Feedback** (ALS)
- **Popularity-Based** (Top Rated, Most Watched, Trending)

## 🚀 GPU Acceleration Support

This system supports GPU acceleration for faster training:
- **SBERT** embeddings on CUDA
- **FAISS** GPU for similarity search
- **PyTorch** based Matrix Factorization
- **Implicit ALS** GPU implementation

## 📁 Project Structure

```
RecommenderSystem/
├── config.py                 # Configuration settings
├── device_config.py          # GPU/CPU device management
├── train.py                  # Training script
├── check_gpu.py              # GPU setup verification
├── run_server.py             # API server runner
├── requirements.txt          # Dependencies
├── requirements-gpu.txt      # GPU-specific dependencies
│
├── preprocessing/            # Data preprocessing
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading
│   ├── text_processor.py     # Text/NLP processing
│   └── matrix_builder.py     # Sparse matrix construction
│
├── models/                   # Recommendation models
│   ├── __init__.py
│   ├── content/              # Content-based filtering
│   │   ├── __init__.py
│   │   └── content_based.py
│   ├── collaborative/        # Collaborative filtering
│   │   ├── __init__.py
│   │   ├── item_based_cf.py
│   │   └── matrix_factorization.py
│   ├── implicit/             # Implicit feedback
│   │   ├── __init__.py
│   │   └── als_implicit.py
│   ├── popularity/           # Popularity-based
│   │   ├── __init__.py
│   │   └── popularity_model.py
│   └── hybrid/               # Hybrid engine
│       ├── __init__.py
│       └── hybrid_engine.py
│
├── evaluation/               # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py
│
├── api/                      # REST API
│   ├── __init__.py
│   └── routes.py
│
├── static/                   # Web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/                # Jupyter notebooks
│   └── demo.ipynb
│
├── saved_models/             # Trained models
└── cache/                    # Cached data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Basic installation (CPU)
pip install -r requirements.txt

# GPU installation (CUDA 11.8)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify GPU setup
python check_gpu.py
```

### 2. Train Models

```bash
# Full training with GPU (auto-detected)
python train.py

# Force CPU mode
python train.py --force-cpu

# Use PyTorch for Matrix Factorization (GPU accelerated)
python train.py --torch-svd

# Quick training (skip SBERT for faster training)
python train.py --skip-sbert

# Training with smaller sample
python train.py --sample-size 1000000 --skip-sbert
```

### 3. Run API Server

```bash
python api_server.py

# With auto-reload for development
python api_server.py --reload
```

### 4. Access API

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend/anime/{name}` | GET | Get similar anime recommendations |
| `/recommend/anime/id/{id}` | GET | Get similar anime by MAL ID |
| `/recommend/user/{user_id}` | GET | Get personalized recommendations |
| `/popular` | GET | Get popular anime |
| `/search?q={query}` | GET | Search anime by name |
| `/weights` | GET/PUT | Get/Update hybrid weights |
| `/explain/{user_id}/{anime_id}` | GET | Get recommendation explanation |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Data settings
data_config.rating_sample_size = 5_000_000  # Sample size for ratings
data_config.min_user_ratings = 5            # Min ratings per user

# Model settings
model_config.tfidf_max_features = 5000      # TF-IDF vocabulary size
model_config.svd_factors = 100              # SVD latent factors
model_config.sbert_model_name = "all-MiniLM-L6-v2"  # SBERT model

# Hybrid weights
model_config.hybrid_weights = {
    "content": 0.3,
    "collaborative": 0.4,
    "implicit": 0.2,
    "popularity": 0.1
}
```

## 📈 Evaluation

```python
from evaluation import RecommenderMetrics

# Evaluate recommendations
results = RecommenderMetrics.evaluate_recommendations(
    recommendations,  # Dict[user_id, List[anime_id]]
    relevant_items,   # Dict[user_id, Set[anime_id]]
    k_values=[5, 10, 20]
)

print(results)
# {'Precision@5': 0.15, 'Recall@5': 0.08, 'NDCG@5': 0.12, ...}
```

## 🎯 Recommendation Strategies

### For New Users (Cold Start)
- Uses **Content-Based** + **Popularity** recommendations
- Can specify preferred genres

### For Existing Users
- Full **Hybrid** approach combining all models
- Weighted combination: `Content + Collaborative + Implicit + Popularity`

## 📦 Dataset

Using [MyAnimeList Dataset 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020):

| File | Rows | Description |
|------|------|-------------|
| anime.csv | 17,562 | Anime metadata |
| anime_with_synopsis.csv | 16,214 | Anime synopses |
| rating_complete.csv | 57M | User ratings |
| animelist.csv | 109M | User watch lists |

## 🛠️ Technical Stack

- **Python 3.10+**
- **Pandas** - Data manipulation
- **Scikit-learn** - TF-IDF, cosine similarity
- **Sentence-Transformers** - SBERT embeddings
- **SciPy** - Sparse matrices
- **FAISS** - Fast similarity search
- **FastAPI** - REST API
- **Uvicorn** - ASGI server

## 📝 License

MIT License
