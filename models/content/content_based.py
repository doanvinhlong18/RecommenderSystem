"""
Content-Based Recommender using TF-IDF and Sentence-BERT.

GPU-accelerated version with FAISS GPU support.
"""
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, model_config
from device_config import (
    get_device, is_gpu_available, get_optimal_batch_size,
    get_faiss_gpu_resources, faiss_index_to_gpu, log_gpu_memory
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-based recommendation using TF-IDF and Sentence-BERT.

    GPU-accelerated with FAISS GPU and SBERT GPU support.

    Computes anime-to-anime similarity based on:
    - Genre information
    - Synopsis text
    - Metadata (type, source, studio)

    Attributes:
        anime_df: DataFrame with anime information
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF feature matrix
        sbert_embeddings: SBERT embeddings (optional)
        faiss_index: FAISS index for fast similarity search
        device: Device for computation ("cuda" or "cpu")
    """

    def __init__(self, device: str = None):
        """
        Initialize ContentBasedRecommender.

        Args:
            device: Device for computation ("cuda" or "cpu"). Auto-detected if None.
        """
        self.device = device or get_device()
        self.anime_df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.sbert_model = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.faiss_index_sbert = None

        # GPU resources for FAISS
        self._faiss_gpu_resources = None
        self._use_faiss_gpu = False

        # Mappings
        self._id_to_idx: Dict[int, int] = {}
        self._idx_to_id: Dict[int, int] = {}
        self._name_to_idx: Dict[str, int] = {}

        logger.info(f"ContentBasedRecommender initialized with device: {self.device}")

    def fit(
        self,
        anime_df: pd.DataFrame,
        use_tfidf: bool = True,
        use_sbert: bool = True,
        use_faiss: bool = True
    ) -> "ContentBasedRecommender":
        """
        Fit the content-based model.

        Args:
            anime_df: DataFrame with anime data (must have MAL_ID, Name, Genres, synopsis)
            use_tfidf: Whether to use TF-IDF
            use_sbert: Whether to use Sentence-BERT
            use_faiss: Whether to use FAISS for fast search

        Returns:
            Self for chaining
        """
        self.anime_df = anime_df.reset_index(drop=True)

        # Create mappings
        self._id_to_idx = {
            row['MAL_ID']: idx
            for idx, row in self.anime_df.iterrows()
        }
        self._idx_to_id = {idx: aid for aid, idx in self._id_to_idx.items()}
        self._name_to_idx = {
            row['Name'].lower(): idx
            for idx, row in self.anime_df.iterrows()
        }

        # Add English names to mapping
        if 'English name' in self.anime_df.columns:
            for idx, row in self.anime_df.iterrows():
                eng_name = row.get('English name', '')
                if pd.notna(eng_name) and eng_name:
                    self._name_to_idx[str(eng_name).lower()] = idx

        # Combine text features
        combined_text = self._combine_features()

        # Fit TF-IDF
        if use_tfidf:
            self._fit_tfidf(combined_text)

        # Fit SBERT
        if use_sbert:
            self._fit_sbert(combined_text)

        # Build FAISS index
        if use_faiss:
            self._build_faiss_index()

        logger.info(f"ContentBasedRecommender fitted on {len(self.anime_df)} anime")
        return self

    def _combine_features(self) -> List[str]:
        """
        Combine text features for each anime.

        Returns:
            List of combined text strings
        """
        combined = []

        for _, row in self.anime_df.iterrows():
            parts = []

            # Genres (repeated for emphasis)
            genres = str(row.get('Genres', ''))
            if genres and genres != 'nan':
                parts.extend([genres] * 2)

            # Synopsis
            synopsis = str(row.get('synopsis', ''))
            if synopsis and synopsis != 'nan':
                parts.append(synopsis)

            # Type
            type_val = str(row.get('Type', ''))
            if type_val and type_val not in ['nan', 'Unknown']:
                parts.append(type_val)

            # Source
            source = str(row.get('Source', ''))
            if source and source not in ['nan', 'Unknown']:
                parts.append(source)

            # Studios
            studios = str(row.get('Studios', ''))
            if studios and studios not in ['nan', 'Unknown']:
                parts.append(studios)

            combined.append(' '.join(parts) if parts else '')

        return combined

    def _fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer.

        Args:
            texts: List of text documents
        """
        logger.info("Fitting TF-IDF vectorizer...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=model_config.tfidf_max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def _fit_sbert(self, texts: List[str]) -> None:
        """
        Generate Sentence-BERT embeddings with GPU support.

        Args:
            texts: List of text documents
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed. Skipping SBERT.")
            return

        logger.info(f"Loading SBERT model: {model_config.sbert_model_name}...")
        logger.info(f"SBERT running on {self.device.upper()}")

        # Load SBERT model on specified device
        self.sbert_model = SentenceTransformer(
            model_config.sbert_model_name,
            device=self.device
        )

        # Get optimal batch size based on device
        batch_size = get_optimal_batch_size("sbert")
        logger.info(f"Encoding {len(texts)} texts with SBERT (batch_size={batch_size})...")

        # Log GPU memory before encoding
        log_gpu_memory("Before SBERT encoding: ")

        self.sbert_embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Log GPU memory after encoding
        log_gpu_memory("After SBERT encoding: ")

        logger.info(f"SBERT embeddings shape: {self.sbert_embeddings.shape}")

    def _build_faiss_index(self) -> None:
        """Build FAISS index for fast similarity search with GPU support."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed. Using sklearn cosine similarity.")
            return

        # Try to get FAISS GPU resources
        self._faiss_gpu_resources, self._use_faiss_gpu = get_faiss_gpu_resources()
        if self._use_faiss_gpu:
            logger.info("FAISS GPU enabled")
        else:
            logger.info("FAISS running on CPU")

        # Build index for SBERT embeddings (preferred)
        if self.sbert_embeddings is not None:
            logger.info("Building FAISS index for SBERT embeddings...")
            d = self.sbert_embeddings.shape[1]

            # Use IVF for large datasets
            if len(self.sbert_embeddings) > 10000:
                nlist = min(model_config.faiss_nlist, len(self.sbert_embeddings) // 100)
                quantizer = faiss.IndexFlatIP(d)
                index_sbert = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                index_sbert.train(self.sbert_embeddings.astype(np.float32))
                index_sbert.add(self.sbert_embeddings.astype(np.float32))
                index_sbert.nprobe = model_config.faiss_nprobe
            else:
                index_sbert = faiss.IndexFlatIP(d)
                index_sbert.add(self.sbert_embeddings.astype(np.float32))

            # Move to GPU if available
            if self._use_faiss_gpu:
                self.faiss_index_sbert = faiss_index_to_gpu(index_sbert, self._faiss_gpu_resources)
            else:
                self.faiss_index_sbert = index_sbert

            logger.info(f"FAISS SBERT index built with {self.faiss_index_sbert.ntotal} vectors")

        # Build index for TF-IDF (dense)
        if self.tfidf_matrix is not None:
            logger.info("Building FAISS index for TF-IDF...")
            tfidf_dense = self.tfidf_matrix.toarray().astype(np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(tfidf_dense, axis=1, keepdims=True)
            norms[norms == 0] = 1
            tfidf_dense = tfidf_dense / norms

            d = tfidf_dense.shape[1]
            index_tfidf = faiss.IndexFlatIP(d)
            index_tfidf.add(tfidf_dense)

            # Move to GPU if available
            if self._use_faiss_gpu:
                self.faiss_index = faiss_index_to_gpu(index_tfidf, self._faiss_gpu_resources)
            else:
                self.faiss_index = index_tfidf

            logger.info(f"FAISS TF-IDF index built with {self.faiss_index.ntotal} vectors")

    def get_similar_anime(
        self,
        anime_identifier: Union[int, str],
        top_k: int = 10,
        method: str = "hybrid"
    ) -> List[Dict]:
        """
        Get similar anime recommendations.

        Args:
            anime_identifier: Anime ID (int) or name (str)
            top_k: Number of recommendations
            method: "tfidf", "sbert", or "hybrid"

        Returns:
            List of recommendation dictionaries
        """
        # Find anime index
        idx = self._get_anime_idx(anime_identifier)
        if idx is None:
            logger.warning(f"Anime not found: {anime_identifier}")
            return []

        # Get similarities based on method
        if method == "tfidf":
            similar_indices, scores = self._get_tfidf_similar(idx, top_k + 1)
        elif method == "sbert":
            similar_indices, scores = self._get_sbert_similar(idx, top_k + 1)
        else:  # hybrid
            similar_indices, scores = self._get_hybrid_similar(idx, top_k + 1)

        # Build results (exclude query anime)
        results = []
        for sim_idx, score in zip(similar_indices, scores):
            if sim_idx == idx:
                continue
            if len(results) >= top_k:
                break

            anime_row = self.anime_df.iloc[sim_idx]
            results.append({
                'mal_id': int(anime_row['MAL_ID']),
                'name': anime_row['Name'],
                'english_name': anime_row.get('English name', anime_row['Name']),
                'genres': anime_row.get('Genres', ''),
                'score': float(anime_row.get('Score', 0)),
                'similarity': float(score),
                'type': anime_row.get('Type', 'Unknown')
            })

        return results

    def _get_anime_idx(self, identifier: Union[int, str]) -> Optional[int]:
        """
        Get anime index from ID or name.

        Args:
            identifier: Anime ID or name

        Returns:
            Index or None if not found
        """
        if isinstance(identifier, int):
            return self._id_to_idx.get(identifier)

        # Try exact match
        name_lower = str(identifier).lower()
        if name_lower in self._name_to_idx:
            return self._name_to_idx[name_lower]

        # Try partial match
        for name, idx in self._name_to_idx.items():
            if name_lower in name or name in name_lower:
                return idx

        return None

    def _get_tfidf_similar(
        self,
        idx: int,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get similar items using TF-IDF."""
        if self.faiss_index is not None:
            # Use FAISS
            query = self.tfidf_matrix[idx].toarray().astype(np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
            scores, indices = self.faiss_index.search(query, top_k)
            return indices[0], scores[0]
        else:
            # Use sklearn
            query = self.tfidf_matrix[idx:idx+1]
            similarities = cosine_similarity(query, self.tfidf_matrix).flatten()
            indices = similarities.argsort()[::-1][:top_k]
            return indices, similarities[indices]

    def _get_sbert_similar(
        self,
        idx: int,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get similar items using SBERT."""
        if self.sbert_embeddings is None:
            return self._get_tfidf_similar(idx, top_k)

        if self.faiss_index_sbert is not None:
            # Use FAISS
            query = self.sbert_embeddings[idx:idx+1].astype(np.float32)
            scores, indices = self.faiss_index_sbert.search(query, top_k)
            return indices[0], scores[0]
        else:
            # Use sklearn
            query = self.sbert_embeddings[idx:idx+1]
            similarities = cosine_similarity(query, self.sbert_embeddings).flatten()
            indices = similarities.argsort()[::-1][:top_k]
            return indices, similarities[indices]

    def _get_hybrid_similar(
        self,
        idx: int,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get similar items using hybrid of TF-IDF and SBERT."""
        # Get both similarities
        tfidf_indices, tfidf_scores = self._get_tfidf_similar(idx, top_k * 2)

        if self.sbert_embeddings is not None:
            sbert_indices, sbert_scores = self._get_sbert_similar(idx, top_k * 2)

            # Normalize scores
            tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
            sbert_scores = (sbert_scores - sbert_scores.min()) / (sbert_scores.max() - sbert_scores.min() + 1e-8)

            # Combine scores
            combined_scores = {}
            for i, score in zip(tfidf_indices, tfidf_scores):
                combined_scores[i] = combined_scores.get(i, 0) + 0.4 * score
            for i, score in zip(sbert_indices, sbert_scores):
                combined_scores[i] = combined_scores.get(i, 0) + 0.6 * score

            # Sort by combined score
            sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            indices = np.array([i for i, _ in sorted_items[:top_k]])
            scores = np.array([s for _, s in sorted_items[:top_k]])

            return indices, scores
        else:
            return tfidf_indices[:top_k], tfidf_scores[:top_k]

    def search_anime(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for anime by name.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching anime
        """
        query_lower = query.lower()
        results = []

        for idx, row in self.anime_df.iterrows():
            name = str(row['Name']).lower()
            eng_name = str(row.get('English name', '')).lower()

            if query_lower in name or query_lower in eng_name:
                results.append({
                    'mal_id': int(row['MAL_ID']),
                    'name': row['Name'],
                    'english_name': row.get('English name', row['Name']),
                    'genres': row.get('Genres', ''),
                    'score': float(row.get('Score', 0)),
                    'type': row.get('Type', 'Unknown')
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'anime_df': self.anime_df,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'sbert_embeddings': self.sbert_embeddings,
            'id_to_idx': self._id_to_idx,
            'idx_to_id': self._idx_to_id,
            'name_to_idx': self._name_to_idx
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"ContentBasedRecommender saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "ContentBasedRecommender":
        """
        Load model from file.

        Args:
            filepath: Path to saved file

        Returns:
            Self for chaining
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.anime_df = state['anime_df']
        self.tfidf_vectorizer = state['tfidf_vectorizer']
        self.tfidf_matrix = state['tfidf_matrix']
        self.sbert_embeddings = state['sbert_embeddings']
        self._id_to_idx = state['id_to_idx']
        self._idx_to_id = state['idx_to_id']
        self._name_to_idx = state['name_to_idx']

        # Rebuild FAISS index
        self._build_faiss_index()

        logger.info(f"ContentBasedRecommender loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Test content-based recommender
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from preprocessing import DataLoader

    loader = DataLoader()
    merged_df = loader.get_merged_anime_data()

    recommender = ContentBasedRecommender()
    recommender.fit(merged_df, use_sbert=False)  # Skip SBERT for quick test

    # Test recommendations
    test_anime = "Naruto"
    print(f"\nRecommendations similar to '{test_anime}':")
    recommendations = recommender.get_similar_anime(test_anime, top_k=5, method="tfidf")
    for rec in recommendations:
        print(f"  {rec['name']} (Score: {rec['score']}, Similarity: {rec['similarity']:.4f})")

    # Test search
    print(f"\nSearch results for 'death':")
    search_results = recommender.search_anime("death", top_k=5)
    for result in search_results:
        print(f"  {result['name']} (Score: {result['score']})")
