"""
Text processor for NLP-based feature extraction.
"""
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CACHE_DIR, model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles text processing for content-based recommendations.

    Supports TF-IDF vectorization and Sentence-BERT embeddings.

    Attributes:
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF feature matrix
        sbert_model: Sentence-BERT model
        sbert_embeddings: SBERT embeddings matrix
    """

    def __init__(self):
        """Initialize TextProcessor."""
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.sbert_model = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self._anime_ids: Optional[List[int]] = None

    def fit_tfidf(
        self,
        texts: Union[pd.Series, List[str]],
        anime_ids: Optional[List[int]] = None,
        max_features: int = None
    ) -> np.ndarray:
        """
        Fit TF-IDF vectorizer on texts.

        Args:
            texts: List or Series of text documents
            anime_ids: Corresponding anime IDs for mapping
            max_features: Maximum number of features

        Returns:
            TF-IDF matrix
        """
        max_features = max_features or model_config.tfidf_max_features

        logger.info(f"Fitting TF-IDF with max_features={max_features}...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # Convert to list and handle NaN
        if isinstance(texts, pd.Series):
            texts = texts.fillna('').tolist()
        texts = [str(t) if t else '' for t in texts]

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self._anime_ids = list(anime_ids) if anime_ids is not None else None

        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix

    def fit_sbert(
        self,
        texts: Union[pd.Series, List[str]],
        anime_ids: Optional[List[int]] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate Sentence-BERT embeddings.

        Args:
            texts: List or Series of text documents
            anime_ids: Corresponding anime IDs
            batch_size: Batch size for encoding

        Returns:
            SBERT embeddings matrix
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

        logger.info(f"Loading SBERT model: {model_config.sbert_model_name}...")
        self.sbert_model = SentenceTransformer(model_config.sbert_model_name)

        # Convert to list and handle NaN
        if isinstance(texts, pd.Series):
            texts = texts.fillna('').tolist()
        texts = [str(t) if t else '' for t in texts]

        logger.info(f"Encoding {len(texts)} texts with SBERT...")
        self.sbert_embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self._anime_ids = list(anime_ids) if anime_ids is not None else None

        logger.info(f"SBERT embeddings shape: {self.sbert_embeddings.shape}")
        return self.sbert_embeddings

    def get_tfidf_similarity(
        self,
        idx: int,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Get most similar items based on TF-IDF.

        Args:
            idx: Index of the query item
            top_k: Number of similar items to return

        Returns:
            List of (index, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF not fitted. Call fit_tfidf first.")

        # Compute similarity for this item
        item_vector = self.tfidf_matrix[idx:idx+1]
        similarities = cosine_similarity(item_vector, self.tfidf_matrix).flatten()

        # Get top-k (excluding self)
        similar_indices = similarities.argsort()[::-1][1:top_k+1]

        return [(i, similarities[i]) for i in similar_indices]

    def get_sbert_similarity(
        self,
        idx: int,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Get most similar items based on SBERT embeddings.

        Args:
            idx: Index of the query item
            top_k: Number of similar items to return

        Returns:
            List of (index, similarity_score) tuples
        """
        if self.sbert_embeddings is None:
            raise ValueError("SBERT not fitted. Call fit_sbert first.")

        # Compute similarity
        item_vector = self.sbert_embeddings[idx:idx+1]
        similarities = cosine_similarity(item_vector, self.sbert_embeddings).flatten()

        # Get top-k (excluding self)
        similar_indices = similarities.argsort()[::-1][1:top_k+1]

        return [(i, similarities[i]) for i in similar_indices]

    def combine_text_features(
        self,
        anime_df: pd.DataFrame,
        include_genres: bool = True,
        include_synopsis: bool = True,
        include_metadata: bool = True
    ) -> pd.Series:
        """
        Combine multiple text features into single text representation.

        Args:
            anime_df: DataFrame with anime data
            include_genres: Include genre information
            include_synopsis: Include synopsis text
            include_metadata: Include other metadata (type, studio, etc.)

        Returns:
            Series of combined text
        """
        combined = []

        for _, row in anime_df.iterrows():
            parts = []

            if include_genres and 'Genres' in anime_df.columns:
                genres = str(row.get('Genres', ''))
                if genres:
                    # Repeat genres for emphasis
                    parts.append(genres + ' ' + genres)

            if include_synopsis and 'synopsis' in anime_df.columns:
                synopsis = str(row.get('synopsis', ''))
                if synopsis:
                    parts.append(synopsis)

            if include_metadata:
                # Add type
                if 'Type' in anime_df.columns:
                    type_val = str(row.get('Type', ''))
                    if type_val and type_val != 'Unknown':
                        parts.append(type_val)

                # Add source
                if 'Source' in anime_df.columns:
                    source = str(row.get('Source', ''))
                    if source and source != 'Unknown':
                        parts.append(source)

                # Add studios
                if 'Studios' in anime_df.columns:
                    studios = str(row.get('Studios', ''))
                    if studios and studios != 'Unknown':
                        parts.append(studios)

            combined.append(' '.join(parts))

        return pd.Series(combined)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save processor state.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        state = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'sbert_embeddings': self.sbert_embeddings,
            'anime_ids': self._anime_ids
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"TextProcessor saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load processor state.

        Args:
            filepath: Path to saved file
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.tfidf_vectorizer = state['tfidf_vectorizer']
        self.tfidf_matrix = state['tfidf_matrix']
        self.sbert_embeddings = state['sbert_embeddings']
        self._anime_ids = state['anime_ids']

        logger.info(f"TextProcessor loaded from {filepath}")


if __name__ == "__main__":
    # Test text processor
    from data_loader import DataLoader

    loader = DataLoader()
    merged_df = loader.get_merged_anime_data()

    processor = TextProcessor()

    # Combine text features
    combined_text = processor.combine_text_features(merged_df)
    print(f"Combined text samples:\n{combined_text.head()}")

    # Fit TF-IDF
    processor.fit_tfidf(combined_text, anime_ids=merged_df['MAL_ID'].tolist())

    # Get similar anime
    similar = processor.get_tfidf_similarity(0, top_k=5)
    print(f"\nSimilar to '{merged_df.iloc[0]['Name']}':")
    for idx, score in similar:
        print(f"  {merged_df.iloc[idx]['Name']}: {score:.4f}")
