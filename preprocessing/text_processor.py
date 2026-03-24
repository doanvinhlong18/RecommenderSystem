"""
Text processor for NLP-based feature extraction.
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import CACHE_DIR, model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SBERT all-MiniLM-L6-v2 hard limit = 512 tokens ≈ 380 words ≈ 1800 chars
# Dùng 1500 chars để an toàn — tránh bị tokenizer cắt silently giữa chừng
_MAX_TEXT_CHARS = 1500


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
        # self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        # self.tfidf_matrix: Optional[np.ndarray] = None
        self.sbert_model = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self._anime_ids: Optional[List[int]] = None

    def fit_sbert(self, texts, anime_ids=None, batch_size=128):
        from sentence_transformers import SentenceTransformer

        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(model_config.sbert_model_name)

        if isinstance(texts, pd.Series):
            texts = texts.fillna("").tolist()

        # FIX: Truncate trước khi encode — SBERT hard limit 512 tokens
        # synopsis dài bị tokenizer cắt silently, mất phần cuối mà không báo lỗi
        texts = [str(t)[:_MAX_TEXT_CHARS] for t in texts]

        embeddings = self.sbert_model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        # L2 normalize
        embeddings = normalize(embeddings)

        self.sbert_embeddings = embeddings
        self._anime_ids = list(anime_ids) if anime_ids is not None else None

        return embeddings

    def get_sbert_similarity(self, idx: int, top_k: int = 10):
        if self.sbert_embeddings is None:
            raise ValueError("SBERT not fitted.")

        item_vector = self.sbert_embeddings[idx]

        # Dot product vì đã normalize
        similarities = self.sbert_embeddings @ item_vector

        similarities[idx] = -1  # loại self

        top_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [(i, float(similarities[i])) for i in top_indices]

    def combine_text_features(self, anime_df: pd.DataFrame) -> pd.Series:
        """
        Combine Name + Synopsis for SBERT embedding.
        Truncate synopsis trước (~1400 chars) để title prefix không bị đẩy ra ngoài token limit.
        """
        name = anime_df["Name"].fillna("")
        # FIX: truncate synopsis riêng trước khi combine
        synopsis = anime_df["synopsis"].fillna("").str.slice(0, 1400)

        combined = "Title: " + name + ". Synopsis: " + synopsis
        return combined

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save processor state.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        state = {
            "sbert_embeddings": self.sbert_embeddings,
            "anime_ids": self._anime_ids,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"TextProcessor saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load processor state.

        Args:
            filepath: Path to saved file
        """
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # self.tfidf_vectorizer = state['tfidf_vectorizer']
        # self.tfidf_matrix = state['tfidf_matrix']
        self.sbert_embeddings = state["sbert_embeddings"]
        self._anime_ids = state["anime_ids"]

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
