"""
FAISS Approximate Nearest Neighbor Index.

Wrapper dùng chung cho cả ContentBasedRecommender và ItemBasedCF.

Chiến lược chọn index:
  - n_vectors < FLAT_THRESHOLD  → IndexFlatIP   (exact, nhanh với tập nhỏ)
  - n_vectors >= FLAT_THRESHOLD → IndexIVFFlat  (ANN, dùng config nlist/nprobe)

Config nlist / nprobe được lấy từ model_config (config.py).
GPU tự động được dùng nếu faiss-gpu được cài (device_config.py).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Dưới ngưỡng này dùng exact search, không cần IVF
FLAT_THRESHOLD = 10_000


def _import_faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        return None


class FaissANNIndex:
    """
    FAISS ANN index dùng chung.

    Parameters
    ----------
    nlist : int
        Số cluster cho IndexIVFFlat (lấy từ config.faiss_nlist).
    nprobe : int
        Số cluster tìm lúc query (lấy từ config.faiss_nprobe).
        Tăng nprobe → độ chính xác cao hơn nhưng chậm hơn.
    use_gpu : bool
        Thử move index lên GPU nếu True và faiss-gpu được cài.
    """

    def __init__(self, nlist: int = 100, nprobe: int = 10, use_gpu: bool = False):
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu

        self._index = None  # faiss index object
        self._is_ivf = False  # True nếu đang dùng IVFFlat
        self._d: int = 0  # embedding dimension
        self._n: int = 0  # số vector đã add
        self._available: bool = False  # False nếu faiss không được cài

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray) -> "FaissANNIndex":
        """
        Build index từ ma trận vectors (n, d).

        Vectors SẼ ĐƯỢC normalize (L2) bên trong — dot product == cosine sim.
        Không cần normalize trước khi gọi.

        Parameters
        ----------
        vectors : np.ndarray, shape (n, d)
        """
        faiss = _import_faiss()
        if faiss is None:
            logger.warning("faiss chưa được cài. Cài bằng: pip install faiss-cpu")
            self._available = False
            return self

        self._available = True
        vecs = normalize(np.asarray(vectors, dtype=np.float32))
        n, d = vecs.shape
        self._d = d
        self._n = n

        if n < FLAT_THRESHOLD:
            # Exact search — IndexFlatIP
            idx = faiss.IndexFlatIP(d)
            self._is_ivf = False
            logger.info(f"FaissANNIndex: dùng IndexFlatIP (n={n} < {FLAT_THRESHOLD})")
        else:
            # ANN — IndexIVFFlat
            # nlist khuyến nghị ≈ sqrt(n), nhưng không vượt n/39 (heuristic FAISS)
            nlist = min(self.nlist, max(1, n // 39))
            quantizer = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(
                f"FaissANNIndex: dùng IndexIVFFlat (n={n}, nlist={nlist}, nprobe={self.nprobe})"
            )
            idx.train(vecs)
            idx.nprobe = self.nprobe
            self._is_ivf = True

        idx.add(vecs)

        # Thử move lên GPU
        if self.use_gpu:
            try:
                from device_config import faiss_index_to_gpu

                idx, moved = faiss_index_to_gpu(idx)
                if moved:
                    logger.info("FaissANNIndex: đã move lên GPU")
            except Exception as e:
                logger.debug(f"Không move được lên GPU: {e}")

        self._index = idx
        logger.info(f"FaissANNIndex built: {n} vectors, dim={d}")
        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        exclude_ids: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tìm top_k vectors gần nhất với query.

        Parameters
        ----------
        query : np.ndarray, shape (d,) hoặc (1, d)
        top_k : int
        exclude_ids : list of int
            Các index cần loại khỏi kết quả (VD: chính query item).

        Returns
        -------
        scores : np.ndarray, shape (top_k,)  — cosine similarity [0, 1]
        indices : np.ndarray, shape (top_k,) — index trong index
        """
        if not self._available or self._index is None:
            return np.array([]), np.array([])

        q = normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))

        # Lấy thêm để bù cho exclude_ids
        fetch_k = top_k + (len(exclude_ids) if exclude_ids else 0) + 1
        fetch_k = min(fetch_k, self._n)

        scores, indices = self._index.search(q, fetch_k)
        scores = scores[0]
        indices = indices[0]

        # Lọc invalid (-1) và exclude_ids
        exclude_set = set(exclude_ids or [])
        mask = (indices >= 0) & (~np.isin(indices, list(exclude_set)))
        scores = scores[mask][:top_k]
        indices = indices[mask][:top_k]

        # Clip về [0, 1] để tránh floating point noise
        scores = np.clip(scores, 0.0, 1.0)
        return scores, indices

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Lưu index ra file."""
        faiss = _import_faiss()
        if faiss is None or not self._available or self._index is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Nếu index đang ở GPU, chuyển về CPU trước khi lưu
        try:
            cpu_idx = faiss.index_gpu_to_cpu(self._index)
        except Exception:
            cpu_idx = self._index
        faiss.write_index(cpu_idx, str(path) + ".faiss")
        meta = {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "d": self._d,
            "n": self._n,
            "is_ivf": self._is_ivf,
        }
        with open(str(path) + ".meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"FaissANNIndex saved to {path}")

    def load(self, path) -> "FaissANNIndex":
        """Load index từ file."""
        faiss = _import_faiss()
        if faiss is None:
            self._available = False
            return self
        path = Path(path)
        faiss_path = str(path) + ".faiss"
        meta_path = str(path) + ".meta.pkl"
        if not Path(faiss_path).exists():
            logger.warning(f"Không tìm thấy FAISS index tại {faiss_path}")
            return self
        self._index = faiss.read_index(faiss_path)
        if self._is_ivf:
            self._index.nprobe = self.nprobe
        if Path(meta_path).exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._d = meta.get("d", 0)
            self._n = meta.get("n", 0)
            self._is_ivf = meta.get("is_ivf", False)
        self._available = True
        logger.info(f"FaissANNIndex loaded from {path}")
        return self

    @property
    def is_available(self) -> bool:
        return self._available and self._index is not None

    def __repr__(self):
        kind = "IVFFlat" if self._is_ivf else "FlatIP"
        return f"FaissANNIndex(type={kind}, n={self._n}, d={self._d}, available={self._available})"
