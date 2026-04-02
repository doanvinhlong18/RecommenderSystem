"""Hybrid recommendation engine module."""

# LearnedHybridEngine thay thế hoàn toàn HybridEngine.
# Import với alias HybridEngine để toàn bộ code khác (routes.py, train.py, ...)
# không cần sửa gì — vẫn dùng "from models.hybrid import HybridEngine".

from .learned_hybrid import LearnedHybridEngine as HybridEngine
from .learned_hybrid import LearnedHybridEngine, train_learned_hybrid

__all__ = ["HybridEngine", "LearnedHybridEngine", "train_learned_hybrid"]
