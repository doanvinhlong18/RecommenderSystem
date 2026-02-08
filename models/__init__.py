"""
Models module for recommendation algorithms.
"""
from .content.content_based import ContentBasedRecommender
from .collaborative.item_based_cf import ItemBasedCF
from .collaborative.matrix_factorization import MatrixFactorization
from .implicit.als_implicit import ALSImplicit
from .popularity.popularity_model import PopularityModel
from .hybrid.hybrid_engine import HybridEngine

__all__ = [
    "ContentBasedRecommender",
    "ItemBasedCF",
    "MatrixFactorization",
    "ALSImplicit",
    "PopularityModel",
    "HybridEngine"
]
