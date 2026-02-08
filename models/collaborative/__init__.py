"""Collaborative filtering module."""
from .item_based_cf import ItemBasedCF
from .matrix_factorization import MatrixFactorization

__all__ = ["ItemBasedCF", "MatrixFactorization"]
