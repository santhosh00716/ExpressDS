"""
Validation Layer - GIGO (Garbage In, Garbage Out) protection
"""
from .validator import validate_dataset, ValidationResult

__all__ = ["validate_dataset", "ValidationResult"]
