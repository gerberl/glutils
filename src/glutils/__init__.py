"""
glutils: Consolidated ML, Information Theory, and Functional Utilities

A collection of utilities for machine learning, information theory, and 
functional programming patterns.

This package provides:
- ml: data normalization, evaluation metrics, cross-validation
- info_theory: entropy, KL divergence, mutual information
- funcitertools: pairwise operations, dictionary utilities
"""

__version__ = "0.1.0"
__author__ = "Luciano Gerber"
__email__ = "L.Gerber@mmu.ac.uk"

# Import submodules
from . import ml
from . import info_theory
from . import funcitertools

__all__ = [
    "ml",
    "info_theory", 
    "funcitertools",
]