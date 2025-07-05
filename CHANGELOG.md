# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-05

### Added
- Initial package structure with modern Python packaging (src layout)
- Consolidated three separate utility packages into glutils:
  - `ml` submodule: Machine learning utilities from LG_MLUtils
    - Data normalization functions (min_max_norm, zscore_norm)
    - Pairwise column operations for feature analysis
    - Evaluation metrics and cross-validation utilities
    - Plotting utilities for model evaluation
    - Hyperparameter search result formatting
  - `info_theory` submodule: Information theory utilities from LG_InfoTUtils
    - Entropy calculation using KDE estimators
    - KL divergence and JS divergence computation
    - Mutual information estimation
    - Vectorized metrics for multiple distributions
  - `funcitertools` submodule: Functional/itertools utilities from disfunctools
    - Pairwise list operations (combinations, chain, first vs rest)
    - Dictionary utilities (slice, drop, merge)
    - Utility functions (tuple operations, key functions, pretty printing)
- Clean API design with submodule imports in main package
- Consistent `__init__.py` files across all submodules with proper `__all__` declarations
- Modern packaging configuration with `pyproject.toml`
- Standard `.gitignore` for Python projects
- Basic test structure with placeholder files for each submodule
- Comprehensive documentation in `CLAUDE.md` for future development

### Technical Details
- Package structure follows modern Python best practices with src layout
- Dependencies: numpy, scipy, scikit-learn, matplotlib, pandas
- Python 3.8+ compatibility
- Minimal configuration focusing on stable package foundation
- Source code preserved from original packages with refactored imports