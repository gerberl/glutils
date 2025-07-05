#!/usr/bin/env python3
"""Simple test script for ml submodule."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_ml_import():
    """Test that ml submodule imports correctly."""
    try:
        from glutils.ml import (
            min_max_norm,
            zscore_norm,
            pairwise_X_cols_combinations,
            pairwise_X_cols_chain,
            pairwise_X_cols_1st_vs_rest,
            get_names_func_eval_metrics,
            get_names_of_eval_scores,
            evaluate_regr_metrics,
            cross_validate_it,
            pivot_cross_validated_stats,
            get_CV_train_test_scores,
            plot_true_vs_predicted,
            plot_rf_feat_imp_barh,
            format_cv_results,
            format_optuna_hs_results
        )
        print("✓ All ml imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of some functions."""
    try:
        import numpy as np
        from glutils.ml import min_max_norm, zscore_norm, pairwise_X_cols_combinations
        
        # Test min_max_norm
        test_data = np.array([[1, 2], [3, 4], [5, 6]])
        normalized = min_max_norm(test_data)
        print(f"✓ min_max_norm works: shape {normalized.shape}")
        
        # Test zscore_norm
        z_normalized = zscore_norm(test_data)
        print(f"✓ zscore_norm works: shape {z_normalized.shape}")
        
        # Test pairwise_X_cols_combinations
        result = list(pairwise_X_cols_combinations(test_data))
        print(f"✓ pairwise_X_cols_combinations works: {len(result)} combinations")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing ml submodule...")
    
    success = True
    success &= test_ml_import()
    success &= test_basic_functionality()
    
    if success:
        print("\n✓ All ml tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some ml tests failed!")
        sys.exit(1)