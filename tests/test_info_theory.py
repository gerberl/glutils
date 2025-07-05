#!/usr/bin/env python3
"""Simple test script for info_theory submodule."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_info_theory_import():
    """Test that info_theory submodule imports correctly."""
    try:
        from glutils.info_theory import (
            entropy_cp,
            KL_div_ccp,
            JS_div_ccp,
            mutual_information_ccp,
            apply_unary_metric_cP,
            apply_binary_metric_cP,
            apply_unary_metric_cX,
            apply_binary_metric_cX,
            apply_unary_metric_cd,
            apply_binary_metric_ccd,
            entropy_cX,
            entropy_cd,
            KL_div_ccX,
            KL_div_ccd,
            JS_div_ccX,
            JS_div_ccd
        )
        print("✓ All info_theory imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of some functions."""
    try:
        import numpy as np
        from scipy.stats import gaussian_kde
        from glutils.info_theory import entropy_cp, KL_div_ccp
        
        # Create sample data and KDE for testing
        sample_data = np.array([1, 2, 3, 4, 5, 2, 3, 4, 3])
        kde = gaussian_kde(sample_data)
        
        # Test entropy_cp with KDE
        entropy_result = entropy_cp(kde, min_x=0, max_x=6)
        print(f"✓ entropy_cp with KDE = {entropy_result:.4f}")
        
        # Test KL divergence with KDEs
        sample_data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 3.5])
        kde2 = gaussian_kde(sample_data2)
        kl_result = KL_div_ccp(kde, kde2, min_x=0, max_x=6)
        print(f"✓ KL_div_ccp works: {kl_result:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing info_theory submodule...")
    
    success = True
    success &= test_info_theory_import()
    success &= test_basic_functionality()
    
    if success:
        print("\n✓ All info_theory tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some info_theory tests failed!")
        sys.exit(1)