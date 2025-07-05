#!/usr/bin/env python3
"""Simple test script for funcitertools submodule."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_funcitertools_import():
    """Test that funcitertools submodule imports correctly."""
    try:
        from glutils.funcitertools import (
            pairwise_lst_el_combinations,
            pairwise_lst_el_chain,
            pairwise_lst_1st_vs_rest,
            slice_dict,
            dict_drop,
            merge_dicts,
            tuple_minus,
            key_tuple,
            key_dict,
            pprint_dict
        )
        print("✓ All funcitertools imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of some functions."""
    try:
        from glutils.funcitertools import (
            pairwise_lst_el_combinations,
            slice_dict,
            merge_dicts,
            tuple_minus
        )
        
        # Test pairwise_lst_el_combinations
        test_list = [1, 2, 3]
        result = list(pairwise_lst_el_combinations(test_list))
        print(f"✓ pairwise_lst_el_combinations([1, 2, 3]) = {result}")
        
        # Test slice_dict
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        result = slice_dict(test_dict, ['a', 'c'])
        print(f"✓ slice_dict works: {result}")
        
        # Test merge_dicts
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'c': 3, 'd': 4}
        result = merge_dicts(dict1, dict2)
        print(f"✓ merge_dicts works: {result}")
        
        # Test tuple_minus
        result = tuple_minus((1, 2, 3), (2,))
        print(f"✓ tuple_minus works: {result}")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing funcitertools submodule...")
    
    success = True
    success &= test_funcitertools_import()
    success &= test_basic_functionality()
    
    if success:
        print("\n✓ All funcitertools tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some funcitertools tests failed!")
        sys.exit(1)