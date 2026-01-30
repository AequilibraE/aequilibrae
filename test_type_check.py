#!/usr/bin/env python
"""
Simple test to verify the type check modification in path_results.py
Tests that both Python int and numpy integer types are accepted,
while non-integer types are rejected.
"""

import numpy as np
import sys

def test_type_check():
    """Test the type check logic"""
    print("Testing type check modification for destination parameter...")
    print("-" * 70)
    
    test_cases = [
        ("Python int", 42, True),
        ("Python int (zero)", 0, True),
        ("Python int (negative)", -5, True),
        ("numpy.int32", np.int32(42), True),
        ("numpy.int64", np.int64(42), True),
        ("numpy.int16", np.int16(42), True),
        ("numpy.uint32", np.uint32(42), True),
        ("numpy.uint64", np.uint64(42), True),
        ("float", 42.5, False),
        ("string", "42", False),
        ("list", [42], False),
        ("dict", {"value": 42}, False),
        ("None", None, False),
    ]
    
    all_passed = True
    
    for test_name, value, should_pass in test_cases:
        try:
            # Simulate the type check from path_results.py
            if not isinstance(value, (int, np.integer)):
                raise TypeError("destination needs to be an integer")
            
            result = "✓ ACCEPTED"
            if not should_pass:
                print(f"✗ FAILED: {test_name:30} - Expected rejection but was accepted")
                all_passed = False
            else:
                print(f"✓ PASSED: {test_name:30} - {result}")
        except TypeError as e:
            result = "✗ REJECTED"
            if should_pass:
                print(f"✗ FAILED: {test_name:30} - Expected acceptance but was rejected")
                all_passed = False
            else:
                print(f"✓ PASSED: {test_name:30} - {result}")
    
    print("-" * 70)
    if all_passed:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = test_type_check()
    sys.exit(exit_code)
