"""
Test Surprise with explicit import order and error handling
"""
import sys
print("Python:", sys.version)
print("Python executable:", sys.executable)

print("\n[1/5] Importing NumPy...")
import numpy as np
print(f"✓ NumPy {np.__version__}")

print("\n[2/5] Importing SciPy...")
import scipy
print(f"✓ SciPy {scipy.__version__}")

print("\n[3/5] Importing pandas...")
import pandas as pd
print(f"✓ pandas {pd.__version__}")

print("\n[4/5] Importing joblib...")
import joblib
print(f"✓ joblib {joblib.__version__}")

print("\n[5/5] Importing Surprise...")
try:
    import surprise
    print(f"✓ Surprise {surprise.__version__}")
    print("\n[6/6] Testing Surprise Dataset...")
    from surprise import Dataset
    data = Dataset.load_builtin('ml-100k')
    print("✓ Built-in dataset loads successfully!")
    print("\n✅ ALL TESTS PASSED - Surprise is working!")
except Exception as e:
    print(f"✗ Surprise import/test failed:")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
