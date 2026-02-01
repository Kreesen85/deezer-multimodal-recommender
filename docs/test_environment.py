#!/usr/bin/env python
"""
Test script to verify all required packages are installed and working.
Run this to confirm your Python environment is set up correctly.
"""

import sys

def test_imports():
    """Test importing all required packages."""
    print("=" * 60)
    print("Testing Python Environment")
    print("=" * 60)
    print(f"\nPython Version: {sys.version}\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Core packages
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        ("sklearn", "Scikit-learn"),
        ("scipy", "SciPy"),
        ("jupyter", "Jupyter"),
        ("IPython", "IPython"),
        ("tqdm", "tqdm"),
    ]
    
    for package_name, display_name in packages:
        try:
            module = __import__(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {display_name:20s} {version}")
            tests_passed += 1
        except ImportError as e:
            print(f"❌ {display_name:20s} NOT INSTALLED")
            tests_failed += 1
    
    # Test scikit-surprise (optional)
    try:
        from surprise import SVD
        print(f"✅ {'scikit-surprise':20s} (installed)")
        tests_passed += 1
    except ImportError:
        print(f"⚠️  {'scikit-surprise':20s} NOT INSTALLED (optional - use scikit-learn instead)")
    
    print("\n" + "=" * 60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 All required packages are installed and working!")
        print("✅ You're ready to run the project!")
    else:
        print("\n⚠️  Some packages are missing. Please install them:")
        print("   pip install -r requirements.txt")
    
    print("=" * 60)
    
    return tests_failed == 0

def test_project_imports():
    """Test importing project modules."""
    print("\n" + "=" * 60)
    print("Testing Project Modules")
    print("=" * 60 + "\n")
    
    try:
        from src.data.preprocessing import (
            add_temporal_features,
            add_release_features,
            add_duration_features,
        )
        print("✅ Preprocessing module imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import preprocessing module: {e}")
        return False

if __name__ == "__main__":
    # Test package imports
    packages_ok = test_imports()
    
    # Test project imports
    project_ok = test_project_imports()
    
    # Exit code
    if packages_ok and project_ok:
        print("\n✅ Environment test completed successfully!\n")
        sys.exit(0)
    else:
        print("\n❌ Environment test failed. Please check the errors above.\n")
        sys.exit(1)
