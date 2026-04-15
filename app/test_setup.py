"""
Test script to verify Flask app configuration and dependencies
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    
    required_packages = {
        'flask': 'Flask',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'joblib': 'Joblib',
        'werkzeug': 'Werkzeug',
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not installed)")
            failed.append(package)
    
    return len(failed) == 0


def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    
    directories = {
        'templates': 'HTML templates',
        'static': 'Static files',
        'uploads': 'Upload directory (will be created)',
    }
    
    for directory, description in directories.items():
        path = os.path.join(os.path.dirname(__file__), directory)
        if os.path.exists(path):
            print(f"✓ {description}: {path}")
        elif directory == 'uploads':
            print(f"! {description}: {path} (will be created on first use)")
        else:
            print(f"✗ {description}: {path} (missing!)")
            return False
    
    return True


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        # Set to development mode for testing
        os.environ['FLASK_ENV'] = 'development'
        from config import get_config
        config = get_config()
        print(f"✓ Configuration loaded")
        print(f"  - Environment: development")
        print(f"  - Debug: {config.DEBUG}")
        print(f"  - Max upload size: {config.MAX_CONTENT_LENGTH / (1024*1024)}MB")
        print(f"  - Allowed extensions: {config.ALLOWED_EXTENSIONS}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_app_creation():
    """Test if Flask app can be created"""
    print("\nTesting Flask app creation...")
    
    try:
        from flask import Flask
        app = Flask(__name__, template_folder='templates', static_folder='static')
        print(f"✓ Flask app created successfully")
        return True
    except Exception as e:
        print(f"✗ Flask app creation error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Cancer Diagnosis Frontend - Tests")
    print("=" * 50)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Directories", test_directories()))
    results.append(("Configuration", test_config()))
    results.append(("App Creation", test_app_creation()))
    
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed! You can start the app with: python app.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
