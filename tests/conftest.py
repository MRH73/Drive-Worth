import sys
from pathlib import Path


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the project folder to Python's import path during tests.
# This lets tests import app.py correctly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
