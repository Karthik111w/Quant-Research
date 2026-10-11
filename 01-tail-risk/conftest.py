"""conftest.py — make src/ importable from tests/ without modifying sys.path manually."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
