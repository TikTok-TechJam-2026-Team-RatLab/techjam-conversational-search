"""TechJam Search Source Package"""
import sys
from pathlib import Path

_libs_dir = str(Path(__file__).resolve().parent.parent / "libs")
sys.path = [_libs_dir] + [p for p in sys.path if "Python313" not in p and p != _libs_dir]