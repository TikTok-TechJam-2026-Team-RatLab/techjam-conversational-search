"""TechJam Search Source Package"""
import sys
from pathlib import Path

_root_dir = str(Path(__file__).resolve().parent.parent)
_libs_dir = str(Path(__file__).resolve().parent.parent / "libs")

if sys.version_info[:2] == (3, 12):
    sys.path = [_libs_dir, _root_dir] + [
        p for p in sys.path if "Python313" not in p and "Roaming" not in p and p not in (_libs_dir, _root_dir)
    ]
else:
    if _root_dir not in sys.path:
        sys.path.insert(0, _root_dir)