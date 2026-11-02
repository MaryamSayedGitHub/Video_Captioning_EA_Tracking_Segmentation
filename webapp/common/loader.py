"""Loads Python files whose names start with a digit (1_detection.py, ...).

Python cannot `import 1_detection`, so we load by path - the same trick the original
Final_Video_with_voice_over.py uses. Used for (a) the numbered step files inside each
feature folder and (b) the original numbered scripts in the project root.
"""
import importlib.util
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # holds 1_Frame_Extraction.py ... 6_burn_captions.py
_cache = {}


def load(path):
    """Load a module from a file path (cached)."""
    path = Path(path).resolve()
    if path in _cache:
        return _cache[path]
    spec = importlib.util.spec_from_file_location("step_" + path.stem.lower().replace(" ", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cache[path] = module
    return module


def load_root(file_name):
    """Load one of the original project scripts, matching the file name case-insensitively
    (so 6_Burn_Captions.py vs 6_burn_captions.py does not matter on Linux/macOS)."""
    match = next((f for f in os.listdir(PROJECT_ROOT) if f.lower() == file_name.lower()), None)
    if match is None:
        raise FileNotFoundError(f"{file_name} not found in {PROJECT_ROOT}")
    return load(PROJECT_ROOT / match)


def step(feature_dir, file_name):
    """Load a numbered step file that sits next to a feature's pipeline.py."""
    return load(Path(feature_dir) / file_name)
