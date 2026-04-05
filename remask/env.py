"""Shared environment setup — import this BEFORE any HuggingFace module."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

os.environ.setdefault("HF_HOME", _DATA_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_DATA_DIR, "datasets"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
