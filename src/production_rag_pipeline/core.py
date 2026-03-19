"""
Internal runtime settings and shared optional dependency state.

Implementation code lives in focused modules: search, fetch, extract,
rerank, pipeline, and prompts.
"""

from __future__ import annotations

import re

# Optional dependencies
try:
    import trafilatura  # noqa: F401

    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    print("[WARN] trafilatura not installed — falling back to basic extraction")
    print("       pip install production-rag-pipeline[extraction]")

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    HAS_EMBEDDINGS = True
    HAS_CROSS_ENCODER = True
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None
    HAS_EMBEDDINGS = False
    HAS_CROSS_ENCODER = False
    print("[WARN] sentence-transformers not installed — using lexical fallbacks")
    print("       pip install production-rag-pipeline[semantic]")


# Configurable runtime knobs
NUM_PER_ENGINE = 15
TOP_N_FETCH = 10
MAX_CONTENT_CHARS = 12000
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_CHUNKS_PER_PAGE = 3
TOTAL_CONTEXT_CHUNKS = 15
FETCH_WORKERS = 10
FETCH_TIMEOUT = 12

BM25_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3
POSITION_BONUS = True
TITLE_MATCH_BONUS = 1.2
EARLY_CHUNK_BONUS = 1.1
FACTUAL_DATA_BONUS = 1.15
FRESH_CONTENT_BONUS = 1.3
RECENT_CONTENT_BONUS = 1.15

IMPERSONATE = [
    "chrome110",
    "chrome116",
    "chrome120",
    "chrome123",
    "chrome124",
    "safari15_5",
    "safari17_0",
]

__all__ = [
    "trafilatura",
    "SentenceTransformer",
    "CrossEncoder",
]


def _apply_runtime_config(config=None, config_path=None):
    if config is None and config_path is None:
        return None

    from .config import apply_config, load_config

    resolved = config if config is not None else load_config(config_path)
    apply_config(resolved)
    return resolved


def save_prompt_to_file(query, prompt_text):
    from datetime import datetime
    from pathlib import Path

    safe_query = re.sub(r"[^\w\s-]", "", query).strip()
    safe_query = re.sub(r"[-\s]+", "_", safe_query)[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"prompt_{safe_query}_{timestamp}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(f"Query: {query}\n")
            handle.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write("=" * 80 + "\n\n")
            handle.write(prompt_text)
        print(f"[SAVED] Prompt saved to: {filename}")
        return str(filename)
    except Exception as exc:
        print(f"[ERROR] Failed to save prompt: {exc}")
        return None
