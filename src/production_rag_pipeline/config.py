from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import get_type_hints

import yaml


@dataclass(frozen=True)
class RAGConfig:
    num_per_engine: int = 15
    top_n_fetch: int = 10
    max_content_chars: int = 12000
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_chunks_per_page: int = 3
    total_context_chunks: int = 15
    fetch_workers: int = 10
    fetch_timeout: int = 12
    bm25_weight: float = 0.7
    semantic_weight: float = 0.3
    position_bonus: bool = True
    title_match_bonus: float = 1.2
    early_chunk_bonus: float = 1.1
    factual_data_bonus: float = 1.15
    fresh_content_bonus: float = 1.3
    recent_content_bonus: float = 1.15


DEFAULT_CONFIG = RAGConfig()
_CURRENT_CONFIG = DEFAULT_CONFIG
_FIELD_TYPES = get_type_hints(RAGConfig)
_CORE_ATTRS = {
    "num_per_engine": "NUM_PER_ENGINE",
    "top_n_fetch": "TOP_N_FETCH",
    "max_content_chars": "MAX_CONTENT_CHARS",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "top_chunks_per_page": "TOP_CHUNKS_PER_PAGE",
    "total_context_chunks": "TOTAL_CONTEXT_CHUNKS",
    "fetch_workers": "FETCH_WORKERS",
    "fetch_timeout": "FETCH_TIMEOUT",
    "bm25_weight": "BM25_WEIGHT",
    "semantic_weight": "SEMANTIC_WEIGHT",
    "position_bonus": "POSITION_BONUS",
    "title_match_bonus": "TITLE_MATCH_BONUS",
    "early_chunk_bonus": "EARLY_CHUNK_BONUS",
    "factual_data_bonus": "FACTUAL_DATA_BONUS",
    "fresh_content_bonus": "FRESH_CONTENT_BONUS",
    "recent_content_bonus": "RECENT_CONTENT_BONUS",
}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _coerce_value(name, value):
    target_type = _FIELD_TYPES[name]
    if target_type is bool:
        return _parse_bool(value)
    return target_type(value)


def _normalize_overrides(overrides):
    normalized = {}
    for key, value in (overrides or {}).items():
        key = key.lower()
        if key not in _FIELD_TYPES:
            continue
        normalized[key] = _coerce_value(key, value)
    return normalized


def _yaml_overrides(path):
    if not path:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return _normalize_overrides(data)


def _env_overrides(prefix="RAG_"):
    overrides = {}
    for key in _FIELD_TYPES:
        env_name = f"{prefix}{key.upper()}"
        if env_name in os.environ:
            overrides[key] = _coerce_value(key, os.environ[env_name])
    return overrides


def load_config(path: str | os.PathLike | None = None, env_prefix: str = "RAG_", overrides=None) -> RAGConfig:
    config_path = path or os.environ.get(f"{env_prefix}CONFIG_PATH")
    merged = asdict(DEFAULT_CONFIG)
    merged.update(_yaml_overrides(config_path))
    merged.update(_env_overrides(prefix=env_prefix))
    merged.update(_normalize_overrides(overrides))
    return RAGConfig(**merged)


def apply_config(config: RAGConfig) -> RAGConfig:
    global _CURRENT_CONFIG

    from . import core

    for field_name, core_attr in _CORE_ATTRS.items():
        setattr(core, core_attr, getattr(config, field_name))

    _CURRENT_CONFIG = config
    return config


def configure(path: str | os.PathLike | None = None, env_prefix: str = "RAG_", overrides=None) -> RAGConfig:
    return apply_config(load_config(path=path, env_prefix=env_prefix, overrides=overrides))


def current_config() -> RAGConfig:
    return _CURRENT_CONFIG


def update_config(**overrides) -> RAGConfig:
    config = replace(_CURRENT_CONFIG, **_normalize_overrides(overrides))
    return apply_config(config)
