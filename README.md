# Production RAG Pipeline

A Python library that gives any local LLM free access to web search — like Perplexity, but self-hosted and without API costs.

Searches Bing + DuckDuckGo, filters noise before fetching, extracts clean content, reranks by relevance, and outputs a complete LLM-ready prompt with inline citations. Plug it into Ollama, LM Studio, or any LLM API and get cited, structured answers from the internet.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/project/production-rag-pipeline/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Sentence Transformers](https://img.shields.io/badge/sentence--transformers-supported-orange.svg)](https://www.sbert.net/)
[![LLM Ready](https://img.shields.io/badge/output-LLM--ready%20prompts-purple.svg)](#how-it-works)

---

## How It Works

```
Query → Language Detection → Search Engines (Bing + DDG)
   ↓
Semantic Pre-Filter (embeddings, threshold 0.25–0.30)
   ↓ (11/20 filtered)
Parallel Fetch (10 workers, 12s timeout)
   ↓
Content Quality Check (structure + semantic relevance)
   ↓
Chunking (semantic topic-based, 600 chars)
   ↓
Hybrid Reranking (BM25 + Semantic + Cross-encoder)
   ↓
Freshness Penalties (news queries: −2 for >30 days, −1 for >7 days)
   ↓
Context Building (filter ghosts, sync references, add warnings)
   ↓
LLM Prompt (clean sources + outdated warnings)
```

---

## Key Features

### Dual Search Engines

Bing + DuckDuckGo searched in parallel, results merged with position-based scoring. Bing works well for financial/factual queries, DDG gives free access and a news mode. When the pipeline detects news-related keywords (`новост`, `news`, `latest`, etc.), DDG switches to its News index — returns actual articles instead of homepages.

### Semantic Pre-Filtering

Each search result is checked for relevance **before** the page is fetched. Cosine similarity between query and title+snippet embeddings, threshold 0.25 (RU) / 0.30 (EN). In practice, 11 out of 20 results get dropped pre-fetch — saves about 6 seconds.

Example: query "LLM agents news" — `flutrackers.com` (sim=0.12) filtered, `llm-stats.com` (sim=0.68) fetched.

### Context-Aware Content Detection

Two-stage check for price lists and tables. First structural: does >30% of lines look like numbers? Then semantic: is this price list actually relevant? This way `cbr.ru` exchange rates pass for a "курс евро" query (sim=0.75) but `steamcommunity.com` CS:GO prices get rejected (sim=0.05). No hardcoded domain whitelists.

### Freshness Tracking

Activated only for news queries (auto-detected via keyword patterns). Content older than 30 days gets −2 to confidence score, older than 7 days gets −1. Outdated sources are flagged in the LLM prompt with exact age. Non-news queries (prices, how-to) are unaffected.

### Multilingual Intelligence

Auto-detects language by Cyrillic character ratio (10% threshold) and switches models accordingly. Russian queries get `paraphrase-multilingual-MiniLM-L12-v2` for embeddings and a multilingual cross-encoder (13 languages). English queries get the faster `all-MiniLM-L6-v2`. Models download automatically on first run.

### Quality Control

Failed fetches excluded from context. Boilerplate, navigation, ads, and newsletter patterns filtered. Citation numbers always match actual sources — no phantom `[4]`, `[5]` references when only 3 sources exist.

---

## Package Overview

The package is structured as a library first:

- `production_rag_pipeline.search`
  Bing/DDG search, result merging, language filtering
- `production_rag_pipeline.fetch`
  HTTP client, retries, parallel page fetching
- `production_rag_pipeline.extract`
  content extraction, publish-date parsing, chunking, quality filtering
- `production_rag_pipeline.rerank`
  BM25, semantic relevance, answer-span detection, MMR, cross-encoder reranking
- `production_rag_pipeline.pipeline`
  end-to-end orchestration: search → fetch → extract → rerank → context
- `production_rag_pipeline.prompts`
  final LLM prompt assembly
- `production_rag_pipeline.core`
  internal runtime settings and optional dependency state

The pipeline is designed to degrade gracefully:

- without semantic models, lexical ranking still works
- without `trafilatura`, extraction falls back to `BeautifulSoup`
- configuration can come from dataclasses, YAML, or environment variables

---

## Installation

```bash
git clone https://github.com/KazKozDev/production_rag_pipeline.git
cd production_rag_pipeline
pip install .
```

Optional extras:

```bash
pip install .[extraction]
pip install .[semantic]
pip install .[full]
```

Profiles:

- `base`: `beautifulsoup4`, `curl-cffi`, `PyYAML`
- `extraction`: adds `trafilatura`
- `semantic`: adds `sentence-transformers`, `scikit-learn`, `numpy`
- `full`: installs both optional groups

---

## Quick Start

```python
from production_rag_pipeline import build_llm_prompt

prompt = build_llm_prompt("latest AI news", lang="en")
print(prompt)
```

CLI:

```bash
production-rag-pipeline "latest AI news"
production-rag-pipeline "Bitcoin price" --mode search
production-rag-pipeline "новости ИИ" --mode read --lang ru
```

---

## Public API

High-level API:

```python
from production_rag_pipeline import (
    build_llm_context,
    build_llm_prompt,
    search_and_read,
    search_extract_rerank,
)
```

Module-level API:

```python
from production_rag_pipeline.search import search, search_bing, search_ddg
from production_rag_pipeline.fetch import fetch_pages_parallel
from production_rag_pipeline.extract import extract_content, chunk_text
from production_rag_pipeline.rerank import rerank_chunks
from production_rag_pipeline.pipeline import search_extract_rerank, build_llm_context
from production_rag_pipeline.prompts import build_llm_prompt
```

---

## Example Pipeline Usage

```python
from production_rag_pipeline.pipeline import build_llm_context, search_extract_rerank

chunks, results, fetched_urls = search_extract_rerank(
    query="latest AI news",
    num_fetch=8,
    lang="en",
    debug=True,
)

context, source_mapping, grouped_sources = build_llm_context(
    chunks,
    results,
    fetched_urls=fetched_urls,
    renumber_sources=True,
)
```

---

## Configuration

Use `RAGConfig` directly:

```python
from production_rag_pipeline import RAGConfig, build_llm_prompt

config = RAGConfig(
    num_per_engine=12,
    top_n_fetch=8,
    fetch_timeout=10,
    total_context_chunks=12,
)

prompt = build_llm_prompt("latest AI news", config=config)
```

Or load from YAML / env:

```bash
production-rag-pipeline "latest AI news" --config config.example.yaml
```

Environment variables follow the `RAG_` prefix convention, for example:

```bash
export RAG_TOP_N_FETCH=8
export RAG_FETCH_TIMEOUT=10
```

---

## Design Notes

Key behaviors:

- semantic pre-filtering before fetch
- content validation after fetch
- adaptive weighting by query type
- MMR-based diversity control
- optional cross-encoder final rerank
- freshness penalties for news-like queries
- current date and time are injected into the generated prompt so the LLM can reason correctly about time-sensitive queries and source freshness
- no hardcoded trusted-domain whitelist
- no hardcoded dated few-shot examples in prompt generation

---

## Development

Basic verification:

```bash
python -m compileall src
```

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
