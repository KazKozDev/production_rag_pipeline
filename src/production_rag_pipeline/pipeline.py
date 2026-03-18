from __future__ import annotations


def search_extract_rerank(query, num_fetch=None, lang="en", debug=False, config=None, config_path=None):
    """
    Full pipeline:
      1. Search Bing+DDG → merge
      2. Take top N results
      3. Parallel fetch + extraction
      4. Chunk extracted text
      5. Rerank chunks vs query
      6. Return ranked context chunks with metadata
    """
    from . import core
    from .extract import chunk_text, filter_low_quality_chunks
    from .fetch import fetch_pages_parallel
    from .rerank import _detect_query_type, filter_results_by_relevance, rerank_chunks
    from .search import search

    core._apply_runtime_config(config=config, config_path=config_path)

    if num_fetch is None:
        num_fetch = core.TOP_N_FETCH

    print(f"\n{'=' * 60}")
    print(f"  QUERY: {query}")
    print(f"{'=' * 60}")

    all_results = search(query, num=20, lang=lang, debug=debug)
    if not all_results:
        return [], [], set()

    threshold = 0.25 if lang == "ru" else 0.30
    print(f"\n[PIPELINE] Pre-filtering {len(all_results)} results by semantic relevance (threshold={threshold})...")
    all_results = filter_results_by_relevance(query, all_results, threshold=threshold, lang=lang)

    if not all_results:
        print("[PIPELINE] No relevant results after pre-filtering")
        return [], [], set()

    top_results = all_results[:num_fetch]
    print(f"\n[PIPELINE] Top {len(top_results)} results to fetch:")
    for i, result in enumerate(top_results, 1):
        engines = ", ".join(result["engines"])
        print(f"  {i}. [{engines}] (score={result['score']}) {result['title'][:60]}")
        print(f"     {result['url'][:80]}")

    print(f"\n[PIPELINE] Fetching {len(top_results)} pages in parallel...")
    urls = [result["url"] for result in top_results]
    fetched = fetch_pages_parallel(urls, query=query, lang=lang)

    if not fetched:
        print("[PIPELINE] No pages fetched successfully — using snippets only")
        chunks_with_meta = []
        snippet_urls = set()
        for i, result in enumerate(top_results):
            if result["snippet"]:
                chunks_with_meta.append(
                    {
                        "text": result["snippet"],
                        "source_idx": i,
                        "source_url": result["url"],
                        "source_title": result["title"],
                        "chunk_idx": 0,
                    }
                )
                snippet_urls.add(result["url"])
        adaptive_chunk_count = _detect_query_type(query)
        ranked = rerank_chunks(query, chunks_with_meta, top_k=adaptive_chunk_count, lang=lang)
        return ranked, top_results, snippet_urls

    print(f"\n[PIPELINE] Chunking {len(fetched)} pages...")
    all_chunks = []

    for i, result in enumerate(top_results):
        page_data = fetched.get(result["url"])
        if not page_data:
            if result["snippet"]:
                all_chunks.append(
                    {
                        "text": result["snippet"],
                        "source_idx": i,
                        "source_url": result["url"],
                        "source_title": result["title"],
                        "chunk_idx": 0,
                        "pub_date": None,
                    }
                )
            continue

        text = page_data["text"]
        pub_date = page_data.get("pub_date")
        chunks = filter_low_quality_chunks(chunk_text(text, lang=lang))
        print(f"  Source {i + 1}: {len(chunks)} chunks from «{result['title'][:50]}»")

        for ci, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "source_idx": i,
                    "source_url": result["url"],
                    "source_title": result["title"],
                    "chunk_idx": ci,
                    "pub_date": pub_date,
                }
            )

    print(f"  Total chunks: {len(all_chunks)}")

    adaptive_chunk_count = _detect_query_type(query)
    rerank_method = "Hybrid (BM25 + Semantic)" if core.HAS_EMBEDDINGS else "BM25 only"
    print(f"\n[PIPELINE] Reranking {len(all_chunks)} chunks with {rerank_method}...")
    print(f"  Query type → selecting top {adaptive_chunk_count} chunks")
    ranked = rerank_chunks(query, all_chunks, top_k=adaptive_chunk_count, lang=lang)
    print(f"  Selected top {len(ranked)} chunks:")
    for chunk in ranked:
        if core.HAS_EMBEDDINGS:
            print(
                f"    src={chunk['source_idx'] + 1} chunk={chunk['chunk_idx']} "
                f"rel={chunk['relevance']:.3f} (bm25={chunk['bm25']:.3f} sem={chunk['semantic']:.3f}) "
                f"— {chunk['text'][:50]}..."
            )
        else:
            print(
                f"    src={chunk['source_idx'] + 1} chunk={chunk['chunk_idx']} "
                f"rel={chunk['relevance']:.4f} — {chunk['text'][:60]}..."
            )

    fetched_urls = set(fetched.keys()) if fetched else set()
    return ranked, top_results, fetched_urls


def _is_source_relevant(chunks, min_relevance=0.3):
    if not chunks:
        return False
    avg_relevance = sum(chunk["relevance"] for chunk in chunks) / len(chunks)
    max_relevance = max(chunk["relevance"] for chunk in chunks)
    return avg_relevance >= min_relevance or max_relevance >= 0.5


def build_llm_context(ranked_chunks, search_results, fetched_urls=None, renumber_sources=True):
    if not ranked_chunks:
        return "No relevant content found.", {}, {}

    by_source = {}
    skipped_ghost_sources = set()
    for chunk in ranked_chunks:
        idx = chunk["source_idx"]
        url = chunk["source_url"]
        if fetched_urls is not None and url not in fetched_urls:
            skipped_ghost_sources.add((idx, chunk["source_title"], url))
            continue
        if idx not in by_source:
            by_source[idx] = {"title": chunk["source_title"], "url": url, "chunks": []}
        by_source[idx]["chunks"].append(chunk)

    if skipped_ghost_sources:
        print(f"\n[CONTEXT] Filtered out {len(skipped_ghost_sources)} sources with failed fetch:")
        for idx, title, url in sorted(skipped_ghost_sources):
            print(f"  [{idx + 1}] {title[:60]}... — {url[:60]}")

    filtered_sources = {}
    for idx, source_data in by_source.items():
        if _is_source_relevant(source_data["chunks"]):
            filtered_sources[idx] = source_data
        else:
            print(f"  [FILTER] Removed source [{idx + 1}]: {source_data['title'][:60]} (low relevance)")

    by_source = filtered_sources
    if not by_source:
        return "No relevant sources found.", {}, {}

    if renumber_sources:
        source_mapping = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(sorted(by_source.keys()))}
    else:
        source_mapping = {idx: idx + 1 for idx in by_source.keys()}

    parts = []
    for old_idx in sorted(by_source.keys()):
        src = by_source[old_idx]
        src_num = source_mapping[old_idx]
        parts.append(f"[{src_num}] {src['title']}")
        for chunk in src["chunks"]:
            parts.append(chunk["text"].replace("**", ""))
        parts.append("")

    return "\n".join(parts), source_mapping, by_source


def search_and_read(query, num_results=None, lang="en", debug=False, config=None, config_path=None):
    from . import core

    core._apply_runtime_config(config=config, config_path=config_path)
    if num_results is None:
        num_results = core.TOP_N_FETCH

    ranked_chunks, search_results, fetched_urls = search_extract_rerank(
        query,
        num_fetch=num_results,
        lang=lang,
        debug=debug,
    )
    context, _, _ = build_llm_context(ranked_chunks, search_results, fetched_urls=fetched_urls)
    return context
