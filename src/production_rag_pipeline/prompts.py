from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse


def _categorize_query(query):
    query_lower = query.lower()
    brief_keywords = ["курс", "цена", "стоимость", "сколько", "когда", "где", "price", "cost", "rate", "when", "where", "what is", "who is"]
    comprehensive_keywords = ["обзор", "новости", "история", "сравнение", "анализ", "overview", "news", "history", "comparison", "analysis", "review"]

    if any(kw in query_lower for kw in brief_keywords):
        query_type = "Factual/Brief"
        expected_length = "1-2 paragraphs"
    elif any(kw in query_lower for kw in comprehensive_keywords):
        query_type = "Comprehensive"
        expected_length = "3-5 paragraphs with detailed analysis"
    else:
        query_type = "Medium complexity"
        expected_length = "2-3 paragraphs"

    if any(kw in query_lower for kw in ["курс", "цена", "price", "cost", "rate", "стоимость"]):
        intent = "Get current price/rate with market data"
        critical_attrs = ["freshness (< 24h preferred)", "numerical accuracy", "source credibility"]
    elif any(kw in query_lower for kw in ["новости", "news", "события", "events"]):
        intent = "Get latest news and developments"
        critical_attrs = ["freshness (today/yesterday)", "factual accuracy", "multiple perspectives"]
    elif any(kw in query_lower for kw in ["что такое", "what is", "кто", "who is"]):
        intent = "Get definition and background information"
        critical_attrs = ["comprehensiveness", "accuracy", "authoritative sources"]
    elif any(kw in query_lower for kw in ["как", "how to", "каким образом"]):
        intent = "Get step-by-step instructions or explanation"
        critical_attrs = ["clarity", "completeness", "practical examples"]
    else:
        intent = "General information retrieval"
        critical_attrs = ["relevance", "accuracy", "comprehensive coverage"]

    return {
        "type": query_type,
        "intent": intent,
        "expected_length": expected_length,
        "critical_attributes": critical_attrs,
    }


def _build_source_list(seen_sources):
    return "\n".join(
        f"  [{idx}] {source['title']} — {source['url']}"
        for idx, source in sorted(seen_sources.items())
    )


def _build_freshness_warning(query, seen_sources, is_news_query, now):
    if not is_news_query(query) or not seen_sources:
        return ""

    outdated_sources = []
    for src_num, src_data in sorted(seen_sources.items()):
        src_chunks = src_data.get("chunks", [])
        dates = [chunk.get("pub_date") for chunk in src_chunks if chunk.get("pub_date")]
        if not dates:
            continue

        oldest_date = min(dates)
        days_old = (now - oldest_date).days
        if days_old > 30:
            month_year = oldest_date.strftime("%b %Y")
            outdated_sources.append(f"[{src_num}] ({month_year}, {days_old} days old)")

    if not outdated_sources:
        return ""

    total_sources = len(seen_sources)
    outdated_count = len(outdated_sources)
    fresh_count = total_sources - outdated_count

    if fresh_count == 0:
        return (
            "\n⚠️ FRESHNESS WARNING: This is a news query, but all dated sources are older than 30 days.\n"
            + "\n".join(f"  {item}" for item in outdated_sources)
            + "\nUse cautious wording and tell the user current coverage may be incomplete.\n"
        )

    fresh_nums = [
        src_num
        for src_num in sorted(seen_sources.keys())
        if not any(f"[{src_num}]" in item for item in outdated_sources)
    ]
    fresh_refs = ", ".join(f"[{num}]" for num in fresh_nums)
    return (
        "\n⚠️ OUTDATED SOURCES WARNING: Some sources are older than 30 days.\n"
        + "\n".join(f"  {item}" for item in outdated_sources)
        + f"\nPrioritize fresher reporting from {fresh_refs}.\n"
    )


def build_llm_prompt(query, lang="en", config=None, config_path=None):
    """Search, rerank, and build a concise LLM-ready prompt."""
    from . import core
    from .pipeline import build_llm_context, search_extract_rerank
    from .rerank import _has_factual_data, _is_news_query

    core._apply_runtime_config(config=config, config_path=config_path)

    ranked_chunks, search_results, fetched_urls = search_extract_rerank(query, lang=lang)
    context, source_mapping, by_source = build_llm_context(
        ranked_chunks,
        search_results,
        fetched_urls=fetched_urls,
        renumber_sources=True,
    )

    seen_sources = {
        source_mapping[old_idx]: {
            "title": src["title"],
            "url": src["url"],
            "chunks": src["chunks"],
        }
        for old_idx, src in by_source.items()
    }

    source_list = _build_source_list(seen_sources)
    now = datetime.now()
    current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    day_of_week = now.strftime("%A")
    freshness_warning = _build_freshness_warning(query, seen_sources, _is_news_query, now)
    query_profile = _categorize_query(query)

    if ranked_chunks:
        unique_sources = len(seen_sources)
        unique_domains = len(set(urlparse(chunk.get("source_url", "")).netloc for chunk in ranked_chunks))
        factual_count = sum(1 for chunk in ranked_chunks if _has_factual_data(chunk["text"]))
        dated_count = sum(1 for chunk in ranked_chunks if chunk.get("pub_date"))
    else:
        unique_sources = unique_domains = factual_count = dated_count = 0

    return f"""Answer the user's question using ONLY the search results provided below.

CURRENT DATE: {current_datetime} ({day_of_week}){freshness_warning}
WRITING REQUIREMENTS:
1. Use only the provided sources.
2. Organize the answer with clear `##` headings.
3. Cite every factual claim with `[N]`.
4. For news queries, prioritize sources from the last 7 days.
5. If sources conflict, describe both versions clearly.
6. Include concrete details: dates, names, numbers, prices, locations, and percentages when available.
7. Use all relevant sources instead of relying on only one or two.
8. End with `## Источники:` and list the sources actually used.

QUALITY SIGNALS FROM THE PIPELINE:
- Context chunks: {len(ranked_chunks)}
- Unique sources: {unique_sources}
- Unique domains: {unique_domains}
- Chunks with factual data: {factual_count}
- Chunks with explicit dates: {dated_count}

QUERY PROFILE:
- Type: {query_profile['type']}
- Intent: {query_profile['intent']}
- Expected answer shape: {query_profile['expected_length']}
- Critical attributes: {", ".join(query_profile['critical_attributes'])}

QUESTION: {query}

SOURCES:
{context}

References:
{source_list}

ANSWER FORMAT:
[Detailed answer with inline citations]

## Источники:
[1] Source Title — URL
[2] Another Source — URL
"""


def ask_with_search(query, llm_fn=None, lang="en", config=None, config_path=None):
    prompt = build_llm_prompt(query, lang=lang, config=config, config_path=config_path)
    return llm_fn(prompt) if llm_fn else prompt
