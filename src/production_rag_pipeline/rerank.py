from __future__ import annotations

import math
import re
from collections import Counter

from .confidence import calculate_confidence

_EMBEDDING_MODEL = None
_CROSS_ENCODER_MODEL = None


def _tokenize(text):
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    stopwords_ru = {
        "в",
        "и",
        "на",
        "с",
        "по",
        "для",
        "не",
        "что",
        "это",
        "как",
        "из",
        "за",
        "к",
        "до",
        "от",
        "при",
        "или",
        "но",
        "а",
        "то",
        "все",
        "так",
        "может",
        "быть",
        "год",
        "года",
        "уже",
        "более",
    }
    stopwords_en = {
        "the",
        "is",
        "at",
        "which",
        "on",
        "a",
        "an",
        "as",
        "are",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
    }
    return [t for t in tokens if len(t) > 2 and t not in (stopwords_ru | stopwords_en)]


def _build_idf(documents):
    n = len(documents)
    if n == 0:
        return {}
    df = Counter()
    for doc_tokens in documents:
        for token in set(doc_tokens):
            df[token] += 1
    return {t: math.log((n + 1) / (freq + 1)) + 1 for t, freq in df.items()}


def _tfidf_vector(tokens, idf):
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {t: (count / total) * idf.get(t, 1.0) for t, count in tf.items()}


def _cosine_sim(v1, v2):
    if not v1 or not v2:
        return 0.0
    common = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _bm25_score(query_tokens, doc_tokens, idf, avg_doc_len, k1=1.5, b=0.75):
    score = 0.0
    doc_len = len(doc_tokens)
    tf = Counter(doc_tokens)
    for token in query_tokens:
        if token in tf:
            freq = tf[token]
            norm = 1 - b + b * (doc_len / avg_doc_len)
            score += idf.get(token, 0) * (freq * (k1 + 1)) / (freq + k1 * norm)
    return score


def _get_embedding_model(lang="en"):
    from . import core

    global _EMBEDDING_MODEL
    if not core.HAS_EMBEDDINGS:
        return None
    if _EMBEDDING_MODEL is None:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if lang == "ru":
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"[EMBEDDINGS] Loading {model_name} (multilingual for Russian)...")
        else:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"[EMBEDDINGS] Loading {model_name} (English)...")
        _EMBEDDING_MODEL = core.SentenceTransformer(model_name)
        print("[EMBEDDINGS] Model loaded ✓")
    return _EMBEDDING_MODEL


def _get_cross_encoder_model(lang="en"):
    from . import core

    global _CROSS_ENCODER_MODEL
    if not core.HAS_CROSS_ENCODER:
        return None
    if _CROSS_ENCODER_MODEL is None:
        if lang == "ru":
            model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            print(f"[CROSS-ENCODER] Loading {model_name} (multilingual for Russian)...")
        else:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            print(f"[CROSS-ENCODER] Loading {model_name} (English)...")
        _CROSS_ENCODER_MODEL = core.CrossEncoder(model_name)
        print("[CROSS-ENCODER] Model loaded ✓")
    return _CROSS_ENCODER_MODEL


def _semantic_similarity(query, texts, lang="en"):
    model = _get_embedding_model(lang)
    if model is None:
        return [0.0] * len(texts)
    try:
        query_embedding = model.encode(query, convert_to_tensor=False)
        text_embeddings = model.encode(texts, convert_to_tensor=False)
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(text_embeddings))[0].tolist()
    except Exception as exc:
        print(f"[EMBEDDINGS] Error: {exc}")
        return [0.0] * len(texts)


def filter_results_by_relevance(query, results, threshold=0.25, lang="en"):
    from . import core

    if not results or not core.HAS_EMBEDDINGS:
        return results
    try:
        model = _get_embedding_model(lang)
        if not model:
            return results
        texts = [r["title"] + " " + r.get("snippet", "") for r in results]
        query_emb = model.encode(query, convert_to_tensor=False, show_progress_bar=False)
        text_embs = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(np.array(query_emb).reshape(1, -1), np.array(text_embs))[0]
        filtered = []
        dropped_count = 0
        for result, sim in zip(results, sims):
            if sim >= threshold:
                filtered.append(result)
            else:
                dropped_count += 1
                print(f"  [PRE-FILTER] Dropped (sim={sim:.3f}): {result['title'][:60]}...")
        if dropped_count > 0:
            print(f"  [PRE-FILTER] Kept {len(filtered)}/{len(results)} results (threshold={threshold})")
        return filtered
    except Exception as exc:
        print(f"  [PRE-FILTER] Error: {exc}, returning all results")
        return results


def _detect_query_type(query):
    query_lower = query.lower()
    brief_keywords = [
        "курс",
        "цена",
        "стоимость",
        "сколько",
        "когда",
        "где",
        "price",
        "cost",
        "rate",
        "when",
        "where",
        "what is",
        "who is",
        "date",
        "время",
        "адрес",
        "контакт",
    ]
    if any(kw in query_lower for kw in brief_keywords):
        return 8
    comprehensive_keywords = [
        "обзор",
        "новости",
        "история",
        "сравнение",
        "анализ",
        "все о",
        "overview",
        "news",
        "history",
        "comparison",
        "analysis",
        "review",
        "все",
        "полный",
        "подробно",
        "complete",
        "comprehensive",
        "detailed",
    ]
    if any(kw in query_lower for kw in comprehensive_keywords):
        return 30
    return 12


def _is_news_query(query):
    return any(
        kw in query.lower()
        for kw in [
            "новост",
            "news",
            "сегодня",
            "today",
            "вчера",
            "yesterday",
            "сейчас",
            "now",
            "актуальн",
            "current",
            "latest",
            "recent",
            "событи",
            "events",
            "происходит",
            "happening",
        ]
    )


def _has_factual_data(text):
    flags = [
        bool(re.search(r"\d+[.,]\d+", text)),
        bool(re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}", text)),
        bool(re.search(r"\d+\s*%", text)),
        bool(re.search(r"[\$€₽£¥]\s*\d+|руб|usd|eur|btc", text.lower())),
        bool(re.search(r"\d{4,}", text)),
        bool(re.search(r"\d{1,2}:\d{2}", text)),
    ]
    return sum(flags) >= 2


def _calculate_confidence(chunk, avg_relevance, query=None):
    return calculate_confidence(
        chunk,
        avg_relevance,
        query=query,
        has_factual_data=_has_factual_data,
        is_news_query=_is_news_query,
    )


def _extract_answer_span(query, chunk_text, window=200):
    factoid_patterns = [r"(курс|цена|price|cost)\s+\S+", r"(когда|when)\s+", r"(сколько|how much|how many)", r"(где|where)", r"(кто|who)"]
    if not any(re.search(pattern, query.lower()) for pattern in factoid_patterns):
        return None

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return None

    best_sentence = None
    best_score = 0
    for sent in re.split(r"(?<=[.!?])\s+", chunk_text):
        sent_tokens = set(_tokenize(sent))
        if not sent_tokens:
            continue
        overlap = len(query_tokens & sent_tokens) / len(query_tokens)
        score = overlap + (0.3 if bool(re.search(r"\d+", sent)) else 0)
        if score > best_score:
            best_score = score
            best_sentence = sent

    if best_sentence and best_score > 0.3:
        idx = chunk_text.find(best_sentence)
        if idx >= 0:
            start = max(0, idx - window // 2)
            end = min(len(chunk_text), idx + len(best_sentence) + window // 2)
            return {"span": chunk_text[start:end].strip(), "score": best_score, "full_sentence": best_sentence}
    return None


def _mmr_diversify(query, chunks_with_scores, top_k, lambda_param=0.7):
    if not chunks_with_scores or len(chunks_with_scores) <= top_k:
        return chunks_with_scores

    selected = [chunks_with_scores[0]]
    remaining = chunks_with_scores[1:]

    while len(selected) < top_k and remaining:
        best_score = -1
        best_idx = -1
        for i, candidate in enumerate(remaining):
            rel_score = candidate["relevance"]
            max_sim = 0
            candidate_tokens = set(_tokenize(candidate["text"]))
            for sel in selected:
                sel_tokens = set(_tokenize(sel["text"]))
                if candidate_tokens and sel_tokens:
                    intersection = len(candidate_tokens & sel_tokens)
                    union = len(candidate_tokens | sel_tokens)
                    max_sim = max(max_sim, intersection / union if union > 0 else 0)
            mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
    print(f"  [MMR] Diversified from {len(chunks_with_scores)} to {len(selected)} chunks")
    return selected


def _cross_encoder_rerank(query, chunks, top_k, lang="en"):
    model = _get_cross_encoder_model(lang)
    if not model or len(chunks) <= top_k:
        return chunks
    try:
        scores = model.predict([[query, chunk["text"]] for chunk in chunks])
        for i, chunk in enumerate(chunks):
            chunk["cross_encoder_score"] = float(scores[i])
            chunk["relevance"] = chunk["relevance"] * 0.6 + scores[i] * 0.4
        chunks.sort(key=lambda x: -x["relevance"])
        print(f"  [CROSS-ENCODER] Reranked {len(chunks)} chunks")
        return chunks[:top_k]
    except Exception as exc:
        print(f"  [WARN] Cross-encoder reranking failed: {exc}")
        return chunks[:top_k]


def _adaptive_weights(query):
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["курс", "цена", "price", "cost", "когда", "where", "сколько"]):
        return {"bm25": 0.8, "semantic": 0.2}
    if any(kw in query_lower for kw in ["что такое", "what is", "explain", "why", "how does", "как работает"]):
        return {"bm25": 0.5, "semantic": 0.5}
    if any(kw in query_lower for kw in ["новости", "news", "события", "latest"]):
        return {"bm25": 0.65, "semantic": 0.35}
    return {"bm25": 0.7, "semantic": 0.3}


def _group_related_chunks(chunks):
    if not chunks:
        return chunks
    grouped = []
    current_group = [chunks[0]]
    for curr_chunk in chunks[1:]:
        prev_chunk = current_group[-1]
        if curr_chunk["source_idx"] == prev_chunk["source_idx"] and abs(curr_chunk["chunk_idx"] - prev_chunk["chunk_idx"]) <= 2:
            current_group.append(curr_chunk)
        else:
            if len(current_group) > 1:
                merged_chunk = current_group[0].copy()
                merged_chunk["text"] = "\n\n".join(c["text"] for c in current_group)
                merged_chunk["relevance"] = max(c["relevance"] for c in current_group)
                grouped.append(merged_chunk)
            else:
                grouped.append(current_group[0])
            current_group = [curr_chunk]
    if len(current_group) > 1:
        merged_chunk = current_group[0].copy()
        merged_chunk["text"] = "\n\n".join(c["text"] for c in current_group)
        merged_chunk["relevance"] = max(c["relevance"] for c in current_group)
        grouped.append(merged_chunk)
    else:
        grouped.append(current_group[0])
    if len(chunks) != len(grouped):
        print(f"  [GROUPING] Merged {len(chunks) - len(grouped)} adjacent chunks")
    return grouped


def rerank_chunks(query, chunks_with_meta, top_k=None, lang="en"):
    from . import core

    if top_k is None:
        top_k = core.TOTAL_CONTEXT_CHUNKS
    if not chunks_with_meta:
        return []

    query_tokens = _tokenize(query)
    all_token_lists = [_tokenize(c["text"]) for c in chunks_with_meta] + [query_tokens]
    idf = _build_idf(all_token_lists)
    avg_doc_len = sum(len(tl) for tl in all_token_lists) / len(all_token_lists)
    weights = _adaptive_weights(query)

    bm25_scores = [_bm25_score(query_tokens, all_token_lists[i], idf, avg_doc_len) for i in range(len(chunks_with_meta))]
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    if max_bm25 > 0:
        bm25_scores = [score / max_bm25 for score in bm25_scores]

    semantic_scores = [0.0] * len(chunks_with_meta)
    if core.HAS_EMBEDDINGS:
        semantic_scores = _semantic_similarity(query, [c["text"] for c in chunks_with_meta], lang=lang)

    scored = []
    for i, chunk in enumerate(chunks_with_meta):
        hybrid_score = weights["bm25"] * bm25_scores[i] + weights["semantic"] * semantic_scores[i]
        answer_span = _extract_answer_span(query, chunk["text"])
        if answer_span:
            hybrid_score *= 1.5
            chunk["answer_span"] = answer_span
        if core.POSITION_BONUS:
            if chunk["chunk_idx"] < 3:
                hybrid_score *= core.EARLY_CHUNK_BONUS
            if any(token in chunk.get("source_title", "").lower() for token in query_tokens):
                hybrid_score *= core.TITLE_MATCH_BONUS
        if _has_factual_data(chunk["text"]):
            hybrid_score *= core.FACTUAL_DATA_BONUS
        if _is_news_query(query) and chunk.get("pub_date"):
            from datetime import datetime

            age_hours = (datetime.now() - chunk["pub_date"]).total_seconds() / 3600
            if age_hours < 24:
                hybrid_score *= core.FRESH_CONTENT_BONUS
            elif age_hours < 24 * 7:
                hybrid_score *= core.RECENT_CONTENT_BONUS
        scored.append(
            {
                **chunk,
                "relevance": round(hybrid_score, 4),
                "bm25": round(bm25_scores[i], 4),
                "semantic": round(semantic_scores[i], 4),
            }
        )

    scored.sort(key=lambda x: (-x["relevance"], x["source_idx"], x["chunk_idx"]))

    deduplicated = []
    for chunk in scored:
        chunk_tokens = _tokenize(chunk["text"])
        is_duplicate = False
        for existing in deduplicated:
            existing_tokens = _tokenize(existing["text"])
            if chunk_tokens and existing_tokens:
                common = set(chunk_tokens) & set(existing_tokens)
                similarity = len(common) / max(len(chunk_tokens), len(existing_tokens))
                if similarity > 0.85:
                    is_duplicate = True
                    break
        if not is_duplicate:
            deduplicated.append(chunk)
    if len(scored) - len(deduplicated) > 0:
        print(f"  [DEDUP] Removed {len(scored) - len(deduplicated)} duplicate chunks")

    from urllib.parse import urlparse

    chunks_per_page = 5 if top_k >= 25 else 4 if top_k >= 15 else core.TOP_CHUNKS_PER_PAGE
    per_source = {}
    per_domain = {}
    selected = []

    for chunk in deduplicated:
        src = chunk["source_idx"]
        if per_source.get(src, 0) >= chunks_per_page:
            continue
        try:
            domain = urlparse(chunk.get("source_url", "")).netloc.replace("www.", "")
            if per_domain.get(domain, 0) >= 3 and selected:
                avg_rel = sum(c["relevance"] for c in selected) / len(selected)
                if chunk["relevance"] < avg_rel * 0.9:
                    continue
        except Exception:
            domain = "unknown"
        per_source[src] = per_source.get(src, 0) + 1
        per_domain[domain] = per_domain.get(domain, 0) + 1
        selected.append(chunk)
        if len(selected) >= top_k * 2:
            break

    selected = _mmr_diversify(query, selected, top_k=min(int(top_k * 1.5), len(selected)))
    selected = _cross_encoder_rerank(query, selected, top_k=top_k, lang=lang) if core.HAS_CROSS_ENCODER else selected[:top_k]
    grouped = _group_related_chunks(selected)

    if grouped:
        avg_relevance = sum(c["relevance"] for c in grouped) / len(grouped)
        for chunk in grouped:
            chunk["confidence"] = _calculate_confidence(chunk, avg_relevance, query=query)

        unique_domains = len(set(per_domain.keys()))
        print(f"  [DIVERSITY] {len(grouped)} chunks from {len(per_source)} sources, {unique_domains} domains")

    return grouped
