from __future__ import annotations

import json
import re
from collections import Counter

from bs4 import BeautifulSoup


def _fallback_extract(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select(
        "script, style, nav, footer, header, aside, iframe, noscript, "
        "form, button, [role='navigation'], [role='banner'], "
        "[role='complementary'], .sidebar, .menu, .nav, .ad, .ads, "
        ".cookie, .popup, .modal"
    ):
        tag.decompose()

    main = soup.select_one("article") or soup.select_one("main") or soup.select_one("[role='main']")
    target = main if main else soup.body if soup.body else soup

    lines = []
    for el in target.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "blockquote", "pre"]):
        text = el.get_text(" ", strip=True)
        if len(text) > 20:
            lines.append(text)
    return "\n\n".join(lines)


def _extract_publish_date(html):
    from datetime import datetime

    soup = BeautifulSoup(html, "html.parser")
    date_str = None

    meta_selectors = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "publication_date"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"property": "og:published_time"}),
        ("meta", {"name": "date"}),
        ("meta", {"itemprop": "datePublished"}),
        ("time", {"datetime": True}),
    ]

    for tag, attrs in meta_selectors:
        elem = soup.find(tag, attrs)
        if elem:
            date_str = elem.get("content") or elem.get("datetime")
            if date_str:
                break

    if not date_str:
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    date_str = data.get("datePublished") or data.get("dateCreated")
                    if date_str:
                        break
            except Exception:
                pass

    if date_str:
        try:
            if "T" in date_str or "-" in date_str:
                date_str = date_str.split("+")[0].split("Z")[0]
                return datetime.fromisoformat(date_str.replace("T", " ")[:19])
        except Exception:
            pass

    return None


def extract_content(html, url=None):
    from . import core

    if core.HAS_TRAFILATURA:
        text = core.trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
            output_format="markdown",
        )
        if text and len(text) > 100:
            return text

    return _fallback_extract(html)


def _semantic_chunking(text, max_chunk_size=None, lang="en"):
    from . import core
    from .rerank import _get_embedding_model

    if max_chunk_size is None:
        max_chunk_size = core.CHUNK_SIZE
    if not core.HAS_EMBEDDINGS:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 3:
        return [text]

    try:
        model = _get_embedding_model(lang)
        if not model:
            return None

        embeddings = model.encode(sentences, convert_to_tensor=False, show_progress_bar=False)

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return None

        similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = np.array(embeddings[i]).reshape(1, -1)
            emb2 = np.array(embeddings[i + 1]).reshape(1, -1)
            similarities.append(cosine_similarity(emb1, emb2)[0][0])

        if not similarities:
            return None

        threshold = np.percentile(similarities, 30)
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])

        for i, sent in enumerate(sentences[1:], 1):
            sent_len = len(sent)
            topic_changed = i - 1 < len(similarities) and similarities[i - 1] < threshold
            size_exceeded = current_size + sent_len > max_chunk_size

            if topic_changed or size_exceeded:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_size = sent_len
            else:
                current_chunk.append(sent)
                current_size += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks if len(chunks) > 1 else None
    except Exception:
        return None


def chunk_text(text, chunk_size=None, overlap=None, lang="en"):
    from . import core

    if chunk_size is None:
        chunk_size = core.CHUNK_SIZE
    if overlap is None:
        overlap = core.CHUNK_OVERLAP
    if not text:
        return []

    semantic_chunks = _semantic_chunking(text, chunk_size, lang=lang)
    if semantic_chunks:
        print(f"  [SEMANTIC] Created {len(semantic_chunks)} semantic chunks")
        return semantic_chunks

    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

        while len(current) > chunk_size * 1.5:
            split_pos = chunk_size
            for delim in [". ", "! ", "? ", ".\n", ";\n", "\n"]:
                pos = current.rfind(delim, 0, chunk_size + 50)
                if pos > chunk_size * 0.3:
                    split_pos = pos + len(delim)
                    break

            chunk_part = current[:split_pos].strip()
            if chunk_part:
                chunks.append(chunk_part)

            remainder = current[split_pos:].strip()
            if overlap > 0 and len(chunk_part) > overlap:
                current = chunk_part[-overlap:] + " " + remainder
            else:
                current = remainder

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if len(chunk) > 40]


def _is_incomplete_chunk(text):
    if not text or len(text) < 50:
        return True

    text_stripped = text.strip()
    if len(text_stripped) < 100 and text_stripped[0].islower():
        return True
    if text_stripped[0] in ".,-—–…":
        return True
    if text_stripped.endswith("-") or text_stripped.endswith("—"):
        return True

    if len(text_stripped) < 150 and text_stripped[-1] not in '.!?"»':
        words = text_stripped.split()
        if words and words[-1][-1].isalpha() and len(text_stripped) < 100:
            return True
    return False


def _remove_garbage_lines(text):
    lines = text.split("\n")
    cleaned_lines = []
    garbage_patterns = [
        "подписаться", "подпис", "subscribe", "sign up", "telegram", "whatsapp",
        "вконтакте", "в max", "следите за нами", "follow us", "поделиться",
        "share on", "tweet", "facebook", "twitter", "комментар", "comment",
        "оставьте отзыв", "читайте также", "read also", "related articles",
        "рекомендуем", "recommended", "похожие статьи", "новости края", "новости региона",
    ]

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        line_lower = line_stripped.lower()
        if len(line_stripped) < 10 and not any(c.isdigit() for c in line_stripped):
            continue
        if any(pattern in line_lower for pattern in garbage_patterns) and len(line_stripped) < 50:
            continue
        cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)


def _is_low_quality_chunk(text):
    if not text or len(text) < 50:
        return True

    words = text.split()
    if not words:
        return True
    if len(words) / len(text) < 0.08:
        return True

    text_lower = text.lower()
    boilerplate_patterns = [
        "cookie policy", "privacy policy", "terms of service", "all rights reserved", "© 20",
        "copyright ©", "subscribe to our newsletter", "sign up for", "follow us on",
        "share this article", "advertisement", "sponsored content", "click here to",
        "read more »", "loading...", "please wait", "javascript is disabled",
        "enable javascript", "accept cookies", "we use cookies", "get latest news",
        "in your inbox", "sign up for our", "newsletter", "get the latest",
        "subscribe now", "join our newsletter", "email updates", "daily digest",
    ]
    if sum(1 for pattern in boilerplate_patterns if pattern in text_lower) >= 2:
        return True

    word_counts = Counter(w.lower() for w in words if len(w) > 3)
    if word_counts and max(word_counts.values()) > len(words) * 0.3:
        return True

    long_words = [w for w in words if len(w) > 4]
    if len(long_words) / len(words) < 0.2:
        return True
    return False


def is_content_page(text, query=None, lang="en", min_avg_sentence_len=40, min_sentences=5, relevance_threshold=0.35):
    from . import core
    from .rerank import _get_embedding_model

    if not text or len(text) < 200:
        return False

    lines = text.split("\n")
    numeric_lines = sum(1 for line in lines if re.match(r"^\s*[\d.,₽$€£¥₸\s\-—]+\s*$", line.strip()))
    looks_like_pricelist = numeric_lines / max(len(lines), 1) > 0.3

    if not looks_like_pricelist:
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 20]
        if len(sentences) < min_sentences:
            return False
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        return avg_len >= min_avg_sentence_len

    if query and core.HAS_EMBEDDINGS:
        try:
            model = _get_embedding_model(lang)
            if model:
                preview = text[:500]
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity

                query_emb = model.encode(query, convert_to_tensor=False, show_progress_bar=False)
                text_emb = model.encode(preview, convert_to_tensor=False, show_progress_bar=False)
                sim = cosine_similarity(np.array(query_emb).reshape(1, -1), np.array(text_emb).reshape(1, -1))[0][0]

                if sim >= relevance_threshold:
                    print(f"  [CONTENT-CHECK] Price list structure but RELEVANT (sim={sim:.3f}) - keeping")
                    return True
                print(f"  [CONTENT-CHECK] Price list structure and NOT relevant (sim={sim:.3f}) - rejecting")
        except Exception:
            pass
    return False


def filter_low_quality_chunks(chunks):
    cleaned = []
    for chunk in chunks:
        cleaned_text = _remove_garbage_lines(chunk)
        if cleaned_text and len(cleaned_text) >= 50:
            cleaned.append(cleaned_text)

    filtered = []
    for chunk in cleaned:
        if not _is_incomplete_chunk(chunk) and not _is_low_quality_chunk(chunk):
            filtered.append(chunk)

    removed_count = len(chunks) - len(filtered)
    if removed_count > 0:
        print(f"  [FILTER] Removed {removed_count} low-quality/incomplete chunks")

    return filtered
