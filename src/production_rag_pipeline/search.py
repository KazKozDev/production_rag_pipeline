from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, quote_plus, urlencode, urlparse

from bs4 import BeautifulSoup

from .fetch import _get


def _bing_unwrap_url(href):
    if "bing.com/ck/a" not in href:
        return href
    try:
        parsed = parse_qs(urlparse(href).query)
        if "u" in parsed:
            raw = parsed["u"][0]
            if raw.startswith("a1"):
                raw = raw[2:]
            raw = raw.replace("-", "+").replace("_", "/")
            padding = 4 - len(raw) % 4
            if padding != 4:
                raw += "=" * padding
            decoded = base64.b64decode(raw).decode("utf-8", errors="ignore")
            if decoded.startswith("http"):
                return decoded
        if "r" in parsed and parsed["r"][0].startswith("http"):
            return parsed["r"][0]
    except Exception:
        pass
    return href


def search_bing(query, num=None, lang="en", debug=False):
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    params = {"q": query, "count": min(num + 5, 30), "setlang": lang, "cc": lang}
    if lang == "en":
        params["setmkt"] = "en-US"

    url = f"https://www.bing.com/search?{urlencode(params)}"
    if debug:
        print(f"[BING] {url}")

    try:
        resp = _get(url, lang)
    except Exception as exc:
        print(f"[BING] Request failed: {exc}")
        return []

    if debug:
        with open("debug_bing.html", "w", encoding="utf-8") as handle:
            handle.write(resp.text)
        print(f"[BING] Status {resp.status_code}, {len(resp.text)} bytes → debug_bing.html")

    if resp.status_code != 200:
        print(f"[BING] HTTP {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    def _add(title, href, snippet=""):
        if not title or not href:
            return
        real_url = _bing_unwrap_url(href)
        if not real_url.startswith("http"):
            return
        host = urlparse(real_url).hostname or ""
        if host.endswith("bing.com") or host.endswith("microsoft.com"):
            return
        norm = real_url.split("?")[0].split("#")[0].rstrip("/").lower()
        if norm in seen:
            return
        seen.add(norm)
        results.append({"title": title.strip(), "url": real_url, "snippet": (snippet or "").strip()})

    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a") or li.select_one("a[href]")
        if not a or not a.get("href", ""):
            continue
        title = a.get_text(strip=True)
        link = a["href"]
        snippet = ""
        for sel in ["div.b_caption p", "p.b_lineclamp2", "p.b_lineclamp3", "p.b_lineclamp4", "div.b_caption .b_snippet"]:
            el = li.select_one(sel)
            if el:
                snippet = el.get_text(" ", strip=True)
                break
        if not snippet:
            cap = li.select_one("div.b_caption")
            if cap:
                snippet = cap.get_text(" ", strip=True)[:300]
        _add(title, link, snippet)

    if not results:
        for h2 in soup.select("h2"):
            a = h2.select_one("a[href]")
            if a and a.get("href", ""):
                _add(a.get_text(strip=True), a["href"])

    if debug:
        print(f"[BING] {len(results)} results")
    return results[:num]


def _ddg_extract_real_url(href):
    if "duckduckgo.com" in href and "uddg=" in href:
        parsed = parse_qs(urlparse(href).query)
        if "uddg" in parsed:
            return parsed["uddg"][0]
    return href


def search_ddg(query, num=None, lang="en", debug=False, news_mode=False):
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    if news_mode:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&iar=news&ia=news"
    else:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    if debug:
        print(f"[DDG] {url} (news_mode={news_mode})")

    try:
        resp = _get(url, lang)
    except Exception as exc:
        print(f"[DDG] Request failed: {exc}")
        return []

    if debug:
        with open("debug_ddg.html", "w", encoding="utf-8") as handle:
            handle.write(resp.text)
        print(f"[DDG] Status {resp.status_code}, {len(resp.text)} bytes → debug_ddg.html")

    if resp.status_code != 200:
        print(f"[DDG] HTTP {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for div in soup.select("div.result, div.web-result"):
        a = div.select_one("a.result__a")
        if not a:
            continue
        title = a.get_text(strip=True)
        link = _ddg_extract_real_url(a.get("href", ""))
        sn = div.select_one("a.result__snippet, div.result__snippet")
        snippet = sn.get_text(strip=True) if sn else ""
        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= num:
            break

    if debug:
        print(f"[DDG] {len(results)} results")
    return results[:num]


def _normalize_url(url):
    url = url.split("?")[0].split("#")[0]
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("www.", "")
    return url.rstrip("/").lower()


def _detect_language(text):
    if not text:
        return "other"

    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isalpha() and c.isascii())

    total_alpha = cyrillic + chinese + latin
    if total_alpha == 0:
        return "other"
    if chinese / total_alpha > 0.3:
        return "zh"
    if cyrillic / total_alpha > 0.1:
        return "ru"
    if latin / total_alpha > 0.5:
        return "en"
    return "other"


def merge_results(bing_results, ddg_results, num=20, target_lang=None):
    merged = {}

    for engine_name, results in [("bing", bing_results), ("ddg", ddg_results)]:
        for rank, result in enumerate(results, 1):
            if target_lang:
                detected_lang = _detect_language(result["title"])
                if detected_lang not in [target_lang, "other"]:
                    continue

            norm = _normalize_url(result["url"])
            position_score = 1.0 / rank
            if norm in merged:
                merged[norm]["score"] += position_score
                merged[norm]["engines"].add(engine_name)
                if len(result["snippet"]) > len(merged[norm]["snippet"]):
                    merged[norm]["snippet"] = result["snippet"]
                if len(result["title"]) > len(merged[norm]["title"]):
                    merged[norm]["title"] = result["title"]
            else:
                merged[norm] = {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "score": position_score,
                    "engines": {engine_name},
                }

    for entry in merged.values():
        if len(entry["engines"]) >= 2:
            entry["score"] *= 1.3

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return [
        {
            "title": entry["title"],
            "url": entry["url"],
            "snippet": entry["snippet"],
            "score": round(entry["score"], 3),
            "engines": sorted(entry["engines"]),
        }
        for entry in ranked[:num]
    ]


def search(query, num=20, lang="en", debug=False, config=None, config_path=None):
    from . import core
    from .rerank import _is_news_query

    core._apply_runtime_config(config=config, config_path=config_path)
    bing_r = []
    ddg_r = []

    is_news = _is_news_query(query)
    if is_news:
        print("[SEARCH] News query detected → DDG news mode enabled")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(search_bing, query, core.NUM_PER_ENGINE, lang, debug): "bing",
            pool.submit(search_ddg, query, core.NUM_PER_ENGINE, lang, debug, news_mode=is_news): "ddg",
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                results = future.result()
                if engine == "bing":
                    bing_r = results
                else:
                    ddg_r = results
            except Exception as exc:
                print(f"[{engine.upper()}] Error: {exc}")

    print(f"[MERGE] Bing: {len(bing_r)}, DDG: {len(ddg_r)} → merging")
    return merge_results(bing_r, ddg_r, num=num, target_lang=lang)
