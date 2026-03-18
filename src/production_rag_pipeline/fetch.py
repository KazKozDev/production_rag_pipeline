from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests as http


def _imp():
    from . import core

    return random.choice(core.IMPERSONATE)


def _headers(lang="en"):
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": f"{lang},en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _get(url, lang="en", cookies=None, max_retries=2, timeout=15):
    """Single HTTP GET with TLS impersonation and retries."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return http.get(
                url,
                headers=_headers(lang),
                cookies=cookies,
                impersonate=_imp(),
                timeout=timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(random.uniform(0.5, 1.5))
    raise last_err


def fetch_page(url, lang="en"):
    """Fetch a single page → return (url, raw_html, publish_date, error)."""
    from . import core
    from .extract import _extract_publish_date

    try:
        resp = _get(url, lang, timeout=core.FETCH_TIMEOUT, max_retries=1)
        if resp.status_code == 200:
            pub_date = _extract_publish_date(resp.text)
            return (url, resp.text, pub_date, None)
        return (url, None, None, f"HTTP {resp.status_code}")
    except Exception as exc:
        return (url, None, None, str(exc))


def fetch_pages_parallel(urls, query=None, lang="en"):
    """Fetch multiple pages in parallel and return extracted text by URL."""
    from . import core
    from .extract import extract_content, is_content_page

    results = {}

    with ThreadPoolExecutor(max_workers=core.FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_page, url, lang): url for url in urls}

        for future in as_completed(futures):
            url = futures[future]
            try:
                fetched_url, html, pub_date, err = future.result()
                if err:
                    print(f"  [FETCH] ✗ {url[:60]}... — {err}")
                    continue

                try:
                    text = extract_content(html, url=fetched_url)
                    if text and len(text) > 50:
                        if not is_content_page(text, query=query, lang=lang):
                            print(f"  [FETCH] ✗ {url[:60]}... — low-quality content (navigation/price list)")
                            continue

                        text = text[:core.MAX_CONTENT_CHARS]
                        results[fetched_url] = {"text": text, "pub_date": pub_date}
                        date_str = pub_date.strftime("%Y-%m-%d") if pub_date else "unknown"
                        print(f"  [FETCH] ✓ {url[:60]}... — {len(text)} chars, date={date_str}")
                    else:
                        print(f"  [FETCH] ✗ {url[:60]}... — empty after extraction")
                except Exception as extract_err:
                    print(f"  [FETCH] ✗ {url[:60]}... — extraction failed: {extract_err}")
            except Exception as exc:
                print(f"  [FETCH] ✗ {url[:60]}... — {exc}")

    return results
