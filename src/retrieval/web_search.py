"""Search current health information from a small set of trusted publishers."""

import html
import re
from urllib.parse import urlsplit

import requests

from src.config import settings

SERPAPI_URL = "https://serpapi.com/search.json"
TRUSTED_DOMAINS = (
    "cdc.gov",
    "clevelandclinic.org",
    "fda.gov",
    "healthdirect.gov.au",
    "mayoclinic.org",
    "medlineplus.gov",
    "nih.gov",
    "nhs.uk",
    "who.int",
    "cancer.gov",
)
_search_status = (
    "not_configured" if not settings.serpapi_key else "configured_not_tested"
)


class SearchUnavailableError(RuntimeError):
    """The live search service could not be reached."""


def get_search_status() -> str:
    """Return a sanitized status for the last live-search attempt."""
    return _search_status


def _is_trusted_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme == "https" and any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in TRUSTED_DOMAINS
    )


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", text).strip()


def search_web_health_sources(query: str, k: int = 4) -> list[dict[str, str]]:
    """Return cited search snippets from trusted health publishers only."""
    global _search_status
    if not settings.serpapi_key:
        _search_status = "not_configured"
        return []

    _search_status = "searching"
    site_filter = " OR ".join(f"site:{domain}" for domain in TRUSTED_DOMAINS)
    try:
        response = requests.get(
            SERPAPI_URL,
            params={
                "engine": "google",
                "q": f"{query} ({site_filter})",
                "api_key": settings.serpapi_key,
                "num": min(max(k * 2, k), 10),
            },
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        if isinstance(exc, requests.exceptions.SSLError):
            _search_status = "certificate_error"
        elif isinstance(exc, requests.HTTPError) and exc.response is not None:
            _search_status = f"http_{exc.response.status_code}"
        else:
            _search_status = "network_error"
        raise SearchUnavailableError(
            "Trusted-source web search is unavailable; please retry shortly."
        ) from exc
    except ValueError as exc:
        _search_status = "invalid_response"
        raise RuntimeError("Trusted-source web search returned invalid data.") from exc

    if not isinstance(payload, dict) or payload.get("error"):
        _search_status = "provider_error"
        raise RuntimeError("Trusted-source web search returned an error.")

    _search_status = "connected"
    results = []
    seen_urls = set()
    organic_results = payload.get("organic_results", [])
    if not isinstance(organic_results, list):
        return results
    for item in organic_results:
        if not isinstance(item, dict):
            continue
        url = item.get("link", "")
        title = _clean_text(item.get("title", ""))
        snippet = _clean_text(item.get("snippet", ""))
        if not _is_trusted_url(url) or not title or not snippet or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(
            {
                "id": f"WEB-{len(results) + 1}",
                "text": f"{title}: {snippet}",
                "source": title,
                "urls": [url],
            }
        )
        if len(results) >= k:
            break
    return results
