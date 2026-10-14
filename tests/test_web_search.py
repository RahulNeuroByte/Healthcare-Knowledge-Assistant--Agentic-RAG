import pytest
from src.retrieval import web_search


def test_serpapi_search_uses_configured_key_and_trusted_sources(monkeypatch):
    monkeypatch.setattr(web_search.settings, "serpapi_key", "test-key")

    class StubResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "organic_results": [
                    {
                        "title": "Trusted result",
                        "snippet": "Evidence from a trusted source.",
                        "link": "https://www.cancer.gov/about-cancer",
                    },
                    {
                        "title": "Untrusted result",
                        "snippet": "Do not include this.",
                        "link": "https://example.com/health",
                    },
                    {
                        "title": "Unsafe scheme",
                        "snippet": "Do not include this either.",
                        "link": "http://www.cdc.gov/health",
                    },
                ]
            }

    def fake_get(url, *, params, timeout):
        assert url == web_search.SERPAPI_URL
        assert params["api_key"] == "test-key"
        assert "cancer.gov" in params["q"]
        assert timeout == 12
        return StubResponse()

    monkeypatch.setattr(web_search.requests, "get", fake_get)

    results = web_search.search_web_health_sources("cancer treatment", k=3)

    assert results == [
        {
            "id": "WEB-1",
            "text": "Trusted result: Evidence from a trusted source.",
            "source": "Trusted result",
            "urls": ["https://www.cancer.gov/about-cancer"],
        }
    ]
    assert web_search.get_search_status() == "connected"


def test_search_without_key_skips_external_request(monkeypatch):
    monkeypatch.setattr(web_search.settings, "serpapi_key", "")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Search should be skipped when no key is configured")

    monkeypatch.setattr(web_search.requests, "get", fail_if_called)

    assert web_search.search_web_health_sources("general health", k=3) == []


def test_search_request_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr(web_search.settings, "serpapi_key", "test-key")

    def fail_request(*_args, **_kwargs):
        raise web_search.requests.Timeout("request URL contains private parameters")

    monkeypatch.setattr(web_search.requests, "get", fail_request)

    with pytest.raises(web_search.SearchUnavailableError) as error:
        web_search.search_web_health_sources("health question")
    assert str(error.value) == (
        "Trusted-source web search is unavailable; please retry shortly."
    )
    assert web_search.get_search_status() == "network_error"


def test_search_reports_certificate_policy_failure(monkeypatch):
    monkeypatch.setattr(web_search.settings, "serpapi_key", "test-key")
    monkeypatch.setattr(
        web_search.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            web_search.requests.exceptions.SSLError("certificate check failed")
        ),
    )

    with pytest.raises(web_search.SearchUnavailableError):
        web_search.search_web_health_sources("health question")

    assert web_search.get_search_status() == "certificate_error"
