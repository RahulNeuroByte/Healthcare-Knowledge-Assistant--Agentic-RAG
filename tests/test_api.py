from fastapi.testclient import TestClient
from api.main import app


def test_health():
    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 200
        assert "web_search_configured" in response.json()
        assert "web_search_status" in response.json()


def test_query_and_feedback(monkeypatch):
    monkeypatch.setattr(
        "src.tools.rag_tool.search_web_health_sources",
        lambda *_args, **_kwargs: [
            {
                "id": "WEB-1",
                "text": "Vaccines help your immune system recognize and fight pathogens.",
                "source": "CDC",
                "urls": ["https://www.cdc.gov/vaccines/"],
            }
        ],
    )
    with TestClient(app) as c:
        r = c.post("/query", json={"query": "How do vaccines help protect people?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] and body["sources"] and body["agents_invoked"]
        assert "WEB-1" in body["answer"]
        assert "WEB-1: https://www.cdc.gov/vaccines/" in body["sources"]
        f = c.post(
            "/feedback",
            json={"query": "How do vaccines help protect people?", "rating": 1},
        )
        assert f.status_code == 200
