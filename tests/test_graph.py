import pytest
from src.graph.workflow import build_graph


@pytest.fixture
def disable_web_search(monkeypatch):
    monkeypatch.setattr(
        "src.tools.rag_tool.search_web_health_sources", lambda *_args, **_kwargs: []
    )


def test_workflow_returns_grounded_trace(disable_web_search):
    result = build_graph().invoke(
        {
            "query": "How do vaccines help protect people?",
            "session_id": "test",
            "history": [],
            "trace": [],
            "retries": 0,
        }
    )
    assert (
        result["answer"]
        and "retriever_agent" in result["trace"]
        and "grounding_validator" in result["trace"]
    )
    assert result["sources"]


def test_workflow_answers_from_web_evidence_without_an_llm(monkeypatch):
    monkeypatch.setattr(
        "src.tools.rag_tool.search_web_health_sources",
        lambda *_args, **_kwargs: [
            {
                "id": "WEB-1",
                "text": "Cancer treatment depends on cancer type and stage.",
                "source": "National Cancer Institute",
                "urls": ["https://www.cancer.gov/about-cancer"],
            }
        ],
    )

    result = build_graph().invoke(
        {
            "query": "How is cancer treated?",
            "session_id": "test-web",
            "history": [],
            "trace": [],
            "retries": 0,
        }
    )

    assert "Cancer treatment depends on cancer type and stage." in result["answer"]
    assert "[WEB-1]" in result["answer"]
    assert result["sources"][0] == "WEB-1"


def test_workflow_falls_back_to_local_faq_when_web_search_is_blocked(monkeypatch):
    from src.retrieval.web_search import SearchUnavailableError

    def fail_search(*_args, **_kwargs):
        raise SearchUnavailableError("search is blocked")

    monkeypatch.setattr("src.tools.rag_tool.search_web_health_sources", fail_search)

    result = build_graph().invoke(
        {
            "query": "What should I do if online health advice conflicts with my clinician?",
            "session_id": "test-offline",
            "history": [],
            "trace": [],
            "retries": 0,
        }
    )

    assert result["answer"]
    assert "Live web search is temporarily unavailable" in result["answer"]
    assert "FAQ-085" in result["sources"]
    assert "NOTICE-SEARCH-UNAVAILABLE" not in result["sources"]


def test_workflow_does_not_answer_unrelated_faq_for_unsupported_topic(monkeypatch):
    from src.retrieval.web_search import SearchUnavailableError

    def fail_search(*_args, **_kwargs):
        raise SearchUnavailableError("search is blocked")

    monkeypatch.setattr("src.tools.rag_tool.search_web_health_sources", fail_search)

    result = build_graph().invoke(
        {
            "query": "How to cure cancer?",
            "session_id": "test-no-topic-match",
            "history": [],
            "trace": [],
            "retries": 0,
        }
    )

    assert "can't provide a cure or treatment plan" in result["answer"]
    assert "FAQ-089" in result["answer"]
    assert "FAQ-085" in result["answer"]
    assert "physical activity" not in result["answer"].lower()
    assert set(result["sources"]) == {"FAQ-089", "FAQ-085"}


def test_exact_local_question_returns_only_matching_faq(monkeypatch):
    monkeypatch.setattr(
        "src.tools.rag_tool.search_web_health_sources",
        lambda *_args, **_kwargs: [],
    )

    from src.tools.rag_tool import retrieve_health_evidence
    import json

    evidence = json.loads(
        retrieve_health_evidence.invoke(
            {"query": "How can I find reliable health information online?"}
        )
    )

    assert [item["id"] for item in evidence] == ["FAQ-009"]


def test_diagnosis_requests_are_bounded(disable_web_search):
    result = build_graph().invoke(
        {
            "query": "Can you diagnose what I have?",
            "session_id": "test",
            "history": [],
            "trace": [],
            "retries": 0,
        }
    )
    assert "cannot identify" in result["answer"].lower()
