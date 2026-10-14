"""RAG is exposed as a reusable LangChain tool callable by specialist agents."""

import json
import logging
import re
from langchain_core.tools import tool
from src.retrieval.web_search import (
    SearchUnavailableError,
    search_web_health_sources,
)
from src.retrieval.store import HybridRetriever

logger = logging.getLogger(__name__)
retriever = HybridRetriever.load_or_build()
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "do",
    "for",
    "get",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "how",
}


@tool
def retrieve_health_evidence(query: str, k: int = 4) -> str:
    """Retrieve trusted current web evidence and relevant local FAQ material."""
    search_unavailable = False
    try:
        web_results = search_web_health_sources(query, k=k)
    except SearchUnavailableError:
        logger.warning("Live health search unavailable; falling back to local FAQs.")
        web_results = []
        search_unavailable = True

    docs = retriever.search(query, k=k)
    local_results = [
        {
            "text": d.page_content,
            "id": d.metadata["id"],
            "source": d.metadata["source"],
            "urls": d.metadata["source_urls"],
        }
        for d in docs
    ]
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOPWORDS
    doc_questions = {d.metadata["id"]: d.metadata["question"] for d in docs}
    exact_question_matches = (
        [
            item
            for item in local_results
            if query_terms
            <= set(re.findall(r"[a-z0-9]+", doc_questions[item["id"]].lower()))
        ]
        if query_terms
        else []
    )
    if exact_question_matches:
        local_results = exact_question_matches
    if search_unavailable:
        relevant_results = [
            item
            for item in local_results
            if query_terms & set(re.findall(r"[a-z0-9]+", item["text"].lower()))
        ]
        if not relevant_results:
            fallback_ids = {"FAQ-085", "FAQ-089"}
            local_results = [
                {
                    "text": d.page_content,
                    "id": d.metadata["id"],
                    "source": d.metadata["source"],
                    "urls": d.metadata["source_urls"],
                }
                for d in retriever.documents
                if d.metadata["id"] in fallback_ids
            ]
            local_results.insert(
                0,
                {
                    "text": "The local FAQ library has no topic-specific answer for this question.",
                    "id": "NOTICE-FAQ-NO-MATCH",
                    "source": "Local FAQ coverage notice",
                    "urls": [],
                },
            )
        else:
            local_results = relevant_results
        local_results.insert(
            0,
            {
                "text": "Live web search is temporarily unavailable; this response uses only the local FAQ library.",
                "id": "NOTICE-SEARCH-UNAVAILABLE",
                "source": "Search availability notice",
                "urls": [],
            },
        )
    return json.dumps(web_results + local_results, ensure_ascii=False)
