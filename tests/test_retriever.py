from src.retrieval.store import HybridRetriever
from src.ingestion.loader import load_faqs


def test_has_100_faqs():
    assert len(load_faqs()) == 100


def test_retrieval_finds_vaccine_topic():
    r = HybridRetriever.build()
    docs = r.search("vaccination schedule for adults", k=3)
    assert docs and any("vaccine" in d.page_content.lower() for d in docs)


def test_retrieval_prefers_exact_health_information_faq():
    docs = HybridRetriever.build().search(
        "How can I find reliable health information online?", k=4
    )

    assert docs[0].metadata["id"] == "FAQ-009"


def test_metadata_sources_present():
    d = HybridRetriever.build().search(
        "how can I find reliable health information", k=1
    )[0]
    assert d.metadata["source_urls"]
