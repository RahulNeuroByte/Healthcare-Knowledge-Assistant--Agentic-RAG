"""Healthcare specialist functions. These are decision-support roles, not clinicians."""

from src.tools.rag_tool import retrieve_health_evidence


def retriever_agent(query, k=4):
    return retrieve_health_evidence.invoke({"query": query, "k": k})


def consultation_agent(query, evidence):
    import json

    items = json.loads(evidence)
    if not items:
        return "I could not find relevant material in the knowledge base. Please consult a qualified healthcare professional."
    search_unavailable = any(
        item["id"] == "NOTICE-SEARCH-UNAVAILABLE" for item in items
    )
    no_topic_match = any(item["id"] == "NOTICE-FAQ-NO-MATCH" for item in items)
    items = [
        item
        for item in items
        if item["id"] not in {"NOTICE-SEARCH-UNAVAILABLE", "NOTICE-FAQ-NO-MATCH"}
    ]
    if no_topic_match:
        return (
            "Live web search is unavailable, and the local FAQ library has no "
            "topic-specific answer, so I can't provide a cure or treatment plan. "
            "Please discuss cancer care with a qualified clinician. You can ask "
            "about getting a second medical opinion (FAQ-089), and do not change "
            "treatment based only on online claims; ask your clinician to explain "
            "the evidence (FAQ-085)."
        )
    web_items = [item for item in items if item["id"].startswith("WEB-")]
    if web_items:
        excerpts = "\n\n".join(
            f"- {item['text']} [{item['id']}]" for item in web_items[:3]
        )
        return (
            "Here is general information from current, selected health sources. "
            "This is educational, not a diagnosis or personal treatment plan.\n\n"
            f"{excerpts}\n\n"
            "For personal medical decisions, speak with a qualified healthcare "
            "professional. Seek urgent care for severe or rapidly worsening symptoms."
        )
    fallback_notice = (
        "Live web search is temporarily unavailable, so this reply uses only the "
        "local FAQ library and may not fully answer your question. "
        if search_unavailable
        else ""
    )
    return fallback_notice + "\n\n".join(f"{x['text']} [{x['id']}]" for x in items[:2])


def diagnosis_support_agent(query, evidence):
    # Deliberately bounded: refuses individualized diagnosis and reframes toward professional assessment.
    return "I cannot identify or rank possible conditions for an individual. A qualified clinician can assess symptoms, history, and examination findings. I can share general educational material from the cited sources."
