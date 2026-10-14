"""Planner → retriever tool → consultation specialist → grounding validator LangGraph."""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.config import settings
from src.agents.specialists import (
    retriever_agent,
    consultation_agent,
    diagnosis_support_agent,
)


class RAGState(TypedDict, total=False):
    query: str
    session_id: str
    history: list
    plan: str
    evidence: str
    answer: str
    confidence: float
    sources: list
    trace: list
    retries: int
    valid: bool


def _lexical_overlap(query, evidence):
    import re

    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    e = set(re.findall(r"[a-z0-9]+", evidence.lower()))
    return len(q & e) / max(1, len(q))


def build_graph():
    def planner(s):
        q = s["query"].strip()
        kind = (
            "safety escalation"
            if any(
                x in q.lower() for x in ["diagnos", "what do i have", "symptoms mean"]
            )
            else "general health education"
        )
        return {
            "plan": f"Classify as {kind}; retrieve public FAQ evidence; answer only within evidence; validate grounding.",
            "trace": s.get("trace", []) + ["planner"],
        }

    def retrieve(s):
        k = settings.top_k * (2 if s.get("retries", 0) > 0 else 1)
        evidence = retriever_agent(s["query"], k)
        return {
            "evidence": evidence,
            "trace": s.get("trace", []) + ["retriever_agent", "rag_tool"],
        }

    def consult(s):
        import json

        if "diagnos" in s["query"].lower() or "what do i have" in s["query"].lower():
            answer = diagnosis_support_agent(s["query"], s.get("evidence", ""))
        else:
            answer = consultation_agent(s["query"], s.get("evidence", "[]"))
        try:
            sources = list(
                dict.fromkeys(
                    x["id"]
                    for x in json.loads(s.get("evidence", "[]"))
                    if x["id"]
                    not in {"NOTICE-SEARCH-UNAVAILABLE", "NOTICE-FAQ-NO-MATCH"}
                )
            )
        except Exception:
            sources = []
        return {
            "answer": answer,
            "sources": sources,
            "trace": s.get("trace", []) + ["consultation_agent"],
        }

    def validate(s):
        import json

        evidence = s.get("evidence", "")
        overlap = _lexical_overlap(s["query"], evidence)
        try:
            count = len(json.loads(evidence))
        except Exception:
            count = 0
        valid = count > 0 and overlap >= 0.04
        confidence = min(0.99, 0.45 + 0.12 * count + 0.35 * overlap) if valid else 0.2
        return {
            "valid": valid,
            "validation_passed": valid,
            "confidence": confidence,
            "trace": s.get("trace", []) + ["grounding_validator"],
        }

    def route(s):
        if not s.get("valid") and s.get("retries", 0) < settings.max_retries:
            return "retry"
        return "done"

    def retry(s):
        return {
            "retries": s.get("retries", 0) + 1,
            "trace": s.get("trace", []) + ["query_rewrite_retry"],
        }

    g = StateGraph(RAGState)
    for name, fn in [
        ("planner", planner),
        ("retriever", retrieve),
        ("consultation", consult),
        ("validator", validate),
        ("retry", retry),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "consultation")
    g.add_edge("consultation", "validator")
    g.add_conditional_edges("validator", route, {"retry": "retry", "done": END})
    g.add_edge("retry", "retriever")
    return g.compile()
