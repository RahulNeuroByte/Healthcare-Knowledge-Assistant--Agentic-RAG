from contextlib import asynccontextmanager
from uuid import uuid4
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from src.config import settings
from src.graph.workflow import build_graph
from src.graph.memory import get_history, remember
from src.tools.rag_tool import retriever
from src.retrieval.web_search import get_search_status
from api.schemas import QueryRequest, QueryResponse, FeedbackRequest


@asynccontextmanager
async def lifespan(app):
    app.state.graph = build_graph()
    yield


app = FastAPI(
    title="Healthcare Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
    description="Educational FAQ RAG. Not for diagnosis or emergencies.",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "web_search_configured": bool(settings.serpapi_key),
        "web_search_status": get_search_status(),
        "faq_count": len(retriever.documents),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    sid = req.session_id or str(uuid4())
    try:
        result = app.state.graph.invoke(
            {
                "query": req.query,
                "session_id": sid,
                "history": get_history(sid),
                "trace": [],
                "retries": 0,
            }
        )
        evidence_items = json.loads(result.get("evidence", "[]"))
        context = [
            item["text"]
            for item in evidence_items
            if item["id"] not in {"NOTICE-SEARCH-UNAVAILABLE", "NOTICE-FAQ-NO-MATCH"}
        ]
        source_ids = result.get("sources", [])
        source_map = {item["id"]: item.get("urls", []) for item in evidence_items}
        source_urls = list(
            dict.fromkeys(
                f"{source_id}: {url}"
                for source_id in source_ids
                for url in source_map.get(source_id, [])
            )
        )
        audit_path = Path(settings.feedback_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as audit:
            audit.write(
                json.dumps(
                    {
                        "event": "validator_verdict",
                        "session_id": sid,
                        "validation_passed": result.get("validation_passed", False),
                        "confidence_heuristic": result.get("confidence", 0.2),
                        "source_ids": source_ids,
                    }
                )
                + "\n"
            )
        remember(sid, req.query, result["answer"])
        return QueryResponse(
            session_id=sid,
            answer=result["answer"],
            sources=source_urls,
            retrieved_context=context,
            agents_invoked=result.get("trace", []),
            confidence=result.get("confidence", 0.2),
            validation_passed=result.get("validation_passed", False),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Query pipeline unavailable: {exc}"
        ) from exc


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    record = req.model_dump()
    record["created_at"] = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat()
    )
    p = Path(settings.feedback_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "status": "recorded",
        "message": "Feedback saved for offline review; it does not automatically alter medical content.",
    }
