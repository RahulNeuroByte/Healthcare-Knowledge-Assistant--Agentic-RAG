"""Load the vetted FAQ corpus and convert records to LangChain documents."""

import json
from pathlib import Path
from langchain_core.documents import Document
from src.config import settings


def load_faqs(path=None):
    records = json.loads(Path(path or settings.data_path).read_text(encoding="utf-8"))
    return [
        Document(
            page_content=f"Question: {r['question']}\nAnswer: {r['answer']}\nCategory: {r['category']}",
            metadata={
                "id": r["id"],
                "category": r["category"],
                "source": r["source"],
                "source_urls": r["source_urls"],
                "question": r["question"],
            },
        )
        for r in records
    ]


def chunk_documents(docs, chunk_size=700, overlap=100):
    # FAQ documents are kept whole; split only unusually long items, preserving metadata.
    out = []
    for doc in docs:
        text = doc.page_content
        if len(text) <= chunk_size:
            out.append(doc)
            continue
        start = 0
        while start < len(text):
            out.append(
                Document(
                    page_content=text[start : start + chunk_size],
                    metadata=dict(doc.metadata),
                )
            )
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap
    return out
