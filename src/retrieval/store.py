"""FAISS semantic index plus lightweight BM25-style keyword retrieval and rank fusion."""

import math
import re
from collections import Counter
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.config import settings
from src.ingestion.embeddings import get_embeddings
from src.ingestion.loader import load_faqs


class HybridRetriever:
    def __init__(self, documents=None, index=None):
        self.documents = documents or []
        self.index = index
        self.embeddings = get_embeddings()

    @classmethod
    def build(cls, documents=None):
        docs = documents or load_faqs()
        return cls(docs, FAISS.from_documents(docs, get_embeddings()))

    def save(self, path=None):
        if self.index:
            self.index.save_local(str(path or settings.index_path))

    @classmethod
    def load_or_build(cls):
        p = Path(settings.index_path)
        if p.exists():
            try:
                idx = FAISS.load_local(
                    str(p), get_embeddings(), allow_dangerous_deserialization=True
                )
                return cls(load_faqs(), idx)
            except Exception:
                pass
        obj = cls.build()
        obj.save()
        return obj

    @staticmethod
    def _tokens(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    def search(self, query, k=None):
        k = k or settings.top_k
        if self.index is None:
            self.index = FAISS.from_documents(self.documents, self.embeddings)
        sem = self.index.similarity_search_with_score(
            query, k=min(max(k * 3, k), max(1, len(self.documents)))
        )
        # BM25-like score, with length normalization and inverse document frequency.
        q = set(self._tokens(query))
        corpus = [self._tokens(d.page_content) for d in self.documents]
        n = max(1, len(corpus))
        df = Counter(t for terms in corpus for t in set(terms))
        scores = []
        avg = sum(map(len, corpus)) / n if corpus else 1
        for doc, terms in zip(self.documents, corpus):
            counts = Counter(terms)
            score = 0.0
            for term in q:
                if term in counts:
                    idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    score += (
                        idf
                        * counts[term]
                        * 2.2
                        / (counts[term] + 1.2 * (0.25 + 0.75 * len(terms) / (avg or 1)))
                    )
            scores.append((doc, score))
        sem_docs = [d for d, _ in sem]
        kw_docs = [
            d for d, s in sorted(scores, key=lambda x: x[1], reverse=True) if s > 0
        ][: max(k * 3, k)]
        # Reciprocal-rank fusion combines semantic and lexical ordering robustly.
        fused = {}
        for rank, doc in enumerate(sem_docs, 1):
            fused[doc.metadata["id"]] = [doc, 1 / (60 + rank)]
        for rank, doc in enumerate(kw_docs, 1):
            entry = fused.setdefault(doc.metadata["id"], [doc, 0.0])
            entry[1] += 1 / (60 + rank)
        query_terms = set(self._tokens(query)) - {
            "a",
            "about",
            "an",
            "and",
            "are",
            "can",
            "do",
            "for",
            "how",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }

        def relevance(entry):
            doc, fused_score = entry
            question_terms = set(self._tokens(doc.metadata.get("question", "")))
            question_coverage = len(query_terms & question_terms) / max(
                1, len(query_terms)
            )
            return fused_score + question_coverage * 0.03

        return [d for d, _ in sorted(fused.values(), key=relevance, reverse=True)[:k]]
