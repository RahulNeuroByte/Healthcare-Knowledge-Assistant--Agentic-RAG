"""Deterministic local hashing embeddings; no external embedding service required."""

import hashlib
import math
import re
from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dimensions
        words = re.findall(r"[a-z0-9]+", text.lower())
        for word in words:
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            v[index] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    return HashEmbeddings()
