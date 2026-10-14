"""Simple in-process, session-scoped conversation memory; replace with durable storage for production."""

from collections import defaultdict, deque

_SESSIONS = defaultdict(lambda: deque(maxlen=8))


def get_history(session_id):
    return list(_SESSIONS[session_id])


def remember(session_id, query, answer):
    _SESSIONS[session_id].append({"query": query, "answer": answer})
