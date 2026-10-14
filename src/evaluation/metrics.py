"""Dependency-light BLEU-1, ROUGE-L F1, and evidence relevance metrics."""

import re
from collections import Counter


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bleu1(reference, prediction):
    ref, hyp = tokens(reference), tokens(prediction)
    if not hyp:
        return 0.0
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    precision = overlap / len(hyp)
    bp = (
        1.0
        if len(hyp) >= len(ref)
        else __import__("math").exp(1 - len(ref) / max(1, len(hyp)))
    )
    return bp * precision


def rouge_l(reference, prediction):
    a, b = tokens(reference), tokens(prediction)
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    lcs = dp[-1]
    if not a or not b:
        return 0.0
    p = lcs / len(b)
    r = lcs / len(a)
    return 2 * p * r / (p + r) if p + r else 0.0


def relevance(query, context):
    q = set(tokens(query))
    c = set(tokens(context))
    return len(q & c) / max(1, len(q))
