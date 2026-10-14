"""Run transparent baseline-vs-agentic comparison on selected FAQ questions."""

import json
import csv
from pathlib import Path
from src.config import settings
from src.retrieval.store import HybridRetriever
from src.evaluation.metrics import bleu1, rouge_l, relevance
from src.graph.workflow import build_graph


def run():
    rows = json.loads(Path(settings.data_path).read_text(encoding="utf-8"))[:20]
    retriever = HybridRetriever.load_or_build()
    graph = build_graph()
    results = []
    for row in rows:
        docs = retriever.search(row["question"], k=2)
        baseline = " ".join(d.page_content for d in docs[:1])
        state = graph.invoke(
            {
                "query": row["question"],
                "session_id": "eval",
                "history": [],
                "trace": [],
                "retries": 0,
            }
        )
        for mode, pred in [("baseline", baseline), ("agentic", state["answer"])]:
            ctx = " ".join(d.page_content for d in docs)
            results.append(
                {
                    "id": row["id"],
                    "mode": mode,
                    "bleu1": bleu1(row["answer"], pred),
                    "rouge_l": rouge_l(row["answer"], pred),
                    "relevance": relevance(row["question"], ctx),
                }
            )
    out = Path("reports/evaluation.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for mode in ("baseline", "agentic"):
        group = [r for r in results if r["mode"] == mode]
        print(
            mode,
            {
                k: round(sum(x[k] for x in group) / len(group), 3)
                for k in ("bleu1", "rouge_l", "relevance")
            },
        )
    print(
        f"Saved {out}; configure Azure credentials to evaluate generated agent answers."
    )


if __name__ == "__main__":
    run()
