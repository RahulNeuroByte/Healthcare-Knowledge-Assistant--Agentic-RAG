# Healthcare Knowledge Assistant — Agentic RAG

A modular healthcare FAQ assistant built with Python 3.12, LangGraph, LangChain tools, FAISS, FastAPI, Streamlit, and GitHub Actions. It uses SerpAPI to search current information from selected public-health sources, combines results with a local FAQ corpus, and does not call Gemini or OpenAI. The included public-health corpus contains exactly 100 FAQs.

## Project structure

```text
healthcare-agentic-rag/
├── api/                    # FastAPI schemas and REST endpoints
├── data/                   # 100 FAQs, FAQ PDF, generated index and feedback log
├── docs/                   # workflow diagram and project report
├── reports/                # evaluation output and report PDF
├── src/
│   ├── agents/             # consultation and diagnosis-boundary agents
│   ├── evaluation/         # BLEU-1, ROUGE-L, relevance evaluation
│   ├── graph/              # LangGraph workflow and session memory
│   ├── ingestion/          # corpus loading, chunking and embeddings
│   ├── retrieval/          # FAISS semantic + keyword hybrid search
│   └── tools/               # reusable LangChain RAG tool
├── tests/                   # retrieval, graph and API tests
├── ui/main.py               # Streamlit chat interface
├── .github/workflows/       # CI and optional Azure App Service deployment
├── brain.md                 # continuation and execution checkpoint
├── Execution guide.md       # Windows 11 / VS Code setup
├── requirements.txt
└── README.md
```

## Capabilities

- 100 public-facing healthcare FAQs in JSON and PDF, with source links and safety notes.
- FAISS retrieval combined with BM25-style keyword matching and reciprocal-rank fusion.
- Local deterministic hashing embeddings for startup without an embedding API.
- LangGraph planner, trusted-source web and FAQ retrieval, extractive cited answers, grounding validator and bounded retry. Diagnosis requests receive a safety boundary response.
- FastAPI `/query`, `/health` and `/feedback` endpoints; Streamlit chat shows retrieved context, references and feedback controls.
- GitHub Actions runs Python 3.12 tests and lint. On `main`, it can optionally deploy the API source ZIP to Azure App Service without containers.

## Run locally

Follow [Execution guide.md](Execution%20guide.md). Quick start in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn api.main:app --reload
```

Set `SERPAPIKEY` in your existing `.env` to enable live search; do not overwrite the file if it already contains your key. An Azure/OpenAI key is not a SerpAPI key and is not used by this application. Live search also requires outbound HTTPS access to `serpapi.com` with valid certificate verification; the UI reports whether the failure is a key, quota, or network/certificate issue. Results are restricted to HTTPS pages from selected health publishers. If web search is unavailable, the app falls back to relevant local FAQs and says when the FAQ corpus has no topic-specific answer instead of returning an API error. Search results are summarized extractively; no Gemini/OpenAI calls are made.

Open a second terminal, activate the environment, and run `streamlit run ui/main.py`. API docs: `http://localhost:8000/docs`; UI: `http://localhost:8501`.

## Evaluation and safety

Run `python -m src.evaluation.run_evaluation`; results are saved to `reports/evaluation.csv`. In the local no-Azure smoke run, baseline BLEU-1 / ROUGE-L / relevance were 0.514 / 0.615 / 0.977, while evidence-only agentic output scored 0.272 / 0.411 / 0.977. Longer evidence excerpts lower text-overlap metrics. These small sample scores are not clinical validation or a model benchmark. Do not treat heuristic confidence as medical certainty.

**Not medical advice.** The system provides general education only. It does not diagnose, prescribe, or respond to emergencies. Do not submit patient-identifiable information. Contact a qualified clinician for personal health decisions and local emergency services in an emergency.

## GitHub Actions deployment

The workflow tests and lints on pushes and pull requests. To deploy the FastAPI API to Azure App Service, configure repository secret `AZURE_WEBAPP_PUBLISH_PROFILE` and repository variable `AZURE_WEBAPP_NAME`. Configure the App Service runtime as Python 3.12 and set `SERPAPIKEY` as an application setting. Optional variable `DEPLOY_URL` enables a post-deployment `/health` check. Streamlit can be run locally as shown above or hosted as a separate Python web app. Deployment uses a Python source ZIP, not containers.
