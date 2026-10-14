# Project report: Healthcare Knowledge Assistant

## Problem and use case
The system is a healthcare knowledge assistant for general health education. It retrieves FAQ evidence, composes an answer, validates whether evidence was found, and returns references and an agent trace. It is not a medical device or clinical decision maker and refuses to diagnose an individual.

## Dataset, chunking and embeddings
The dataset contains 100 manually authored, conservative FAQs across prevention, vaccines, nutrition, activity, mental health, medicines, chronic conditions, reproductive/family health, and care navigation. Records include category, source citations and disclaimer. The corpus is public-facing material and includes no patient notes or private data. FAQs remain intact as chunks unless oversized; generic chunking supports overlap. Embeddings use deterministic feature hashing to permit local startup without an embedding API. When configured, SerpAPI provides current results that are filtered to HTTPS URLs from selected public-health publishers.

## Retrieval design
FAISS cosine/similarity search ranks semantic candidates. A lightweight BM25-style keyword scorer provides lexical ranking. Reciprocal-rank fusion combines them. A configurable top-k limits context. The project intentionally omits reranking to keep dependencies lean; it can be added after evaluation.

## Agents and graph
LangGraph sequences planner, retriever agent, consultation specialist, and grounding validator. The retriever invokes the reusable LangChain `retrieve_health_evidence` tool to combine current web results with local FAQ material. Consultation returns cited evidence excerpts without a generative model. Diagnosis-seeking wording routes to a boundary response rather than a diagnosis. Validator checks retrieval presence and a lexical overlap heuristic; one retry increases k. In-memory session history is kept as groundwork for conversation context but no personal health detail should be entered.

## Feedback loop
The Streamlit interface offers helpful/not-helpful controls. FastAPI `/feedback` persists rating and optional correction to JSONL for offline analysis. Validator traces are included in query results. Feedback does not automatically alter the corpus, avoiding unreviewed changes to health content.

## Evaluation
`python -m src.evaluation.run_evaluation` compares first-document baseline retrieval to agentic output over the first 20 FAQs and writes BLEU-1, ROUGE-L F1, and query-context lexical relevance. In the included no-Azure, evidence-only run, baseline BLEU-1 was 0.514, ROUGE-L 0.615, and relevance 0.977. Agentic BLEU-1 was 0.272, ROUGE-L 0.411, and relevance 0.977. The lower overlap scores reflect that responses include longer evidence excerpts. This small lexical smoke test is not a meaningful model benchmark or clinical validation.

## API and UI
FastAPI exposes `/health`, `/query`, `/feedback` and Swagger UI. Query responses contain session ID, answer, source URLs, retrieved context, agent trace, heuristic confidence and validation status. Streamlit provides a chat interface with context disclosure and feedback controls. Run the API and UI as two Python processes in separate terminals.

## CI/CD, conclusion and future work
GitHub Actions runs Ruff and pytest on pull requests and pushes. On `main`, it packages the API source as a ZIP artifact and can deploy it to Azure App Service using a publish-profile secret and app-name variable. An optional `DEPLOY_URL` enables `/health` polling. Rollback is performed by redeploying a previous application revision in Azure; provider-level automation requires the configured Azure service. This project does not use containers. Future work includes clinician review, jurisdiction-specific content updates, durable secure memory, true embedding deployment, measured model evaluation, reranking, privacy controls, and platform-specific deployment automation.

## References
- World Health Organization, Health topics: https://www.who.int/health-topics
- Centers for Disease Control and Prevention, Health topics: https://www.cdc.gov/health-topics.html
- LangGraph quickstart: https://docs.langchain.com/oss/python/langgraph/quickstart
- LangChain FAISS integration: https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
