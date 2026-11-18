# Advanced RAG Agent with Context Compression & Semantic Memory

A FastAPI backend implementing an advanced Retrieval-Augmented Generation (RAG) pipeline: hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion, parent-document retrieval, Cohere Rerank context compression, Mem0 long-term semantic memory and per-user persona adaptation, with per-request latency telemetry.

## 1. Overview and problem statement

Basic RAG (top-k vector similarity) has well-known weaknesses: it misses exact terminology, returns tiny chunks with too little context, passes irrelevant passages to the LLM (wasting tokens and inviting hallucination), and treats every user identically. This project addresses each one:

| Problem | Solution |
|---|---|
| Misses keywords / identifiers | BM25 sparse retrieval alongside dense vectors |
| Merging two result lists | Reciprocal Rank Fusion (no score normalisation needed) |
| Chunks too small to answer from | Parent-document retriever (index ~200-char children, return ~1000-char parents) |
| Irrelevant context, wasted tokens | Cohere `rerank-english-v3.0`: 20 candidates -> score threshold -> max 5 |
| No personalisation | Mem0 extracts durable user facts; persona is injected into the system prompt on every new session |

## 2. Features

- PDF/TXT ingestion with deterministic IDs; re-running ingestion never duplicates records
- Dense retrieval (Qdrant) + BM25 sparse retrieval, merged by a standalone RRF module
- Parent-document retrieval with de-duplication and missing-parent handling
- Cohere rerank via LangChain `CohereRerank` + `ContextualCompressionRetriever`, configurable top-N and minimum relevance score
- Mem0 semantic memory (OSS SDK with local Qdrant, or hosted platform) keyed by `user_id`, independent of `session_id`
- Dynamic persona built from stored memories (nothing hard-coded)
- Configurable LLM (`openai` / `anthropic` / `cohere`) and embedding (`openai` / `cohere`) providers
- Telemetry measured with `time.perf_counter()`: retrieval, rerank, generation, total (+ memory stages and debug counters)
- Secrets redacted from errors and logs; graceful handling of API outages

## 3. Technology stack

Python 3.12 (Docker image; developed on 3.13), FastAPI, Uvicorn, Pydantic, LangChain (`langchain`, `-core`, `-community`, `-classic`), Qdrant (local persistent storage or server), `rank-bm25`, Cohere Rerank, Mem0, OpenAI/Anthropic/Cohere SDKs, python-dotenv, pytest + pytest-mock, Docker.

## 4. Architecture and RAG workflow

```
POST /api/chat {query, user_id, session_id}
  |
  1. Mem0: get all memories for user_id (NOT filtered by session) -> build persona
  2. Dense search (Qdrant, child chunks)  +  Sparse search (BM25, child chunks)
  3. Reciprocal Rank Fusion  -> ranked child chunks
  4. Child -> parent mapping (deduplicated)  -> top-20 parent candidates
  5. Cohere rerank (rerank-english-v3.0) -> drop score < MIN_RELEVANCE_SCORE -> keep <= 5
  6. Prompt = grounding rules + persona + numbered contexts + question
  7. LLM generates the answer
  8. Mem0 extracts durable facts from the user's message and persists them
  -> {answer, sources, telemetry}
```

Layers are separated: `app/ingestion`, `app/retrieval`, `app/compression`, `app/memory`, `app/agent`, `app/services` (orchestration + telemetry), `app/api` (HTTP only). Clients are wired in `app/api/deps.py` and replaced via FastAPI dependency overrides in tests.

### Hybrid retrieval
Dense vectors capture meaning ("how do I combine search methods" ~ "hybrid search"); BM25 captures exact terms such as `rerank-english-v3.0`. They fail differently, so fusing both improves recall.

### Parent-child retrieval
Documents are split into ~1000-char **parents** and each parent into ~200-char **children** (`CHILD_CHUNK_SIZE`, `PARENT_CHUNK_SIZE`). Only children are embedded/indexed, each carrying `document_id`, `parent_id`, `child_id`, `source`, `page`. A matching child is replaced by its full parent from the local repository (`data/parents/`); several children of one parent yield one parent. If a parent can't be found, the child chunk is kept and flagged `parent_missing`.

### Reciprocal Rank Fusion
`RRF_score(d) = sum_i 1 / (k + rank_i(d))` (`RRF_K`, default 60; ranks start at 1) in `app/retrieval/rrf.py`. A chunk present in both lists is merged into one result with a summed score.

### Cohere Rerank and compression
`ThresholdCohereRerank` (`app/compression/reranker.py`) wraps LangChain's `CohereRerank`, sorts by `relevance_score`, drops scores below `MIN_RELEVANCE_SCORE`, and hard-caps at `RERANK_TOP_N` (5). Source metadata is preserved and `relevance_score` is added. `build_compression_retriever()` composes it with a `ContextualCompressionRetriever`. The chat service calls the retriever and compressor as two explicit stages so retrieval and rerank latency can be reported separately. If Cohere fails, the request degrades to the RRF order (top 5) and `telemetry.rerank_degraded` is `true`. Telemetry reports `retrieved_candidates` (<=20) and `compressed_contexts` (<=5).

### Mem0 semantic memory and persona adaptation
`Mem0Manager` offers `add_memory`, `search_memory`, `get_all_memories`, `update_memory`, `delete_memory`. Extraction is done by Mem0's LLM with custom instructions that keep only durable preferences/facts (not greetings or one-off questions). Every call is scoped by `user_id`; results from other users are filtered out defensively, and update/delete verify ownership. On each chat, all of the user's memories are turned into a profile block ("Programming language: Prefers Python", "Formatting: Prefers code-only responses", ...) that is injected into the system prompt. Categories come from keyword rules in `app/memory/persona.py`. A memory outage never fails a chat request.

### Telemetry
Each successful `/api/chat` response has `retrieval_latency_ms`, `rerank_latency_ms`, `generation_latency_ms`, `total_latency_ms` (measured with `perf_counter`; total covers the whole request), plus `memory_retrieval_latency_ms`, `memory_update_latency_ms` and counters. Timing lives in `app/services/telemetry_service.py` / `app/utils/timing.py`.

## 5. Project structure

```
app/            api/ core/ ingestion/ retrieval/ compression/ memory/ agent/ services/ schemas/ utils/
data/           documents/ (input files)  qdrant_storage/  parents/  mem0/  (runtime state, git-ignored)
scripts/        ingest_documents.py
tests/          test_rrf / reranker / retrieval / memory / chat_api / telemetry
postman/        Advanced_RAG_API_Collection.json
Dockerfile  docker-compose.yml  requirements.txt  pytest.ini  .env.example
```

## 6. Prerequisites and installation

- Python 3.12+ (3.13 also works), Git, optionally Docker
- API keys: **Cohere** (rerank - always required), **Gemini** (default LLM; embeddings run locally with HuggingFace `all-MiniLM-L6-v2`) or OpenAI/Anthropic/Cohere if you switch providers

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in your keys
```

## 7. Environment variables

Loaded with python-dotenv from `.env` (never committed). See `.env.example`.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY` | Provider keys (only those for the providers you select; Cohere is always needed for rerank) |
| `LLM_PROVIDER`, `LLM_MODEL` | `gemini` (default) / `openai` / `anthropic` / `cohere` and model name |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `huggingface` (default, local) / `gemini` / `openai` / `cohere` and model name |
| `RERANK_MODEL` (`rerank-english-v3.0`), `RETRIEVAL_TOP_K` (20), `RERANK_TOP_N` (5), `MIN_RELEVANCE_SCORE` | Compression settings |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant server; empty `QDRANT_URL` = local storage at `QDRANT_PATH` |
| `MEM0_API_KEY` | Set to use hosted Mem0; empty = open-source Mem0 stored under `MEM0_PATH` |
| `CHILD_CHUNK_SIZE`, `PARENT_CHUNK_SIZE`, `RRF_K`, `DENSE_TOP_K`, `SPARSE_TOP_K` | Chunking / retrieval tuning |

Missing or invalid values produce an explicit error (HTTP 503 from the API, exit code 2 from the ingest script) naming each problem.
If you change `EMBEDDING_MODEL`/provider after ingesting, re-ingest with `--force` into a fresh `QDRANT_COLLECTION` (vector sizes differ).

## 8. Ingest documents

Put `.pdf` / `.txt` files in `data/documents/` (a sample `rag_overview.txt` is included), then:

```bash
python scripts/ingest_documents.py            # everything in data/documents
python scripts/ingest_documents.py my.pdf     # one file
python scripts/ingest_documents.py --force    # re-ingest unchanged files
```

Unchanged files are skipped; changed files replace their old records. **With local Qdrant storage only one process can open the store: stop the API server before ingesting** (or set `QDRANT_URL` to a Qdrant server, as docker-compose does).

## 9. Run the application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs  (OpenAPI JSON at `/openapi.json`)
- Health: http://localhost:8000/health

## 10. API

### `POST /api/chat`
```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"query":"Explain hybrid search","user_id":"usr_99","session_id":"session_001"}'
```
Response shape (values vary; latencies are real measurements):
```json
{
  "answer": "...",
  "user_id": "usr_99",
  "session_id": "session_001",
  "sources": [{"source": "rag_overview.txt", "page": null, "parent_id": "...-p0", "relevance_score": 0.93, "snippet": "..."}],
  "telemetry": {"retrieval_latency_ms": 0, "rerank_latency_ms": 0, "generation_latency_ms": 0, "total_latency_ms": 0,
                "retrieved_candidates": 20, "compressed_contexts": 5, "rerank_degraded": false,
                "memories_used": 0, "memories_saved": 0}
}
```
Status codes: 200, 422 (invalid body), 502 (LLM/retrieval/memory provider failed), 503 (configuration missing).

### `GET /api/users/{user_id}/memories`
```bash
curl http://localhost:8000/api/users/usr_99/memories
```
```json
{"user_id": "usr_99", "memories": [{"id": "...", "fact": "Prefers Python", "category": "programming_language"}]}
```
Unknown users return `200` with an empty list.

### Multi-session persona example
```bash
curl -X POST localhost:8000/api/chat -H "Content-Type: application/json" -d '{"query":"Hello! I work in Python and prefer brief, code-only responses.","user_id":"usr_99","session_id":"session_001"}'
curl localhost:8000/api/users/usr_99/memories
curl -X POST localhost:8000/api/chat -H "Content-Type: application/json" -d '{"query":"Write a quick merge sort routine.","user_id":"usr_99","session_id":"session_002"}'
```

## 11. Postman collection

Import `postman/Advanced_RAG_API_Collection.json` (Postman -> Import). It defines `base_url`, `user_id`, `session_id` variables, requests for both endpoints (plus `/health`), example responses, and basic test scripts.

## 12. Tests

```bash
pytest -v
```
Covers RRF, Cohere rerank filtering (mocked client), parent retrieval, ingestion idempotency, memory isolation/persistence, the chat API and telemetry. No real Cohere/OpenAI/Mem0 calls are made: external services are mocked in `tests/conftest.py`.

## 13. Docker

```bash
docker build -t advanced-rag-agent .
docker run -p 8000:8000 --env-file .env advanced-rag-agent
```
With Compose (API + Qdrant server, persistent volumes):
```bash
docker compose up --build -d
docker compose exec app python scripts/ingest_documents.py
docker compose down            # add -v to also delete volumes
```
The image has a `/health` HEALTHCHECK and never contains secrets.

## 14. Known limitations

- Running the full pipeline requires real Cohere and LLM/embedding API keys; the automated tests mock them. Live end-to-end behaviour against those APIs depends on your keys and models.
- Memory categories are assigned by keyword rules, not by the LLM; unusual facts fall into `general`.
- The BM25 index is held in memory and rebuilt from `data/children.json` when chunks change; very large corpora would call for Qdrant sparse vectors instead.
- Local Qdrant storage is single-process (see ingestion note). Hosted Mem0 (`MEM0_API_KEY`) support follows the SDK's documented interface but was not exercised against the live service.
- Hosted Mem0 (`MEM0_API_KEY`) is eventually consistent: `add` returns `PENDING` and facts appear ~2-5 s later, so `memories_saved` is 0 for that request and a *new session started immediately* may not see them yet. The open-source Mem0 path is synchronous.
- Mem0 extraction quality depends on the configured LLM; a stated preference is stored only if the model judges it durable.
- Only PDF and TXT are supported; PDFs without a text layer (scans) yield no chunks.
