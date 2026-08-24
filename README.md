# 🏥 Hospital Assistant RAG v3

**Production-oriented conversational hospital assistant** built with Retrieval-Augmented Generation (RAG), hybrid retrieval, cross-encoder reranking, and a fully API-driven, containerized architecture.

Version 3 evolves the project from a single-file Streamlit prototype into a **modular, configurable, evaluated, and Dockerized GenAI application**, powered by Qdrant, FastAPI, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)]()
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)]()
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)]()
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-orange)]()

---

## 🎥 Demo

**▶️ ![Watch the full demo video](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/hospital-assistant-rag-v3-demo.mp4)** — conversational Q&A, retrieval-mode switching, reranking, source transparency, and Docker deployment in action.

---

## 📸 Screenshots

### 💬 Conversational Chat Interface

![Hospital Assistant Chat](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/hospital-assistant-chat.png)

A welcome screen with guided example questions, live retrieval-mode/LLM configuration in the sidebar, and grounded, hospital-policy-aware answers.

### 🗂️ Multi-Chat Management

![Multi-chat management](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/multi-chat-management.png)

Multiple persistent conversations with full chat history, so users can keep separate threads for different topics.

### 🗑️ Chat Deletion Confirmation

![Delete chat confirmation](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/delete-chat-confirmation.png)

Guarded chat-deletion flow to prevent accidental loss of conversation history.

### ⚙️ Retrieval Mode & Reranker Settings

![Retrieval mode and reranker settings](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/retrieval-mode-reranker-settings.png)

Users can switch between **Semantic**, **Semantic + Reranker**, **Hybrid**, and **Hybrid + Reranker** retrieval modes directly from the UI.

### 🤖 LLM Provider & Model Selection

![LLM provider and model selection](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/llm-provider-model-selection.png)

Swap between LLM providers and models at runtime without touching the underlying RAG pipeline.

### 📋 Retrieved Sources Panel

![Retrieved sources](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/retrieved-sources.png)

Every answer is paired with the retrieved source documents, chunk rank, and page number for full traceability.

### ⚡ FastAPI Swagger UI

![FastAPI UI](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/fastapi-api-ui.png)
![FastAPI Schemas](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/fastapi-api-schemas.png)

Auto-generated OpenAPI documentation for the `/health` and `/chat` endpoints, including request/response schemas.

### 🐳 Docker Deployment

![Docker images](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/docker_images.png)
![Docker containers](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/docker-containers.png)
![Docker logs](https://github.com/Namitha-C-Anto/hospital-assistant-v3/raw/main/assets/docker-logs.png)

Streamlit, FastAPI, and Qdrant running as independent, orchestrated Docker Compose services with live container logs.

---

## ✨ Key Features

- 🧠 **End-to-end RAG** for hospital policy and patient-service questions
- 🗄️ **Qdrant vector database** for persistent, production-grade vector storage
- 🔎 **Semantic retrieval** using Qdrant
- 🔀 **Hybrid retrieval** using Qdrant + BM25
- 🏆 **Reciprocal Rank Fusion (RRF)** for merging hybrid results
- 🎯 **Optional cross-encoder reranking** for semantic or hybrid retrieval
- 📋 **Retrieval/source metadata** displayed in the UI (document, page, chunk rank)
- 🤖 **Multiple LLM providers and models** selectable at runtime through the UI
- ⚡ **FastAPI REST API** with dependency injection and lifespan management
- 💬 **Streamlit conversational interface** with multi-chat management
- 📝 **Structured terminal logging**
- 🧪 **Automated tests** for API and Qdrant functionality
- 📊 **RAGAS evaluation** with experiment tracking (7 recorded runs)
- 🐳 **Docker and Docker Compose** for one-command deployment
- 🔐 **Environment-based API key configuration**

---

## 🏗️ Architecture

```
                        User
                          │
                          ▼
                 ┌─────────────────┐
                 │    Streamlit    │
                 │    Frontend     │
                 └────────┬────────┘
                          │
                      HTTP / REST
                          │
                          ▼
                 ┌─────────────────┐
                 │     FastAPI     │
                 │     Backend     │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   RAGService    │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Retrieval Strategy       LLM Provider
              │                  + Model
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
Semantic             Hybrid
    │                   │
 Qdrant          ┌──────┴──────┐
    │             │             │
    │          Qdrant          BM25
    │             │             │
    │             └──────┬──────┘
    │                    │
    │                   RRF
    │                    │
    └──────────┬─────────┘
               │
               ▼
        Optional Reranker
               │
               ▼
        Top-K Documents
               │
               ▼
       Retrieved Context
               │
               ▼
          Selected LLM
               │
               ▼
       Answer + Metadata
```

### 📄 Document Ingestion

```
Hospital Documents → Loading → Text Extraction → Splitting → Embeddings
                                                            ↓
                                             ┌──────────────┴──────────────┐
                                             ▼                             ▼
                                        Qdrant Vector Store              BM25
```

### 🔍 Retrieval Pipeline

**Semantic:** `Query → Embedding → Qdrant Similarity Search → Optional Reranker → Top-K Context`

**Hybrid:** `Query → [Qdrant Search ‖ BM25 Search] → RRF → Deduplication → Optional Reranker → Top-K Context`

Retrieval mode is configurable (`qdrant` or `hybrid`), and reranking is **independent** of retrieval mode — it can be toggled on for either.

---

## 🎯 Reranking

Retrieved documents can optionally be reranked with a cross-encoder:

```
Default model: BAAI/bge-reranker-base
```

Controlled via `USE_RERANKER`, `RERANKER_MODEL`, `RERANKER_TOP_N`.

---

## 🤖 Multiple LLM Providers & Models

Supported providers: **Groq**, **OpenAI**, **OpenRouter** — selectable through the UI/API without changing pipeline code. The selected LLM receives the user question, conversation history, and retrieved context, and returns a grounded answer with source metadata.

---

## ⚡ FastAPI Backend

A dedicated `RAGService` separates API handling from RAG logic, wired up via **FastAPI lifespan** initialization and **dependency injection**.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /chat` | Chat with the assistant (question, provider, model, retrieval mode, reranker toggle, history) |
| `/docs` | Swagger / OpenAPI documentation |

---

## 🗄️ Qdrant Vector Database

Version 3 **migrated the vector store from FAISS to Qdrant**, running as an independent Docker service with persistent storage that survives container recreation — a key step toward production readiness.

---

## 🐳 Docker Architecture

| Service   | Purpose                | Port |
| --------- | ----------------------- | ---- |
| Streamlit | User interface          | 8501 |
| FastAPI   | REST API + RAG service  | 8000 |
| Qdrant    | Vector database          | 6333 |

Services communicate using Docker Compose service names rather than `localhost`.

---

## 📊 RAG Evaluation

Evaluation is powered by **RAGAS** across **22 test questions** covering hospital policy and patient-service scenarios (`dataset_v2`), using `sentence-transformers/all-mpnet-base-v2` embeddings. The same dataset is reused across pipeline versions to track quality regressions/improvements over time.

### Metrics

| Metric | Purpose |
|---|---|
| Faithfulness | Whether the answer is supported by retrieved context |
| Answer Relevancy | How well the answer addresses the question |
| Context Precision | Relevance of retrieved documents |
| Context Recall | Whether the required information was retrieved |

### Experiment Results

*(source: [`evaluate/evaluation_results/experiment_summary.csv`](evaluate/evaluation_results/experiment_summary.csv))*

| Experiment | Retrieval | Reranker | Generation LLM | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg. Pipeline Latency (s) |
|---|---|---|---|---|---|---|---|---|
| `faiss_...no_reranker` | FAISS (semantic) | ❌ | gpt-4o-mini | 0.961 | 0.914 | 0.555 | 0.955 | 1.91 |
| `hybrid_20260712_..._reranker` | Hybrid (Qdrant + BM25) | ✅ BAAI/bge-reranker-base | gpt-4o-mini | 0.879 | 0.906 | 0.853 | 0.909 | 10.13 |
| `hybrid_20260713_..._reranker` | Hybrid | ✅ | gpt-4o-mini | 0.841 | 0.910 | 0.863 | 0.864 | 5.24 |
| `hybrid_20260714_..._reranker` | Hybrid | ✅ | gpt-4o-mini | 0.878 | 0.890 | 0.857 | 0.864 | 5.91 |
| `hybrid_20260822_004553_reranker` | Hybrid | ✅ | qwen/qwen3-235b-a22b | 0.817 | 0.932 | 0.861 | 0.864 | 16.09 |
| `hybrid_20260822_012606_reranker` | Hybrid | ✅ | openai/gpt-oss-120b | 0.853 | **0.944** | 0.853 | 0.877 | 15.24 |
| `hybrid_20260822_015905_reranker` (top_k=20) | Hybrid | ✅ | openai/gpt-oss-120b | 0.827 | 0.873 | 0.855 | 0.864 | 14.47 |

**Takeaways:**
- Pure semantic (FAISS) retrieval gives the highest **context recall** (0.955) but the weakest **context precision** (0.555) — it pulls in a lot of irrelevant context.
- Adding **hybrid retrieval + reranking** roughly **doubles context precision** (0.85–0.86 vs. 0.555), at the cost of extra pipeline latency from the cross-encoder pass.
- Swapping the generation LLM (`gpt-4o-mini` → `qwen3-235b` / `gpt-oss-120b`) shifts the faithfulness/relevancy trade-off and significantly raises latency and token usage — useful for comparing cost/quality across providers before choosing a production model.
- Each run captures retrieval, generation, prompt, and reranker latency, plus full token accounting, enabling data-driven configuration decisions rather than guesswork.

### Evaluation Pipeline Outputs

- Per-question RAGAS metrics
- Overall experiment summaries (CSV)
- JSON experiment metadata
- Retrieval / generation / reranker latency breakdowns
- Token usage statistics

---

## 🧪 Testing

Automated tests cover API and Qdrant functionality:

- Chat endpoint
- Health endpoint
- Qdrant connectivity, collection, search, retrieval, retriever

```bash
pytest
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| RAG Framework | LangChain |
| Vector Database | Qdrant |
| Embeddings | Hugging Face / Sentence Transformers |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Qdrant + BM25 + RRF |
| Reranker | `BAAI/bge-reranker-base` |
| LLM Providers | Groq, OpenAI, OpenRouter |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| PDF Processing | PyMuPDF |
| Testing | Pytest |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```
hospital-assistant-v3/
│
├── app/
│   ├── api/routes.py
│   ├── schemas/chat.py
│   ├── services/rag_service.py
│   ├── dependencies.py
│   └── main.py
│
├── assets/              # Screenshots + demo video
├── docs/                # Source hospital documents
├── evaluate/            # RAGAS evaluation pipeline + results
├── llm/                 # LLM provider integrations
├── memory/              # Conversation memory
├── prompts/             # Prompt templates
│
├── rag/
│   ├── builder.py
│   ├── embeddings.py
│   ├── generation.py
│   ├── initializer.py
│   ├── loader.py
│   ├── models.py
│   ├── pipeline.py
│   ├── qdrant_retriever.py
│   ├── qdrant_store.py
│   ├── reranker.py
│   ├── retrieval.py
│   ├── retriever.py
│   ├── rrf.py
│   ├── splitter.py
│   └── storage.py
│
├── tests/                # Pytest suite (API + Qdrant)
├── ui/                   # Streamlit UI
├── utils/
│
├── app.py
├── build_db.py
├── config.py
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Configuration is environment-variable driven.

```bash
# RAG
DATASET
EMBEDDING_MODEL
CHUNK_SIZE
CHUNK_OVERLAP
TOP_K

# Retrieval — supported values: qdrant | hybrid
RETRIEVAL_MODE

# Reranking
RERANKER_MODEL
RERANKER_TOP_N
USE_RERANKER

# LLM
LLM_PROVIDER
LLM_MODEL
TEMPERATURE

# API Keys
GROQ_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
```

API keys are loaded through environment variables and are never hardcoded or logged.

---

## 🚀 Running the Project

### 1. Clone

```bash
git clone https://github.com/Namitha-C-Anto/hospital-assistant-v3.git
cd hospital-assistant-v3
```

### 2. Configure Environment

Create a `.env` file with the required API keys and configuration values.

### 3. Run with Docker Compose

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Qdrant | http://localhost:6333 |

---

## 🆕 What Changed From v1 → v2 → v3

This project has evolved through three iterations, each adding a distinct layer of production maturity.

### 🔹 v1 — [hospital-assistant-rag](https://github.com/Namitha-C-Anto/hospital-assistant-rag) (Foundation)
The original prototype focused on core RAG mechanics:
- PDF ingestion and recursive text chunking
- HuggingFace embeddings + **FAISS** vector store
- Basic semantic retrieval
- Session-based conversational memory
- Single-file Streamlit chat interface
- No evaluation, no reranking, no hybrid search

### 🔹 v2 — [hospital-assistant-rag-v2](https://github.com/Namitha-C-Anto/hospital-assistant-rag-v2) (Retrieval Quality)
Building on v1, v2 focused on improving **retrieval and answer quality**:
- **Hybrid retrieval**: FAISS + BM25 combined via **Reciprocal Rank Fusion (RRF)**
- **Cross-encoder reranking** (`BAAI/bge-reranker-base`)
- **Source attribution** — page + document shown alongside answers
- **RAGAS evaluation framework** introduced (faithfulness, relevancy, precision, recall)
- Multi-chat Streamlit UI (create/switch/rename/delete conversations)
- Modular `rag/`, `llm/`, `memory/`, `prompts/`, `evaluate/` package structure
- Deployed to Streamlit Community Cloud

### 🔹 v3 — hospital-assistant-v3 (this repo) — Production Architecture
v3 turns the v2 prototype into a **modular, API-driven, configurable, and containerized production application**:
- **Migrated vector store: FAISS → Qdrant**, with persistent Docker-native storage
- **New FastAPI backend** with `RAGService`, dependency injection, lifespan events, `/health` and `/chat` endpoints, and Swagger/OpenAPI docs — the UI now talks to a real REST API instead of calling the pipeline in-process
- **Decoupled frontend**: Streamlit now consumes the FastAPI backend over HTTP instead of running the pipeline directly
- **Multiple LLM providers & models selectable at runtime** (Groq, OpenAI, OpenRouter) — v2 was OpenAI-only
- **Configurable retrieval modes** (`qdrant` semantic vs. `hybrid`) toggled live from the UI, with reranking as an **independent, composable** option for either mode
- **Full Docker Compose orchestration** — Streamlit, FastAPI, and Qdrant run as three separate, networked services (v2 had no containerization)
- **Structured terminal logging** and centralized error handling at the API/service boundary
- **Expanded automated test suite** covering both API endpoints and Qdrant connectivity/search/retrieval
- **Deeper evaluation tooling**: experiment-tracked RAGAS runs (7 recorded experiments) capturing latency breakdowns (pipeline, retrieval, generation, reranker), token usage, and per-configuration comparisons — beyond v2's single-run metrics table
- **Environment-based configuration** across RAG, retrieval, reranking, LLM, and API-key settings, replacing v2's simpler `.env`

**In short:** v1 proved the concept, v2 made retrieval and evaluation rigorous, and v3 makes the system deployable, observable, and configurable — the shape of a real production service.

---

## 🚀 Future Improvements

- 🔎 **Metadata Filtering** — filter retrieved documents by department, document type, category, etc.
- ✍️ **Query Rewriting** — rewrite conversational/ambiguous queries before retrieval
- 🧠 **Agentic RAG** — agents for tool selection, multi-step reasoning, task-specific workflows
- ☁️ **Cloud Deployment** — AWS/Azure/GCP managed containers + a cloud-native LLM platform (Bedrock, Vertex AI, Azure AI Foundry)
- 📝 **File-Based Logging** — rotating log files for production troubleshooting/auditing
- 📈 **Observability** — metrics, tracing, latency monitoring, dashboards
- 📊 **Advanced Evaluation** — larger evaluation dataset, systematic retrieval/reranking/LLM comparisons

---

## 🎯 Project Objective

The goal of this project is to demonstrate how a document-based RAG prototype can be transformed into a **modular, configurable, evaluated, API-driven, and containerized GenAI application** — focusing on practical RAG engineering (retrieval strategy selection, hybrid search, RRF, reranking, evaluation, API architecture, testing, logging, and deployment) rather than LLM integration alone.

---

## 👩‍💻 Author

**Namitha C Anto**
GenAI / AI Engineering Portfolio Project

- **GitHub:** [Namitha-C-Anto/hospital-assistant-v3](https://github.com/Namitha-C-Anto/hospital-assistant-v3)
- **v2:** [hospital-assistant-rag-v2](https://github.com/Namitha-C-Anto/hospital-assistant-rag-v2)
- **v1:** [hospital-assistant-rag](https://github.com/Namitha-C-Anto/hospital-assistant-rag)
