# 🏥 Hospital Assistant RAG v3

An AI-powered **Retrieval-Augmented Generation (RAG)** application that answers hospital policy and patient-service questions using trusted document content.

Version 3 evolves the application from a Streamlit-based RAG prototype into a **modular, API-driven, configurable, and containerized application** using Qdrant, FastAPI, Streamlit, and Docker Compose.

---

## 🎥 Demo

> Add your demo video here.

**▶️ [Watch the Demo Video](YOUR_DEMO_VIDEO_LINK)**

The demo showcases:

* 💬 Conversational hospital policy and patient-service questions
* 🔎 Semantic and hybrid retrieval
* 🎯 Optional cross-encoder reranking
* 🤖 LLM provider and model selection
* 📋 Retrieved document metadata displayed in the UI
* ⚡ FastAPI backend
* 🐳 Dockerized deployment

---

## 📸 Screenshots

### 💬 Chat Interface

![Hospital Assistant Chat](assets/screenshots/chat-interface.png)

The response interface displays the generated answer together with **retrieval/source metadata**, making the retrieved context more transparent and easier to verify.

### ⚙️ Model & Retrieval Configuration

![Model and Retrieval Configuration](assets/screenshots/settings.png)

Users can configure the available **LLM provider/model**, retrieval mode, and optional reranking through the application UI.

### ⚡ FastAPI Swagger UI

![FastAPI Swagger UI](assets/screenshots/fastapi-swagger.png)

### 🗄️ Qdrant

![Qdrant](assets/screenshots/qdrant.png)

> Replace the screenshot filenames above with the actual files you add to your repository.

---

## ✨ Key Features

* 🧠 **End-to-end RAG** for hospital policies and patient services
* 🗄️ **Qdrant vector database** for persistent vector storage
* 🔎 **Semantic retrieval** using Qdrant
* 🔀 **Hybrid retrieval** using Qdrant + BM25
* 🏆 **Reciprocal Rank Fusion (RRF)** for hybrid result fusion
* 🎯 **Optional cross-encoder reranking** for semantic or hybrid retrieval
* 📋 **Retrieval/source metadata** displayed in the UI
* 🤖 **Multiple LLM providers and models** selectable through the UI
* ⚡ **FastAPI REST API** with dependency injection and lifespan management
* 💬 **Streamlit conversational interface**
* 📝 **Structured terminal logging**
* 🧪 **Automated tests** for API and Qdrant functionality
* 📊 **RAGAS evaluation**
* 🐳 **Docker and Docker Compose**
* 🔐 **Environment-based API key configuration**

---

## 🏗️ Architecture

```text
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

---

## 📄 Document Ingestion

Hospital documents are processed through the ingestion pipeline:

```text
Hospital Documents
       ↓
Document Loading
       ↓
Text Extraction
       ↓
Text Splitting
       ↓
Embedding Generation
       ↓
      ┌┴───────────────┐
      ▼                ▼
   Qdrant            Chunks
 Vector Store           │
                       ▼
                      BM25
```

The processed chunks and metadata support both semantic and keyword-based retrieval.

---

## 🔍 Retrieval Pipeline

### Semantic Search

```text
User Query
    ↓
Query Embedding
    ↓
Qdrant Similarity Search
    ↓
Retrieved Documents
    ↓
Optional Reranker
    ↓
Top-K Context
```

### Hybrid Search

```text
                    User Query
                        │
                ┌───────┴───────┐
                ▼               ▼
           Qdrant Search      BM25 Search
           Semantic Search    Keyword Search
                │               │
                └───────┬───────┘
                        ▼
                       RRF
                        ↓
                  Deduplication
                        ↓
                 Optional Reranker
                        ↓
                   Top-K Context
```

The retrieval mode can be configured as:

* `qdrant` — semantic vector search
* `hybrid` — Qdrant + BM25 + RRF

The **reranker is independent of the retrieval mode** and can optionally be enabled for either semantic or hybrid retrieval.

---

## 🎯 Reranking

Retrieved documents can optionally be reranked using a cross-encoder.

**Default model:**

```text
BAAI/bge-reranker-base
```

The reranker evaluates the relationship between the **user query and retrieved documents** and improves document ordering before context is passed to the LLM.

Reranking can be controlled through:

```text
USE_RERANKER
RERANKER_MODEL
RERANKER_TOP_N
```

---

## 🤖 Multiple LLM Providers & Models

The application supports multiple LLM providers:

* **Groq**
* **OpenAI**
* **OpenRouter**

The provider and model can be selected through the UI/API without changing the core RAG pipeline.

The selected LLM receives:

```text
User Question
      +
Conversation History
      +
Retrieved Context
      ↓
Selected LLM
      ↓
Grounded Answer
```

This makes it possible to compare different models while keeping retrieval and application logic consistent.

---

## 📋 Retrieval Metadata & Source Transparency

The application returns retrieval information along with the generated answer.

The UI exposes relevant document metadata such as:

* Source document
* Page information
* Retrieved content/chunks
* Retrieval information

This provides greater **transparency, traceability, and debugging capability** for RAG responses.

---

## ⚡ FastAPI Backend

FastAPI provides the API layer between the Streamlit frontend and RAG pipeline.

The backend uses a dedicated `RAGService` to separate API handling from RAG logic.

```text
FastAPI
   │
   ▼
RAGService
   │
   ├── Initialize RAG components
   ├── Select LLM provider/model
   ├── Run retrieval
   ├── Apply optional reranking
   ├── Generate answer
   └── Return answer + retrieval metadata
```

The application also uses:

* **FastAPI lifespan** for application-level initialization
* **Dependency injection** for `RAGService`
* `/health` endpoint
* `/chat` endpoint
* Swagger/OpenAPI documentation

---

## 📝 Logging & Error Handling

The application uses **structured terminal logging** to make application execution and failures easier to monitor and debug.

Logs include useful information such as:

* Application events
* Retrieval execution
* Service initialization
* Errors and exceptions
* Configuration-related information

Application errors are handled at the appropriate service/API boundaries so that failures can be logged and returned as controlled API responses rather than exposing raw internal exceptions.

> API keys and sensitive configuration values are not hardcoded or logged.

---

## 🗄️ Qdrant Vector Database

Version 3 migrated the vector storage layer from **FAISS to Qdrant**.

```text
Previous

Application → FAISS


Version 3

Streamlit
    ↓
FastAPI
    ↓
RAGService
    ↓
Qdrant
```

Qdrant runs as an independent Docker service and uses persistent storage so that vector data survives container recreation.

---

## 🐳 Docker Architecture

The application uses Docker Compose to run separate services:

```text
┌───────────────────────────────────────┐
│            Docker Compose             │
│                                       │
│  ┌──────────────┐                     │
│  │  Streamlit   │ :8501               │
│  └──────┬───────┘                     │
│         │ HTTP                        │
│         ▼                             │
│  ┌──────────────┐                     │
│  │   FastAPI    │ :8000               │
│  └──────┬───────┘                     │
│         │                             │
│         ▼                             │
│  ┌──────────────┐                     │
│  │   Qdrant     │ :6333               │
│  └──────────────┘                     │
│                                       │
└───────────────────────────────────────┘
```

| Service   | Purpose                | Port |
| --------- | ---------------------- | ---: |
| Streamlit | User interface         | 8501 |
| FastAPI   | REST API + RAG service | 8000 |
| Qdrant    | Vector database        | 6333 |

Inside Docker, services communicate using their **Docker Compose service names** rather than `localhost`.

---

## 📊 RAG Evaluation

The project uses **RAGAS** to evaluate the quality of retrieved context and generated answers.

The evaluation dataset contains **22 test questions** covering hospital policy and patient-service scenarios. The same dataset is used to compare different retrieval configurations and measure improvements across RAG pipeline versions.

### Metrics

| Metric            | Purpose                                                       |
| ----------------- | ------------------------------------------------------------- |
| Faithfulness      | Measures whether the answer is supported by retrieved context |
| Answer Relevancy  | Measures how well the answer addresses the question           |
| Context Precision | Measures the relevance of retrieved documents                 |
| Context Recall    | Measures whether the required information was retrieved       |


### Evaluation Flow

```text
22 Test Questions
       ↓
   RAG Pipeline
       ↓
 Generated Answers + Retrieved Context
       ↓
      RAGAS
       ↓
  Quality Metrics

## 🧪 Testing

The project includes automated tests covering API and Qdrant functionality.

Current test areas include:

* Chat endpoint
* Health endpoint
* Qdrant connectivity
* Qdrant collection
* Qdrant search
* Qdrant retrieval
* Qdrant retriever

Run tests with:

```bash
pytest
```

---

## 🛠️ Tech Stack

| Category          | Technology                           |
| ----------------- | ------------------------------------ |
| Language          | Python                               |
| RAG Framework     | LangChain                            |
| Vector Database   | Qdrant                               |
| Embeddings        | Hugging Face / Sentence Transformers |
| Keyword Retrieval | BM25                                 |
| Hybrid Retrieval  | Qdrant + BM25 + RRF                  |
| Reranker          | `BAAI/bge-reranker-base`             |
| LLM Providers     | Groq, OpenAI, OpenRouter             |
| Backend           | FastAPI                              |
| Frontend          | Streamlit                            |
| Evaluation        | RAGAS                                |
| PDF Processing    | PyMuPDF                              |
| Testing           | Pytest                               |
| Containerization  | Docker                               |
| Orchestration     | Docker Compose                       |
| Version Control   | Git & GitHub                         |

---

## 📂 Project Structure

The following structure reflects the current repository organization.

```text
hospital-assistant-v3/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── schemas/
│   │   └── chat.py
│   ├── services/
│   │   └── rag_service.py
│   ├── dependencies.py
│   └── main.py
│
├── assets/
├── docs/
├── evaluate/
├── llm/
├── memory/
├── prompts/
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
├── tests/
│   ├── test_chat.py
│   ├── test_health.py
│   ├── test_qdrant.py
│   ├── test_qdrant_collection.py
│   ├── test_qdrant_retrieval.py
│   ├── test_qdrant_retriever.py
│   └── test_qdrant_search.py
│
├── ui/
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

The application uses environment variables for configuration.

### RAG

```text
DATASET
EMBEDDING_MODEL
CHUNK_SIZE
CHUNK_OVERLAP
TOP_K
```

### Retrieval

```text
RETRIEVAL_MODE
```

Supported values:

```text
qdrant
hybrid
```

### Reranking

```text
RERANKER_MODEL
RERANKER_TOP_N
USE_RERANKER
```

### LLM

```text
LLM_PROVIDER
LLM_MODEL
TEMPERATURE
```

### API Keys

```text
GROQ_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
```

API keys are loaded through environment variables and should never be hardcoded or committed to the repository.

---

## 🚀 Running the Project

### 1. Clone

```bash
git clone https://github.com/Namitha-C-Anto/hospital-assistant-v3.git
cd hospital-assistant-v3
```

### 2. Configure Environment

Create a `.env` file with the required API keys and configuration.

### 3. Run with Docker Compose

```bash
docker compose up --build
```

The application starts:

* Streamlit
* FastAPI
* Qdrant

Access the applications from the configured host ports:

```text
Streamlit → http://localhost:8501
FastAPI   → http://localhost:8000
Swagger   → http://localhost:8000/docs
Qdrant    → http://localhost:6333
```

---

## 🚀 Future Improvements

The current architecture provides a foundation for further production enhancements.

* 🔎 **Metadata Filtering** — filter retrieved documents using metadata such as department, document type, category, or other attributes.
* ✍️ **Query Rewriting** — rewrite conversational or ambiguous queries before retrieval to improve retrieval quality.
* 🧠 **Agentic RAG** — introduce agents for tool selection, multi-step reasoning, and task-specific workflows.
* ☁️ **Cloud Deployment** — Deploy the containerized application to AWS/Azure/GCP using managed container services and integrate a cloud-native LLM platform such as Amazon Bedrock, Google Vertex AI, or Azure AI Foundry.
* 📝 **File-Based Logging** — persist application logs to rotating log files for production troubleshooting and auditing.
* 📈 **Observability** — add metrics, tracing, latency monitoring, and production dashboards.
* 📊 **Advanced Evaluation** — expand the evaluation dataset and systematically compare retrieval, reranking, and LLM configurations.

---

## 🎯 Project Objective

The goal of this project is to demonstrate how a document-based RAG prototype can be transformed into a **modular, configurable, evaluated, API-driven, and containerized GenAI application**.

The project focuses on practical RAG engineering rather than only LLM integration, including **retrieval strategy selection, hybrid search, RRF, reranking, evaluation, API architecture, testing, logging, and deployment**.

---

## 👩‍💻 Author

**Namitha C Anto**

GenAI / AI Engineering Portfolio Project

**GitHub:** [Namitha-C-Anto/hospital-assistant-v3](https://github.com/Namitha-C-Anto/hospital-assistant-v3)
