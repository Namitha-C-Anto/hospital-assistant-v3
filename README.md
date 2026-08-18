# 🏥 Hospital Assistant RAG v3

An AI-powered conversational assistant that answers hospital policy and patient-service questions using **Retrieval-Augmented Generation (RAG)**.

Version 3 evolves the application from a Streamlit-based RAG prototype into a more **modular, API-driven, and containerized architecture** using **Qdrant, FastAPI, Streamlit, and Docker Compose**.

The system retrieves relevant information from hospital documents, optionally combines semantic and keyword retrieval, reranks the retrieved documents, and uses an LLM to generate a grounded response.

---

## 🚀 Highlights

* 🧠 End-to-end Retrieval-Augmented Generation
* 🗄️ Qdrant vector database
* 🔍 Semantic retrieval using Qdrant
* 🔀 Optional hybrid retrieval using Qdrant + BM25 + RRF
* 🎯 Cross-Encoder reranking
* ⚡ FastAPI REST API
* 💬 Streamlit conversational interface
* 🐳 Docker and Docker Compose
* 🤖 Multiple LLM provider support
* 📊 RAGAS evaluation
* 📝 Structured logging
* 🧪 Automated tests
* 🔐 Environment-based configuration
* 📚 Retrieval result and source metadata

---

# 🎯 Project Overview

The Hospital Assistant is designed to answer questions about:

* Hospital policies
* Admissions
* Insurance
* Patient services
* Other information contained in the hospital knowledge base

Instead of asking an LLM to answer directly from its internal knowledge, the application first retrieves relevant information from the hospital document collection.

```text
User Question
      │
      ▼
Document Retrieval
      │
      ▼
Relevant Hospital Context
      │
      ▼
LLM
      │
      ▼
Grounded Answer
```

This RAG approach helps keep responses grounded in the available hospital knowledge base.

---

# 🏗️ Architecture

Version 3 separates the frontend, API layer, RAG service, and vector database.

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
                  │    RAGService   │
                  └────────┬────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌───────────────┐      ┌───────────────┐
        │    Qdrant     │      │      LLM      │
        │ Vector Search │      │    Provider   │
        └───────┬───────┘      └───────────────┘
                │
                ▼
         Retrieved Context
                │
                ▼
          Context + Question
                │
                ▼
          Grounded Answer
```

---

# 📄 Document Ingestion

Hospital documents are processed and stored in the vector database through the ingestion pipeline.

```text
Hospital Documents
        │
        ▼
Document Loading
        │
        ▼
Text Extraction
        │
        ▼
Text Splitting
        │
        ▼
Embedding Generation
        │
        ├──────────────► Saved Chunks
        │                  │
        │                  ▼
        │                 BM25
        │
        ▼
      Qdrant
```

The chunks are stored separately so that BM25 keyword retrieval can be used when hybrid retrieval is enabled.

---

# 🔍 Retrieval Pipeline

The retrieval system supports two modes.

### Qdrant Retrieval

```text
User Question
      │
      ▼
Embedding
      │
      ▼
Qdrant Semantic Search
      │
      ▼
Relevant Documents
```

### Hybrid Retrieval

```text
                   User Question
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Qdrant Search           BM25 Search
        Semantic Search         Keyword Search
              │                     │
              └──────────┬──────────┘
                         ▼
              Reciprocal Rank Fusion
                         │
                         ▼
                  Deduplication
                         │
                         ▼
                Cross-Encoder
                   Reranking
                         │
                         ▼
                Top Ranked Context
```

The retrieval mode can be configured through the `RETRIEVAL_MODE` environment variable.

Supported modes:

```text
qdrant
hybrid
```

---

# 🔀 Reciprocal Rank Fusion

When hybrid retrieval is enabled, the system combines:

* Qdrant semantic search
* BM25 keyword search

The results are merged using **Reciprocal Rank Fusion (RRF)**.

RRF allows documents that appear highly ranked across multiple retrieval methods to receive stronger combined rankings.

This helps combine:

* semantic similarity
* exact keyword matching

into a single retrieval result.

---

# 🎯 Cross-Encoder Reranking

Retrieved documents can be reranked using a Cross-Encoder.

Default model:

```text
BAAI/bge-reranker-base
```

The reranker evaluates the relationship between the user query and retrieved documents and returns the most relevant documents to the generation stage.

Reranking can be enabled or disabled through:

```text
USE_RERANKER
```

The number of documents returned by the reranker can be configured using:

```text
RERANKER_TOP_N
```

---

# 🤖 Answer Generation

The application supports multiple LLM providers.

Currently supported providers include:

* Groq
* OpenAI
* OpenRouter

The provider and model can be selected through the API request and environment configuration.

The LLM receives:

```text
User Question
+
Conversation History
+
Retrieved Context
```

and generates the final response.

---

# ⚡ FastAPI Backend

FastAPI provides the REST API layer between the frontend and RAG pipeline.

The application uses a dedicated `RAGService` to keep the RAG logic separate from the API routes.

```text
FastAPI
   │
   ▼
RAGService
   │
   ├── Initialize RAG components
   ├── Select LLM provider
   ├── Run retrieval pipeline
   ├── Generate answer
   └── Return retrieval metadata
```

---

## ❤️ Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy."
}
```

---

## 💬 Chat Endpoint

```http
POST /chat
```

Example request:

```json
{
  "question": "What is the hospital visitor policy?",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "chat_history": []
}
```

The endpoint returns:

```json
{
  "answer": "Generated answer based on the hospital knowledge base.",
  "retrieval_result": {}
}
```

The actual response includes the retrieval information generated by the RAG pipeline.

---

# 🧩 FastAPI Lifespan

The application initializes `RAGService` during FastAPI startup using the **lifespan mechanism**.

```text
Application Startup
        │
        ▼
FastAPI Lifespan
        │
        ▼
RAGService()
        │
        ▼
Initialize RAG Components
        │
        ▼
Application Ready
```

This avoids repeatedly initializing expensive RAG components for every API request.

---

# 🔗 Dependency Injection

The `/chat` endpoint receives `RAGService` through FastAPI dependency injection.

```text
/chat
  │
  ▼
Depends(get_rag_service)
  │
  ▼
RAGService
```

This keeps route handling separate from service initialization and makes the application easier to test and maintain.

---

# 🗄️ Qdrant Vector Database

Version 3 migrates the vector storage layer from **FAISS to Qdrant**.

### Previous Architecture

```text
Application
    │
    ▼
FAISS
```

### Version 3

```text
Application
    │
    ▼
RAG Service
    │
    ▼
Qdrant
```

Qdrant runs as an independent service, allowing the vector database to be separated from the application.

This makes the architecture better suited for containerized and future cloud deployment.

---

# 🐳 Docker Architecture

The application uses Docker Compose to run separate services.

```text
┌────────────────────────────────────────────┐
│               Docker Compose               │
│                                            │
│  ┌─────────────────┐                       │
│  │    Streamlit    │                       │
│  │                 │                       │
│  │     :8501       │                       │
│  └────────┬────────┘                       │
│           │                                │
│           │ HTTP                           │
│           ▼                                │
│  ┌─────────────────┐                       │
│  │     FastAPI     │                       │
│  │                 │                       │
│  │     :8000       │                       │
│  └────────┬────────┘                       │
│           │                                │
│           ▼                                │
│  ┌─────────────────┐                       │
│  │     Qdrant      │                       │
│  │                 │                       │
│  │     :6333       │                       │
│  └─────────────────┘                       │
│                                            │
└────────────────────────────────────────────┘
```

### Services

| Service     | Purpose                  | Port |
| ----------- | ------------------------ | ---: |
| `streamlit` | User interface           | 8501 |
| `fastapi`   | REST API and RAG service | 8000 |
| `qdrant`    | Vector database          | 6333 |

---

# 🌐 Docker Networking

The services communicate using Docker Compose service names.

Streamlit communicates with FastAPI using:

```text
http://fastapi:8000
```

FastAPI communicates with Qdrant using:

```text
qdrant:6333
```

This is different from local development, where `localhost` is used.

```text
Inside Docker:

streamlit
    │
    ▼
fastapi:8000
    │
    ▼
qdrant:6333
```

From the host machine:

```text
localhost:8501  → Streamlit
localhost:8000  → FastAPI
localhost:6333  → Qdrant
```

---

# 💾 Persistent Vector Storage

Qdrant uses persistent storage so that vector data is not lost when the container is recreated.

```text
Docker Volume
     │
     ▼
Qdrant Storage
     │
     ▼
Vector Collection
```

The Docker Compose configuration uses a persistent Qdrant volume.

---

# 📊 RAG Evaluation

The project includes an automated evaluation framework using **RAGAS**.

The evaluation process measures both retrieval quality and generated-answer quality.

## Metrics

| Metric                | Description                                                                  |
| --------------------- | ---------------------------------------------------------------------------- |
| **Faithfulness**      | Measures whether the generated answer is supported by the retrieved context. |
| **Answer Relevancy**  | Measures how well the answer addresses the user's question.                  |
| **Context Precision** | Measures the relevance of retrieved documents.                               |
| **Context Recall**    | Measures whether the required information was retrieved.                     |

Evaluation artifacts include experiment results and comparison data used during RAG development.

---

# 🧪 Testing

The project includes automated tests covering the API and Qdrant-related functionality.

Test areas include:

* Chat endpoint
* Health endpoint
* Qdrant connectivity
* Qdrant collection
* Qdrant search
* Qdrant retrieval
* Qdrant retriever

Run the tests using:

```bash
pytest
```

---

# 🛠️ Tech Stack

| Category          | Technology                           |
| ----------------- | ------------------------------------ |
| Language          | Python                               |
| RAG Framework     | LangChain                            |
| Vector Database   | Qdrant                               |
| Embeddings        | Hugging Face / sentence-transformers |
| Keyword Retrieval | BM25                                 |
| Hybrid Retrieval  | Qdrant + BM25 + RRF                  |
| Reranker          | BAAI/bge-reranker-base               |
| LLM Providers     | Groq, OpenAI, OpenRouter             |
| API               | FastAPI                              |
| Frontend          | Streamlit                            |
| Evaluation        | RAGAS                                |
| PDF Processing    | PyMuPDF                              |
| Testing           | Pytest                               |
| Containerization  | Docker                               |
| Orchestration     | Docker Compose                       |
| Version Control   | Git & GitHub                         |

---

# 📂 Project Structure

```text
hospital-assistant-v3
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── schemas/
│   │   └── chat.py
│   │
│   ├── services/
│   │   └── rag_service.py
│   │
│   ├── dependencies.py
│   └── main.py
│
├── assets/
├── db/
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
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# ⚙️ Configuration

The application uses environment variables for configuration.

### RAG Configuration

```text
DATASET
EMBEDDING_MODEL
CHUNK_SIZE
CHUNK_OVERLAP
TOP_K
```

### Retrieval Configuration

```text
RETRIEVAL_MODE
```

Supported values:

```text
qdrant
hybrid
```

### Reranker Configuration

```text
RERANKER_MODEL
RERANKER_TOP_N
USE_RERANKER
```

### LLM Configuration

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

### Logging

```text
LOG_LEVEL
DEBUG
```

API keys are loaded through environment variables and should never be hardcoded or committed to the repository.

---

# 🏃 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/Namitha-C-Anto/hospital-assistant-v3.git
cd hospital-assistant-v3
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Add the remaining configuration variables as required.

## 5. Start Qdrant

```bash
docker compose up -d qdrant
```

## 6. Build the vector database

```bash
python build_db.py
```

This loads the hospital documents, creates chunks, generates embeddings, and stores the vectors in Qdrant.

## 7. Start FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## 8. Start Streamlit

```bash
streamlit run app.py
```

---

# 🐳 Running with Docker Compose

Build and start all services:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

Stop the services:

```bash
docker compose down
```

Access the application:

```text
Streamlit:
http://localhost:8501

FastAPI:
http://localhost:8000

Swagger:
http://localhost:8000/docs

Qdrant:
http://localhost:6333
```

---

# 🔐 Security

API keys are provided through environment variables.

Do not commit:

```text
.env
```

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
```

---

# 🔄 Project Evolution

The project was developed incrementally.

## Version 1 — Basic RAG

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
LLM
    ↓
Streamlit
```

Focused on understanding the fundamentals of RAG.

---

## Version 2 — Advanced Retrieval

```text
Documents
    ↓
FAISS + BM25
    ↓
Hybrid Retrieval
    ↓
RRF
    ↓
Cross-Encoder Reranking
    ↓
LLM
    ↓
Streamlit
```

Focused on improving retrieval quality and evaluating different retrieval strategies.

---

## Version 3 — Modular & Containerized RAG

```text
Streamlit
    ↓
FastAPI
    ↓
RAGService
    ↓
Qdrant
    ↓
Optional BM25 + RRF
    ↓
Cross-Encoder Reranking
    ↓
LLM
```

Version 3 focuses on:

* Migrating from FAISS to Qdrant
* Separating API and UI layers
* Introducing a dedicated RAG service
* Supporting configurable LLM providers
* Containerizing the application
* Running multiple services with Docker Compose
* Adding automated tests
* Preparing the application for future cloud deployment

---

# 📚 Key Engineering Concepts Demonstrated

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Document ingestion
* Text chunking
* Embeddings
* Semantic search
* Vector databases
* Qdrant
* BM25
* Hybrid retrieval
* Reciprocal Rank Fusion
* Cross-Encoder reranking
* LLM integration
* RAGAS evaluation
* FastAPI
* REST APIs
* Dependency injection
* FastAPI lifespan
* Docker
* Docker Compose
* Docker networking
* Persistent volumes
* Environment configuration
* Structured logging
* Automated testing
* Modular architecture

---

# 🚀 Future Improvements

Potential future improvements include:

* Authentication and authorization
* Metadata-based filtering
* Streaming responses
* Improved observability
* Cloud deployment
* Managed Qdrant
* Automated evaluation in CI/CD
* CI/CD pipeline
* Agentic AI workflows
* Advanced conversational memory

---

# ☁️ Deployment Architecture

The current containerized architecture can be extended to cloud infrastructure.

```text
                         Users
                           │
                           ▼
                    Application Layer
                           │
                           ▼
                        FastAPI
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
             Qdrant                   LLM
          Vector Database           Provider
```

The application, API, and vector database are separated into services, making the architecture suitable for future deployment on cloud infrastructure.

---

# 🎯 Project Goal

The goal of this project is to demonstrate the complete lifecycle of a practical RAG application:

```text
Document Ingestion
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector Database
       ↓
Retrieval
       ↓
Hybrid Search
       ↓
Reranking
       ↓
LLM Generation
       ↓
RAGAS Evaluation
       ↓
FastAPI
       ↓
Streamlit
       ↓
Docker Compose
       ↓
Deployment-Ready Architecture
```

Rather than building only a chatbot, this project demonstrates how a RAG application can be **evaluated, modularized, exposed through an API, connected to a dedicated vector database, tested, and containerized**.

---

# 👩‍💻 Author

**Namitha C Anto**

AI Engineer | Generative AI | RAG | Agentic AI

**GitHub:**
https://github.com/Namitha-C-Anto

**LinkedIn:**
https://www.linkedin.com/in/namitha-c-anto-79442b103
