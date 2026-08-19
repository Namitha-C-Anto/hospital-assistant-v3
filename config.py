import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv(override=True)

#---------------------------------------------------
# Logging Configuration
#---------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s | %(levelname)s | %(name)s | %(message)s") 

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET = os.getenv("DATASET", "dataset_v1")

DOCS_PATH = BASE_DIR / "docs" / DATASET  

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "hospital_documents"

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

EVALUATION_PATH = BASE_DIR / "evaluate" / "evaluation_results"
RAGAS_RESULTS_PATH = EVALUATION_PATH / "ragas_results" / DATASET
COMPARISON_PATH = EVALUATION_PATH / "comparisons"
REPORT_PATH = EVALUATION_PATH / "reports"
DATASET_PATH = EVALUATION_PATH / "datasets"

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# --------------------------------------------------
# Text Splitting
# --------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# --------------------------------------------------
# Retriever Configuration
# --------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 5))
# --------------------------------------------------
# Reranker Configuration
# --------------------------------------------------

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)

RERANKER_TOP_N = int(
    os.getenv("RERANKER_TOP_N", 3)
)

#--------------------------------------------------Toggle
USE_RERANKER = os.getenv(
    "USE_RERANKER",
    "True"
).lower() == "true"

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "qdrant").lower()
 
DEBUG = os.getenv(
    "DEBUG",
    "false"
).lower() == "true"

# --------------------------------------------------
# Large Language Model
# --------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.getenv("GROQ_API_KEY")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
# --------------------------------------------------
# Evaluation LLM
# --------------------------------------------------
JUDGE_PROVIDER = os.getenv(
    "JUDGE_PROVIDER",
    "groq"
)
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    LLM_MODEL
)
JUDGE_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# Streamlit Application
# --------------------------------------------------

# 1. Define the variable with the SVG code
APP_TITLE = """
        <h2>
        <svg width="35" height="35" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block; vertical-align: -6px; margin-right:5px;">
            <rect x="2" y="10" width="13" height="5" rx="1.5" fill="#EE0000"/>
            <rect x="5" y="7" width="6" height="12" rx="1.5" fill="#FF0000"/>
            <path d="M 19 1 Q 19 6 24 6 Q 19 6 19 11 Q 19 6 14 6 Q 19 6 19 1 Z" fill="#FF0000"/>
            <path d="M 15 0 Q 15 2 17 2 Q 15 2 15 4 Q 15 2 13 2 Q 15 2 15 0 Z" fill="#FF0000"/>
            <path d="M 22 9 Q 22 11 24 11 Q 22 11 22 13 Q 22 11 20 11 Q 22 11 22 9 Z" fill="#FF0000"/>
        </svg>
        Hospital Assistant
        </h2>
        """ 
SUB_TITLE= """
        <h2 style="margin-bottom:0;">👋 Welcome!</h2>

        <p style="font-size:18px; color:#B0B0B0; margin-top:0.5rem;">
        Ask questions about <b>hospital policies</b>,
        <b>admissions</b>, <b>insurance</b>, and
        <b>patient services</b>.
        </p>
        """

WELCOME_TITLE = "Welcome!"

WELCOME_SUBTITLE = (
    "Ask questions about hospital policies, admissions, insurance, and patient services."
)

WELCOME_CAPTION = "⚡AI-powered answers grounded in your hospital knowledge base."
# --------------------------------------------------
# API Keys
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --------------------------------------------------
# Validation
# --------------------------------------------------
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found.")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found.")

if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY not found.")

RUN_DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
