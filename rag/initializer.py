from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    JUDGE_PROVIDER, 
    JUDGE_MODEL, 
    JUDGE_API_KEY, 
    EMBEDDING_MODEL,)
 
from rag.retriever import create_retriever
from rag.models import PipelineComponents, EvaluationComponents
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper 
from utils.logger import logger
from llm.llm import get_llm

def initialize_rag() -> PipelineComponents:
    
    """
    Initialize and return all components required for the RAG pipeline.

    This includes:
    - Creating the Qdrant semantic retriever
    - Creating the optional BM25 keyword retriever
    - Initializing the application RAG components

    Returns:
        PipelineComponents: Container holding the initialized
        retrieval components.
    """
    try: 
        
        logger.info("Initializing RAG components.")
        # -------------------------------------------------
        # 1. Load vector store and create retrievers
        # -------------------------------------------------
        retriever = create_retriever()

        # Individual retrievers available for experimentation and retrieval strategies
        qdrant_retriever = retriever["qdrant"]
        bm25_retriever = retriever["bm25"]

        # -------------------------------------------------
        # 2. Return all initialized components
        # -------------------------------------------------
        pipeline_components =  PipelineComponents( 
            retriever=retriever,
            qdrant_retriever=qdrant_retriever,
            bm25_retriever=bm25_retriever, 
        )

        logger.info("RAG components initialized successfully.")
        return pipeline_components

    except Exception:
        logger.exception("Failed to initialize RAG components.")
        raise
    
# -----------------------------------------------------------------------------------------------
def initialize_ragas() -> EvaluationComponents:
    """
    Initialize and return all components required for the RAGAS Evaluation.

    This includes:
    - Initializing the evaluation (RAGAS) LLM and embedding model

    Returns:
        EvaluationComponents: Container holding all initialized ragas evaluation objects.
    """

    # -------------------------------------------------
    # 1. Initialize LLM used as the RAGAS evaluation judge
    # -------------------------------------------------
    
    logger.info("Initializing RAGAS components.")

    logger.info(
        "Initializing RAGAS judge: provider = '%s', model = '%s', api_key_provided = '%s'.",
        JUDGE_PROVIDER,
        JUDGE_MODEL,
        bool(JUDGE_API_KEY)
        )

    judge_llm = get_llm(
        provider= JUDGE_PROVIDER,
        model=JUDGE_MODEL,
        api_key=JUDGE_API_KEY,
    )

    judge_llm.temperature = 0
    # judge_llm.max_tokens = 4096
    
    # -------------------------------------------------
    # 2. Initialize embedding model for RAGAS metrics
    # -------------------------------------------------
    ragas_llm = LangchainLLMWrapper(judge_llm) 

    logger.info("Initializing embeddings: model = '%s'", EMBEDDING_MODEL)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        hf_embeddings
    )

    logger.info("RAGAS evaluation components initialized successfully.")
    # -------------------------------------------------
    # 3. Return all initialized components
    # -------------------------------------------------
    return EvaluationComponents(
        judge_llm=judge_llm,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )
    
