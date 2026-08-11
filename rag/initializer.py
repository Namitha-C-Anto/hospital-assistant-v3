from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    JUDGE_PROVIDER, 
    JUDGE_MODEL, 
    JUDGE_API_KEY, 
    EMBEDDING_MODEL,)

from rag.vectorstore import load_vectorstore
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
    - Loading the persisted vector store
    - Creating retrieval components (FAISS + BM25)
    - Initializing the application LLM 

    Returns:
        PipelineComponents: Container holding all initialized pipeline objects.
    """
    try: 
        
        logger.info("Initializing RAG components.")
        # -------------------------------------------------
        # 1. Load vector store and create retrievers
        # -------------------------------------------------
        vectorstore = load_vectorstore()

        retriever = create_retriever(vectorstore)

        # Individual retrievers available for experimentation and retrieval strategies
        faiss_retriever = retriever["faiss"]
        bm25_retriever = retriever["bm25"]

        # # -------------------------------------------------
        # # 2. Initialize the application LLM used for answer generation
        # # -------------------------------------------------
        # app_llm = ChatOpenAI(
        #     model=LLM_MODEL,
        #     api_key=OPENAI_API_KEY,
        #     )

        # -------------------------------------------------
        # 3. Return all initialized components
        # -------------------------------------------------
        pipeline_components =  PipelineComponents(
            vectorstore=vectorstore,
            retriever=retriever,
            faiss_retriever=faiss_retriever,
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
    
