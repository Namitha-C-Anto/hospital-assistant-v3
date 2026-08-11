import streamlit as st
from rag.models import PipelineComponents, RagPipelineResult
from rag.retrieval import run_retrieval_pipeline
from rag.generation import generate_answer
from llm.llm import get_llm
from utils.logger import logger
from config import LLM_PROVIDER, LLM_MODEL

def run_rag_pipeline(
    question: str,
    rag_components: PipelineComponents,
    chat_history: str = "",
) -> RagPipelineResult:

    """
    Execute the complete RAG pipeline.

    This includes:
    - Document retrieval
    - Deduplication
    - Reranking
    - Context construction
    - Answer generation

    Args:
        question: User question.
        rag_components: Initialized RAG components.

    Returns:
        RagPipelineResult containing the generated answer,
        retrieved documents, latency measurements, and token usage.
    """
    # -------------------------------------------------
    # 1. Retrieve and rerank relevant documents
    # -------------------------------------------------
    retrieval_result, retrieval_time, reranker_time = run_retrieval_pipeline(
            question,
            rag_components.faiss_retriever,
            rag_components.bm25_retriever, 
        )

    # Combine retrieved contexts into a single prompt for the LLM.
    context_text = "\n\n---\n\n".join(
        document.content
        for document in retrieval_result.reranked_documents
        )
    
    # -------------------------------------------------
    # 2. Create the selected LLM
    # -------------------------------------------------        
    
    logger.info(
        "LLM settings: provider=%s, model=%s, api_key_provided=%s",
        st.session_state.get("provider", LLM_PROVIDER),
        st.session_state.get("model", LLM_MODEL),
        bool(st.session_state.get("api_key")),
    )

    try:
        app_llm = get_llm(
            provider=st.session_state.get("provider", LLM_PROVIDER),
            model=st.session_state.get("model", LLM_MODEL),
            api_key=st.session_state.get("api_key"),
        )
    except Exception:
        logger.exception("Failed to initialize selected LLM.")
        st.error(
            "Unable to initialize the selected LLM. "
            "Please check the provider, model, and API key."
        )
        st.stop()

    # -------------------------------------------------
    # 3. Generate answer
    # -------------------------------------------------
    answer, usage, prompt_time, generation_time = generate_answer(
        question,
        context_text,
        app_llm,
        chat_history,
    ) 
 
    logger.info(
        "RAG pipeline completed: retrieval=%.4fs, reranker=%.4fs, "
        "prompt=%.4fs, generation=%.4fs",
        retrieval_time,
        reranker_time,
        prompt_time,
        generation_time,
    )

    return RagPipelineResult(
        answer=answer,
        context=context_text,
        retrieval_result=retrieval_result,
        usage=usage,
        retrieval_time=retrieval_time,
        reranker_time=reranker_time,
        prompt_time=prompt_time,
        generation_time=generation_time,
    )