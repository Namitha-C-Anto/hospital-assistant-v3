from config import RETRIEVAL_MODE, USE_RERANKER
from rag.models import PipelineComponents, RagPipelineResult
from rag.retrieval import run_retrieval_pipeline
from rag.generation import generate_answer
from utils.logger import logger
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

async def run_rag_pipeline(
    question: str,
    rag_components: PipelineComponents,
    app_llm: BaseChatModel,
    chat_history: list[BaseMessage] | None = None,
    retrieval_mode: str = RETRIEVAL_MODE,
    use_reranker:bool = USE_RERANKER,
) -> RagPipelineResult:

    """
   
    Execute the complete RAG pipeline.

    The pipeline retrieves relevant documents, optionally reranks them,
    constructs the LLM context, and generates a grounded answer.

    Args:
        question: Current user question.
        rag_components: Initialized retrieval components.
        app_llm: Configured chat model used for answer generation.
        chat_history: Recent conversation history as LangChain messages.
        retrieval_mode: Retrieval strategy to use.
        use_reranker: Whether to apply the cross-encoder reranker.

    Returns:
        RagPipelineResult containing the generated answer, retrieval
        results, context, token usage, and latency measurements.
    """
    
    # -------------------------------------------------
    # 1. Retrieve and rerank relevant documents
    # -------------------------------------------------
    retrieval_result, retrieval_time, reranker_time = run_retrieval_pipeline(
            question,
            rag_components.qdrant_retriever,
            rag_components.bm25_retriever, 
            retrieval_mode,
            use_reranker,
        )

    # Combine retrieved contexts into a single prompt for the LLM.
    context_text = "\n\n---\n\n".join(
        document.content
        for document in retrieval_result.reranked_documents
        )

    # -------------------------------------------------
    # 3. Generate answer
    # -------------------------------------------------
    answer, usage, prompt_time, generation_time = await generate_answer(
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