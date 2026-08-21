import time
from typing import Any
from langchain_core.language_models import BaseChatModel 
from config import (
    DEBUG,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_API_KEY,
    USE_RERANKER,
    RETRIEVAL_MODE)

from utils.logger import logger
from evaluate.logging import (
    log_pipeline_stats, 
    log_debug_info,)
from rag.models import (
    PipelineResults, 
    Latency, 
    RetrievalStats, 
    TokenUsage, 
    PipelineComponents,)
from rag.pipeline import run_rag_pipeline
from llm.llm import get_llm

async def run_test_questions(
    test_data: list[dict[str, Any]],
    rag_components: PipelineComponents,
) -> list[PipelineResults]:

    """
    Run the RAG pipeline for a collection of evaluation test questions.

    The application LLM is initialized once and reused for all test
    questions. Each question is processed through the complete RAG
    pipeline, and the resulting answers, retrieval information,
    latency measurements, and token usage are collected into
    structured PipelineResults objects.

    If an individual question fails during processing, the error is
    logged and that question is skipped so that the remaining test
    questions can continue to be evaluated.

    Args:
        test_data: List of test cases containing questions, ground-truth
            answers, and related evaluation information.
        rag_components: Initialized RAG components required for document
            retrieval and reranking.

    Returns:
        List of PipelineResults for successfully processed test
        questions. Failed questions are excluded from the returned list.

    Raises:
        Exception: If the application LLM cannot be initialized.
        
    """
    pipeline_results = []

    logger.info(
        "LLM settings: provider=%s, model=%s, api_key_provided=%s",
        LLM_PROVIDER,
        LLM_MODEL,
        bool(LLM_API_KEY),
    )
    
    try:
        app_llm = get_llm(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
        )
    except Exception:
        logger.exception("Failed to initialize the LLM.")
        raise

    logger.info("Running RAG on test questions.")

    # Execute the RAG pipeline for each test question.
    for item in test_data:

        logger.info("Processing test question: '%s'.", item["id"])
        result = await process_test_questions(item, rag_components, app_llm)

        # Skip questions that failed during pipeline execution.
        if result is not None:
            pipeline_results.append(result)

    return pipeline_results

#------------------------------------------------------
async def process_test_questions(
    item: dict[str, Any],
    rag_components: PipelineComponents,
    app_llm: BaseChatModel,
) -> PipelineResults | None:

    """
    Execute the complete RAG pipeline for a single test question.

    This includes:
    - Document retrieval
    - Reranking
    - Answer generation
    - Latency measurement
    - Token usage tracking

    Args:
        item: Test question containing the input question and ground truth.
        rag_components: Initialized RAG pipeline components.

    Returns:
        PipelineResults containing the generated answer and evaluation
        metadata, or None if the pipeline execution fails.
    """

    question = item["question"]
    # Start timer to measure end-to-end pipeline latency.
    pipeline_start = time.perf_counter()

    try:
        
        pipeline_result = await run_rag_pipeline(
            question,
            rag_components,
            app_llm,
            RETRIEVAL_MODE,
            USE_RERANKER,
        )

        # Calculate total pipeline execution time.
        pipeline_time = round(time.perf_counter() - pipeline_start,4)

        # -------------------------------------------------
        # 3. Build structured pipeline result
        # -------------------------------------------------
        result = PipelineResults(
                    question=question,
                    answer=pipeline_result.answer,
                    reference=item["ground_truth"],
                    ground_truth = item["ground_truth"],
                    
                    latency=Latency(
                        retrieval_seconds=pipeline_result.retrieval_time,
                        reranker_seconds=pipeline_result.reranker_time,
                        generation_seconds=pipeline_result.generation_time,
                        pipeline_seconds=pipeline_time,
                        prompt_seconds=pipeline_result.prompt_time,
                    ),

                    retrieval_stats=RetrievalStats(
                        retrieved=len(pipeline_result.retrieval_result.retrieved_documents),
                        after_reranker=len(pipeline_result.retrieval_result.reranked_documents)
                    ),

                    retrieval=pipeline_result.retrieval_result, 

                    usage=TokenUsage(
                        prompt_tokens=pipeline_result.usage.get("prompt_tokens", 0),
                        completion_tokens=pipeline_result.usage.get("completion_tokens", 0),
                        total_tokens=pipeline_result.usage.get("total_tokens", 0),
                    ), 
                    metrics = None,
                )

        # Log pipeline performance metrics.
        log_pipeline_stats(
            question,
            pipeline_result.retrieval_time,
            pipeline_result.reranker_time,
            pipeline_result.generation_time,
            pipeline_time,
        )

        # Log detailed debugging information when debug mode is enabled.
        
        if DEBUG:
            log_debug_info(
                question,
                pipeline_result.answer,
                item["ground_truth"],
                pipeline_result.context,
                pipeline_result.retrieval_result,
            )
        return result

    except Exception:
        logger.exception("Failed to process test question: %s", question)
        return None