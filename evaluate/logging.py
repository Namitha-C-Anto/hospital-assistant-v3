from utils.logger import logger
from rag.models import RetrievalResult
import pandas as pd

def log_pipeline_stats(
    question: str,
    retrieval_time: float,
    reranker_time: float,
    generation_time: float,
    pipeline_time: float,
) -> None:
     
    logger.info(
        "Question: %s; Retrieval time = %.4f s; Reranker time = %.4f s; Generation time = %.4f s; Pipeline time = %.4f s.",
        question,
        retrieval_time,
        reranker_time,
        generation_time,
        pipeline_time,
        )

#-------------------------------------------------------------
def log_debug_info(
    question:str,
    answer:str,
    reference:str,
    context_text:str,
    retrieval_result:RetrievalResult,
) -> None:

    """Log detailed information about a RAG pipeline result."""
    
    logger.info("=" * 80)

    logger.info("Question: %s", question) 

    logger.info("Response: %s", answer)

    logger.info("Reference: %s", reference) 

    logger.info("Context: %s", context_text)

    logger.info("=" * 80)

    for i, ctx in enumerate(retrieval_result.reranked_documents, 1):
        logger.info("Chunk %d", i)
        logger.info("Metadata: %s", ctx.metadata)
        logger.info("Content: %s", ctx.content)

#--------------------------------------------------------------------------
def log_df_info(df: pd.DataFrame) -> None:

    columns = [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    available_columns = [
        col for col in columns
        if col in df.columns
    ]

    
    logger.info(
        "\n%s",
        df[available_columns].to_string(index=False)
    )

#------------------------------------------------------------------------
def log_avg_scores(summary: dict[str, float]) -> None:
    logger.info("\n========== AVERAGE SCORES ==========")
    
    for metric, score in summary.items():
        logger.info(f"{metric}: {score}")
