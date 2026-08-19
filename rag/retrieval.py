import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import USE_RERANKER, TOP_K, RETRIEVAL_MODE
from rag.models import RetrievalResult, DocumentInfo
from rag.retriever import retrieve_documents
from rag.reranker import reranker

##--------------------DEDUPLICATION
def deduplicate_documents(
    documents: list[Document]
) -> list[Document]:

    """Remove duplicate documents based on source, page and chunk_id."""
         
    seen = set()
    unique_documents = []

    for document in documents:
        
        key = (
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.metadata.get("chunk_id", document.page_content)
        )

        if key not in seen:
            seen.add(key)
            unique_documents.append(document)

    return unique_documents

# ----------------------------------------------------------------------
def build_retrieval_result(
    retrieved_documents: list[Document], 
    reranked_documents: list[Document]
) -> RetrievalResult:
    
    """
    Convert retrieved and reranked LangChain documents into a RetrievalResult.
    """

    return RetrievalResult(
        retrieved_documents=[
            DocumentInfo(
                    content=document.page_content,
                    metadata=document.metadata,
                )
                for document in retrieved_documents
            ],
            reranked_documents=[
                DocumentInfo(
                    content=document.page_content,
                    metadata=document.metadata,
                )
                for document in reranked_documents
            ],
        )

# -------------------------------------------------------------------------------
def run_retrieval_pipeline(
    question: str,
    qdrant_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever | None,
    retrieval_mode: str = RETRIEVAL_MODE,
    use_reranker: bool = USE_RERANKER,
) -> tuple[RetrievalResult, float, float]:

    """
    Run the document retrieval pipeline.

    The pipeline supports semantic retrieval using Qdrant and hybrid
    retrieval using Qdrant and BM25 with Reciprocal Rank Fusion (RRF).
    Retrieved documents are deduplicated and can optionally be reranked
    using a cross-encoder reranker.

    Args:
        question: User's query.
        qdrant_retriever: Qdrant-based semantic retriever.
        bm25_retriever: BM25 keyword retriever, or None when hybrid
            retrieval is not used.
        retrieval_mode: Retrieval strategy to use. Supported values are
            "qdrant" for semantic search and "hybrid" for hybrid search.
        use_reranker: Whether to apply the optional cross-encoder
            reranker to the retrieved documents.

    Returns:
        A tuple containing:
            - RetrievalResult containing the retrieved and optionally
              reranked documents.
            - Retrieval latency in seconds.
            - Reranker latency in seconds.
    """
    
    # ----------------------------------------
    # Retrieve documents
    # ---------------------------------------- 
    retrieval_start = time.perf_counter()
    retrieved_documents = retrieve_documents(
        question,
        qdrant_retriever,
        bm25_retriever,
        retrieval_mode,
    )
    retrieval_time = round(
        time.perf_counter() - retrieval_start, 
        4,
    )

    # ----------------------------------------
    # Remove duplicate documents
    # ----------------------------------------
    unique_documents = deduplicate_documents(retrieved_documents)

    # ----------------------------------------
    # Rerank documents (optional)
    # ----------------------------------------
    reranker_time = 0.0

    # Rerank them
    if use_reranker:
        reranker_start = time.perf_counter()

        reranked_documents  = reranker.compress_documents(
            documents=unique_documents,
            query=question
        )
        reranker_time = round(
            time.perf_counter() - reranker_start,
            4,
        )
    else:
        reranked_documents  = unique_documents[:TOP_K]

    # ----------------------------------------
    # Build retrieval metadata
    # ----------------------------------------
    retrieval_result = build_retrieval_result(
        retrieved_documents=unique_documents,
        reranked_documents=reranked_documents,
    )
 
    return (
        retrieval_result,
        retrieval_time,
        reranker_time,
    )