from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from rag.qdrant_retriever import QdrantRetriever, load_documents_from_qdrant
from rag.qdrant_store import get_qdrant_client
from rag.embeddings import get_embeddings

from utils.logger import logger

from config import (
    TOP_K,
    RETRIEVAL_MODE,
) 
from rag.rrf import reciprocal_rank_fusion
 

def create_retriever() -> dict[str, BaseRetriever | None]:

    """
    Create retrievers based on the configured retrieval mode.

    Returns:
        Dictionary containing:
            - "qdrant": Qdrant retriever
            - "bm25": BM25 retriever or None
    """

    logger.info("Creating retriever using '%s'.", RETRIEVAL_MODE)

    if RETRIEVAL_MODE not in {"qdrant", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode: {RETRIEVAL_MODE}")

    embeddings = get_embeddings()
    client = get_qdrant_client()

    qdrant_retriever = QdrantRetriever(
        client=client,
        embeddings=embeddings,
        top_k=TOP_K,
    )

    bm25_retriever = None
    
    if RETRIEVAL_MODE == "hybrid": 
        documents = load_documents_from_qdrant(client)

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = TOP_K

    return {
        "qdrant": qdrant_retriever,
        "bm25": bm25_retriever,
    }

##-----------------------------------------------------------------------------------------

def retrieve_documents(
    question: str,
    qdrant_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever | None,
) -> list[Document]:

    """
    Retrieve documents using the configured retrieval mode.

    In Qdrant mode, retrieves semantically relevant documents using
    the Qdrant vector retriever. In hybrid mode, combines Qdrant
    semantic search with BM25 keyword search using Reciprocal Rank
    Fusion (RRF).

    Args:
        question: User's query.
        qdrant_retriever: Qdrant-based semantic retriever.
        bm25_retriever: BM25 keyword retriever, required for hybrid mode.

    Returns:
        A list of retrieved LangChain Document objects.
    """

    question = question.strip()

    if RETRIEVAL_MODE == "qdrant":
        return qdrant_retriever.invoke(question)

    if bm25_retriever is None:
        raise ValueError(
            "Hybrid retrieval requires a BM25 retriever."
        )
        
    qdrant_docs = qdrant_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    return reciprocal_rank_fusion(
        [qdrant_docs, bm25_docs],
        top_n=TOP_K,
    )