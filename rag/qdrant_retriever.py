from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rag.qdrant_store import search_qdrant


class QdrantRetriever(BaseRetriever):
    """LangChain retriever backed by Qdrant."""

    client: QdrantClient
    embeddings: HuggingFaceEmbeddings
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:

        results = search_qdrant(
            client=self.client,
            query=query,
            embeddings=self.embeddings,
            top_k=self.top_k,
        )

        return [
            document
            for document, score in results
        ]