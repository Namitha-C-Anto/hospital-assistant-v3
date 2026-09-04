from typing import Any

from utils.logger import logger
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rag.qdrant_store import search_qdrant
from config import COLLECTION_NAME


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
            for document, _ in results
        ]

# ----------------------------------------------------------------------------------------

def load_documents_from_qdrant(
    client: QdrantClient,
) -> list[Document]:
    """Load all document chunks from Qdrant."""

    documents = []
    offset = None

    try:
        logger.info(
            "Loading documents from Qdrant collection %s.",
            COLLECTION_NAME,
        )

        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                documents.append(
                    Document(
                        page_content=payload.get("text", ""),
                        metadata={
                            key: value
                            for key, value in payload.items()
                            if key != "text"
                        },
                    )
                )

            if offset is None:
                break

        logger.info(
            "Loaded %d chunks from Qdrant.",
            len(documents),
        )

        return documents

    except Exception:
        logger.exception(
            "Failed to load documents from Qdrant."
        )
        raise