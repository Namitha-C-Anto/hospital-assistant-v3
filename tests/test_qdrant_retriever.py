from langchain_core.documents import Document

from rag.qdrant_retriever import QdrantRetriever
from rag.qdrant_store import get_qdrant_client
from rag.embeddings import get_embeddings


def test_qdrant_retriever():

    client = get_qdrant_client()
    embeddings = get_embeddings()

    try:
        retriever = QdrantRetriever(
            client=client,
            embeddings=embeddings,
            top_k=5,
        )

        documents = retriever.invoke(
            "What are the admission guidelines?"
        )

        print("\nRetrieved documents:")

        for document in documents:
            print("\nSource:", document.metadata.get("source"))
            print("Page:", document.metadata.get("page"))
            print("Text:", document.page_content[:300])

        assert len(documents) == 5
        assert all(
            isinstance(document, Document)
            for document in documents
        )

    finally:
        client.close()