from config import COLLECTION_NAME
from langchain_core.documents import Document
from rag.qdrant_store import (
    get_qdrant_client,
    search_qdrant,
)
from rag.embeddings import get_embeddings


def test_search_qdrant():

    client = get_qdrant_client()
    embeddings = get_embeddings()

    try:
        results = search_qdrant(
            client=client,
            query="What are the admission guidelines?",
            embeddings=embeddings,
            top_k=5,
        )

        print("\nSearch results:")

        for document, score in results:
            print("\nScore:", score)
            print("Source:", document.metadata.get("source"))
            print("Page:", document.metadata.get("page"))
            print("Text:", document.page_content[:300])

        assert len(results) == 5
        assert all(
            isinstance(document, Document)
            for document, score in results
        )


    finally:
        client.close()