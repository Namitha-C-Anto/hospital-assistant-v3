from qdrant_client import QdrantClient

from config import QDRANT_PATH
from rag.qdrant_store import (
    COLLECTION_NAME,
    get_qdrant_client,
    get_embeddings,
)


def test_search_qdrant():
    client = get_qdrant_client()

    try:
        embeddings = get_embeddings()

        query = "What are the guidelines for hospital admission?"

        query_vector = embeddings.embed_query(query)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1,
        )

        print("\nSearch result:")
        print(results)

        assert len(results.points) == 1

        point = results.points[0]

        print("\nPayload:")
        print(point.payload)

        assert point.payload is not None
        assert "text" in point.payload
        assert "source" in point.payload

    finally:
        client.close()