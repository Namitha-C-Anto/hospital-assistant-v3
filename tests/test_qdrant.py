from qdrant_client import QdrantClient
from rag.qdrant_store import get_qdrant_client


def test_qdrant_local():
    client = get_qdrant_client()

    collections = client.get_collections()

    print(collections)

    client.close()