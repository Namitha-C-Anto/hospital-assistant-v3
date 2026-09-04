from config import COLLECTION_NAME
from rag.qdrant_store import(
    get_qdrant_client,
    create_collection,
)
from rag.embeddings import get_embeddings

def test_create_qdrant_collection():

    client = get_qdrant_client()
    embeddings = get_embeddings()

    create_collection(client, embeddings)

    try:
        assert client.collection_exists(COLLECTION_NAME)

        collection = client.get_collection(COLLECTION_NAME)

        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"Collection info: {collection}")

    finally:
        client.close()