from config import COLLECTION_NAME
from rag.qdrant_store import(
    get_qdrant_client,
    create_collection,
)

def test_create_qdrant_collection():

    create_collection()

    client = get_qdrant_client()

    try:
        assert client.collection_exists(COLLECTION_NAME)

        collection = client.get_collection(COLLECTION_NAME)

        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"Collection info: {collection}")

    finally:
        client.close()