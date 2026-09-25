from config import COLLECTION_NAME
from rag.qdrant_store import(
    get_qdrant_client,
    create_collection,
)
from rag.embeddings import get_embeddings

def test_create_qdrant_collection():

    client = get_qdrant_client()
    embeddings = get_embeddings()

    try:
        create_collection(client, embeddings)

        assert client.collection_exists(COLLECTION_NAME)

        collection = client.get_collection(COLLECTION_NAME)

        assert collection is not None 

    finally:
        client.close()