from langchain_core import embeddings
from config import CHUNKS_PATH
from rag.qdrant_store import (
    COLLECTION_NAME,
    add_document,
    get_qdrant_client,
)
from rag.storage import load_chunks
from rag.vectorstore import get_embeddings


def test_insert_document():
    chunks = load_chunks(CHUNKS_PATH)

    client = get_qdrant_client()
    embeddings = get_embeddings()

    try:
        add_document(
            client=client,
            document=chunks[0],
            point_id=1,
            embeddings=embeddings,
        )

        collection = client.get_collection(COLLECTION_NAME)

        print(f"\nPoints count: {collection.points_count}")

        assert collection.points_count == 1

    finally:
        client.close()
