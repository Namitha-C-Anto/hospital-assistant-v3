from config import CHUNKS_PATH
from rag.qdrant_store import (
    COLLECTION_NAME,
    add_documents,
    get_qdrant_client,
)
from rag.storage import load_chunks
from rag.vectorstore import get_embeddings


def test_ingest_all_chunks():

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = get_embeddings()
    client = get_qdrant_client()

    try:
        add_documents(
            client=client,
            documents=chunks,
            embeddings=embeddings,
            batch_size=32,
        )

        collection = client.get_collection(COLLECTION_NAME)

        print(f"\nNumber of chunks: {len(chunks)}")
        print(f"Points count: {collection.points_count}")

        assert collection.points_count == len(chunks)

    finally:
        client.close()