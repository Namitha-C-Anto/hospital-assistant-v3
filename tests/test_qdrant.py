from qdrant_client import QdrantClient
from config import QDRANT_PATH


def test_qdrant_local():
    client = QdrantClient(path = QDRANT_PATH)

    collections = client.get_collections()

    print(collections)

    client.close()