from rag.qdrant_store import get_qdrant_client

def test_qdrant_local():

    try:
        client = get_qdrant_client()

        collections = client.get_collections()
    
        assert len(collections.collections) > 0 

    finally:

        client.close()