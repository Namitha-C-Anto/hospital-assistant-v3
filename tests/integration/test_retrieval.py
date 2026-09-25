from rag.qdrant_retriever import load_documents_from_qdrant
from rag.qdrant_store import get_qdrant_client
from langchain_community.retrievers import BM25Retriever

def test_qdrant_chunks_and_bm25_retrieval():
        
    client = get_qdrant_client()

    documents = load_documents_from_qdrant(client)


    # Verify documents are loaded from Qdrant
    assert len(documents) > 0
    assert documents[0].page_content

    # Verify BM25 can be built from the retrieved documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    results = bm25_retriever.invoke(
        "What are the admission guidelines?"
    )

    # Verify BM25 returns results
    assert len(results) > 0
    assert len(results) <= 5

    # Verify returned documents contain content
    assert all(doc.page_content for doc in results)
    