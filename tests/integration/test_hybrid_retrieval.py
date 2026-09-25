from rag.initializer import initialize_rag

def test_hybrid_retrieval():
 
    retrievers = initialize_rag()
 
    query = "What are the key processes for effective elective admissions?"

    qdrant_results = retrievers.qdrant_retriever.invoke(query)
    bm25_results = retrievers.bm25_retriever.invoke(query)

    assert len(qdrant_results) > 0
    assert len(bm25_results) > 0
 
    assert qdrant_results[0].page_content
    assert bm25_results[0].page_content 