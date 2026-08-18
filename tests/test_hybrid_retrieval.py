from rag.initializer import initialize_rag
from config import RETRIEVAL_MODE

 
print(f"Retrieval mode: {RETRIEVAL_MODE}")

retrievers = initialize_rag()

print("Qdrant retriever:", retrievers.qdrant_retriever)
print("BM25 retriever:", retrievers.bm25_retriever)

query = "What are the key processes for effective elective admissions?"

qdrant_results = retrievers.qdrant_retriever.invoke(query)
bm25_results = retrievers.bm25_retriever.invoke(query)

print("\nQdrant results:", len(qdrant_results))
print("BM25 results:", len(bm25_results))

print("\n--- Qdrant Result ---")
print(qdrant_results[0])

print("\n--- BM25 Result ---")
print(bm25_results[0])