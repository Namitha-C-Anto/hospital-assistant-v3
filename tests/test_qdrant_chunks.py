from rag.qdrant_retriever import load_documents_from_qdrant
from rag.qdrant_store import get_qdrant_client
from langchain_community.retrievers import BM25Retriever

client = get_qdrant_client()

documents = load_documents_from_qdrant(client)

print("Number of documents:", len(documents))

print(documents[0])

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

results = bm25_retriever.invoke(
    "What are the admission guidelines?"
)

print("BM25 results:", len(results))

for doc in results:
    print(doc)