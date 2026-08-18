from langchain_core.documents import Document
from typing import Sequence

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from rag.embeddings import get_embeddings
from utils.logger import logger

def get_qdrant_client() -> QdrantClient:
    """Create and return a local Qdrant client."""

    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

def create_collection() -> None:
    """Create the hospital documents collection if it does not exist."""

    client = get_qdrant_client()

    try:
        if client.collection_exists(COLLECTION_NAME):
            logger.info(
                "Qdrant collection %s already exists.",
                COLLECTION_NAME,
            )
            return

        embeddings = get_embeddings()

        # Generate one embedding to determine vector dimension.
        sample_vector = embeddings.embed_query("test_docuement")

        vector_size = len(sample_vector)

        logger.info(
            "Creating Qdrant collection %s with vector size %d.",
            COLLECTION_NAME,
            vector_size,
        )

        client.create_collection(
            collection_name = COLLECTION_NAME,
            vectors_config = VectorParams(
                size = vector_size,
                distance = Distance.COSINE,

            ),
        )

        logger.info(
            "Qdrant collection %s created successfully.",
            COLLECTION_NAME,
        )

    finally:
        client.close()


def add_documents(
    client: QdrantClient,
    documents: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = 32,
) -> None:
    
    """Add multiple LangChain documents to Qdrant in batches."""

    for start in range(0, len(documents), batch_size):

        batch = documents[start:start + batch_size]

        texts = [
            document.page_content
            for document in batch
        ]

        vectors = embeddings.embed_documents(texts)

        points = []

        for index, (document, vector) in enumerate(
            zip(batch, vectors),
            start=start,
        ):
            payload = {
                "text": document.page_content,
                **document.metadata,
            }

            points.append(
                PointStruct(
                    id=index,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        logger.info(
            "Inserted chunks %d-%d into Qdrant.",
            start,
            start + len(batch) - 1,
        )

def search_qdrant(
    client: QdrantClient,
    query: str,
    embeddings: HuggingFaceEmbeddings,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Search Qdrant using semantic similarity."""

    query_vector = embeddings.embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        (
            Document(
                page_content=result.payload.get("text", ""),
                metadata={
                    key: value
                    for key, value in result.payload.items()
                    if key != "text"
                },
            ),
            result.score,
        )
        for result in results
    ]