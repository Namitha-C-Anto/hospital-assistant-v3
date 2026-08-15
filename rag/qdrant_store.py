from langchain_core.documents import Document
from typing import Sequence

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_PATH
from rag.vectorstore import get_embeddings
from utils.logger import logger

COLLECTION_NAME = "hospital_documents"

def get_qdrant_client() -> QdrantClient:
    """Create and return a local Qdrant client."""

    return QdrantClient(path = str(QDRANT_PATH))

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


def add_document(
    client: QdrantClient,
    document: Document,
    point_id: int,
    embeddings: HuggingFaceEmbeddings,
) -> None:

    """Add a single langchain document to Qdrant."""
 
    vector = embeddings.embed_query(document.page_content)

    payload = {
        "text": document.page_content,
        **document.metadata,
    }

    client.upsert(
        collection_name = COLLECTION_NAME,
        points = [
            PointStruct(
                id = point_id,
                vector = vector,
                payload = payload,
            )
        ],
    )

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