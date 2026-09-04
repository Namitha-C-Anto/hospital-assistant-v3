from config import DOCS_PATH
 
from rag.loader import load_documents_from_folder
from rag.splitter import split_documents
from rag.qdrant_store import (
    get_qdrant_client,
    create_collection,
    add_documents,
)
from rag.embeddings import get_embeddings
from utils.logger import logger

def build_vector_database() -> None:
    
    """
    Load, chunk, embed, and ingest documents into Qdrant.

    Raises:
        Exception: If any step in the vector database build process fails.
    """

    client = None


    try:
        logger.info("Building vector database.")

        # Step 1: Load documents
        documents = load_documents_from_folder(DOCS_PATH)  

        # Step 2: Split into chunks 
        chunks = split_documents(documents)  

        # Step 3: Create Qdrant client and embedding model
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Step 4: Create Qdrant collection
        create_collection(client, embeddings)

        # Step 5: Insert chunks into Qdrant
        add_documents(
            client=client,
            documents=chunks,
            embeddings=embeddings,
        )

        logger.info("Vector database created successfully.") 

    except Exception:
        logger.exception("Failed to build vector database.")
        raise

    finally:
        if client is not None:
            client.close()

