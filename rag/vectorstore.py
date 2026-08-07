from pathlib import Path
from typing import Sequence
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    EMBEDDING_MODEL, 
    DB_PATH,)
from utils.logger import logger

#------------------------------------------------------------------------------------------
#Create embedding model
#------------------------------------------------------------------------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and return the configured embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


#------------------------------------------------------------------------------------------
#Create Vector Store
#------------------------------------------------------------------------------------------
def create_vectorstore(documents: Sequence[Document]) -> FAISS:
    """
    Create a FAISS vector store from the provided document chunks.

    Args:
        documents: Collection of document chunks.

    Returns:
        Initialized FAISS vector store.
    """

    try:

        logger.info("Creating vector store from %d chunks.", len(documents))

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(
            documents=documents, 
            embedding=embeddings,
            )

        logger.info("Vector store created successfully.")

        return vectorstore

    except Exception:
        logger.exception("Failed to create vector store.")
        raise
     
#------------------------------------------------------------------------------------------
#Save Vector Store
#------------------------------------------------------------------------------------------
def save_vectorstore(
    vectorstore: FAISS, 
    path: str | Path = DB_PATH
) -> None:
    """
    Save the FAISS vector store to disk.

    Args:
        vectorstore: FAISS vector store to save.
        path: Destination directory.
    """

    try:
        logger.info("Saving vector store to '%s'.", path)

        vectorstore.save_local(path)
        logger.info("Vector store saved to '%s'.", path)

    except Exception:
        logger.exception("Failed to save vector store.")
        raise

#---------------------------------------------------------------------------------------
def load_vectorstore(path: str = DB_PATH) -> FAISS:
    """
    Load and return the FAISS vector database from disk.
    """

    logger.info(f"Loading vector database from '{path}'.")

    embeddings = get_embeddings()
    return FAISS.load_local(
        path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )