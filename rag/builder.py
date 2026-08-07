from config import (
    DOCS_PATH, 
    CHUNKS_PATH,)
 
from rag.loader import load_documents_from_folder
from rag.splitter import split_documents
from rag.storage import save_chunks
from rag.vectorstore import (
    create_vectorstore, 
    save_vectorstore,)
from utils.logger import logger

def build_vector_database() -> None:
    
    """
    Load documents, split them into chunks, save the chunks,
    create a FAISS vector database, and persist it to disk.
    """

    try:
        logger.info("Building vector database.")

        # Step 1: Load documents
        documents = load_documents_from_folder(DOCS_PATH)  

        # Step 2: Split into chunks 
        chunks = split_documents(documents)  

        # Save chunks for BM25 / hybrid retrieval. 
        save_chunks(chunks, CHUNKS_PATH)

        # Step 3: Create vector store 
        vectorstore = create_vectorstore(chunks)

        # Step 4: Save vector DB 
        save_vectorstore(vectorstore)

        logger.info("Vector database created successfully.") 

    except Exception:
        logger.exception("Failed to build vector database.")
        raise

