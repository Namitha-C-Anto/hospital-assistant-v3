from typing import Sequence
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.logger import logger

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def split_documents(
    documents: Sequence[Document],
) -> list[Document]:
    """
    Split documents into overlapping chunks for indexing.

    Args:
        documents: Collection of loaded documents.

    Returns:
        List of chunked documents.
    """
    try:

        logger.info(
            "Splitting %d documents.",
            len(documents),
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        chunks = text_splitter.split_documents(documents)

        logger.info(
            "Split %d documents into %d chunks.",
            len(documents),
            len(chunks),
        )

        return chunks

    except Exception:
        logger.exception("Failed to split documents.")
        raise