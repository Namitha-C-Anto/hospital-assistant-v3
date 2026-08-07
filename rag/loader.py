from pathlib import Path
import fitz
import re
import hashlib
from langchain_core.documents import Document 
from utils.logger import logger

def load_documents_from_folder(
    folder_path: str | Path
) -> list[Document]:

    """
    Load PDF documents from a folder.

    Each page is extracted, normalized, and converted into a
    LangChain Document with metadata for the source PDF,
    page number, and document identifier.

    Args:
        folder_path: Directory containing PDF documents.

    Returns:
        List of loaded LangChain documents.
    """
    logger.info("Loading documents.")

    documents: list[Document] = []
    
    folder = Path(folder_path)

    for pdf_path in sorted(folder.glob("*.pdf")):

        file = pdf_path.name
 
        try:
            
            with fitz.open(pdf_path) as pdf:

                for page_number, page in enumerate(pdf):

                    text = page.get_text(sort=True)
                    # Normalize whitespace extracted from the PDF.
                    text = re.sub(r"\s+", " ", text).strip()

                    if not text:
                         
                        logger.warning(
                            "No text found in '%s', page %d.", 
                            file, 
                            page_number + 1
                        )
                        continue

                    doc_id = hashlib.md5(
                        f"{file}_{page_number}_{text}".encode("utf-8")
                        ).hexdigest()

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file,
                                "page": page_number + 1,     # Store pages using 1-based numbering.
                                "doc_id": doc_id
                                }
                            )
                    )
        except Exception:
            logger.exception("Failed to process '%s'.", file)

     
    logger.info(
        "Loaded %d pages from '%s'.",
        len(documents),
        folder_path
    )
    
    return documents