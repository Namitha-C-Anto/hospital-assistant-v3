from langchain_huggingface import HuggingFaceEmbeddings
from config import  EMBEDDING_MODEL

#------------------------------------------------------------------------------------------
#Create embedding model
#------------------------------------------------------------------------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and return the configured embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

