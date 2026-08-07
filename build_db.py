from rag.builder import build_vector_database
from utils.logger import logger

def main()->None:

    try: 
        build_vector_database() 
    except Exception:
        logger.exception("Application failed")
        raise
    
if __name__ == "__main__":    
    main()