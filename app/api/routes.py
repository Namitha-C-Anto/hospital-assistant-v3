from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_rag_service
from app.services.rag_service import RAGService

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,)

from utils.logger import logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health_status():
    return {
        "status":"healthy."
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request:ChatRequest, 
    rag_service: RAGService = Depends(get_rag_service)
    ):
 
    try:
        answer, retrieval_result  = await rag_service.ask(
            question = request.question, 
            provider = request.provider,
            model = request.model, 
            chat_history = request.chat_history,
            )

        return {
            "answer": answer,
            "retrieval_result": retrieval_result,
        }

    except ValueError as e:

        logger.warning("Invalid chat request: %s", e)

        raise HTTPException(
            status_code = 400,
            detail = str(e)
        )

    except Exception:

        logger.exception("RAG request failed.")

        raise HTTPException(
            status_code = 500,
            detail = "Failed to generate a response."
        )