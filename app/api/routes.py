from fastapi import APIRouter, Request

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,)



router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health_status():
    return {
        "status":"healthy."
    }

@router.post("/chat", response_model=ChatResponse)
def chat(request:ChatRequest, app_request:Request):

    rag_service = app_request.app.state.rag_service

    answer = rag_service.ask(
        question = request.question, 
        provider = request.provider,
        model = request.model,
        api_key = request.api_key,
        )

    return {
        "answer" : answer
    }