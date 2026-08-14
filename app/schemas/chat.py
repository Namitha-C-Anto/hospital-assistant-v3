from typing import Any
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length = 1)
    provider: str
    model: str
    api_key: str
    chat_history: str

class DocumentInfoResponse(BaseModel):
    content: str
    metadata: dict[str, Any]

class RetrievalResultResponse(BaseModel):
    retrieved_documents: list[DocumentInfoResponse]
    reranked_documents: list[DocumentInfoResponse]

class ChatResponse(BaseModel):
    answer: str
    retrieval_result: RetrievalResultResponse

class HealthResponse(BaseModel):
    status: str