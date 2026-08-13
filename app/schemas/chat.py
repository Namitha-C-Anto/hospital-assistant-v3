from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length = 1)
    provider: str
    model: str
    api_key: str

class ChatResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str