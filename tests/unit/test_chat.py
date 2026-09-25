from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.main import app
from app.dependencies import get_rag_service
from rag.models import RetrievalResult, DocumentInfo

mock_retrieval_result = RetrievalResult(
    retrieved_documents=[
        DocumentInfo(
            content="Hospital visiting hours are from 2 PM to 4 PM.",
            metadata={
                "source": "hospital_policy.pdf",
                "page": 5,
            },
        )
    ],
    reranked_documents=[
        DocumentInfo(
            content="Hospital visiting hours are from 2 PM to 4 PM.",
            metadata={
                "source": "hospital_policy.pdf",
                "page": 5,
            },
        )
    ],
)

mock_rag_service = AsyncMock()
 
mock_rag_service.ask.return_value = (
    "This is a test answer.",
    mock_retrieval_result,
)

def override_get_rag_service():
    return mock_rag_service

app.dependency_overrides[get_rag_service] = override_get_rag_service

client = TestClient(app)

def test_chat():

    response = client.post(
        "/chat",
        json={
            "question": "What are the visiting hours?",
            "provider": "groq",
            "model": "test-model", 
            "chat_history": [],
            "retrieval_mode": "semantic",
            "use_reranker": False,
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."
    assert "retrieval_result" in data

    mock_rag_service.ask.assert_awaited_once_with(
        question="What are the visiting hours?",
        provider="groq",
        model="test-model", 
        chat_history=[],
        retrieval_mode="semantic",
        use_reranker=False,
    )

    app.dependency_overrides.clear()