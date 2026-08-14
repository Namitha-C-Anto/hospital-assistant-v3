from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.main import app
from app.dependencies import get_rag_service

mock_rag_service = AsyncMock()

mock_rag_service.ask.return_value = "This is a test answer."

def override_get_rag_service():
    return mock_rag_service

app.dependency_overrides[get_rag_service] = override_get_rag_service

client = TestClient(app)

def test_chat():

    response = client.post(
        "/chat",
        json = {
            "question": "What are the visiting hours?",
            "provider": "groq",
            "model": "test-model",
            "api_key": "fake-key",
        }
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a test answer."

    mock_rag_service.ask.assert_awaited_once_with(
        question="What are the visiting hours?",
        provider="groq",
        model="test-model",
        api_key="fake-key",
    )

    app.dependency_overrides.clear()