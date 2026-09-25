import requests

def test_chat_e2e():
    response = requests.post(
        url = "http://localhost:8000/chat",
        json = {
            "question": "What are the guidelines for hospital admission?",
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "chat_history": [],
            "retrieval_mode": "semantic",
            "use_reranker": False,
        },
        timeout= 120,
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert data["answer"]

    assert "retrieval_result" in data
    assert data["retrieval_result"] is not None