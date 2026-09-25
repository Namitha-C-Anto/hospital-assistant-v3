from app.schemas.chat import ChatHistoryItem
from langchain_core.messages import HumanMessage, AIMessage

from memory.chat_manager import convert_chat_history


def test_convert_chat_history():
    chat_history = [
        ChatHistoryItem(
            question="What are the admission guidelines?",
            answer="The admission guidelines include..."
        ),
        ChatHistoryItem(
            question="What documents are required?",
            answer="The required documents include..."
        ),
    ]

    messages = convert_chat_history(chat_history)

    assert len(messages) == 4

    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "What are the admission guidelines?"

    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "The admission guidelines include..."

    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "What documents are required?"

    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "The required documents include..."