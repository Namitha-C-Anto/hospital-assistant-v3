from app.schemas.chat import ChatHistoryItem
from memory.chat_manager import convert_chat_history
from prompts.prompt_template import prompt

def test_convert_chat_history():

    history = [
        ChatHistoryItem(
            question="What are the visiting hours?",
            answer="Visiting hours are from 4 PM to 6 PM."
        ),
        ChatHistoryItem(
            question="Can children visit?",
            answer="Children can visit with an accompanying adult."
        ),
    ]

    messages = convert_chat_history(history)

    assert len(messages) == 4