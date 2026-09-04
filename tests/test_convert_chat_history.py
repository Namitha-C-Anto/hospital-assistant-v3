from app.schemas.chat import ChatHistoryItem
from memory.chat_manager import convert_chat_history
from prompts.prompt_template import prompt

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

for message in messages:
    print(type(message).__name__, ":", message.content)
    
messages = prompt.format_messages(
    context="Visiting hours are from 4 PM to 6 PM.",
    question="What are the visiting hours?",
    chat_history=messages,
)

for message in messages:
    print(type(message).__name__, ":", message.content)