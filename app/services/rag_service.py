from config import GROQ_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
from rag.initializer import initialize_rag
from rag.pipeline import run_rag_pipeline
from llm.llm import get_llm
from rag.models import RetrievalResult
from memory.chat_manager import convert_chat_history
from app.schemas.chat import ChatHistoryItem

class RAGService:

    def __init__(self):
        self.components = initialize_rag()

    async def ask(
        self, 
        question:str,
        provider:str,
        model:str, 
        chat_history:list[ChatHistoryItem],
        retrieval_mode:str,
        use_reranker:bool,
    ) -> tuple[str, RetrievalResult]:

        if provider == "groq":
            api_key = GROQ_API_KEY

        elif provider == "openai":
            api_key = OPENAI_API_KEY

        elif provider == "openrouter":
            api_key = OPENROUTER_API_KEY

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm = get_llm(
            provider,
            model,
            api_key,
        )

        # Keep only the most recent 5 conversation turns.
        recent_history = chat_history[-5:]

        # Convert API models to LangChain messages.
        messages = convert_chat_history(recent_history)
        
        pipeline_result = await run_rag_pipeline(
            question=question,
            rag_components=self.components,
            app_llm=llm,
            chat_history=messages, # Only use the last 10 messages for context.
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
        )

        return pipeline_result.answer, pipeline_result.retrieval_result