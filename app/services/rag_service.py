from config import GROQ_API_KEY, OPENAI_API_KEY
from rag.initializer import initialize_rag
from rag.pipeline import run_rag_pipeline
from llm.llm import get_llm
from rag.models import RetrievalResult

class RAGService:

    def __init__(self):
        self.components = initialize_rag()

    async def ask(
        self, 
        question:str,
        provider:str,
        model:str, 
        chat_history:str,
    ) -> tuple[str, RetrievalResult]:

        if provider == "groq":
            api_key = GROQ_API_KEY

        elif provider == "openai":
            api_key = OPENAI_API_KEY

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm = get_llm(
            provider,
            model,
            api_key,
        )

        pipeline_result = await run_rag_pipeline(
            question,
            self.components,
            llm,
            chat_history = chat_history,

        )

        return pipeline_result.answer, pipeline_result.retrieval_result