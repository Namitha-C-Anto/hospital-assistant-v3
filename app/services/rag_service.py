from rag.initializer import initialize_rag
from rag.pipeline import run_rag_pipeline
from llm.llm import get_llm

class RAGService:

    def __init__(self):
        self.components = initialize_rag()

    def ask(
        self, 
        question:str,
        provider:str,
        model:str,
        api_key:str,
    ) -> str:

        llm = get_llm(
            provider,
            model,
            api_key,
        )

        pipeline_result = run_rag_pipeline(
            question,
            self.components,
            llm,
            chat_history = "",

        )

        return pipeline_result.answer