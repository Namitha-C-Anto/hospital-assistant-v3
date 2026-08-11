from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from config import TEMPERATURE, LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, GROQ_API_KEY


def get_llm(
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
    api_key: str | None = None,
):
    provider = provider.lower()

    if not api_key:
        raise ValueError(
            f"API key is required for {provider}."
        )

    if provider == "groq":
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=TEMPERATURE,
        )

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=TEMPERATURE,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")