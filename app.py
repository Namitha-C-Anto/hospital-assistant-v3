import requests
from pathlib import Path
import streamlit as st
from config import (
    DB_PATH,
    LLM_PROVIDER,
    LLM_MODEL,)  
from utils.logger import logger     
from rag.pipeline import run_rag_pipeline  
from rag.initializer import initialize_rag 
from llm.llm import get_llm

from memory.chat_manager import (
    initialize_chat_sessions,
    rename_chat,
    get_chat_history, 
    format_chat_history, 
    save_chat,)
from rag.builder import build_vector_database
from rag.models import PipelineComponents
from ui.styles import load_css
from ui.sidebar import render_sidebar
from ui.welcome import render_welcome
from ui.example_questions import render_example_questions
from ui.sources import render_sources 

# -------------------------------------------------------------
FASTAPI_URL = "http://localhost:8000"

def call_chat_api(
    question: str,
    provider: str,
    model: str,
    api_key: str,
    chat_history: str,
) -> str:

    response = requests.post(
        f"{FASTAPI_URL}/chat",
        json={
            "question": question,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "chat_history": chat_history,
        },
        timeout=300,
    )

    response.raise_for_status()

    return response.json()
# -------------------------------------------------------------

load_css()
# --------------------------------------------------------------
# Load and cache initialized RAG components.
# This prevents reloading the vector database and
# models on every Streamlit rerun.
# ---------------------------------------------------------------
@st.cache_resource
def load_rag_components() -> PipelineComponents:
    """
    Initialize and cache RAG pipeline components.
    """
    try: 
        return initialize_rag() 

    except Exception:
        logger.exception("Failed to load RAG components.")
        raise

#------------------------------------------------------------------

def main() -> None:
    """
    Run the Streamlit Hospital Policy RAG application.
    """
    
    # Step 1: Initialize the session state
    initialize_chat_sessions()
        
    # -------------------------------------------------
    # Create the vector database on first launch if it
    # does not already exist.
    # -------------------------------------------------
    if not Path(DB_PATH).exists():
        build_vector_database()
        
    # -------------------------------------------------
    # Display the Side bar
    # -------------------------------------------------
    with st.sidebar:
        render_sidebar()
        
        if "last_retrieval_result" in st.session_state:
            render_sources(
                st.session_state.last_retrieval_result
            )
    # -------------------------------------------------
    # Load cached RAG components.
    # -------------------------------------------------
    rag_components = load_rag_components()

    # -------------------------------------------------
    # Create the selected LLM
    # -------------------------------------------------        
    
    logger.info(
        "LLM settings: provider=%s, model=%s, api_key_provided=%s",
        st.session_state.get("provider", LLM_PROVIDER),
        st.session_state.get("model", LLM_MODEL),
        bool(st.session_state.get("api_key")),
    )

    try:
        app_llm = get_llm(
            provider=st.session_state.get("provider", LLM_PROVIDER),
            model=st.session_state.get("model", LLM_MODEL),
            api_key=st.session_state.get("api_key"),
        )
    except Exception:
        logger.exception("Failed to initialize selected LLM.")
        st.error(
            "Unable to initialize the selected LLM. "
            "Please check the provider, model, and API key."
        )
        st.stop()
    # -------------------------------------------------
    # Restore and display previous chat messages.
    # -------------------------------------------------
    chat_history = get_chat_history()

    if not chat_history:
        render_welcome()

    for chat in chat_history:

        with st.chat_message("user"):
            st.write(
                chat["question"]
            )
        with st.chat_message("assistant"):
            st.write(
                chat["answer"]
            )

    #------------------------------------------------------------    
    if not chat_history:
        st.markdown("##### 💡 Try asking")
        
        example_questions = render_example_questions()
        columns = st.columns(2)

        for i, question in enumerate(example_questions):
            with columns[i % 2]:
                if st.button(question, use_container_width=True):
                    st.session_state.selected_question = question

    # -------------------------------------------------
    # Accept a new user question.
    # -------------------------------------------------
    question = st.chat_input(
        "Ask your hospital-related question."
    )

    # If an example question was clicked, use it instead
    if "selected_question" in st.session_state:
        question = st.session_state.pop("selected_question")

    if question:
         
        logger.info("Processing question: %s.",question)

        # -------------------------------------------------
        # Format conversation history and execute the
        # complete RAG pipeline.
        # -------------------------------------------------
        history_text = format_chat_history(chat_history)

        # -------------------------------------------------
        # Display the latest conversation.
        # -------------------------------------------------
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and generating answer..."):
                    
                    # pipeline_result = run_rag_pipeline(
                    #     question,
                    #     rag_components,
                    #     app_llm,
                    #     chat_history=history_text,
                    # )
                                    
                    api_result = call_chat_api(
                        question=question,
                        provider=st.session_state.get("provider", LLM_PROVIDER),
                        model=st.session_state.get("model", LLM_MODEL),
                        api_key=st.session_state.get("api_key"),
                        chat_history=history_text
                    )

                    answer = api_result["answer"]
                    retrieval_result = api_result["retrieval_result"]

            except Exception as e:
                logger.exception("RAG Pipeline failed.")

                error_message = str(e).lower()

                if "rate limit" in error_message or "429" in error_message:
                    st.error(
                        "The LLM provider has reached its rate limit. "
                        "Please try again later or switch to another provider."
                    )

                elif "api key" in error_message or "authentication" in error_message:
                    st.error(
                        "The LLM API key is invalid or missing. "
                        "Please check your API key in Settings."
                    )

                else:
                    st.error(
                        "The LLM request failed. "
                        "Please check your settings and try again."
                    )

                st.stop()
                
            # st.write(pipeline_result.answer)
            st.write(answer)
            # st.session_state.last_pipeline_result = pipeline_result
            st.session_state.last_retrieval_result = retrieval_result
    
        # Save conversation for future turns.
        # save_chat(question, pipeline_result.answer)
        save_chat(question, answer)
        rename_chat(question)
                  
        st.rerun()
            
if __name__ == "__main__":
    main()