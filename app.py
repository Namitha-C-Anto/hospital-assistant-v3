import requests 
import streamlit as st
from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    FASTAPI_URL,
    RETRIEVAL_MODE,
    USE_RERANKER
)  
from utils.logger import logger      

from memory.chat_manager import (
    initialize_chat_sessions,
    rename_chat,
    get_chat_history,  
    save_chat,)  

from ui.styles import load_css
from ui.sidebar import render_sidebar
from ui.welcome import render_welcome
from ui.example_questions import render_example_questions
from ui.sources import render_sources 

# -------------------------------------------------------------
def call_chat_api(
    question: str,
    provider: str,
    model: str,
    chat_history: list[dict[str, str]],
    retrieval_mode: str,
    use_reranker: bool,
) -> dict:

    response = requests.post(
        f"{FASTAPI_URL}/chat",
        json={
            "question": question,
            "provider": provider,
            "model": model, 
            "chat_history": chat_history,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
        },
        timeout=300,
    )

    response.raise_for_status()

    return response.json()
# -------------------------------------------------------------

load_css()

def main() -> None:
    """
    Run the Streamlit Hospital Policy RAG application.
    """
    
    # Step 1: Initialize the session state
    initialize_chat_sessions()

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
    # Create the selected LLM
    # -------------------------------------------------        
    
    logger.info(
        "LLM settings: provider=%s, model=%s",
        st.session_state.get("provider", LLM_PROVIDER),
        st.session_state.get("model", LLM_MODEL), 
    )

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
        # Eexecute the complete RAG pipeline.
        # -------------------------------------------------
         
        # -------------------------------------------------
        # Display the latest conversation.
        # -------------------------------------------------
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and generating answer..."):
                    
                    api_result = call_chat_api(
                        question=question,
                        provider=st.session_state.get("provider", LLM_PROVIDER),
                        model=st.session_state.get("model", LLM_MODEL),
                        chat_history=chat_history,
                        retrieval_mode=st.session_state.get("retrieval_mode", RETRIEVAL_MODE),
                        use_reranker=st.session_state.get("use_reranker", USE_RERANKER),
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
                 
            st.write(answer) 
            st.session_state.last_retrieval_result = retrieval_result
    
        # Save conversation for future turns. 
        save_chat(question, answer)
        rename_chat(question)
                  
        st.rerun()
            
if __name__ == "__main__":
    main()