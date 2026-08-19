import streamlit as st
from config import APP_TITLE
from memory.chat_manager import (
    switch_chat,
    list_chats,
    delete_chat,)
from utils.logger import logger

def render_sidebar() -> None:
    st.markdown(APP_TITLE, unsafe_allow_html=True)
 
    # -----------------------------
    # New Chat
    # -----------------------------

    #Step 2: Create a new chat
    if st.button(
        "➕ New Chat",
        use_container_width=True,
        type="primary",
    ):        
        switch_chat(None)
        st.rerun()

    with st.expander("💬Recents", expanded=True):

        chats = list_chats()

        if not chats:
            st.caption("No conversations yet.")

        #Step 3: Show the active chat
        for chat_id, chat in  chats:
                    
            title = chat["title"]

            if chat_id == st.session_state.current_chat:
                title = f"🩺{title}"

            if st.button(
                title,
                key=chat_id,
                use_container_width=True,
                type="secondary",
            ):
                switch_chat(chat_id)
                st.rerun()

        if st.session_state.current_chat is not None:
            st.divider() 

            if st.button(
                "🗑️ Delete Chat",
                use_container_width=True,
            ):
                st.session_state.confirm_delete = True

            if st.session_state.get("confirm_delete", False):
                st.warning("Delete this chat?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes"):
                        delete_chat(st.session_state.current_chat)
                        st.session_state.confirm_delete = False
                        st.rerun()

                with col2:
                    if st.button("Cancel"):
                        st.session_state.confirm_delete = False
                        st.rerun()

    # ---------------------------------------------------
    # Settings
    # ---------------------------------------------------

    with st.expander("⚙️Settings"):

        # -------------------------------------------------
        # Retrieval Mode
        # -------------------------------------------------

        retrieval_options = {
            "Semantic": {
                "retrieval_mode": "qdrant",
                "use_reranker": False,
            },
            "Semantic + Reranker": {
                "retrieval_mode": "qdrant",
                "use_reranker": True,
            },
            "Hybrid": {
                "retrieval_mode": "hybrid",
                "use_reranker": False,
            },
            "Hybrid + Reranker": {
                "retrieval_mode": "hybrid",
                "use_reranker": True,
            },
        }

        mode_description = {
            "Semantic": "Qdrant semantic search",
            "Semantic + Reranker": "Qdrant semantic search + cross-encoder reranking",
            "Hybrid": "Qdrant + BM25 → RRF fusion",
            "Hybrid + Reranker": "Qdrant + BM25 → RRF → cross-encoder reranking",
        }

        retrieval_level = st.selectbox(
            "🔎Retrieval Mode",
            options=list(retrieval_options.keys()),
            index=0,
            help=(
                "Semantic: Qdrant vector search only.\n\n"
                "Semantic + Reranker: Qdrant vector search followed by cross-encoder reranking.\n\n"
                "Hybrid: Qdrant + BM25 combined using Reciprocal Rank Fusion (RRF).\n\n"
                "Hybrid + Reranker: Hybrid retrieval followed by cross-encoder reranking."
            ),
        )

        selected_retrieval = retrieval_options[retrieval_level]

        st.caption(
            f"**{retrieval_level}:** {mode_description[retrieval_level]}"
        )

        retrieval_mode = selected_retrieval["retrieval_mode"]
        use_reranker = selected_retrieval["use_reranker"]

        st.session_state["retrieval_mode"] = retrieval_mode.lower()
        st.session_state["use_reranker"] = use_reranker

        logger.info(
            "Retrieval mode: level=%s, search=%s, reranker=%s",
            retrieval_level,
            retrieval_mode,
            use_reranker,
        )

        provider = st.selectbox(
            "🏢LLM Provider",
            ["Groq", "OpenAI", "OpenRouter"],
            key="llm_provider",
        )
        st.session_state["provider"] = provider.lower()

        if provider == "Groq":
            model = st.selectbox(
                "🧠Model",
                [
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3.6-27b",
                ],
                key="llm_model",
            )
            
        elif provider == "OpenRouter":
            model = st.selectbox(
                "🧠Model",
                [
                    "openai/gpt-oss-120b",
                    "qwen/qwen3-235b-a22b",
                ],
                key="llm_model",
            )

        st.session_state["model"] = model
