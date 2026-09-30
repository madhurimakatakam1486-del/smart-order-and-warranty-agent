import streamlit as st
from langchain_ollama import ChatOllama


# ============================================================
# MODEL CONFIGURATION
# ============================================================

OLLAMA_MODEL = "llama3.2:latest"


# ============================================================
# CACHED LLM
# ============================================================

@st.cache_resource
def get_llm():
    """
    Returns one cached ChatOllama instance.

    Using Streamlit cache prevents the model client
    from being recreated on every Streamlit rerun.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=100,
    )


# ============================================================
# SIMPLE MODEL TEST
# ============================================================

def test_llm():

    llm = get_llm()

    response = llm.invoke(
        "Reply with exactly: "
        "Smart Order Agent is ready."
    )

    return response.content