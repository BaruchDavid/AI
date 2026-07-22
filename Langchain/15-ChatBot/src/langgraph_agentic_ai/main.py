import streamlit as st
from src.langgraph_agentic_ai.ui.load_ui import LoadStreamlitUI


def load_langgraph_agentic_ai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    this function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use cae, and displays the output while
    implementing exception handling for robustness.
    """

    ## Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input.get("selected_llm"):
        st.warning("Error: failed to load user input from the UI.")
        return

    user_message = st.text_input("Enter your message:", key="user_message")
