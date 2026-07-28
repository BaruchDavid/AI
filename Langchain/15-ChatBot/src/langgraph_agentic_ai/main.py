import streamlit as st
from src.langgraph_agentic_ai.ui.load_ui import LoadStreamlitUI
from src.langgraph_agentic_ai.LLMs.groq_llm import GroqLLM
from src.langgraph_agentic_ai.graph.graph_builder import GraphBuilder
from src.langgraph_agentic_ai.ui.streamlitui.display_result import DisplayResultSteamlit


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

    if user_message:

        try:
            ## Configure LLM model based on user selection
            groqLLM = GroqLLM(user_controls_input=user_input)
            model = groqLLM.get_llm_models()

            if not model:
                st.error("Error: failed to initialize the LLM model.")
                return

            ## Initialize and set up the graph based on use case
            usecase = user_input.get("selected_usecase")

            if not usecase:
                st.error("Error: No use case selected.")
                return

            ## Graph Builder
            graph_builder = GraphBuilder(model)
            try:

                graph = graph_builder.setup_graph(usecase.strip())

            except Exception as e:
                st.error(f"Error setting up the graph: {e}")
                return

            ## Display the result in Streamlit UI
            display_result = DisplayResultSteamlit(usecase, graph, user_message)
            display_result.display_result()

        except Exception as e:
            st.error(f"Error processing input: {e}")
            if st.session_state.get("messages"):
                st.session_state["messages"][-1]["content"] = "Error processing your message."
