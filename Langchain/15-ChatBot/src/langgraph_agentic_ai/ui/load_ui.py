import streamlit as st
import os
from dotenv import load_dotenv


from src.langgraph_agentic_ai.ui.config_reader import Config

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class LoadStreamlitUI:

    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.config.get_page_title(), layout="wide")
        st.title(self.config.get_page_title())
        st.header("Welcome to LangGraph: Build Stateful Agentic AI graph")
        st.session_state.IsFetchButtonClicked = False  # Initialize the session state variable
        st.session_state.timeframe = None  # Initialize the session state variable

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM selection
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            if self.user_controls["selected_llm"] == "Groq":

                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Groq Model", model_options)
                self.user_controls["selected_groq_api_key"] = st.text_input("API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
                st.session_state["GROQ_API_KEY"] = self.user_controls["selected_groq_api_key"]

                # Validate key
                if not self.user_controls["selected_groq_api_key"]:
                    st.warning("Please enter your Groq API key. refer https://console.qroq.com/keys")
                else:
                    st.success("Groq API key entered successfully.")

            # Usecase selection
            self.user_controls["selected_usecase"] = st.selectbox("Select Use Case", usecase_options)

            if self.user_controls["selected_usecase"] == "Chatboot with Web" or self.user_controls["selected_usecase"] == "AI News":
                self.user_controls["TAVILY_API_KEY "] = st.session_state["TAVILY_API_KEY"] = st.text_input(
                    "TAVILY_API_KEY", type="password", value=os.environ.get("TAVILY_API_KEY", "")
                )

                if not self.user_controls["TAVILY_API_KEY"]:
                    st.warning("Please enter your TAVILY API key. refer https://tavily.com/")

            if self.user_controls["selected_usecase"].strip() == "AI News":
                st.subheader("AI News Use Case")
                with st.sidebar:
                    time_frame = st.selectbox("Select Time Frame", ["Last 24 hours", "Last 7 days", "Last 30 days"], index=0)

                if st.button("Fetch AI News", use_container_width=True):
                    st.session_state.timeframe = time_frame
                    st.session_state.IsFetchButtonClicked = True
        return self.user_controls
