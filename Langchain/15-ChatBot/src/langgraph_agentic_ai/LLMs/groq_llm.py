import os
import streamlit as st
from langchain_groq import ChatGroq


class GroqLLM:

    def __init__(self, user_controls_input):
        self.user_controls = user_controls_input

    def get_llm_models(self):
        """
        Returns the LLM model based on the user selection.
        If the selected LLM is 'Groq', it initializes and returns a ChatGroq instance
        with the selected Groq model and API key.
        """
        try:
            groq_api_key = self.user_controls.get["GROQ_API_KEY"]
            selected_groq_model = self.user_controls.get["GROQ_MODEL"]
            if groq_api_key == "" and os.environ["GROQ_API_KEY"] == "":
                st.error("Error: Groq API key is not set. Please provide a valid API key.")

            llm = ChatGroq(model=selected_groq_model, api_key=groq_api_key)

        except Exception as e:
            raise RuntimeError(f"Error initializing Groq LLM: {e}")
        return llm
