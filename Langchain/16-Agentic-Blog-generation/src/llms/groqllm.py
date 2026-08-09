from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


class GroqLLM:

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL")
        self.client = ChatGroq(api_key=self.api_key, model_name=self.groq_model)

    def get_llm(self):
        return self.client
