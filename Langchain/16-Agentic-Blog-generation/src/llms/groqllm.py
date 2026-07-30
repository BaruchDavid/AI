from langchain_groq import GroqLLM
import os
from dotenv import load_dotenv


class GroqLLM:

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL_NAME", "groq-llm-1.0")
        self.client = GroqLLM(api_key=self.api_key, model_name=self.model_name)

    def get_llm(self):
        return self.client
