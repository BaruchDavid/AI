import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from my_chat_gpt import MyChatGpt


load_dotenv()

## set envs for keeping app in LangSmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


myChatGpt = MyChatGpt("gemma:2b")
result = myChatGpt.execute_prompt("wonderful")

print(f"content: {result.text} , with modell: {result.meta_daten["model"]}")


st.title("Langchain with gemma:2b")
input_text = st.text_input("What question you have in mind?")


if input_text:
    st.write(myChatGpt.execute_chain(input_text))
