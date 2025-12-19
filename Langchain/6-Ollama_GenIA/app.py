import os
from typing import Optional
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from my_chat_gpt import MyChatGpt
from diagnostic.llm_Diagnostics_Util import LlmDiagnosticUtil
from diagnostic.diagnosis_Mode import DiagnosisMode
from llm_Result import LlmResult

load_dotenv()

## set envs for keeping app in LangSmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


llm_result: Optional[LlmResult] = None
myChatGpt = MyChatGpt("llama3.2:latest")


st.title("Langchain with gemma:2b")
input_text = st.text_input("What question you have in mind?")


if input_text:
    llm_result = myChatGpt.execute_chain(input_text)
    st.write(llm_result.text)


llm_diagnostic = LlmDiagnosticUtil(
    llm=myChatGpt.get_llm(),
    max_expected_completion_tokens=400,
    max_prompt_tokens=30,
    slow_latency_ms=3000,
)

if llm_result is not None:
    llm_diagnostic.diagnose(
        prompt_tokens=llm_result.meta_daten["prompt_tokens"],
        completion_tokens=llm_result.meta_daten["completion_tokens"],
        latency_ms=llm_result.meta_daten["latency_ms"],
        task_type="analys",
        diagnose_mode=DiagnosisMode.RULES_AND_LLM,
    )
