import os
import uuid
import logging
import logging.config
import json
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
from my_chat_gpt import MyChatGpt
from diagnostic.llm_Diagnostics_Util import LlmDiagnosticUtil
from diagnostic.model.diagnosis_Mode import DiagnosisMode
from llm_Result import LlmResult
from history.session_history_store import SessionHistoryStore
from loaders.load_config import load_diagnostic_config
from view.diagnosis_renderer import render_combined_diagnosis

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "logging_config.json"

with open(config_path) as config:
    config = json.load(config)

logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

## set envs for keeping app in LangSmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


llm_result: Optional[LlmResult] = None

st.title("Langchain with gemma:2b")
input_text = st.text_input("What question you have in mind?")
# --- Session init FIRST ---
if "session_id" not in st.session_state:
    new_session_id = str(uuid.uuid4())
    st.session_state.session_id = new_session_id
    logger.debug(f"New session_id created: {new_session_id}")

if "chat_gpt" not in st.session_state:
    st.session_state.chat_gpt = MyChatGpt(
        llm_name="llama3.2", history_store=SessionHistoryStore()
    )


if input_text:
    logger.debug(f"passing current session_id to llm: {st.session_state.session_id}")
    llm_result = st.session_state.chat_gpt.execute_chain(
        message=input_text, session_id=st.session_state.session_id
    )
    st.write(llm_result.text)


if llm_result is not None:

    config_path = BASE_DIR / "config.yaml"
    config = load_diagnostic_config(config_path)
    llm_diagnostic = LlmDiagnosticUtil(
        llm=st.session_state.chat_gpt.get_llm(),
        max_expected_completion_tokens=400,
        max_prompt_tokens=30,
        config=config,
    )

    diagnostic_result = llm_diagnostic.diagnose(
        prompt_tokens=llm_result.meta_daten["prompt_tokens"],
        completion_tokens=llm_result.meta_daten["completion_tokens"],
        latency_ms=llm_result.meta_daten["latency_ms"],
        task_type="analys",
        diagnose_mode=DiagnosisMode.RULES_AND_LLM,
    )

    render_combined_diagnosis(diagnostic_result)
