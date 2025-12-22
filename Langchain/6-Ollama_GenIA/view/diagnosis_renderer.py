from diagnostic.model.combined_Diagnosis import CombinedDiagnosis
from diagnostic.model.llm_Diagnosis import LlmDiagnosis
from view.diagnosis_styles import ISSUE_STYLE
import streamlit as st


def render_combined_diagnosis(diagnosis: CombinedDiagnosis):
    col1, col2, col3 = st.columns(3)

    with col1:
        if diagnosis.rule_based:
            render_diagnosis("Rule-based", diagnosis.rule_based)
        else:
            st.markdown("### ⚪ Rule-based")
            st.caption("Nicht verfügbar")

    with col2:
        render_diagnosis("LLM-based", diagnosis.llm_based)

    with col3:
        st.markdown("## ✅ Final")
        render_diagnosis("Final Decision", diagnosis.final)


def render_diagnosis(title: str, diagnosis: LlmDiagnosis) -> None:
    emoji, font_color = ISSUE_STYLE[diagnosis.issue]

    st.markdown(
        f"### <span style='color:{font_color}'>{emoji} {title}</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        **Issue:** `{diagnosis.issue}`  
        **Confidence:** `{diagnosis.confidence:.2f}`
        """
    )

    st.progress(diagnosis.confidence)

    with st.expander("Begründung"):
        st.write(diagnosis.reason)
