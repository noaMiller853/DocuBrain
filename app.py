import os
import logging

import streamlit as st
from dotenv import load_dotenv

from modules import (
    load_skill,
    has_saved_pdf,
    list_saved_pdfs,
    load_all_saved_pdfs,
    process_pdf_file,
    build_hybrid_agent
)


# ============================================================
# Load environment variables from .env
# ============================================================

load_dotenv()


# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title="DocuBrain Pro",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# Design
# ============================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E88E5;
        font-weight: bold;
    }

    .source-badge {
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown(
    '<p class="main-header">🧠 DocuBrain Pro — Hybrid RAG Agent</p>',
    unsafe_allow_html=True
)


# ============================================================
# Load Skill
# ============================================================

skill_data = load_skill(
    "skills/hybrid_rag_skill.yaml"
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("🔑 הגדרות ומפתחות API")

    st.markdown("""
    **איך משיגים את המפתחות?**

    1. **Claude API Key:** מפתח API של Claude
    2. **Anthropic Workspace ID:** מזהה Workspace של Anthropic
    3. **Tavily Key:** מפתח לחיפוש באינטרנט
    """)

    # ========================================================
    # Anthropic API Key
    # ========================================================

    env_anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if env_anthropic_key:

        anthropic_key = env_anthropic_key

        st.success(
            "✅ Claude API Key נטען מה-.env"
        )

    else:

        anthropic_key = st.text_input(
            "Claude API Key:",
            type="password",
            help="לא נמצא מפתח ב-.env, לכן יש להזין מפתח ידנית"
        )


    # ========================================================
    # Anthropic Workspace ID
    # ========================================================

    env_workspace_id = os.getenv(
        "ANTHROPIC_WORKSPACE_ID"
    )

    if env_workspace_id:

        anthropic_workspace_id = env_workspace_id

        st.success(
            "✅ Workspace ID נטען מה-.env"
        )

    else:

        anthropic_workspace_id = st.text_input(
            "Anthropic Workspace ID:",
            type="password",
            help="נדרש עבור API Key שאינו משויך ל-Workspace"
        )


    # ========================================================
    # Tavily API Key
    # ========================================================

    env_tavily_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if env_tavily_key:

        tavily_key = env_tavily_key

        st.success(
            "✅ Tavily API Key נטען מה-.env"
        )

    else:

        tavily_key = st.text_input(
            "Tavily API Key (אופציונלי):",
            type="password",
            help="מאפשר לסוכן לחפש באינטרנט"
        )


    st.divider()


    # ========================================================
    # Skill
    # ========================================================

    st.header("📜 סטטוס Skill")

    if skill_data:

        st.success(
            f"טעון: {skill_data.get('name')}"
        )

        with st.expander("הצג הנחיות Skill"):

            st.code(
                skill_data.get("instructions"),
                language="yaml"
            )

    else:

        st.error(
            "קובץ ה-Skill לא נמצא!"
        )


# ============================================================
# Configuration checks
# ============================================================

if not skill_data:

    st.error(
        "לא ניתן להפעיל את המערכת ללא Skill."
    )

    st.stop()


if not anthropic_key:

    st.warning(
        "🔑 יש להזין Claude API Key כדי להמשיך."
    )

    st.stop()


# ============================================================
# Main application
# ============================================================

saved_pdf_names = list_saved_pdfs()
st.caption(
    f"מסמכים שמורים: {len(saved_pdf_names)}. החיפוש כולל את כולם."
)

upload_new_option = "העלאת קובץ PDF חדש"
document_options = saved_pdf_names + [upload_new_option]
selected_document = st.selectbox(
    "בחרי מסמך שמור או העלי מסמך חדש:",
    document_options,
    index=0,
)

uploaded_file = None
if selected_document == upload_new_option:
    uploaded_file = st.file_uploader(
        "העלי קובץ PDF",
        type=["pdf"],
    )

if uploaded_file:
    upload_signature = os.path.normcase(uploaded_file.name)
    already_saved = has_saved_pdf(uploaded_file.name)
    if st.session_state.get("processed_upload_signature") != upload_signature:
        if already_saved:
            st.info("הקובץ בשם הזה כבר נשמר; משתמשת באינדקס הקיים.")
        else:
            try:
                with st.spinner("מעבד ושומר את המסמך..."):
                    process_pdf_file(uploaded_file)
            except Exception as e:
                logging.exception("Failed to process uploaded PDF %r", uploaded_file.name)
                st.error(f"שגיאה בעיבוד המסמך ({type(e).__name__}): {e}")
                st.stop()

        st.session_state.processed_upload_signature = upload_signature
        if not already_saved:
            st.session_state.pop("retriever", None)
            st.session_state.pop("agent_executor", None)

    saved_pdf_names = list_saved_pdfs()

if saved_pdf_names:
    retriever_signature = tuple(saved_pdf_names)

    if (
        "retriever" not in st.session_state
        or st.session_state.get("retriever_signature") != retriever_signature
    ):
        try:
            st.session_state.retriever = load_all_saved_pdfs()
            st.session_state.retriever_signature = retriever_signature
        except Exception as e:
            st.error(f"שגיאה בטעינת המסמכים השמורים: {e}")
            st.stop()

    agent_signature = (
        retriever_signature,
        anthropic_key,
        anthropic_workspace_id,
        tavily_key,
    )

    if (
        "agent_executor" not in st.session_state
        or
        st.session_state.get("agent_signature") != agent_signature
    ):

        try:

            st.session_state.agent_executor = build_hybrid_agent(
                retriever=st.session_state.retriever,
                skill_instructions=skill_data["instructions"],
                anthropic_api_key=anthropic_key,
                anthropic_workspace_id=anthropic_workspace_id,
                tavily_api_key=tavily_key
            )

            st.session_state.agent_signature = agent_signature

        except Exception as e:

            st.error(
                f"שגיאה ביצירת הסוכן: {e}"
            )

            st.stop()

    agent_executor = st.session_state.agent_executor


    # ========================================================
    # Quenstion
    # ========================================================

    user_query = st.text_input(
        "שאלי שאלה עסקית או כללית:"
    )


    if user_query:

        with st.spinner(
            "הסוכן חוקר ומנסח תשובה..."
        ):

            try:

                response = agent_executor.invoke(
                    {
                        "input": user_query
                    }
                )

                st.markdown(
                    "### תשובת המערכת:"
                )

                st.write(
                    response["output"]
                )

            except Exception as e:

                st.error(
                    f"שגיאה בהרצת הסוכן: {e}"
                )
else:
    st.info("העלי מסמך PDF כדי להתחיל.")