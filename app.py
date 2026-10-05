import os

import streamlit as st
from dotenv import load_dotenv

from modules import (
    load_skill,
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

# Upload PDF

uploaded_file = st.file_uploader(
    "העלי קובץ PDF לבדיקה",
    type=["pdf"]
)


if uploaded_file:

    # ========================================================
    # Create a Retriever only if it is a new PDF.
    # ========================================================

    if (
        "retriever" not in st.session_state
        or
        st.session_state.get("file_name") != uploaded_file.name
    ):

        with st.spinner(
            "מעבד מסמך ובונה אינדקס וקטורי..."
        ):

            try:

                st.session_state.retriever = (
                    process_pdf_file(uploaded_file)
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.success(
                    "המסמך נקלט במערכת!"
                )

            except Exception as e:

                st.error(
                    f"שגיאה בעיבוד המסמך: {e}"
                )

                st.stop()


    # ========================================================
    # Create Agent
    # ========================================================

    try:

        agent_executor = build_hybrid_agent(
            retriever=st.session_state.retriever,
            skill_instructions=skill_data["instructions"],
            anthropic_api_key=anthropic_key,
            anthropic_workspace_id=anthropic_workspace_id,
            tavily_api_key=tavily_key
        )

    except Exception as e:

        st.error(
            f"שגיאה ביצירת הסוכן: {e}"
        )

        st.stop()


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