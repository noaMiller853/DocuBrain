import hashlib
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ================================================================
# Where persisted vector stores live on disk. Each processed PDF
# gets its own subfolder, named after a hash of its content - so
# the same file processed twice (even across app restarts) reuses
# the same folder instead of re-embedding from scratch.
# ================================================================

PERSIST_ROOT = "chroma_store"


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    """
    טוען את מודל ה-embeddings פעם אחת בלבד לכל חיי האפליקציה.

    זו הנקודה הקריטית לביצועים: בלי caching, המודל (כ-90MB)
    נטען מחדש מהדיסק/מהזיכרון בכל קריאה ל-process_pdf_file,
    כלומר בכל העלאת קובץ חדשה. עם st.cache_resource, Streamlit
    שומר את האובייקט בזיכרון בין ריצות (reruns) ומחזיר את אותו
    מופע בדיוק - המודל נטען פעם אחת בלבד, גם אם המשתמש מעלה
    עשרה קבצים שונים באותה session.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _file_content_hash(file_bytes: bytes) -> str:
    """
    מזהה קובץ לפי תוכן (לא לפי שם) - כך שאם שני משתמשים
    מעלים קובץ זהה בשמות שונים, או אותו משתמש מעלה את אותו
    קובץ שוב בהרצה חדשה של האפליקציה, הוא מזוהה כ"כבר מוטמע"
    ולא עובר עיבוד/embedding כפול.
    """

    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _load_existing_vectorstore(collection_name: str, persist_dir: str, embeddings):
    """
    בודק אם כבר קיים אינדקס וקטורי שמור לקובץ הזה, ואם כן
    טוען אותו מהדיסק במקום לבנות מחדש. מחזיר None אם אין
    אינדקס שמור או שהוא ריק.
    """

    if not (os.path.isdir(persist_dir) and os.listdir(persist_dir)):
        return None

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    if vectorstore._collection.count() > 0:
        return vectorstore

    return None


def process_pdf_file(uploaded_file):
    """
    מקבל קובץ PDF, מחלק אותו לקטעים ובונה (או טוען, אם כבר קיים)
    אינדקס וקטורי שמור בדיסק (Chroma persist_directory).
    """

    file_bytes = uploaded_file.getvalue()
    file_hash = _file_content_hash(file_bytes)

    collection_name = f"doc_{file_hash}"
    persist_dir = os.path.join(PERSIST_ROOT, file_hash)

    embeddings = get_embeddings_model()

    # ====================================================
    # Reuse an existing index for this exact file content,
    # if one was already built before - skips PDF parsing,
    # chunking and embedding entirely.
    # ====================================================

    existing = _load_existing_vectorstore(
        collection_name, persist_dir, embeddings
    )

    if existing is not None:

        return existing.as_retriever(
            search_kwargs={
                "k": 3
            }
        )


    # ====================================================
    # No existing index - process the file from scratch.
    # ====================================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(file_bytes)

        tmp_path = tmp_file.name


    try:

        # ====================================================
        # Load PDF
        # ====================================================

        loader = PyPDFLoader(
            tmp_path
        )

        docs = loader.load()


        # ====================================================
        # Split the document into Chunks
        #
        # PyPDFLoader שם metadata['page'] (0-indexed) על כל
        # Document אוטומטית - נשמר כך שה-Agent יוכל לצטט
        # "עמוד X" בתשובות שלו.
        # ====================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(
            docs
        )


        # ====================================================
        # Chroma Vector Store - persisted to disk under a
        # content-hash folder, so next time this exact file
        # is uploaded, _load_existing_vectorstore finds it.
        # ====================================================

        os.makedirs(persist_dir, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )


        # ====================================================
        # Retriever
        # ====================================================

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )


        return retriever


    finally:

        if os.path.exists(tmp_path):

            os.remove(
                tmp_path
            )
