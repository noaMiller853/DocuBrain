import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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


def process_pdf_file(uploaded_file):
    """
    מקבל קובץ PDF,
    מחלק אותו לקטעים
    ובונה ChromaDB.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getvalue()
        )

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
        # ====================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(
            docs
        )


        # ====================================================
        # Local embeddings (cached - see get_embeddings_model)
        # ====================================================

        embeddings = get_embeddings_model()


        # ====================================================
        # Chroma Vector Store
        # ====================================================

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
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