import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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
        # Local embeddings
        # ====================================================

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


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