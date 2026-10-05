import hashlib
import json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_ROOT = os.path.join(PROJECT_ROOT, "chroma_store")
FILE_REGISTRY = os.path.join(PERSIST_ROOT, "file_registry.json")


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


def file_content_hash(file_bytes: bytes) -> str:
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


def load_saved_pdf(file_name: str):
    """Return the saved retriever for a previously processed filename."""

    if not os.path.isfile(FILE_REGISTRY):
        return None

    with open(FILE_REGISTRY, "r", encoding="utf-8") as registry_file:
        registry = json.load(registry_file)

    entry = registry.get(os.path.normcase(file_name))
    file_hash = entry.get("file_hash") if isinstance(entry, dict) else entry
    if not file_hash:
        return None

    if (
        not isinstance(file_hash, str)
        or len(file_hash) != 16
        or any(character not in "0123456789abcdef" for character in file_hash)
    ):
        raise ValueError(f"Invalid saved index for file: {file_name}")

    persist_dir = os.path.join(PERSIST_ROOT, file_hash)
    vectorstore = _load_existing_vectorstore(
        f"doc_{file_hash}",
        persist_dir,
        get_embeddings_model(),
    )
    if vectorstore is None:
        return None

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def list_saved_pdfs() -> list[str]:
    """List filenames with an available persisted vector index."""

    return sorted(
        {file_name for file_name, _, _ in _saved_file_records()},
        key=str.casefold,
    )


def has_saved_pdf(file_name: str) -> bool:
    normalized_name = os.path.normcase(file_name)
    return any(
        os.path.normcase(saved_name) == normalized_name
        for saved_name, _, _ in _saved_file_records()
    )


def _saved_file_records() -> list[tuple[str, str, str]]:
    if not os.path.isfile(FILE_REGISTRY):
        return []

    with open(FILE_REGISTRY, "r", encoding="utf-8") as registry_file:
        registry = json.load(registry_file)

    records = []
    seen_hashes = set()
    for registry_key, entry in registry.items():
        file_name = (
            entry.get("file_name", registry_key)
            if isinstance(entry, dict)
            else registry_key
        )
        file_hash = entry.get("file_hash") if isinstance(entry, dict) else entry
        if (
            not isinstance(file_name, str)
            or not isinstance(file_hash, str)
            or len(file_hash) != 16
            or any(character not in "0123456789abcdef" for character in file_hash)
            or file_hash in seen_hashes
        ):
            continue

        persist_dir = os.path.join(PERSIST_ROOT, file_hash)
        if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
            continue

        records.append((file_name, file_hash, persist_dir))
        seen_hashes.add(file_hash)

    return records


class SavedDocumentsRetriever:
    def __init__(self, vectorstores: list[tuple[str, Chroma]]):
        self._vectorstores = vectorstores

    def invoke(self, query: str):
        results = []
        for file_name, vectorstore in self._vectorstores:
            for document, score in vectorstore.similarity_search_with_score(
                query,
                k=3,
            ):
                document.metadata["file_name"] = file_name
                results.append((score, document))

        results.sort(key=lambda result: result[0])
        return [document for _, document in results[:6]]


def load_all_saved_pdfs() -> SavedDocumentsRetriever:
    """Load every registered PDF index for combined document search."""

    embeddings = get_embeddings_model()
    vectorstores = []
    for file_name, file_hash, persist_dir in _saved_file_records():
        vectorstore = _load_existing_vectorstore(
            f"doc_{file_hash}",
            persist_dir,
            embeddings,
        )
        if vectorstore is not None:
            vectorstores.append((file_name, vectorstore))

    return SavedDocumentsRetriever(vectorstores)


def _register_file(file_name: str, file_hash: str) -> None:
    os.makedirs(PERSIST_ROOT, exist_ok=True)
    registry = {}
    if os.path.isfile(FILE_REGISTRY):
        with open(FILE_REGISTRY, "r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)

    registry[os.path.normcase(file_name)] = {
        "file_name": file_name,
        "file_hash": file_hash,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=PERSIST_ROOT,
            delete=False,
            suffix=".tmp",
        ) as registry_file:
            temporary_path = registry_file.name
            json.dump(registry, registry_file, ensure_ascii=False)
        os.replace(temporary_path, FILE_REGISTRY)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def process_pdf_file(uploaded_file):
    """
    מקבל קובץ PDF, מחלק אותו לקטעים ובונה (או טוען, אם כבר קיים)
    אינדקס וקטורי שמור בדיסק (Chroma persist_directory).
    """

    file_bytes = uploaded_file.getvalue()
    file_hash = file_content_hash(file_bytes)

    collection_name = f"doc_{file_hash}"
    persist_dir = os.path.join(PERSIST_ROOT, file_hash)

    # ====================================================
    # Reuse an existing index for this exact file content,
    # if one was already built before - skips PDF parsing,
    # chunking and embedding entirely.
    # ====================================================

    embeddings = get_embeddings_model()
    existing = _load_existing_vectorstore(collection_name, persist_dir, embeddings)

    if existing is not None:
        _register_file(uploaded_file.name, file_hash)
        return existing.as_retriever(search_kwargs={"k": 3})


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

        _register_file(uploaded_file.name, file_hash)

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
