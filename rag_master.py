from pathlib import Path
import pandas as pd
import re
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

DOCS_PATH = Path("/home/aarav/MicroGPT/documents")


class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        DOCS_PATH.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_documents(self):
        docs = []

        for file in DOCS_PATH.iterdir():
            if not file.is_file():
                continue

            if file.name.startswith('.') or "Zone.Identifier" in file.name:
                continue

            try:
                if file.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file))
                    raw_docs = loader.load()
                    for d in raw_docs:
                        cleaned = self.clean_text(d.page_content)
                        if cleaned:
                            docs.append(Document(page_content=cleaned, metadata=d.metadata))

                elif file.suffix.lower() in [".txt", ".md"]:
                    loader = TextLoader(str(file), encoding="utf-8")
                    raw_docs = loader.load()
                    for d in raw_docs:
                        cleaned = self.clean_text(d.page_content)
                        if cleaned:
                            docs.append(Document(page_content=cleaned, metadata=d.metadata))

                elif file.suffix.lower() == ".docx":
                    loader = UnstructuredWordDocumentLoader(str(file))
                    raw_docs = loader.load()
                    for d in raw_docs:
                        cleaned = self.clean_text(d.page_content)
                        if cleaned:
                            docs.append(Document(page_content=cleaned, metadata=d.metadata))

                elif file.suffix.lower() in [".csv", ".xlsx"]:
                    df = pd.read_excel(file) if file.suffix.lower() == ".xlsx" else pd.read_csv(file)
                    for _, row in df.iterrows():
                        text = " | ".join([str(cell) for cell in row.values])
                        cleaned = self.clean_text(text)
                        if cleaned:
                            docs.append(Document(page_content=cleaned))

            except Exception as e:
                print(f"Error loading {file}: {e}")

        return docs

    def get_dynamic_params(self, num_docs):
        if num_docs < 50:
            return 400, 80, 4
        elif num_docs < 200:
            return 600, 100, 5
        else:
            return 800, 150, 7

    def reload(self):
        print("🔄 Reloading RAG...")
        documents = self.load_documents()

        if not documents:
            print("⚠ No documents found (RAG inactive)")
            self.vectorstore = None
            self.retriever = None
            return False

        chunk_size, chunk_overlap, top_k = self.get_dynamic_params(len(documents))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = splitter.split_documents(documents)
        chunks = [c for c in chunks if c.page_content.strip()]

        if not chunks:
            self.vectorstore = None
            self.retriever = None
            return False

        self.vectorstore = FAISS.from_documents(chunks, self.embedding)

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 2}
        )

        return True

    def refresh_rag_index(self):
        self.vectorstore = None
        self.retriever = None
        return self.reload()


_rag_engine = None


def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine