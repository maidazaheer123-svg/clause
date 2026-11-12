from pathlib import Path

# from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from core.config import get_settings, get_embeddings, get_qdrant_client


def load(path: str) -> None:
    return PyPDFLoader(file_path=path).load_and_split()

def ingest() -> init:
    settings = get_settings()

    #1. Load
    file = Path(settings.data_path)
    documents = load(path=file)

    #2. chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separator=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    #3. Embeddings
    embed = get_embeddings(settings)

    #4. Store
    client = get_qdrant_client(settings)

    if client.collection_exists(collection_name=settings.collection_name):
        client.delete_collection(collection_name=settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vector_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embed
    )

    return len(chunks)