from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from app.core.config import get_settings, get_embeddings, get_qdrant_client
from qdrant_client.models import Distance, VectorParams

def load(path: Path) -> list[Document]:
    return PyPDFLoader(file_path=path).load()

def ingest() -> int:
    settings = get_settings()

    file = Path(settings.data_path)
    documents = load(path=file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embed = get_embeddings(settings)

    client = get_qdrant_client(settings)

    if client.collection_exists(collection_name=settings.collection_name):
        client.delete_collection(collection_name=settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embed
    )

    vector_store.add_documents(documents=chunks)

    return len(chunks)


if __name__ == "__main__":
    print(ingest(), "chunks indexed!")
