from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


DATA_PATH = "shop_handbook.pdf"           # Client doc
CHAT_PROVIDER = "ollama"                  # AI Provider
CHAT_MODEL = "gemma3:1b"                  # LLM
EMBED_PROVIDER = "ollama"                 # AI Embedding Provider
EMBED_MODEL = "nomic-embed-text:latest"   # Embedding Model
EMBED_DIM = 768                           # Embedding Dimension

COLLECTION_NAME = "shop_handbook"         # Collection name for Qdrant
QDRANT_DATA = "./vector_store"             # Qdrant data path

SYSTEM = """
You are a helpful AI clause shop assistant!
"""

client = QdrantClient(path=QDRANT_DATA)

# Indexing
# 1. Load
pages = PyPDFLoader(DATA_PATH).load_and_split() # List of Documents

# 2. Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(pages)

# 3. Embed
embdd = init_embeddings(
    model=EMBED_MODEL,
    provider=EMBED_PROVIDER
)

# 4. Store

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embdd
)

vector_store.add_documents(documents=chunks)