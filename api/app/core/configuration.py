from pathlib import Path
from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain.embeddings import Embeddings
from pydantic_settings import BaseSettings, settingsConfigDict
from qdrant_client import QdrantClient

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"  #__file__ means current file

class Settings(BaseSettings):

    model_config = settingsConfigDict(env_prefix="RAG_", env_file=ENV_FILE, extra="ignore")

    #App
    app_name: str = "Clause"
    data_path = "data/shop_handbook.pdf"           # Client doc

    #Chat
    chat_provider = "ollama"                     # AI Chat Provider
    chat_model = "gemma3:1b"                  # LLM
    temperature: float = 0.1


    #Embeddings
    embed_provider = "ollama"                 # AI Embedding Provider
    embed_model = "nomic-embed-text:latest"   # Embedding Model
    embed_dim = 768                           # Embedding Dimension
    

    #indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    #Qdrant
    collection_name = "shop_handbook"         # Collection name for Qdrant
    qdrant_data = "./qdrant_data"             # Qdrant data path

    #Retrieval
    top_k: int = 33                            # Number of top results to retrieve
    score_threshold: float = 0.5                 # Minimum score threshold for retrieval

    #Prompts
    system: str = (
    "You are a helpful AI clause shop assistant!"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_embeddings(settings: Settings) -> Embeddings:
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider
    )

def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        path=settings.qdrant_data
        )