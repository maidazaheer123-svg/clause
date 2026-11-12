from pathlib import Path
from functools import lru_cache
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=ENV_FILE, extra="ignore")

    #App
    app_name: str = "Clause"
    data_path: str = "files/shop_handbook.pdf"

    #Chat
    chat_provider: str = "ollama"
    chat_model: str = "gemma3:1b"
    temperature: float = 0.1

    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    #Indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Qdrant
    collection_name: str = "shop_handbook"
    qdrant_data: str = "./vector_stores"

    #Retrival
    top_k: int = 3
    score_threshold: float = 0.5

    #Prompts
    system: str = (
        "You are a helpful AI Clause shop assistant! "
        "Answers only from provided contnt"
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