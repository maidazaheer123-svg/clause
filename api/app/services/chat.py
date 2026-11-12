from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from app.core.config import get_settings, get_embeddings, get_qdrant_client

NO_ANSWER = "I don't know. The clause Shop handler does not cover that"

class Source(BaseModel):
    source: str
    page: int | None = None
    score: float

class Answer(BaseModel):
    answer: str
    sources: list[Source]

class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_qdrant_client(settings=self.settings)

        if not self.client.collection_exists(collection_name=self.settings.collection_name):
            raise RuntimeError(f"Collection {self.settings.collection_name} not fount. Run first")

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings(settings=self.settings)
        )
        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.temperature,
            # google_api_key=self.settings.google_api_key
        )

    def ask(self, question: str) -> Answer | None:
        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty")

        results = self.store.similarity_search_with_score( query=question, k=self.settings.top_k)

        if not results:
            return Answer(answer=NO_ANSWER, sources=[])

        context = "\n\n".join(doc.page_content for doc, score in results)

        messages = [
            {"role": "system", "content": self.settings.system},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]

        return Answer(
            answer=self.llm.invoke(messages).text,
            sources=[
                Source(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(number=score, ndigits=4)
                )
                for doc, score in results
            ]
        )

    def close(self) -> None:
        self.client.close()