from dataclasses import dataclass
import warnings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.core.config import Settings


@dataclass
class RetrievedDocument:
    document: Document
    score: float


class RetrieverService:
    def __init__(self, vectorstore: Chroma, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings

    def search(self, query: str) -> list[RetrievedDocument]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Relevance scores must be between 0 and 1.*",
                category=UserWarning,
            )
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=self.settings.retrieval_k,
            )
        return [RetrievedDocument(document=doc, score=_normalize_score(float(score))) for doc, score in results]


def _normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))
