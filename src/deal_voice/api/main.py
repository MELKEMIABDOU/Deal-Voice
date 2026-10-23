from contextlib import asynccontextmanager
import logging
from typing import List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from http import HTTPStatus

from deal_voice.config import settings
from deal_voice.generation.generator import RAGGenerator
from deal_voice.retrieval.search import LocalSearchEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing search engine and LLM generator")
    app.state.search_engine = LocalSearchEngine()
    app.state.generator = RAGGenerator(model_name=settings.ollama_model_name)
    logger.info("Models initialized successfully")
    yield
    del app.state.search_engine
    del app.state.generator
    logger.info("Models released")


app = FastAPI(
    title="DealVoice Enterprise RAG",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Requête utilisateur limitée pour éviter les abus de bande passante.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Nombre de chunks à récupérer (entre 1 et 50).",
    )


class ChunkResponse(BaseModel):
    content: str
    source: str
    rerank_score: Optional[float] = None
    speakers: Optional[List[str]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    authorized_role: Optional[str] = None
    answer: str
    retrieved_chunks: List[ChunkResponse]


def get_current_user_role(x_user_role: str | None = Header(default=None)) -> str | None:
    if not settings.rbac_enabled:
        return None
    if x_user_role is None:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN, detail="Rôle utilisateur requis."
        )
    role = x_user_role.lower()
    if role not in settings.default_roles:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN, detail="Rôle non autorisé ou invalide."
        )
    return role


@app.post("/api/v1/search", response_model=QueryResponse)
def secure_rag_search(
    request: Request,
    query_request: QueryRequest,
    user_role: str = Depends(get_current_user_role),
):
    search_engine: LocalSearchEngine | None = getattr(
        request.app.state, "search_engine", None
    )
    generator: RAGGenerator | None = getattr(
        request.app.state, "generator", None
    )

    if not search_engine or not generator:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Moteurs non initialisés.")

    try:
        retrieved_docs = search_engine.search(
            query=query_request.query,
            user_role=user_role,
            top_k=query_request.top_k,
        )

        generated_answer = generator.generate(
            question=query_request.query, retrieved_docs=retrieved_docs
        )

        chunks = [
            ChunkResponse(
                content=doc.page_content,
                source=doc.metadata.get("source", "inconnu"),
                rerank_score=doc.metadata.get("rerank_score"),
                speakers=doc.metadata.get("speakers"),
                start_time=doc.metadata.get("start_time"),
                end_time=doc.metadata.get("end_time"),
            )
            for doc in retrieved_docs
        ]

        return QueryResponse(
            query=query_request.query,
            authorized_role=user_role,
            answer=generated_answer,
            retrieved_chunks=chunks,
        )

    except Exception:
        logger.exception("RAG search failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue pendant la recherche.",
        )
