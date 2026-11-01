import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deal_voice.api.documents import router as documents_router
from deal_voice.api.documents import take_operation
from deal_voice.config import settings
from deal_voice.generation.generator import RAGGenerator
from deal_voice.retrieval.search import LocalSearchEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing search engine and LLM generator")
    app.state.search_engine = LocalSearchEngine()
    app.state.jobs = {}
    try:
        app.state.generator = RAGGenerator(model_name=settings.ollama_model_name)
        logger.info("Models initialized successfully")
        yield
    finally:
        app.state.search_engine.client.close()
        del app.state.search_engine
        if hasattr(app.state, "generator"):
            del app.state.generator
        logger.info("Models released")


app = FastAPI(
    title="DealVoice Enterprise RAG",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.operation_lock = Lock()
WEB_DIRECTORY = Path(__file__).resolve().parents[1] / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIRECTORY), name="assets")


@app.get("/", include_in_schema=False)
def client_portal():
    return FileResponse(WEB_DIRECTORY / "index.html")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/api/v1/status")
def server_status():
    return {
        "ready": hasattr(app.state, "search_engine")
        and hasattr(app.state, "generator"),
        "busy": app.state.operation_lock.locked(),
        "model": settings.ollama_model_name,
        "limits": {
            "files": settings.max_upload_files,
            "upload_mb": settings.max_upload_size_mb,
            "audio_minutes": settings.max_audio_duration_minutes,
        },
    }


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
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)


class ChunkResponse(BaseModel):
    content: str
    source: str
    rerank_score: float | None = None
    speakers: list[str] | None = None
    start_time: float | None = None
    end_time: float | None = None
    page: int | None = None
    section: str | None = None
    content_kind: str | None = None
    table_title: str | None = None
    table_complete: bool | None = None


class QueryResponse(BaseModel):
    query: str
    authorized_role: str | None = None
    answer: str
    retrieved_chunks: list[ChunkResponse]
    generation_status: str = "unknown"
    omitted_evidence_count: int = 0


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


app.include_router(documents_router, dependencies=[Depends(get_current_user_role)])


@app.post("/api/v1/search", response_model=QueryResponse)
def secure_rag_search(
    request: Request,
    query_request: QueryRequest,
    user_role: str | None = Depends(get_current_user_role),
):
    search_engine: LocalSearchEngine | None = getattr(
        request.app.state, "search_engine", None
    )
    generator: RAGGenerator | None = getattr(request.app.state, "generator", None)

    if not search_engine or not generator:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Moteurs non initialisés.",
        )

    lock = take_operation(request)
    try:
        search_options = {
            "query": query_request.query,
            "user_role": user_role,
            "top_k": query_request.top_k,
        }
        if query_request.document_ids is not None:
            search_options["document_ids"] = query_request.document_ids
        retrieved_docs = search_engine.search(**search_options)

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
                page=doc.metadata.get("page_number", doc.metadata.get("page")),
                section=doc.metadata.get("section"),
                content_kind=doc.metadata.get("content_kind"),
                table_title=doc.metadata.get("table_title"),
                table_complete=doc.metadata.get("table_complete"),
            )
            for doc in retrieved_docs
        ]

        trace = getattr(generator, "last_trace", {})
        return QueryResponse(
            query=query_request.query,
            authorized_role=user_role,
            answer=generated_answer,
            retrieved_chunks=chunks,
            generation_status=trace.get("status", "unknown"),
            omitted_evidence_count=trace.get("omitted_evidence_count", 0),
        )

    except Exception:
        logger.exception("RAG search failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue pendant la recherche.",
        ) from None
    finally:
        lock.release()
