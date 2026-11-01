"""Bounded uploads and document management for the single-user client portal."""

import hashlib
import logging
import uuid
from pathlib import Path

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from deal_voice.config import settings
from deal_voice.ingestion.uploads import (
    SUPPORTED_SUFFIXES,
    cleanup_upload,
    index_uploaded_file,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def take_operation(request: Request):
    lock = request.app.state.operation_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(
            409, "Une opération est déjà en cours. Réessayez après sa fin."
        )
    return lock


@router.get("/documents")
def list_documents(request: Request):
    lock = take_operation(request)
    try:
        return {"documents": request.app.state.search_engine.list_documents()}
    finally:
        lock.release()


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request):
    lock = take_operation(request)
    try:
        engine = request.app.state.search_engine
        if not any(
            row["document_id"] == document_id for row in engine.list_documents()
        ):
            raise HTTPException(404, "Ce document n'est plus disponible.")
        engine.delete_documents([document_id])
        return Response(status_code=204)
    finally:
        lock.release()


def _index_job(app, document_id: str, path: Path, fingerprint: str) -> None:
    job = app.state.jobs[document_id]
    try:
        index_uploaded_file(path, document_id, fingerprint, app.state.search_engine)
        job.update(
            status="completed",
            document={
                "document_id": document_id,
                "source": path.name,
                "file_type": "pdf" if path.suffix.lower() == ".pdf" else "audio",
            },
        )
    except Exception:
        logger.exception("Document import failed: %s", document_id)
        try:
            cleanup_upload(document_id)
        except Exception:
            logger.exception("Upload cleanup failed: %s", document_id)
        job.update(
            status="failed",
            error="L'import a échoué. Vérifiez le fichier, son format et sa durée, puis contactez l'opérateur si nécessaire.",
        )
    finally:
        app.state.operation_lock.release()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            404,
            "Cet import n'est plus suivi. Actualisez les documents ; le serveur a peut-être redémarré.",
        )
    return dict(job)


@router.put("/documents", status_code=202)
async def upload_document(
    request: Request,
    background: BackgroundTasks,
    filename: str = Query(min_length=1, max_length=240),
):
    if (
        any(char in filename for char in '/\\\0:<>"|?*')
        or len(filename.encode("utf-8")) > 240
        or Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES
    ):
        raise HTTPException(
            422, "Nom ou format non accepté. Utilisez PDF, MP3, WAV, M4A ou FLAC."
        )
    lock = take_operation(request)
    document_id = str(uuid.uuid4())
    path = Path("data") / "raw" / document_id / filename
    try:
        limit = settings.max_upload_size_mb * 1024 * 1024
        await run_in_threadpool(path.parent.mkdir, parents=True, exist_ok=False)
        fingerprint, size = hashlib.sha256(), 0
        async with await anyio.open_file(path, "wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        413, f"Le fichier dépasse {settings.max_upload_size_mb} Mo."
                    )
                fingerprint.update(chunk)
                await handle.write(chunk)
        if not size:
            raise HTTPException(422, "Le fichier est vide.")
        jobs = request.app.state.jobs
        while len(jobs) >= 100:
            del jobs[next(iter(jobs))]
        job = {"job_id": document_id, "source": filename, "status": "processing"}
        jobs[document_id] = job
        background.add_task(
            _index_job, request.app, document_id, path, fingerprint.hexdigest()
        )
        return dict(job)
    except BaseException:
        # Includes a disconnected/cancelled upload; only this new UUID is removed.
        try:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(cleanup_upload, document_id)
        finally:
            lock.release()
        raise
