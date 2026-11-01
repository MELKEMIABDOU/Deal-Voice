"""Client portal contracts using isolated files and no model downloads."""

import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from deal_voice.api import documents as documents_api
from deal_voice.api import main as api
from deal_voice.config import settings

EXISTING_ID = "b7a4eb06-5b7f-46af-9949-9f90fecba3df"


class FakeSearchEngine:
    def __init__(self):
        self.closed = False
        self.client = SimpleNamespace(close=self.close)
        self.embeddings = object()
        self.documents = [
            {"document_id": EXISTING_ID, "source": "rapport.pdf", "file_type": "pdf"}
        ]
        self.search_calls = []
        self.delete_calls = []

    def close(self):
        self.closed = True

    def list_documents(self):
        return list(self.documents)

    def delete_documents(self, document_ids):
        self.delete_calls.append(document_ids)
        self.documents = [
            row for row in self.documents if row["document_id"] not in document_ids
        ]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [
            Document(
                page_content="Les conditions sont décrites dans cette section.",
                metadata={
                    "source": "rapport.pdf",
                    "document_id": EXISTING_ID,
                    "page": 3,
                    "section": "Conditions",
                },
            )
        ]


class FakeGenerator:
    last_trace = {"status": "answer", "omitted_evidence_count": 1}

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "Les conditions figurent dans le rapport."


@pytest.fixture
def web_runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "rbac_enabled", False)
    monkeypatch.setattr(settings, "max_upload_files", 2)
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    monkeypatch.setattr(settings, "max_audio_duration_minutes", 3)
    engine, generator = FakeSearchEngine(), FakeGenerator()
    indexed = []
    monkeypatch.setattr(api, "LocalSearchEngine", lambda: engine)
    monkeypatch.setattr(api, "RAGGenerator", lambda **kwargs: generator)

    def index_file(
        file_path, document_id, content_sha256, search_engine, whisper_model="medium"
    ):
        assert search_engine is engine
        assert whisper_model == "medium"
        path = Path(file_path)
        content = path.read_bytes()
        assert content_sha256 == hashlib.sha256(content).hexdigest()
        assert str(uuid.UUID(document_id)) == document_id
        assert path.parent == Path("data/raw") / document_id or (
            path.parent == (tmp_path / "data/raw" / document_id)
        )
        indexed.append({"path": path, "content": content, "document_id": document_id})
        engine.documents.append(
            {"document_id": document_id, "source": path.name, "file_type": "pdf"}
        )

    monkeypatch.setattr(documents_api, "index_uploaded_file", index_file)
    return SimpleNamespace(
        engine=engine, generator=generator, indexed=indexed, root=tmp_path
    )


@pytest.fixture
def web_client(web_runtime):
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client
    assert web_runtime.engine.closed


def test_portal_and_assets_are_served_without_streamlit(web_client):
    response = web_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DealVoice" in response.text
    for name in ("app.js", "styles.css"):
        assert web_client.get(f"/assets/{name}").status_code == 200
    assert web_client.get("/assets/%2e%2e/%2e%2e/config.py").status_code == 404


def test_status_and_documents_report_current_server_state(web_client):
    status = web_client.get("/api/v1/status")
    assert status.status_code == 200
    assert status.json()["ready"] is True
    assert status.json()["busy"] is False
    assert status.json()["model"] == settings.ollama_model_name
    assert status.json()["limits"] == {"files": 2, "upload_mb": 1, "audio_minutes": 3}
    documents = web_client.get("/api/v1/documents")
    assert documents.status_code == 200
    assert documents.json()["documents"] == [
        {"document_id": EXISTING_ID, "source": "rapport.pdf", "file_type": "pdf"}
    ]


def test_raw_upload_uses_a_server_id_and_shared_ingestion(web_client, web_runtime):
    response = web_client.put(
        "/api/v1/documents",
        params={"filename": "Rapport client.pdf"},
        content=b"%PDF-test",
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "processing"
    completed = web_client.get(f"/api/v1/jobs/{job['job_id']}")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    result = completed.json()["document"]
    assert result["source"] == "Rapport client.pdf"
    assert result["file_type"] == "pdf"
    assert result["document_id"] == web_runtime.indexed[0]["document_id"]
    assert web_runtime.indexed[0]["content"] == b"%PDF-test"
    assert web_client.get("/api/v1/documents").json()["documents"][-1] == result


@pytest.mark.parametrize(
    "filename",
    [
        "../private.pdf",
        "..\\private.pdf",
        "/private.pdf",
        "C:\\private.pdf",
        "file.exe",
    ],
)
def test_upload_rejects_unsafe_or_unsupported_names(web_client, web_runtime, filename):
    response = web_client.put(
        "/api/v1/documents", params={"filename": filename}, content=b"file bytes"
    )
    assert response.status_code in (400, 415, 422)
    assert not web_runtime.indexed
    assert not list((web_runtime.root / "data/raw").glob("*"))


def test_upload_rejects_an_empty_file(web_client, web_runtime):
    response = web_client.put(
        "/api/v1/documents", params={"filename": "empty.pdf"}, content=b""
    )
    assert response.status_code in (400, 422)
    assert not web_runtime.indexed
    assert not list((web_runtime.root / "data/raw").glob("*"))


def test_chunked_upload_cannot_bypass_the_size_limit(web_client, web_runtime):
    def chunks():
        yield b"a" * (512 * 1024)
        yield b"b" * (512 * 1024 + 1)

    response = web_client.put(
        "/api/v1/documents",
        params={"filename": "large.pdf"},
        content=chunks(),
        headers={"Transfer-Encoding": "chunked"},
    )
    assert response.status_code == 413
    assert not web_runtime.indexed
    assert not list((web_runtime.root / "data/raw").glob("*"))


def test_upload_failure_removes_the_new_file_and_hides_internal_errors(
    web_client, web_runtime, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError("private engine details must not reach the client")

    monkeypatch.setattr(documents_api, "index_uploaded_file", fail)
    response = web_client.put(
        "/api/v1/documents", params={"filename": "broken.pdf"}, content=b"%PDF-test"
    )
    assert response.status_code == 202
    result = web_client.get(f"/api/v1/jobs/{response.json()['job_id']}")
    assert result.status_code == 200
    assert result.json()["status"] == "failed"
    assert "private engine details" not in result.text
    assert isinstance(result.json()["error"], str)
    assert not list((web_runtime.root / "data/raw").glob("*"))
    assert not web_client.get("/api/v1/status").json()["busy"]


def test_search_forwards_document_selection_and_exposes_response_scope(
    web_client, web_runtime
):
    response = web_client.post(
        "/api/v1/search",
        json={
            "query": "Quelles conditions ?",
            "top_k": 3,
            "document_ids": [EXISTING_ID],
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["generation_status"] == "answer"
    assert result["omitted_evidence_count"] == 1
    assert result["retrieved_chunks"][0]["page"] == 3
    assert result["retrieved_chunks"][0]["section"] == "Conditions"
    assert web_runtime.engine.search_calls == [
        {
            "query": "Quelles conditions ?",
            "user_role": None,
            "top_k": 3,
            "document_ids": [EXISTING_ID],
        }
    ]


def test_busy_server_rejects_overlapping_expensive_requests(web_client, web_runtime):
    lock = api.app.state.operation_lock
    assert lock.acquire(blocking=False)
    try:
        assert web_client.get("/api/v1/status").json()["busy"]
        assert (
            web_client.post("/api/v1/search", json={"query": "Question"}).status_code
            == 409
        )
        assert (
            web_client.put(
                "/api/v1/documents", params={"filename": "next.pdf"}, content=b"pdf"
            ).status_code
            == 409
        )
        assert not web_runtime.engine.search_calls
        assert not web_runtime.indexed
        assert not list((web_runtime.root / "data/raw").glob("*"))
    finally:
        lock.release()


def test_document_deletion_checks_existence(web_client, web_runtime):
    assert web_client.delete("/api/v1/documents/unknown-id").status_code == 404
    assert not web_runtime.engine.delete_calls
    response = web_client.delete(f"/api/v1/documents/{EXISTING_ID}")
    assert response.status_code == 204
    assert web_runtime.engine.delete_calls == [[EXISTING_ID]]
    assert web_client.get("/api/v1/documents").json() == {"documents": []}


def test_failed_startup_releases_the_local_database_owner(web_runtime, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("generator setup failed")

    monkeypatch.setattr(api, "RAGGenerator", fail)
    with pytest.raises(RuntimeError, match="generator setup failed"):
        with TestClient(api.app):
            pass
    assert web_runtime.engine.closed
