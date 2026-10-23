import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from deal_voice.api.main import app
from deal_voice.config import settings


@pytest.fixture(autouse=True)
def enable_rbac_for_api_tests(monkeypatch):
    monkeypatch.setattr(settings, "rbac_enabled", True)


class FakeSearchEngine:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, user_role: str, top_k: int) -> list[Document]:
        self.calls.append(
            {"query": query, "user_role": user_role, "top_k": top_k}
        )
        if self.should_fail:
            raise RuntimeError("database details must stay private")

        return [
            Document(
                page_content="The Q3 price is $1,200 per seat.",
                metadata={"source": "pricing.pdf", "rerank_score": 0.95},
            )
        ]


class FakeGenerator:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def generate(
        self, question: str, retrieved_docs: list[Document]
    ) -> str:
        self.calls.append(
            {"question": question, "retrieved_docs": retrieved_docs}
        )
        return "The Q3 price is $1,200 per seat."


@pytest.fixture
def api_client():
    app.state.search_engine = FakeSearchEngine()
    app.state.generator = FakeGenerator()
    client = TestClient(app)

    try:
        yield client
    finally:
        client.close()
        del app.state.search_engine
        del app.state.generator


def test_search_route_returns_generated_answer(api_client: TestClient):
    response = api_client.post(
        "/api/v1/search",
        json={"query": "What is the Q3 price?", "top_k": 5},
        headers={"X-User-Role": "SALES"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is the Q3 price?",
        "authorized_role": "sales",
        "answer": "The Q3 price is $1,200 per seat.",
        "retrieved_chunks": [
            {
                "content": "The Q3 price is $1,200 per seat.",
                "source": "pricing.pdf",
                "rerank_score": 0.95,
                "speakers": None,
                "start_time": None,
                "end_time": None,
            }
        ],
    }

    assert api_client.app.state.search_engine.calls == [
        {"query": "What is the Q3 price?", "user_role": "sales", "top_k": 5}
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "top_k": 5},
        {"query": "question", "top_k": 0},
        {"query": "question", "top_k": 51},
    ],
)
def test_search_route_rejects_invalid_payload(
    api_client: TestClient, payload: dict[str, object]
):
    response = api_client.post(
        "/api/v1/search", json=payload, headers={"X-User-Role": "sales"}
    )

    assert response.status_code == 422


def test_search_route_rejects_unknown_role(api_client: TestClient):
    response = api_client.post(
        "/api/v1/search",
        json={"query": "What is the Q3 price?", "top_k": 5},
        headers={"X-User-Role": "unknown"},
    )

    assert response.status_code == 403


def test_search_route_hides_internal_errors():
    app.state.search_engine = FakeSearchEngine(should_fail=True)
    app.state.generator = FakeGenerator()

    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/search",
            json={"query": "What is the Q3 price?", "top_k": 5},
            headers={"X-User-Role": "sales"},
        )
    finally:
        client.close()
        del app.state.search_engine
        del app.state.generator

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Une erreur interne est survenue pendant la recherche."
    }
    assert "database details" not in response.text
