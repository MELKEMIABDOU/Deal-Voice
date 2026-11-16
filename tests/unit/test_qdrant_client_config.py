from deal_voice.ingestion import qdrant_schema


def test_qdrant_client_uses_server_when_url_is_configured(monkeypatch):
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(qdrant_schema, "QdrantClient", fake_client)

    qdrant_schema.create_qdrant_client(
        path="ignored-local-path",
        url="http://qdrant:6333",
        api_key="secret",
    )

    assert calls == [{"url": "http://qdrant:6333", "api_key": "secret"}]


def test_qdrant_client_keeps_embedded_mode_without_url(monkeypatch):
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(qdrant_schema, "QdrantClient", fake_client)

    qdrant_schema.create_qdrant_client(path="local-db")

    assert calls == [{"path": "local-db"}]
