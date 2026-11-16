import shutil
from pathlib import Path

from qdrant_client import QdrantClient, models

from deal_voice.config import settings


def create_qdrant_client(
    *, path: str, url: str | None = None, api_key: str | None = None
) -> QdrantClient:
    """Connect to Qdrant Server when configured, otherwise use local storage."""
    if url:
        return QdrantClient(url=url, api_key=api_key or None)
    return QdrantClient(path=path)


def ensure_collection_schema(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    parent_docstore_path: str,
) -> None:
    """Create or migrate a local collection to the configured vector size."""
    if client.collection_exists(collection_name):
        collection_info = client.get_collection(collection_name)
        configured_vectors = collection_info.config.params.vectors
        current_size = getattr(configured_vectors, "size", None)
        if current_size != vector_size:
            if not settings.allow_destructive_schema_migration:
                raise RuntimeError(
                    "Qdrant vector dimension mismatch. Refusing destructive "
                    "migration; back up the database and recreate it explicitly."
                )
            client.delete_collection(collection_name)
            shutil.rmtree(
                Path(parent_docstore_path),
                ignore_errors=True,
            )

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.allowed_roles",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD
            ),
        )
