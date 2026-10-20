import shutil
from pathlib import Path

from qdrant_client import QdrantClient, models
from deal_voice.config import settings


def ensure_collection_schema(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    db_path: str,
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
                Path(db_path) / "parent_docstore",
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
