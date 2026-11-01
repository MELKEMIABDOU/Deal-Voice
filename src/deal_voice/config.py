from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Base de données Qdrant
    qdrant_storage_path: str = "./local_qdrant_db"
    collection_name: str = "enterprise_knowledge"
    allow_destructive_schema_migration: bool = False
    max_upload_size_mb: int = 100
    max_upload_files: int = 5
    max_audio_duration_minutes: int = 120

    # Modèles multilingues pour les contenus et requêtes en français
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_vector_size: int = 1024
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # Configuration Chunking (Small-to-Big)
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # Sécurité & RBAC
    rbac_enabled: bool = False
    default_roles: list[str] = ["sales", "admin", "executive", "intern", "guest"]

    # Modèle LLM Ollama
    ollama_model_name: str = "qwen3:8b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 512
    ollama_num_thread: int = 4
    ollama_keep_alive: str = "0s"
    ollama_reasoning: bool = False
    ollama_request_timeout_seconds: float = Field(default=120.0, gt=0)

    # WhisperX / Hugging Face
    hf_token: str = ""
    audio_worker_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    audio_worker_cpu_threads: int = Field(default=2, ge=1)
    pdf_worker_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    pdf_worker_cpu_threads: int = Field(default=2, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
