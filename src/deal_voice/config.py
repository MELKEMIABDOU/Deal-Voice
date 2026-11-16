from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Base de données Qdrant
    qdrant_storage_path: str = "./local_qdrant_db"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = Field(default=None, repr=False)
    parent_docstore_path: str = "./local_qdrant_db/parent_docstore"
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
    whisper_model_name: str = Field(default="medium", min_length=1)
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_compute_type: Literal["int8", "float16", "float32", "int8_float16"] = "int8"
    whisper_language: str = Field(default="fr", min_length=2, max_length=16)
    whisper_batch_size: int = Field(default=1, ge=1, le=16)
    audio_worker_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    audio_worker_cpu_threads: int = Field(default=2, ge=1)
    pdf_worker_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    pdf_worker_cpu_threads: int = Field(default=2, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_whisper_runtime(self):
        if self.whisper_device == "cpu" and self.whisper_compute_type in {
            "float16",
            "int8_float16",
        }:
            raise ValueError("CPU transcription requires int8 or float32.")
        return self


settings = Settings()
