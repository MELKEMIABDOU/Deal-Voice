"""Interactive Streamlit client for the Deal-Voice RAG pipeline."""

import atexit
import shutil
import uuid
from typing import Any
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from deal_voice.config import settings
from deal_voice.generation.generator import RAGGenerator
from deal_voice.ingestion.audio import WhisperXTranscriber
from deal_voice.ingestion.document import EnterpriseDocumentParser
from deal_voice.ingestion.indexer import EnterpriseIndexer
from deal_voice.retrieval.search import LocalSearchEngine


st.set_page_config (
    page_title="Deal-Voice RAG",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def load_search_engine() -> LocalSearchEngine:
    return LocalSearchEngine()


@st.cache_resource
def load_generator(model_name: str) -> RAGGenerator:
    return RAGGenerator(model_name=model_name)


def clear_streamlit_resources() -> None:
    """Release cached model and Qdrant resources when the process exits."""
    st.cache_resource.clear()


atexit.register(clear_streamlit_resources)


def render_source(document: Document, index: int) -> None:
    metadata: dict[str, Any] = document.metadata
    source = metadata.get("source", "inconnu")
    st.markdown(f"**Source {index}: `{source}`**")

    citation = []
    speakers = metadata.get("speakers")
    if speakers:
        citation.append(f"Locuteurs : {', '.join(speakers)}")
    if metadata.get("start_time") is not None:
        citation.append(f"Début : {metadata['start_time']:.1f}s")
    if metadata.get("end_time") is not None:
        citation.append(f"Fin : {metadata['end_time']:.1f}s")
    if metadata.get("rerank_score") is not None:
        citation.append(f"Score : {metadata['rerank_score']:.3f}")

    if citation:
        st.caption(" | ".join(citation))
    st.write(document.page_content)


def index_uploaded_files(uploaded_files: list[Any], whisper_model: str) -> list[str]:
    """Persist and index uploads, rolling back files and vectors on failure."""
    if len(uploaded_files) > settings.max_upload_files:
        raise ValueError(f"Maximum {settings.max_upload_files} fichiers par import.")
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    for item in uploaded_files:
        if item.size > max_bytes:
            raise ValueError(
                f"{item.name} dépasse la limite de {settings.max_upload_size_mb} Mo."
            )

    documents: list[Document] = []
    document_ids: list[str] = []
    indexer = None
    try:
        for uploaded_file in uploaded_files:
            document_id = str(uuid.uuid4())
            document_ids.append(document_id)
            file_path = Path("data") / "raw" / document_id / Path(uploaded_file.name).name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(uploaded_file.getbuffer())
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                parsed = EnterpriseDocumentParser(default_roles=[]).parse_pdf(
                    file_path, allowed_roles=None
                )
                for document in parsed:
                    document.metadata.update(
                        {"document_id": document_id, "original_path": str(file_path)}
                    )
                documents.extend(parsed)
            elif suffix in {".mp3", ".wav", ".m4a", ".flac"}:
                import torchaudio

                audio_info = torchaudio.info(str(file_path))
                duration_minutes = audio_info.num_frames / audio_info.sample_rate / 60
                if duration_minutes > settings.max_audio_duration_minutes:
                    raise ValueError(
                        f"{uploaded_file.name} dépasse la durée maximale de "
                        f"{settings.max_audio_duration_minutes} minutes."
                    )
                if not settings.hf_token:
                    raise ValueError(
                        "HF_TOKEN est nécessaire pour la diarisation audio. "
                        "Ajoutez-le dans .env avant l'upload."
                    )
                document = WhisperXTranscriber(
                    model_name=whisper_model,
                    device="cpu",
                    compute_type="int8",
                    language="fr",
                ).transcribe_to_document(
                    file_path,
                    hf_token=settings.hf_token,
                    output_dir=Path("data") / "processed" / document_id,
                )
                document.metadata.update(
                    {"document_id": document_id, "original_path": str(file_path)}
                )
                documents.append(document)
            else:
                raise ValueError(
                    "Format non supporté. Utilisez PDF, MP3, WAV, M4A ou FLAC."
                )

        indexer = EnterpriseIndexer()
        indexer.index_documents(documents)
    except Exception:
        if indexer is not None:
            try:
                indexer.delete_documents(document_ids)
            except Exception:
                pass
        for document_id in document_ids:
            shutil.rmtree(Path("data") / "raw" / document_id, ignore_errors=True)
            shutil.rmtree(Path("data") / "processed" / document_id, ignore_errors=True)
        raise
    finally:
        if indexer is not None:
            indexer.client.close()
    return document_ids


def main() -> None:
    st.session_state.setdefault("selected_documents", [])

    st.title("🎙️ Deal-Voice")
    st.subheader("Multimodal RAG pour appels commerciaux, réunions et documents")
    st.info(
        "Cette interface teste le retrieval Qdrant et la génération LLM Ollama. "
        "Les documents doivent être indexés avant la recherche."
    )

    with st.sidebar:
        st.header("Configuration")
        model_name = st.text_input(
            "Modèle Ollama",
            value=settings.ollama_model_name,
            help="Le modèle doit être installé avec `ollama pull <modèle>`.",
        ).strip()
        top_k = st.slider(
            "Nombre maximum de résultats",
            min_value=1,
            max_value=10,
            value=5,
            help="Nombre maximal de chunks parents envoyés au LLM.",
        )
        user_role = None
        if settings.rbac_enabled:
            role_options = ["Aucun filtre"] + settings.default_roles
            selected_role = st.selectbox("Filtre de rôle", role_options)
            user_role = None if selected_role == "Aucun filtre" else selected_role
        search_engine = load_search_engine()
        available_documents = search_engine.list_documents()
        labels = {doc["document_id"]: doc["source"] for doc in available_documents}
        selected_ids = st.multiselect(
            "Documents à interroger",
            list(labels),
            format_func=lambda doc_id: labels.get(doc_id, doc_id),
            key="selected_documents",
            help="Les fichiers de même nom restent identifiés séparément.",
        )
        whisper_model = st.selectbox(
            "Modèle WhisperX",
            ["tiny", "base", "small"],
            index=1,
            help="base/small peuvent améliorer le français, mais sont plus lents.",
        )

        st.divider()
        if st.button("Migrer les documents historiques sans ID"):
            migrated_count = search_engine.migrate_legacy_documents()
            st.success(
                f"Migration terminée : {migrated_count} document(s) ancien(s) identifié(s)."
            )
            st.rerun()
        if selected_ids and st.button("Supprimer les documents sélectionnés"):
            ids_to_delete = selected_ids
            search_engine.delete_documents(ids_to_delete)
            st.session_state["selected_documents"] = []
            st.success(f"{len(ids_to_delete)} document(s) supprimé(s).")
            st.rerun()
        st.markdown("**Préparation locale**")
        st.code("ollama serve\nollama pull <nom-du-modele>")

    st.header("Ajouter un document")
    uploaded_files = st.file_uploader(
        "Choisissez un ou plusieurs PDF/fichiers audio",
        type=["pdf", "mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True,
        help="Le fichier sera sauvegardé dans data/raw puis indexé dans Qdrant.",
    )
    if uploaded_files and st.button(
        "Indexer les fichiers sélectionnés", use_container_width=True
    ):
        try:
            st.cache_resource.clear()
            with st.spinner("Ingestion et indexation en cours..."):
                indexed_names = index_uploaded_files(uploaded_files, whisper_model)
            st.success(
                f"{len(indexed_names)} fichier(s) indexé(s)."
            )
            st.rerun()
        except Exception as exc:
            st.error("L'indexation a échoué. Vérifiez le fichier et la configuration.")
            st.warning(
                "Si vous venez de changer le modèle d'embedding, arrêtez puis "
                "relancez complètement Streamlit avant de réessayer."
            )
            st.exception(exc)

    query = st.text_area(
        "Posez une question sur les documents indexés",
        placeholder="Exemple : Quelles sont les conditions commerciales évoquées ?",
        height=100,
    )

    if st.button("Rechercher et générer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Saisissez une question avant de lancer la recherche.")
            return
        if not model_name:
            st.error("Le nom du modèle Ollama est obligatoire.")
            return
        if not selected_ids:
            st.warning(
                "Sélectionnez au moins un fichier uploadé dans "
                "'Documents à interroger'."
            )
            return

        try:
            search_engine = load_search_engine()
            generator = load_generator(model_name)
            documents = search_engine.search(
                query=query.strip(),
                user_role=user_role,
                document_ids=selected_ids,
                top_k=top_k,
            )
            answer = generator.generate(
                question=query.strip(),
                retrieved_docs=documents,
            )
        except Exception as exc:
            st.error(
                "La recherche ou la génération a échoué. "
                "Vérifiez Qdrant, Ollama et les modèles locaux."
            )
            st.exception(exc)
            return

        st.header("Réponse du LLM")
        st.markdown(answer)

        st.header(f"Sources récupérées ({len(documents)})")
        if not documents:
            st.warning("Aucune source autorisée n'a été trouvée.")
        else:
            for index, document in enumerate(documents, start=1):
                with st.expander(
                    f"Source {index} — {document.metadata.get('source', 'inconnu')}",
                    expanded=index == 1,
                ):
                    render_source(document, index)


if __name__ == "__main__":
    main()
