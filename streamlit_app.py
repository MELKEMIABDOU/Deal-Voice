"""Interactive Streamlit client for the Deal-Voice RAG pipeline."""

import atexit
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


st.set_page_config(
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
    """Persist and index multiple uploaded PDF or audio files."""
    upload_dir = Path("data") / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    indexed_names: list[str] = []

    for uploaded_file in uploaded_files:
        file_path = upload_dir / Path(uploaded_file.name).name
        file_path.write_bytes(uploaded_file.getbuffer())
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(
                EnterpriseDocumentParser(default_roles=[]).parse_pdf(
                    file_path, allowed_roles=None
                )
            )
        elif suffix in {".mp3", ".wav", ".m4a", ".flac"}:
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
                output_dir=Path("data") / "processed",
            )
            documents.append(document)
        else:
            raise ValueError(
                "Format non supporté. Utilisez PDF, MP3, WAV, M4A ou FLAC."
            )
        indexed_names.append(file_path.name)

    indexer = EnterpriseIndexer()
    try:
        indexer.delete_sources(indexed_names)
        indexer.index_documents(documents)
    finally:
        indexer.client.close()
    return indexed_names


def main() -> None:
    st.session_state.setdefault("uploaded_sources", [])
    st.session_state.setdefault("selected_sources", [])
    pending_sources = st.session_state.pop("pending_sources", None)
    if pending_sources is not None:
        st.session_state["selected_sources"] = pending_sources

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
        role_options = ["Aucun filtre"] + settings.default_roles
        selected_role = st.selectbox("Filtre de rôle", role_options)
        user_role = None if selected_role == "Aucun filtre" else selected_role
        source_options = st.session_state["uploaded_sources"]
        selected_sources = st.multiselect(
            "Documents à interroger",
            source_options,
            key="selected_sources",
            help="La liste contient uniquement les fichiers uploadés dans cette session.",
        )
        source = selected_sources
        whisper_model = st.selectbox(
            "Modèle WhisperX",
            ["tiny", "base", "small"],
            index=1,
            help="base/small peuvent améliorer le français, mais sont plus lents.",
        )

        st.divider()
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
            st.session_state["uploaded_sources"] = sorted(
                set(st.session_state["uploaded_sources"]).union(indexed_names)
            )
            st.session_state["pending_sources"] = indexed_names
            st.success(
                f"{len(indexed_names)} fichier(s) indexé(s) : "
                + ", ".join(indexed_names)
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
        if not selected_sources:
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
                source=source,
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
