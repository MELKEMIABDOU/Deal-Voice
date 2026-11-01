from langchain_core.documents import Document

from deal_voice.config import settings
from deal_voice.ingestion import indexer as indexer_module
from deal_voice.ingestion.indexer import EnterpriseIndexer


def test_audio_parent_documents_keep_turn_boundaries():
    document = Document(
        page_content="ignored",
        metadata={
            "source": "call.mp3",
            "file_type": "audio",
            "turns": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Les fontaines à eau remplacent les bouteilles.",
                    "start_time": 1.0,
                    "end_time": 4.0,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Nous pouvons fixer un rendez-vous mardi.",
                    "start_time": 4.0,
                    "end_time": 8.0,
                },
            ],
        },
    )

    parents = EnterpriseIndexer._audio_parent_documents(document)

    assert len(parents) == 1
    assert "fontaines à eau" in parents[0].page_content
    assert "rendez-vous mardi" in parents[0].page_content
    assert parents[0].metadata["start_time"] == 1.0
    assert parents[0].metadata["end_time"] == 8.0


def test_audio_parent_documents_repeat_complete_turns_at_boundaries(monkeypatch):
    monkeypatch.setattr(settings, "parent_chunk_size", 90)
    monkeypatch.setattr(settings, "parent_chunk_overlap", 35)
    turns = [
        {
            "speaker": f"SPEAKER_{index}",
            "text": text,
            "start_time": float(index),
            "end_time": float(index + 1),
        }
        for index, text in enumerate(
            [
                "Karima gère les achats de l'agence.",
                "Le commercial présente la nouvelle machine.",
                "Karima demande si la machine fait du bruit.",
            ]
        )
    ]
    document = Document(
        page_content="ignored",
        metadata={"source": "call.mp3", "file_type": "audio", "turns": turns},
    )

    parents = EnterpriseIndexer._audio_parent_documents(document)

    assert len(parents) == 2
    assert turns[1]["text"] in parents[0].page_content
    assert turns[1]["text"] in parents[1].page_content
    assert turns[2]["text"] in parents[1].page_content
    assert parents[1].metadata["start_time"] == turns[1]["start_time"]


def test_audio_indexing_keeps_prebuilt_parents_and_child_overlap(monkeypatch):
    captured = {}

    class FakeParentDocumentRetriever:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def add_documents(self, documents, ids=None):
            captured["documents"] = documents

    monkeypatch.setattr(
        indexer_module, "ParentDocumentRetriever", FakeParentDocumentRetriever
    )
    indexer = EnterpriseIndexer.__new__(EnterpriseIndexer)
    indexer.vector_store = object()
    indexer.docstore = object()
    indexer.retriever = object()
    document = Document(
        page_content="ignored",
        metadata={
            "source": "call.mp3",
            "file_type": "audio",
            "turns": [{"speaker": "SPEAKER_00", "text": "Un tour complet."}],
        },
    )

    indexer.index_documents([document])

    assert captured["parent_splitter"] is None
    assert captured["child_splitter"]._chunk_overlap == settings.child_chunk_overlap
    assert len(captured["documents"]) == 1
