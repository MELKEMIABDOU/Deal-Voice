from langchain_core.documents import Document

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
