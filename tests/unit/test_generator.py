from langchain_core.documents import Document

from deal_voice.generation.generator import RAGGenerator


class FakeChain:
    def __init__(self):
        self.payload = None

    def invoke(self, payload):
        self.payload = payload
        return "Réponse de test"


def test_generator_distinguishes_topic_from_transcript_language():
    generator = RAGGenerator.__new__(RAGGenerator)
    generator.system_prompt = (
        "Do not confuse the language of the transcript with its subject. "
        "Describe concrete business topics."
    )
    generator.chain = FakeChain()

    answer = generator.generate(
        "C'est quoi le sujet de la conversation ?",
        [
            Document(
                page_content="Le client demande une assistance pour sa ligne mobile.",
                metadata={
                    "source": "appel.mp3",
                    "file_type": "audio",
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                },
            )
        ],
    )

    assert answer == "Réponse de test"
    assert "ligne mobile" in generator.chain.payload["context"]
    assert "appel.mp3" in generator.chain.payload["context"]


def test_generator_prompt_prioritizes_business_abstraction():
    generator = RAGGenerator.__new__(RAGGenerator)
    generator.system_prompt = (
        "Abstract incidental literal details into the narrowest business problem. "
        "Preserve literal detail for concrete example questions."
    )

    assert "business problem" in generator.system_prompt
    assert "literal detail" in generator.system_prompt


def test_generator_prompt_refuses_missing_specific_facts():
    generator = RAGGenerator.__new__(RAGGenerator)
    generator.system_prompt = (
        "If a specific fact is absent, answer directly that the information is "
        "unavailable instead of giving a general summary."
    )

    assert "specific fact is absent" in generator.system_prompt
    assert "general summary" in generator.system_prompt
