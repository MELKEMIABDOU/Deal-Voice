import json
import pytest
from pathlib import Path
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# Import your pipeline components and the custom judge
from deal_voice.retrieval.search import LocalSearchEngine
from deal_voice.generation.generator import RAGGenerator
from tests.evaluation.local_judge import LocalOllamaJudge

# 1. Load the Golden Dataset
def load_golden_dataset():
    dataset_path = Path("tests/evaluation/golden_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 2. Shared fixture to avoid re-initializing models and locking Qdrant DB concurrently
@pytest.fixture(scope="module")
def rag_pipeline():
    search_engine = LocalSearchEngine()
    generator = RAGGenerator(model_name="llama3")
    local_judge = LocalOllamaJudge(model_name="llama3")
    yield search_engine, generator, local_judge
    search_engine.client.close()

# 3. Pytest Parametrize dynamically creates a test for every item in the JSON
@pytest.mark.parametrize("test_case_data", load_golden_dataset())
def test_rag_pipeline_quality(test_case_data, rag_pipeline):
    search_engine, generator, local_judge = rag_pipeline
    
    # Faithfulness checks if the LLM hallucinated outside the retrieved context
    faithfulness_metric = FaithfulnessMetric(threshold=0.85, model=local_judge)
    
    # Answer Relevancy checks if the LLM actually answered the user's prompt
    relevancy_metric = AnswerRelevancyMetric(threshold=0.85, model=local_judge)
    
    query = test_case_data["query"]
    user_role = test_case_data["role"]
    
    # --- ACT: Run the RAG Pipeline ---
    retrieved_docs = search_engine.search(query=query, user_role=user_role, top_k=5)
    actual_output = generator.generate(question=query, retrieved_docs=retrieved_docs)
    
    # Combine context chunks into a single string for DeepEval
    retrieval_context = [doc.page_content for doc in retrieved_docs]
    
    # --- ASSEMBLE: Create the DeepEval Test Case ---
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    # --- ASSERT: Have the Local Judge score the pipeline ---
    # If the score drops below 0.85, this Pytest will fail the CI/CD build.
    assert_test(test_case, [faithfulness_metric, relevancy_metric])