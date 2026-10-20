
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama

class LocalOllamaJudge(DeepEvalBaseLLM):
    """Overrides DeepEval's OpenAI default to use our local Ollama instance for grading."""
    
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        # Temperature 0 is critical for a Judge LLM so its grading is deterministic
        self.model = ChatOllama(model=model_name, temperature=0.0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """DeepEval calls this synchronous method to grade the test."""
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        """DeepEval calls this async method for parallel testing."""
        res = await self.model.ainvoke(prompt)
        return res.content
        
    def get_model_name(self):
        return f"Local-Ollama-{self.model_name}"