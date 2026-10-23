from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama # The new local integration

class RAGGenerator:
    """Generates grounded answers using retrieved context via local Ollama."""

    def __init__(self, model_name: str = "llama3"):
        self.system_prompt = (
            "You are an enterprise AI assistant for DealVoice. "
            "Answer accurately using ONLY the retrieved context below.\n\n"
            "STRICT RULES:\n"
            "1. Rely strictly on clear facts mentioned in the context. Do not extrapolate or assume.\n"
            "2. If the context does not contain enough information to answer the question, "
            "state in the user's language that the authorized knowledge base does not "
            "contain sufficient information to answer it. In French, use exactly: "
            "'Je ne dispose pas de suffisamment d'informations autorisées dans la base "
            "de connaissances pour répondre à cette question.'\n"
            "3. Always cite your source file names when providing facts.\n\n"
            "4. Answer in the same language as the user's question. If the question is in French, "
            "answer entirely in French. If it is in English, answer entirely in English.\n"
            "5. When the user asks for the subject, topic, or summary of a conversation, "
            "describe the concrete business topics discussed in the context. "
            "Do not confuse the language of the transcript with its subject. "
            "For example, a conversation spoken in French is not automatically about "
            "the French language or French culture.\n"
            "6. Do not invent a topic from the language, filename, or speaker labels. "
            "If the transcript is too unclear to identify a topic, say so explicitly.\n\n"
            "7. When the user asks for the business-level problem, objective, or issue, "
            "abstract incidental literal details into the narrowest business problem "
            "that is directly supported by the context. For example, a product described "
            "with insects or another vivid detail should be summarized as a product "
            "quality or contamination incident, not presented as the business topic by "
            "itself. Mention the literal detail only as a secondary example when it "
            "helps support the summary. Do not invent causes, actors, or legal conclusions.\n"
            "8. When the user asks for a concrete example or exact wording, preserve the "
            "literal detail from the context instead of abstracting it.\n\n"
            "9. If the question asks for a specific fact that is absent from the "
            "retrieved context, answer directly that the information is unavailable. "
            "Do not replace the answer with a general summary of the document, and do "
            "not cite unrelated facts. Use the insufficient-information sentence from "
            "rule 2 when no reliable answer can be given.\n\n"
            "10. Treat Markdown tables as structured data: keep every row's values "
            "associated with its column headers. For requests to list, count, filter, "
            "compare, rank, or calculate across a table, inspect every row of each "
            "retrieved table marked complete. Include every matching row when the user "
            "asks for all items. If a retrieved table is marked incomplete, do not "
            "claim an exhaustive list or a global maximum/minimum; say that the "
            "available table excerpt is incomplete.\n\n"
            "11. For table questions, use the column headers to identify each value. "
            "Never combine prices, quantities, margins, or stocks from different "
            "columns or sections. Preserve numeric values and decimal separators "
            "exactly as written in the selected table; never turn 1,00 € into 100, "
            "or infer values for rows that are not present.\n\n"
            "{answer_language_instruction}\n\n"
            "Retrieved Context:\n{context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])
        
        # Connect directly to your local Ollama instance running on the RX6600M
        print(f"Connecting to local Ollama model: {model_name}...")
        self.llm = ChatOllama(
            model=model_name, 
            temperature=0.0, # 0.0 forces the model to be analytical, not creative
        )
            
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, retrieved_docs: List[Document]) -> str:
        if not retrieved_docs:
            return "No authorized documents found matching your security credentials."

        answer_language_instruction = self._answer_language_instruction(question)
        context_parts = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            metadata_lines = [
                f"Source: {metadata.get('source', 'Unknown')}",
                f"Type: {metadata.get('file_type', 'document')}",
            ]
            if metadata.get("speakers"):
                metadata_lines.append(
                    f"Speakers: {', '.join(metadata['speakers'])}"
                )
            if metadata.get("content_kind") == "table":
                metadata_lines.extend(
                    [
                        f"Table title: {metadata.get('table_title') or 'Untitled'}",
                        f"Table rows: {metadata.get('table_row_count', 'unknown')}",
                        "Table completeness: "
                        + ("complete" if metadata.get("table_complete") else "incomplete"),
                    ]
                )
            context_parts.append(
                "--- Retrieved document ---\n"
                + "\n".join(metadata_lines)
                + f"\nContent:\n{doc.page_content}"
            )
        formatted_context = "\n\n".join(context_parts)

        return self.chain.invoke({
            "context": formatted_context,
            "question": question,
            "answer_language_instruction": answer_language_instruction,
        })

    @staticmethod
    def _answer_language_instruction(question: str) -> str:
        """Give the LLM an explicit output-language constraint."""
        french_markers = {
            "alors", "avec", "c'est", "conversation", "dans", "est-ce",
            "français", "quelle", "quel", "que", "qui", "sujet", "sur",
        }
        words = {
            word.strip("¿?!.,;:()[]{}\"'").lower()
            for word in question.split()
        }
        if any(char in question.lower() for char in "àâçéèêëîïôùûüÿœ"):
            language = "French"
        elif len(words & french_markers) >= 1:
            language = "French"
        else:
            language = "English"

        return (
            f"OUTPUT LANGUAGE: {language}. Every sentence of the answer, including "
            "the explanation, headings, and source descriptions, MUST be written "
            f"in {language}. Do not translate the answer into another language."
        )
