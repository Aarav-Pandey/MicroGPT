from langchain_ollama import ChatOllama
from math_tool import math_tool
from rag_master import get_rag_engine
import datetime

MODEL_NAME = "mgpt"

INTRO_QUERIES = {
    "who are you",
    "who are you?",
    "introduce yourself",
    "what are you"
}


class Memory:
    def __init__(self):
        self.history = []

    def save(self, user, assistant):
        self.history.append((user, assistant))

    def clear_memory(self):
        self.history = []


class MicroGPTAgent:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME)
        self.memory = Memory()

    def _is_math(self, text):
        text = text.lower()
        return any(w in text for w in ["+", "-", "*", "/", "%", "plus", "minus", "multiply", "divide"])

    def _is_time(self, text):
        text = text.lower()
        return any(w in text for w in ["time", "current time", "time now"])

    def _is_doc(self, text):
        text = text.lower()
        return any(w in text for w in [
            "document", "file", "pdf", "docx",
            "csv", "xlsx", "content", "summarize", "list"
        ])

    def run_stream(self, user_input):
        user_input_lower = user_input.lower()

        # INTRO
        if any(w in user_input_lower for w in INTRO_QUERIES):
            yield {"type": "token", "content": "I am MicroGPT, an optimized system designed to address these challenges by enabling efficient local execution of language models. The system focuses on improving performance through resource optimization and intelligent processing, allowing reliable operation on limited hardware while maintaining effective response generation."}
            return

        # TIME
        if self._is_time(user_input):
            now = datetime.datetime.now()
            yield {"type": "token", "content": now.strftime("%I:%M:%S %p")}
            return

        # MATH
        if self._is_math(user_input):
            result = math_tool.solve(user_input)
            yield {"type": "token", "content": str(result)}
            return

        # RAG
        if self._is_doc(user_input):
            rag_engine = get_rag_engine()

            # Ensure index exists
            if rag_engine.retriever is None:
                if not rag_engine.reload():
                    yield {"type": "token", "content": "No documents available. Please upload files first."}
                    return

            docs = rag_engine.retriever.invoke(user_input)

            if not docs:
                yield {"type": "token", "content": "No relevant content found."}
                return

            # Safe extraction
            extracted = []
            for doc in docs:
                content = getattr(doc, "page_content", "")
                if content and content.strip():
                    extracted.append(content.strip())

            if not extracted:
                yield {"type": "token", "content": "No readable content found in documents."}
                return
            
            plain_text = "\n\n".join(extracted)

            # Internal system prompt for MicroGPT (hidden from UI)
            response = (
            "SYSTEM INSTRUCTION FOR MICROGPT LLM\n\n"
            
            "You are MicroGPT, a local AI agent built using a Retrieval-Augmented Generation (RAG) architecture with FAISS vector database and Ollama-based LLM.\n\n"
            
            "Your capabilities:\n"
            "- Answer normal questions based on your training data and memory\n"
            "- Process and understand complex queries related to document content\n"
            "- Summarize and extract key information from large text\n"
            "- Identify and extract relevant information from large documents\n"
            "- Analyze and synthesize information from multiple document sections\n"
            "- Retrieve relevant context using semantic search\n"
            "- Reconstruct structured information from fragmented text\n"
            "- Provide accurate, context-based answers without hallucination\n\n"
            
            "Modes of Operation:\n"
            "1. Normal Chat:\n"
            "   - Respond naturally to user queries\n"
            "   - Stay accurate and contextually aware\n\n"
            "2. Mathematics:\n"
            "   - Perform calculations accurately\n"
            "   - Show steps where appropriate\n\n"
            "3. Self Introduction:\n"
            "   - Introduce yourself as 'MicroGPT'\n"
            "   - Describe your capabilities clearly\n"
            "   - Keep it concise and relevant to the project\n\n"
            "4. RAG based Document QA:\n"
            "   - Analyze provided document context\n"
            "   - Identify headings, key topics, and sections\n"
            "   - Use semantic search to retrieve relevant passages\n"
            "   - Provide concise, structured, and technically accurate answers\n"
            "   - Preserve the meaning of the original text\n\n"
            
            "Strict Guidelines:\n"
            "- Answer only using the information provided in the document for RAG queries\n"
            "- Do NOT hallucinate information\n"
            "- Use headings and bullet points for clarity in structured answers\n"
            "- If an answer is not present in the document, respond: 'I could not find the answer in the document'\n\n"
            
            "User Input:\n"
            f"{user_input}\n\n"
            
            "Document Context (for RAG queries only):\n"
            f"{plain_text}\n\n"
            
            "Instructions for LLM:\n"
            "- Determine the mode based on userInput:\n"
            "   * If it's a math question → use Mathematics mode\n"
            "   * If user asks about your identity → use Self Introduction mode\n"
            "   * If a document query → use RAG mode with context\n"
            "   * Otherwise → Normal Chat mode\n"
            "- Produce output according to the selected mode\n"
            )

            prompt_for_llm = response

        else:
            prompt_for_llm = user_input

        # NORMAL CHAT & RAG STREAMING
        full = ""
        for chunk in self.llm.stream(prompt_for_llm):
            token = getattr(chunk, "content", str(chunk))
            full += token
            yield {"type": "token", "content": token}

        self.memory.save(user_input, full)