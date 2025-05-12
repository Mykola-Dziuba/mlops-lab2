# rag_service.py
# --- imports -------------------------------------------------------------
from typing import List
import os
import google.generativeai as genai

from rag_search import search_pages  # local semantic search function

# --- Gemini initialisation ----------------------------------------------
MODEL_NAME = "gemini-2.0-flash"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
g_model = genai.GenerativeModel(MODEL_NAME)

# --- helpers -------------------------------------------------------------
def build_prompt(contexts: List[str], query: str) -> str:
    """
    Compose final prompt: numbered context chunks + user question.
    """
    joined_ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "Odpowiedz na pytanie korzystając WYŁĄCZNIE z poniższego kontekstu.\n"
        "Jeśli brak potrzebnych informacji, przyznaj, że nie wiesz.\n\n"
        f"{joined_ctx}\n\n"
        f"Pytanie: {query}\n"
        "Odpowiedź:"
    )


def rag_answer(query: str, *, top_k: int = 3) -> str:
    """
    1. Retrieve *top_k* relevant pages from Milvus.
    2. Build a prompt with those pages and the user query.
    3. Ask Gemini and return the model's answer.
    """
    ctx_pages = search_pages(query, top_k=top_k)
    prompt = build_prompt(ctx_pages, query)
    response = g_model.generate_content(prompt)
    return response.text.strip()


# --- CLI test -------------------------------------------------------------
if __name__ == "__main__":
    question = "Jakie są główne rodzaje sztucznej inteligencji?"
    print(rag_answer(question))

