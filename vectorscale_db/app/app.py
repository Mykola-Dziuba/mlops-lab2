# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_service import rag_answer

app = FastAPI(title="AI RAG demo")

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
async def ask_rag(q: Query):
    """Return RAG answer for given question."""
    answer = rag_answer(q.question, top_k=q.top_k)
    return {"answer": answer}

