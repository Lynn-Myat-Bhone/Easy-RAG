from fastapi import FastAPI
from pydantic import BaseModel

from RAG.pipeline import RAG

app = FastAPI(title="Easy RAG API", version="1.0")

rag = RAG()
rag.load_db()

# request schema
class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "RAG API is running "}


@app.post("/ask")
def ask(query: Query):
    answer = rag.ask(query.question)
    return {
        "question": query.question,
        "answer": answer
    }