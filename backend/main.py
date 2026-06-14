from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAG
import config as cfg

app = FastAPI(title="CranioSure Research Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
rag = None


class Query(BaseModel):
    message: str


@app.on_event("startup")
def load():
    global rag
    rag = RAG()


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": cfg.DISCLAIMER}


@app.post("/chat")
def chat(q: Query):
    return rag.answer(q.message)
