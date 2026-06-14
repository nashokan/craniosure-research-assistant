# CranioSure Research Assistant

A retrieval-augmented (RAG) chatbot over the **craniosynostosis / craniofacial research
literature**, pulled from PubMed. Ask a question in plain English and get a grounded,
**cited** answer drawn only from real published abstracts — no digging through search results.

> **Responsible use.** This is a literature-navigation tool for researchers and students. It
> summarizes published research and is **not medical advice, not a diagnostic tool, and not a
> substitute for a clinician.** That constraint is enforced in the model's system prompt and shown in the UI.

Built in the CranioSure domain on purpose: it pairs naturally with the cranial-screening
computer-vision work and gives you a defensible, end-to-end RAG project.

## Architecture

```
PubMed (E-utilities API)
        │  ingest/fetch_pubmed.py   -> data/papers.jsonl
        ▼
Chunk + embed (sentence-transformers)
        │  ingest/build_index.py    -> ChromaDB (data/chroma)
        ▼
Backend  (FastAPI, backend/)
  hybrid retrieval: dense (ChromaDB) + sparse (BM25)
        -> cross-encoder rerank (top 50 -> top 5)
        -> LLM answers ONLY from context, cites [n]
        ▼
Frontend (Streamlit, frontend/app.py)  -> chat UI + clickable citations
Eval     (eval/evaluate.py)            -> RAGAS faithfulness / answer relevancy
```

**Stack:** Python, FastAPI, ChromaDB, sentence-transformers (dense), rank-bm25 (sparse),
cross-encoder reranker, OpenAI-compatible LLM (local Ollama by default), Streamlit, RAGAS.

## Setup (run in order)

```bash
# 0. From the repo root
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # defaults to free local Ollama

# (LLM) easiest free option — install Ollama from ollama.com, then:
ollama pull mistral
#   ...or edit .env to use a hosted key (OpenAI / Mistral). Embeddings + reranker are always local.

# 1. Build the knowledge base (~a few hundred papers, takes a couple minutes)
python ingest/fetch_pubmed.py
python ingest/build_index.py

# 2. Start the backend (from the backend/ folder so imports resolve)
cd backend && uvicorn main:app --reload --port 8000
#   leave this running; open a SECOND terminal for step 3

# 3. Start the frontend (from repo root, venv active)
streamlit run frontend/app.py
```

Open the Streamlit URL it prints (usually http://localhost:8501) and ask away.

## How it works (the 60-second version)

1. **Ingest** — pull abstracts for craniosynostosis / plagiocephaly / cranial-deformity / cranial
   CV topics via PubMed's API, store title + abstract + PMID + link.
2. **Index** — split into chunks, embed locally with MiniLM, store vectors in ChromaDB.
3. **Retrieve (hybrid)** — for each question, get candidates by *meaning* (vector search) **and**
   by *exact terms* (BM25), then a **cross-encoder reranks** them so only the best 5 reach the model.
4. **Answer (grounded)** — the LLM is instructed to use only the retrieved context and cite `[n]`;
   if the answer isn't there, it says so. Citations link back to PubMed.
5. **Evaluate** — `eval/evaluate.py` scores faithfulness (is the answer supported by the sources?)
   and answer relevancy with RAGAS, so you can quote a real accuracy number.

## Talking points (for an application / interview)

- Why **hybrid + reranking** beats naive vector search (exact clinical terms vs. paraphrase).
- **Grounding + citations** as a hallucination defense, with a measured **faithfulness** score.
- A clear, responsible scope: literature navigation, not diagnosis.

## Stretch goals (if you have time)

- Add PMC open-access **full-text** (not just abstracts) for deeper answers.
- Add an **agentic self-check** node that verifies each citation actually supports its claim.
- Show **retrieval metrics** (context precision) on a small labeled set in the README.
- Conversation memory (multi-turn follow-ups).
