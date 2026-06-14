"""Hybrid retrieval (dense + BM25) -> cross-encoder rerank -> grounded, cited answer."""
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
import config as cfg


class RAG:
    def __init__(self):
        self.embedder = SentenceTransformer(cfg.EMBED_MODEL)
        self.reranker = CrossEncoder(cfg.RERANKER_MODEL)
        self.client = chromadb.PersistentClient(path=cfg.CHROMA_PATH)
        self.col = self.client.get_collection(cfg.COLLECTION)
        data = self.col.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.bm25 = BM25Okapi([d.lower().split() for d in self.docs])
        self.id2idx = {i: k for k, i in enumerate(self.ids)}
        self.llm = OpenAI(base_url=cfg.LLM_BASE_URL, api_key=cfg.LLM_API_KEY)

    def _dense(self, query, k=25):
        emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        return self.col.query(query_embeddings=emb, n_results=k)["ids"][0]

    def _sparse(self, query, k=25):
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.ids[i] for i in top]

    def retrieve(self, query, top_n=5):
        # hybrid: union of semantic + keyword candidates, then cross-encoder rerank
        cand_ids = list(dict.fromkeys(self._dense(query) + self._sparse(query)))
        idxs = [self.id2idx[i] for i in cand_ids]
        scores = self.reranker.predict([[query, self.docs[i]] for i in idxs])
        ranked = sorted(zip(idxs, scores), key=lambda x: float(x[1]), reverse=True)[:top_n]
        return [{"text": self.docs[i], **self.metas[i]} for i, _ in ranked]

    def answer(self, query):
        ctx = self.retrieve(query)
        context_block = "\n\n".join(
            f"[{n + 1}] {c['title']} ({c['year']}). {c['text']}" for n, c in enumerate(ctx))
        system = (
            "You are a research assistant for the craniofacial / craniosynostosis literature. "
            "Answer ONLY using the numbered context below, and cite sources inline as [n]. "
            "If the answer is not in the context, say you don't have enough information. "
            "Do NOT give medical advice or diagnoses; this is a literature-navigation tool.")
        user = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer with citations:"
        resp = self.llm.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1)
        return {"answer": resp.choices[0].message.content, "sources": ctx}
