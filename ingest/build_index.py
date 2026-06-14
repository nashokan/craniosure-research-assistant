"""Chunk abstracts, embed locally (no API key), and store in ChromaDB."""
import os, json
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION = os.getenv("COLLECTION", "cranio_lit")
PAPERS = "data/papers.jsonl"


def chunk(text, size=900, overlap=150):
    words = text.split()
    if len(words) <= size:
        return [text]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


def main():
    papers = [json.loads(l) for l in open(PAPERS)]
    print(f"loaded {len(papers)} papers")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for p in papers:
        full = f"{p['title']}. {p['abstract']}"
        for j, ch in enumerate(chunk(full)):
            ids.append(f"{p['pmid']}-{j}")
            docs.append(ch)
            metas.append({"pmid": p["pmid"], "title": p["title"],
                          "journal": p["journal"], "year": p["year"], "url": p["url"]})
    print(f"embedding {len(docs)} chunks...")
    embs = model.encode(docs, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True).tolist()
    for i in range(0, len(docs), 500):
        col.add(ids=ids[i:i + 500], documents=docs[i:i + 500],
                metadatas=metas[i:i + 500], embeddings=embs[i:i + 500])
    print(f"indexed {col.count()} chunks into '{COLLECTION}'")


if __name__ == "__main__":
    main()
