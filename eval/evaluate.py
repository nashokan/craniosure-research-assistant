"""Optional: evaluate retrieval + generation with RAGAS (needs an LLM configured).
Computes faithfulness and answer relevancy (no ground-truth labels required).
Run from repo root:  python eval/evaluate.py"""
import json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from rag import RAG


def main():
    questions = json.load(open(os.path.join(os.path.dirname(__file__), "golden_set.json")))
    rag = RAG()
    rows = []
    for q in questions:
        out = rag.answer(q)
        rows.append({"question": q, "answer": out["answer"],
                     "contexts": [c["text"] for c in out["sources"]]})

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        ds = Dataset.from_list(rows)
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        print("\n=== RAGAS scores ===")
        print(result)
    except Exception as e:
        print(f"\n(RAGAS not run: {e}.)\nShowing raw outputs so you can sanity-check grounding:")
        for r in rows:
            print("\nQ:", r["question"])
            print("A:", r["answer"][:300])


if __name__ == "__main__":
    main()
