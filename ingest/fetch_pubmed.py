"""Fetch craniofacial / craniosynostosis abstracts from PubMed (NCBI E-utilities).
Saves to data/papers.jsonl. No API key required (optional key only raises rate limits)."""
import os, json, time, requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()
NCBI_KEY = os.getenv("NCBI_API_KEY", "").strip()
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Tuned to the CranioSure domain: cranial shape, deformity, and screening methods.
QUERIES = [
    "craniosynostosis",
    "positional plagiocephaly infant",
    "cranial deformity cephalic index",
    "craniosynostosis deep learning",
    "3D photogrammetry craniofacial",
]
PER_QUERY = 120
OUT = "data/papers.jsonl"


def esearch(term, retmax):
    params = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"}
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"].get("idlist", [])


def efetch(pmids):
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY
    r = requests.get(f"{BASE}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.text


def parse(xml_text):
    root = ET.fromstring(xml_text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        abst_nodes = art.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(n.itertext()) for n in abst_nodes).strip()
        journal = art.findtext(".//Journal/Title") or ""
        year = (art.findtext(".//JournalIssue/PubDate/Year")
                or art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")
        if title and abstract:
            out.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "journal": journal, "year": year[:4],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    return out


def main():
    os.makedirs("data", exist_ok=True)
    seen, papers = set(), []
    for q in QUERIES:
        print(f"searching: {q}")
        ids = [i for i in esearch(q, PER_QUERY) if i not in seen]
        seen.update(ids)
        for i in range(0, len(ids), 100):
            papers.extend(parse(efetch(ids[i:i + 100])))
            time.sleep(0.4 if NCBI_KEY else 0.5)
    uniq = {p["pmid"]: p for p in papers}
    with open(OUT, "w") as f:
        for p in uniq.values():
            f.write(json.dumps(p) + "\n")
    print(f"saved {len(uniq)} unique papers -> {OUT}")


if __name__ == "__main__":
    main()
