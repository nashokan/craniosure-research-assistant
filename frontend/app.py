import requests
import streamlit as st

BACKEND = "http://localhost:8000"
st.set_page_config(page_title="CranioSure Research Assistant", page_icon="brain")

st.title("CranioSure Research Assistant")
st.caption("Ask about the craniosynostosis & craniofacial research literature. "
           "Answers are grounded in PubMed abstracts, with citations.")

with st.sidebar:
    st.header("About")
    st.write("A retrieval-augmented assistant over published craniofacial research. "
             "Hybrid retrieval (semantic + keyword) with cross-encoder reranking, and "
             "grounded answers that cite back to PubMed.")
    st.warning("Research / education tool only. Not medical advice or a diagnostic tool.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("e.g. What imaging methods measure cranial asymmetry?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the literature..."):
            try:
                r = requests.post(f"{BACKEND}/chat", json={"message": prompt}, timeout=180)
                data = r.json()
                st.markdown(data["answer"])
                with st.expander("Sources"):
                    for n, s in enumerate(data["sources"], 1):
                        st.markdown(
                            f"**[{n}]** [{s['title']}]({s['url']}) - *{s['journal']}* ({s['year']})")
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"]})
            except Exception as e:
                st.error(f"Backend error: {e}. Is the FastAPI server running on :8000?")
