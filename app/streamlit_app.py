"""FactLens Streamlit application.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import streamlit as st

from factlens.config import load_config
from factlens.data.wiki import title_to_text
from factlens.explain.templates import explain
from factlens.labels import DISPLAY, LABELS
from factlens.retrieval import Retriever, WikiStore

st.set_page_config(page_title="FactLens", page_icon="🔎", layout="wide")
CFG = load_config()
COLORS = {"SUPPORTS": "green", "REFUTES": "red", "NOT ENOUGH INFO": "orange"}
EXAMPLES = [
    "The Eiffel Tower was completed in 1887.",
    "Barack Obama was born in Hawaii.",
    "The Great Wall of China is visible from the Moon with the naked eye.",
]


@st.cache_resource(show_spinner="Loading Wikipedia index ...")
def get_store() -> WikiStore:
    """Opens the SQLite store once per app session."""
    return WikiStore(CFG["paths"]["wiki_db"])


@st.cache_resource(show_spinner="Loading retriever ...")
def get_retriever(sent_ranker: str, doc_top_k: int) -> Retriever:
    """Builds one retriever per ranking mode and page count."""
    return Retriever.from_config(CFG, store=get_store(), sent_ranker=sent_ranker, doc_top_k=doc_top_k)


@st.cache_resource(show_spinner="Loading verification model ...")
def get_verifier(model_path: str):
    """Loads the fine-tuned verifier once per model path."""
    from factlens.models.verifier import Verifier

    return Verifier(model_path, mode="evidence", max_length=CFG["model"]["max_length"])


def wiki_url(page_id: str) -> str:
    """Builds a Wikipedia link from a FEVER page id."""
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title_to_text(page_id).replace(" ", "_"))


with st.sidebar:
    st.header("Settings")
    sent_ranker = st.selectbox("Sentence ranking", ["hybrid", "bm25", "dense"], index=0,
                               help="hybrid gave the best sentence recall; bm25 is fastest on CPU.")
    doc_top_k = st.slider("Pages retrieved", 1, 10, CFG["retrieval"]["doc_top_k"])
    model_path = st.text_input("Model (folder or Hub ID)", CFG["app"]["model_path"])
    st.caption("Knowledge base: FEVER Wikipedia dump (June 2017). Newer facts are out of scope.")

st.title("🔎 FactLens")
st.write("Evidence-grounded claim verification. Enter a factual claim; FactLens retrieves Wikipedia "
         "evidence and a fine-tuned DeBERTa model decides whether the evidence supports or refutes it.")

if not Path(CFG["paths"]["wiki_db"]).exists():
    st.error(f"Wikipedia index not found at {CFG['paths']['wiki_db']}. Build it with "
             "scripts/02_build_wiki_db.py or copy the shared database there.")
    st.stop()

example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
claim = st.text_area("Claim", "" if example.startswith("(") else example, height=80)

if st.button("Verify", type="primary", disabled=not claim.strip()):
    retriever = get_retriever(sent_ranker, doc_top_k)
    with st.spinner("Retrieving evidence ..."):
        result = retriever.retrieve(claim)
    try:
        verifier = get_verifier(model_path)
    except Exception as exc:  # Surfaces loading problems, such as a missing Hub login, to the user.
        st.error(f"Could not load model {model_path}: {exc}")
        st.stop()
    rows = [[e.page_id, e.line, e.text] for e in result.sentences]
    with st.spinner("Verifying ..."):
        label, probs = verifier.predict(claim, rows)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### :{COLORS[label]}[{DISPLAY[label]}]")
        for lab in LABELS:
            st.progress(probs[lab], text=f"{DISPLAY[lab]}: {probs[lab]:.1%}")
    with col2:
        st.info(explain(label, probs[label], [{"page_id": e.page_id, "text": e.text} for e in result.sentences]))

    st.subheader("Evidence")
    if not result.sentences:
        st.write("No evidence sentences were retrieved.")
    for i, e in enumerate(result.sentences, 1):
        st.markdown(f"**{i}. [{title_to_text(e.page_id)}]({wiki_url(e.page_id)})** "
                    f"(sentence {e.line}, score {e.score:.3f})  \n{e.text}")
    with st.expander("Retrieved pages"):
        st.write([title_to_text(p) for p in result.pages])
