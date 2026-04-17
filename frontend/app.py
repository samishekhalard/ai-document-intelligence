"""Streamlit dashboard for the Enterprise AI Document Intelligence Platform."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon=":open_file_folder:",
    layout="wide",
)


# -------- helpers -------- #


def _get(path: str, timeout: float = 10.0):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.sidebar.error(f"GET {path} failed: {exc}")
        return None


def _post_json(path: str, payload: dict, timeout: float = 120.0):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"POST {path} failed: {exc}")
        return None


def _post_file(path: str, files, data=None, timeout: float = 300.0):
    try:
        r = requests.post(f"{API_BASE}{path}", files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"POST {path} failed: {exc}")
        return None


# -------- sidebar: status + upload -------- #


with st.sidebar:
    st.title("AI Document Intelligence")
    st.caption("RAG + NLP + OCR agent platform")

    with st.spinner("Checking services..."):
        health = _get("/health", timeout=5.0)
    if health:
        status = health["status"]
        colour = {"ok": "green", "degraded": "orange", "down": "red"}.get(status, "gray")
        st.markdown(f"**Status:** :{colour}[{status.upper()}]")
        st.markdown(f"**LLM:** `{health['llm_model']}`")
        st.markdown(f"**Embeddings:** `{health['embed_model']}`")
        st.markdown(f"**Documents indexed:** {health['documents_indexed']}")
        st.markdown(f"**LLM ready:** {'yes' if health['llm_ready'] else 'no'}")
    else:
        st.error("Backend unreachable")

    st.divider()
    st.subheader("Upload document")
    uploaded = st.file_uploader(
        "PDF / DOCX / TXT / Image",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tif", "tiff"],
    )
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting..."):
            res = _post_file(
                "/upload",
                files={"file": (uploaded.name, uploaded.getvalue())},
            )
        if res:
            st.success(f"Indexed {res['n_chunks']} chunks — {res['document_id']}")
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    st.divider()
    st.subheader("Indexed documents")
    docs = _get("/documents") or []
    if docs:
        for d in docs:
            st.markdown(
                f"- **{d['title']}** "
                f"(`{d['file_type']}`, {d['doc_type']}, "
                f"{d.get('language') or '?'}, {d['n_chunks']} chunks)"
            )
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    stats = _get("/stats") or {}
    if stats:
        st.subheader("Metrics")
        st.metric("Chunks", stats.get("chroma_chunks", 0))
        st.metric("Queries", stats.get("n_queries", 0))
        st.metric("Avg latency (ms)", f"{stats.get('avg_latency_ms', 0):.0f}")


# -------- main layout -------- #

tab_chat, tab_nlp, tab_ocr = st.tabs(
    [":speech_balloon: RAG Chat", ":mag: NLP Analysis", ":camera: OCR"]
)


# -------- chat tab -------- #


with tab_chat:
    st.subheader("Ask your documents")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for i, s in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**[{i}] {s['metadata'].get('title', 'Untitled')}** "
                            f"(score {s['score']:.3f}, "
                            f"dense {s['dense_score']:.2f}, "
                            f"sparse {s['sparse_score']:.2f})"
                        )
                        st.code(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

    prompt = st.chat_input("Ask a question about your indexed documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                res = _post_json("/query", {"query": prompt})
                elapsed = time.time() - start
            if res:
                st.markdown(res["answer"])
                st.caption(
                    f"confidence {res['confidence']:.2f} • "
                    f"latency {res['latency_ms']:.0f} ms • "
                    f"trace: {' -> '.join(res.get('agent_trace', []))}"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": res["answer"], "sources": res.get("sources", [])}
                )
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Query failed, see error above."}
                )


# -------- nlp tab -------- #


with tab_nlp:
    st.subheader("Run NLP analysis")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        nlp_text = st.text_area("Paste text", height=260, key="nlp_text")
    with col_b:
        doc_options = {d["document_id"]: d["title"] for d in docs} if docs else {}
        doc_choice = st.selectbox(
            "...or pick an indexed document",
            options=["(none)"] + list(doc_options.keys()),
            format_func=lambda k: doc_options.get(k, "(none)"),
        )
        tasks = st.multiselect(
            "Tasks",
            ["entities", "classify", "summarize", "language", "keyphrases"],
            default=["entities", "classify", "summarize", "language", "keyphrases"],
        )
        run_nlp = st.button("Analyse", use_container_width=True)

    if run_nlp:
        payload = {"tasks": tasks}
        if doc_choice != "(none)":
            payload["document_id"] = doc_choice
        elif nlp_text.strip():
            payload["text"] = nlp_text
        else:
            st.warning("Provide text or choose a document.")
            st.stop()
        with st.spinner("Analysing..."):
            res = _post_json("/analyze", payload, timeout=180.0)
        if res:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Classification:** `{res['classification']}`")
                st.markdown(f"**Language:** `{res['language']}`")
                st.markdown("**Key phrases:**")
                st.write(res.get("keyphrases") or [])
            with c2:
                st.markdown("**Entities:**")
                if res.get("entities"):
                    st.dataframe(res["entities"], use_container_width=True, hide_index=True)
                else:
                    st.caption("(none found)")
            st.markdown("**Summary:**")
            st.info(res.get("summary") or "(empty)")


# -------- ocr tab -------- #


with tab_ocr:
    st.subheader("OCR")
    img = st.file_uploader(
        "Upload an image or scanned page",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="ocr_upload",
    )
    lang = st.text_input("Languages (tesseract codes)", value="eng+rus")
    if img and st.button("Extract text", use_container_width=True):
        with st.spinner("Running OCR..."):
            res = _post_file(
                "/extract-ocr",
                files={"file": (img.name, img.getvalue())},
                data={"language": lang},
            )
        if res:
            st.success(f"Extracted {res['n_words']} words in {res['latency_ms']:.0f} ms")
            st.code(res["text"] or "(empty)")
