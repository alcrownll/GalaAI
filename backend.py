"""
backend.py — RAG pipeline and Groq streaming for Gala.AI.

Public API
----------
load_rag_system()   → (kb, embedder, faiss_index)   (cached by Streamlit)
retrieve(query, …)  → list[dict]
stream_groq_response(query, docs, placeholder) → str
"""

import time
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, KB_PATH, TOP_K, SYSTEM_PROMPT


# ─────────────────────────────────────────────────────
# KNOWLEDGE BASE PARSER
# ─────────────────────────────────────────────────────

def parse_knowledge_base(filepath: str = KB_PATH) -> list[dict]:
    """
    Parse a flat-file knowledge base delimited by '=' * 50 separators.

    Expected format per entry:
        CATEGORY: <name>
        TITLE: <name>
        ===…===
        <content text>
        ===…===

    Returns a list of {"category", "title", "content"} dicts.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        raw = fh.read()

    entries: list[dict] = []
    blocks = raw.split("=" * 50)

    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        if block.startswith("CATEGORY:"):
            category, title = "", ""
            for line in block.split("\n"):
                if line.startswith("CATEGORY:"):
                    category = line.replace("CATEGORY:", "").strip()
                elif line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()

            if i + 1 < len(blocks):
                content = blocks[i + 1].strip()
                if category and title and content:
                    entries.append({"category": category, "title": title, "content": content})
                i += 2
            else:
                i += 1
        else:
            i += 1

    return entries


# ─────────────────────────────────────────────────────
# RAG SYSTEM  (Streamlit-cached for performance)
# ─────────────────────────────────────────────────────

@st.cache_resource
def load_rag_system():
    """
    Build and cache:
      - parsed knowledge base
      - SentenceTransformer embedder
      - FAISS flat-L2 index

    Returns (kb, embedder, index).
    """
    kb = parse_knowledge_base()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    docs = [
        f"[{item['category']}] {item['title']}: {item['content']}"
        for item in kb
    ]
    embeddings = embedder.encode(docs, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return kb, embedder, index


def retrieve(
    query: str,
    kb: list[dict],
    embedder: SentenceTransformer,
    index,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed `query`, search FAISS, return the top-k knowledge-base entries.
    Each result is {"title", "category", "content"}.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    return [
        {
            "title":    kb[i]["title"],
            "category": kb[i]["category"],
            "content":  kb[i]["content"],
        }
        for i in indices[0]
    ]


# ─────────────────────────────────────────────────────
# GROQ STREAMING
# ─────────────────────────────────────────────────────

def stream_groq_response(
    query: str,
    retrieved_docs: list[dict],
    placeholder,
    typing_delay: float = 0.4,
) -> str:
    """
    Stream a Groq/Llama response into a Streamlit `st.empty()` placeholder.

    Shows a brief typing-dot animation before the first token arrives,
    then streams tokens into the placeholder as they come.

    Returns the full response string.
    """
    client = Groq(api_key=GROQ_API_KEY)

    context = "\n\n".join(
        f"[{doc['category']}] {doc['title']}:\n{doc['content']}"
        for doc in retrieved_docs
    )

    # Brief typing indicator
    placeholder.markdown(
        '<span class="typing-dot"></span>'
        '<span class="typing-dot"></span>'
        '<span class="typing-dot"></span>',
        unsafe_allow_html=True,
    )
    time.sleep(typing_delay)

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.7,
        max_tokens=650,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            placeholder.markdown(
                full_response.replace("\n", "<br>"),
                unsafe_allow_html=True,
            )

    return full_response