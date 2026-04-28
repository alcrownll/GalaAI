import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv
import time

# ─────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Gala.AI",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# MODERN CHATBOT CSS (CHATGPT / CLAUDE STYLE)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0b0f17;
    --surface: #111827;
    --surface2: #0f172a;
    --border: rgba(255,255,255,0.08);
    --accent: #22c55e;
    --accent-dim: rgba(34,197,94,0.15);
    --ink: #f8fafc;
    --muted: rgba(255,255,255,0.65);
    --muted2: rgba(255,255,255,0.45);
    --radius: 16px;
    --radius-lg: 22px;
    --pill: 999px;
    --shadow: 0 12px 40px rgba(0,0,0,0.45);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: radial-gradient(circle at top, #0f172a, #0b0f17) !important;
    color: var(--ink) !important;
}

/* Hide Streamlit junk */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Main container */
.block-container {
    padding-top: 0.6rem !important;
    max-width: 980px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17,24,39,0.95) !important;
    border-right: 1px solid var(--border) !important;
}

/* Sidebar buttons */
.stButton > button {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.03) !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(34,197,94,0.12) !important;
    border-color: rgba(34,197,94,0.35) !important;
    transform: translateY(-1px);
}

/* Chat message layout */
.chat-wrap {
    padding-bottom: 120px;
}

/* User message */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0;
}
.user-bubble {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    padding: 0.85rem 1.15rem;
    border-radius: var(--radius-lg) var(--radius-lg) 6px var(--radius-lg);
    max-width: 75%;
    line-height: 1.7;
    font-size: 0.95rem;
    box-shadow: var(--shadow);
}

/* Assistant message */
.bot-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin: 1rem 0;
}
.bot-avi {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.bot-bubble {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    padding: 0.95rem 1.15rem;
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 6px;
    max-width: 80%;
    line-height: 1.75;
    font-size: 0.95rem;
}

/* Sources */
.sources {
    margin-top: 0.9rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
.source-pill {
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: var(--pill);
    background: var(--accent-dim);
    border: 1px solid rgba(34,197,94,0.25);
    color: var(--accent);
    font-weight: 600;
}

/* Action buttons */
.action-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.65rem;
}
.action-btn {
    font-size: 0.75rem;
    padding: 0.35rem 0.75rem;
    border-radius: var(--pill);
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.02);
    color: var(--muted);
    cursor: pointer;
}
.action-btn:hover {
    background: rgba(34,197,94,0.12);
    border-color: rgba(34,197,94,0.35);
    color: var(--accent);
}

/* Chat input */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted2) !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 14px !important;
}

/* Mobile */
@media (max-width: 700px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .user-bubble, .bot-bubble { max-width: 92%; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# KNOWLEDGE BASE PARSER
# ─────────────────────────────────────────
def parse_knowledge_base(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = []
    blocks = content.split("=" * 50)

    i = 0
    while i < len(blocks):
        block = blocks[i].strip()

        if block.startswith("CATEGORY:"):
            lines = block.split("\n")
            category, title = "", ""

            for line in lines:
                if line.startswith("CATEGORY:"):
                    category = line.replace("CATEGORY:", "").strip()
                elif line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()

            if i + 1 < len(blocks):
                content_block = blocks[i + 1].strip()
                if category and title and content_block:
                    entries.append({
                        "category": category,
                        "title": title,
                        "content": content_block
                    })
                i += 2
            else:
                i += 1
        else:
            i += 1

    return entries


# ─────────────────────────────────────────
# LOAD RAG SYSTEM
# ─────────────────────────────────────────
@st.cache_resource
def load_rag_system():
    kb = parse_knowledge_base("cebu_tourism.txt")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    docs = [f"[{item['category']}] {item['title']}: {item['content']}" for item in kb]
    embeddings = embedder.encode(docs, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return kb, embedder, index


def retrieve(query, kb, embedder, index, top_k=3):
    qe = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(qe, top_k)

    results = []
    for i in indices[0]:
        results.append({
            "title": kb[i]["title"],
            "category": kb[i]["category"],
            "content": kb[i]["content"]
        })
    return results


# ─────────────────────────────────────────
# STREAMING RESPONSE (MODERN CHATBOT FEEL)
# ─────────────────────────────────────────
def stream_groq_response(query, retrieved_docs, placeholder):
    client = Groq(api_key=GROQ_API_KEY)

    SYSTEM_PROMPT = """
You are Gala.AI 🌴 — a warm, knowledgeable AI travel guide for Cebu, Philippines.
Answer using ONLY the provided context. If context is insufficient, say so and suggest another Cebu topic.
Style: conversational like a friendly local Cebuano, sprinkle Bisaya naturally (Dali!, Maayong biyahe!, Nindot kaayo!),
concise short paragraphs, end with one follow-up suggestion. No bullet walls.
"""

    context = "\n\n".join([
        f"[{doc['category']}] {doc['title']}:\n{doc['content']}"
        for doc in retrieved_docs
    ])

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.7,
        max_tokens=650,
        stream=True
    )

    full_response = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            placeholder.markdown(full_response.replace("\n", "<br>"), unsafe_allow_html=True)

    return full_response


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "pending_regen" not in st.session_state:
    st.session_state.pending_regen = False


# ─────────────────────────────────────────
# LOAD RAG
# ─────────────────────────────────────────
try:
    kb, embedder, faiss_index = load_rag_system()
except Exception:
    st.error("⚠️ Could not load cebu_tourism.txt. Make sure it is in the same folder as app.py.")
    st.stop()


# ─────────────────────────────────────────
# SIDEBAR (CHATGPT STYLE)
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌴 Gala.AI")
    st.caption("Cebu Tourism Chatbot")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query = None
        st.session_state.pending_regen = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕘 Recent Chats")

    if not st.session_state.chat_titles:
        st.caption("No chat history yet.")
    else:
        for i, title in enumerate(reversed(st.session_state.chat_titles[-8:])):
            st.markdown(f"- {title}")

    st.markdown("---")
    st.caption("⚡ Powered by Groq + RAG")
    st.caption("📍 Cebu Tourism Knowledge Base")


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.8rem 0.2rem 1rem 0.2rem;">
    <div>
        <div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.5px;">
            Gala<span style="color:#22c55e;">.AI</span>
        </div>
        <div style="color:rgba(255,255,255,0.65);font-size:0.85rem;">
            Your Cebu travel guide — friendly, local, and fast 🌴
        </div>
    </div>
    <div style="padding:0.3rem 0.8rem;border-radius:999px;
                border:1px solid rgba(255,255,255,0.08);
                background:rgba(255,255,255,0.03);
                font-size:0.75rem;font-weight:600;color:#22c55e;">
        ● Online
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="padding:1.2rem 0 2rem 0;text-align:center;">
        <div style="font-size:2.1rem;font-weight:700;letter-spacing:-1px;">
            Kumusta! 👋
        </div>
        <div style="color:rgba(255,255,255,0.65);font-size:0.95rem;margin-top:0.6rem;">
            Ask me about beaches, festivals, food, history, or hidden gems in Cebu.
        </div>
        <div style="margin-top:1.2rem;font-size:0.8rem;
                    display:inline-block;padding:0.35rem 0.9rem;
                    border-radius:999px;border:1px solid rgba(34,197,94,0.25);
                    background:rgba(34,197,94,0.12);color:#22c55e;font-weight:600;">
            Wagtang kalaay — let's plan your Cebu trip 🌴
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Try asking")
    col1, col2 = st.columns(2)

    suggestions = [
        "🐋 Where can I see whale sharks in Cebu?",
        "🎊 What is Sinulog Festival all about?",
        "🍖 What food should I try in Cebu?",
        "🏝️ Suggest hidden beaches in Cebu",
        "⚔️ Who was Lapu-Lapu?",
        "💰 Give me budget travel tips in Cebu"
    ]

    for i, s in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(s, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.messages.append({"role": "user", "content": s})
                st.session_state.last_query = s
                st.rerun()


# ─────────────────────────────────────────
# CHAT FEED
# ─────────────────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-row">
            <div class="user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        sources_html = ""
        if msg.get("sources"):
            pills = "".join([f'<span class="source-pill">📍 {s}</span>' for s in msg["sources"]])
            sources_html = f'<div class="sources">{pills}</div>'

        content = msg["content"].replace("\n", "<br>")

        st.markdown(f"""
        <div class="bot-row">
            <div class="bot-avi">🌴</div>
            <div class="bot-bubble">
                {content}
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# REGENERATE BUTTON
# ─────────────────────────────────────────
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant":
        if st.button("🔄 Regenerate last response", use_container_width=True):
            st.session_state.pending_regen = True
            st.rerun()


# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
user_input = st.chat_input("Ask anything about Cebu, Philippines...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_query = user_input

    if user_input not in st.session_state.chat_titles:
        st.session_state.chat_titles.append(user_input[:35] + ("..." if len(user_input) > 35 else ""))

    st.rerun()


# ─────────────────────────────────────────
# GENERATE RESPONSE (STREAMING)
# ─────────────────────────────────────────
if st.session_state.last_query and (
    (len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user")
    or st.session_state.pending_regen
):

    # If regenerate, remove last assistant response
    if st.session_state.pending_regen:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()

    query = st.session_state.last_query

    retrieved = retrieve(query, kb, embedder, faiss_index, top_k=3)

    # Placeholder streaming message
    st.markdown("""
    <div class="bot-row">
        <div class="bot-avi">🌴</div>
        <div class="bot-bubble">
    """, unsafe_allow_html=True)

    placeholder = st.empty()

    full_response = stream_groq_response(query, retrieved, placeholder)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": [d["title"] for d in retrieved]
    })

    st.session_state.pending_regen = False
    st.rerun()