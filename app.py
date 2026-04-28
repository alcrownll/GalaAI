import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ─────────────────────────────────────────
# CONFIG — put your Groq key here
# ─────────────────────────────────────────
import os
from dotenv import load_dotenv

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
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg:         #0f1117;
    --surface:    #1a1d27;
    --surface2:   #22263a;
    --border:     rgba(255,255,255,0.08);
    --green:      #4ade80;
    --green-dim:  rgba(74,222,128,0.12);
    --green-glow: rgba(74,222,128,0.25);
    --white:      #f0f2f8;
    --muted:      rgba(240,242,248,0.45);
    --user-bg:    #2a2d3e;
    --radius:     14px;
    --radius-lg:  20px;
    --pill:       999px;
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: var(--bg) !important;
    color: var(--white) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }

/* ════════════════════════════════════════
   APP SHELL — full viewport chat layout
════════════════════════════════════════ */
.app-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 780px;
    margin: 0 auto;
}

/* ── TOP NAV ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    background: rgba(15,17,23,0.85);
    position: sticky;
    top: 0;
    z-index: 50;
}
.topnav-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topnav-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #4ade80, #22c55e);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 16px var(--green-glow);
}
.topnav-name {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--white);
}
.topnav-name span { color: var(--green); }
.topnav-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--green);
    background: var(--green-dim);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: var(--pill);
    padding: 0.25rem 0.75rem;
}
.online-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── CHAT SCROLL AREA ── */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 1.5rem 0;
    scroll-behavior: smooth;
}
.chat-area::-webkit-scrollbar { width: 4px; }
.chat-area::-webkit-scrollbar-track { background: transparent; }
.chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── WELCOME SCREEN ── */
.welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 3rem 1rem 2rem;
    min-height: 55vh;
}
.welcome-glow {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, #4ade80, #16a34a);
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 0 40px var(--green-glow), 0 0 80px rgba(74,222,128,0.1);
}
.welcome-title {
    font-size: clamp(1.6rem, 4vw, 2.2rem);
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}
.welcome-title span { color: var(--green); }
.welcome-sub {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
    max-width: 340px;
    margin-bottom: 0.75rem;
}
.welcome-bisaya {
    font-size: 0.78rem;
    color: rgba(74,222,128,0.7);
    font-style: italic;
    border: 1px solid rgba(74,222,128,0.2);
    background: var(--green-dim);
    border-radius: var(--pill);
    padding: 0.3rem 0.9rem;
    margin-bottom: 2rem;
}

/* ── SUGGESTION CHIPS ── */
.chips-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    text-align: center;
}
.chips-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    width: 100%;
    max-width: 480px;
}
.chip-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--white);
}
.chip-btn:hover {
    background: var(--surface2);
    border-color: rgba(74,222,128,0.3);
    transform: translateY(-1px);
}
.chip-icon { font-size: 1.1rem; margin-bottom: 0.3rem; display: block; }
.chip-text { font-size: 0.8rem; font-weight: 500; color: var(--white); line-height: 1.3; }
.chip-sub  { font-size: 0.7rem; color: var(--muted); margin-top: 0.15rem; }

/* ── MESSAGES ── */
.msg-group {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
}

/* User */
.msg-user-row {
    display: flex;
    justify-content: flex-end;
    animation: msgIn 0.25s ease;
}
.msg-user-bubble {
    background: var(--user-bg);
    border: 1px solid rgba(255,255,255,0.06);
    color: var(--white);
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
    padding: 0.85rem 1.1rem;
    max-width: min(72%, 520px);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Bot */
.msg-bot-row {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    animation: msgIn 0.25s ease;
}
.bot-avi {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #4ade80, #16a34a);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
    box-shadow: 0 0 12px var(--green-glow);
    margin-top: 2px;
}
.msg-bot-bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--white);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
    padding: 0.9rem 1.1rem;
    max-width: min(78%, 560px);
    font-size: 0.9rem;
    line-height: 1.7;
}

/* Typing indicator */
.typing-row {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    animation: msgIn 0.25s ease;
    margin-bottom: 1rem;
}
.typing-bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
    padding: 0.9rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 60px;
}
.typing-dot {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    opacity: 0.4;
    animation: typingPulse 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingPulse {
    0%, 60%, 100% { opacity: 0.4; transform: scale(1); }
    30%            { opacity: 1;   transform: scale(1.25); }
}

/* Sources */
.sources-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.65rem;
    padding-top: 0.65rem;
    border-top: 1px solid var(--border);
}
.source-pill {
    font-size: 0.67rem;
    font-weight: 600;
    color: var(--green);
    background: var(--green-dim);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: var(--pill);
    padding: 0.18rem 0.6rem;
    letter-spacing: 0.2px;
}

/* ── INPUT AREA ── */
.input-area {
    padding: 1rem 1.5rem 1.5rem;
    border-top: 1px solid var(--border);
    background: rgba(15,17,23,0.95);
    backdrop-filter: blur(12px);
    position: sticky;
    bottom: 0;
}
.input-wrap {
    display: flex;
    align-items: flex-end;
    gap: 0.6rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.6rem 0.75rem;
    transition: border-color 0.2s;
}
.input-wrap:focus-within {
    border-color: rgba(74,222,128,0.4);
    box-shadow: 0 0 0 3px rgba(74,222,128,0.08);
}
.input-hint {
    font-size: 0.7rem;
    color: var(--muted);
    text-align: center;
    margin-top: 0.6rem;
    letter-spacing: 0.2px;
}

/* Streamlit chat input overrides */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.2rem 0.5rem !important;
    transition: all 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(74,222,128,0.4) !important;
    box-shadow: 0 0 0 3px rgba(74,222,128,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--white) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    resize: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}
[data-testid="stChatInput"] button {
    background: var(--green) !important;
    border-radius: 8px !important;
    border: none !important;
    color: #0f1117 !important;
}
[data-testid="stChatInput"] button:hover {
    background: #22c55e !important;
}

/* Suggestion buttons */
.stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: var(--radius) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 0.75rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    height: auto !important;
}
.stButton > button:hover {
    background: var(--surface2) !important;
    border-color: rgba(74,222,128,0.3) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
}

/* Clear button */
.clear-btn > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    padding: 0.35rem 0.75rem !important;
    border-radius: var(--pill) !important;
    width: auto !important;
}
.clear-btn > button:hover {
    border-color: rgba(255,100,100,0.3) !important;
    color: #f87171 !important;
    transform: none !important;
    box-shadow: none !important;
}

@keyframes msgIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── MOBILE ── */
@media (max-width: 600px) {
    .topnav { padding: 0.75rem 1rem; }
    .chat-area { padding: 1rem 1rem 0; }
    .input-area { padding: 0.75rem 1rem 1rem; }
    .msg-user-bubble, .msg-bot-bubble { max-width: 85%; font-size: 0.85rem; }
    .chips-grid { grid-template-columns: 1fr 1fr; gap: 0.5rem; }
    .welcome { padding: 2rem 1rem 1rem; min-height: 45vh; }
    .welcome-title { font-size: 1.5rem; }
    .welcome-glow { width: 58px; height: 58px; font-size: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# KNOWLEDGE BASE
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
                    entries.append({"category": category, "title": title, "content": content_block})
                i += 2
            else:
                i += 1
        else:
            i += 1
    return entries


# ─────────────────────────────────────────
# RAG SYSTEM
# ─────────────────────────────────────────
@st.cache_resource
def load_rag_system():
    kb = parse_knowledge_base("cebu_tourism.txt")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    docs = [f"[{item['category']}] {item['title']}: {item['content']}" for item in kb]
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return kb, embedder, idx


def retrieve(query, kb, embedder, index, top_k=3):
    qe = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(qe, top_k)
    return [{"title": kb[i]['title'], "category": kb[i]['category'], "content": kb[i]['content']}
            for i in indices[0]]


def generate_response(query, retrieved_docs):
    client = Groq(api_key=GROQ_API_KEY)
    SYSTEM_PROMPT = """You are Gala.AI 🌴 — a warm, knowledgeable AI travel guide for Cebu, Philippines.
Answer using ONLY the provided context. If context is insufficient, say so and suggest another Cebu topic.
Style: conversational like a friendly local Cebuano, sprinkle Bisaya naturally (Dali!, Maayong biyahe!, Nindot kaayo!), concise short paragraphs, end with one follow-up suggestion. No bullet walls."""

    context = "\n\n".join([
        f"[{doc['category']}] {doc['title']}:\n{doc['content']}"
        for doc in retrieved_docs
    ])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False


# ─────────────────────────────────────────
# LOAD RAG
# ─────────────────────────────────────────
try:
    kb, embedder, faiss_index = load_rag_system()
    rag_ready = True
except Exception as e:
    rag_ready = False
    rag_error = str(e)


# ─────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────
nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])

with nav_col1:
    st.markdown("""
    <div style="padding: 0.8rem 0 0.8rem 1.5rem;">
        <div style="display:flex; align-items:center; gap:0.6rem;">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#4ade80,#22c55e);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem;box-shadow:0 0 16px rgba(74,222,128,0.25);">🌴</div>
            <span style="font-size:1.1rem;font-weight:700;letter-spacing:-0.5px;">Gala<span style="color:#4ade80;">.AI</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;padding-top:0.9rem;">
        <div style="display:flex;align-items:center;gap:0.4rem;font-size:0.72rem;font-weight:500;
                    color:#4ade80;background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.2);
                    border-radius:999px;padding:0.28rem 0.8rem;">
            <div style="width:6px;height:6px;border-radius:50%;background:#4ade80;
                        box-shadow:0 0 6px #4ade80;animation:blink 2s infinite;"></div>
            Cebu Tourism AI · Online
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown('<div style="display:flex;justify-content:flex-end;padding-top:0.6rem;padding-right:1.5rem;">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("✕ Clear", key="clear_nav"):
            st.session_state.messages = []
            st.session_state.is_typing = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:0;">', unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHAT AREA
# ─────────────────────────────────────────
if not rag_ready:
    st.error(f"⚠️ Could not load **cebu_tourism.txt** — make sure it's in the same folder as app.py.")
    st.stop()

# ── WELCOME / EMPTY STATE ──
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-glow">🌴</div>
        <div class="welcome-title">Hello! I'm Gala<span>.AI</span></div>
        <div class="welcome-sub">Your personal AI travel guide to the best of Cebu, Philippines. Ask me anything — beaches, history, food, festivals, hidden gems.</div>
        <div class="welcome-bisaya">Wagtang kalaay, decide your destination through Gala.AI!</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="chips-label">✦ Suggested questions</p>', unsafe_allow_html=True)

    suggestions = [
        ("🐋", "Whale sharks in Oslob", "Watch butanding up close"),
        ("🎊", "What is Sinulog Festival?", "Cebu's biggest celebration"),
        ("💰", "Budget travel tips", "Explore Cebu for less"),
        ("🏝️", "Hidden gems in Cebu", "Off-the-beaten-path spots"),
        ("⚔️", "Who was Lapu-Lapu?", "Cebu's greatest hero"),
        ("🍖", "What to eat in Cebu?", "Food you can't miss"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, label, sub) in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"{icon}  {label}\n{sub}", key=f"chip_{i}"):
                st.session_state.pending = label
                st.rerun()

# ── CHAT MESSAGES ──
else:
    st.markdown('<div style="padding: 1.5rem 1.5rem 0; max-width: 780px; margin: 0 auto;">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user-row">
                <div class="msg-user-bubble">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                pills = "".join([f'<span class="source-pill">📍 {s}</span>' for s in msg["sources"]])
                sources_html = f'<div class="sources-strip">{pills}</div>'

            content = msg["content"].replace("\n\n", "<br><br>").replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-bot-row">
                <div class="bot-avi">🌴</div>
                <div class="msg-bot-bubble">
                    {content}
                    {sources_html}
                </div>
            </div>""", unsafe_allow_html=True)

    # Typing indicator
    if st.session_state.is_typing:
        st.markdown("""
        <div class="typing-row" style="padding: 0 0 1rem;">
            <div class="bot-avi">🌴</div>
            <div class="typing-bubble">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Bottom padding so input doesn't cover messages
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHAT INPUT (bottom, sticky)
# ─────────────────────────────────────────
st.markdown('<div style="max-width:780px;margin:0 auto;">', unsafe_allow_html=True)
user_input = st.chat_input("Ask anything about Cebu, Philippines...")
st.markdown('</div>', unsafe_allow_html=True)

# Handle suggestion click
if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None


# ─────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_typing = True
    st.rerun()

# If typing flag is set, generate the response
if st.session_state.is_typing:
    try:
        last_user = next(
            (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
            None
        )
        if last_user:
            retrieved = retrieve(last_user, kb, embedder, faiss_index, top_k=3)
            response  = generate_response(last_user, retrieved)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": [d["title"] for d in retrieved]
            })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, something went wrong: {e}",
            "sources": []
        })
    finally:
        st.session_state.is_typing = False
        st.rerun()