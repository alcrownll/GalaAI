import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Gala.AI",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg:          #f7f8fc;
    --surface:     #ffffff;
    --surface2:    #f0f2f8;
    --border:      #e4e7f0;
    --border-focus:#4ade80;
    --green:       #16a34a;
    --green-light: #4ade80;
    --green-dim:   rgba(22,163,74,0.08);
    --green-glow:  rgba(74,222,128,0.3);
    --ink:         #111827;
    --muted:       #6b7280;
    --muted-light: #9ca3af;
    --user-bg:     #1b4332;
    --user-text:   #f0fdf4;
    --radius:      14px;
    --radius-lg:   20px;
    --pill:        999px;
    --shadow-sm:   0 1px 4px rgba(0,0,0,0.06);
    --shadow-md:   0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg:   0 8px 32px rgba(0,0,0,0.1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ════════════════════
   TOP NAV
════════════════════ */
.topnav-wrap {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(247,248,252,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
}
.topnav-inner {
    max-width: 860px;
    margin: 0 auto;
    padding: 0.9rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.nav-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4ade80, #16a34a);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 2px 12px var(--green-glow);
    flex-shrink: 0;
}
.nav-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--ink);
}
.nav-title span { color: var(--green); }
.nav-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--green);
    background: var(--green-dim);
    border: 1px solid rgba(22,163,74,0.2);
    border-radius: var(--pill);
    padding: 0.28rem 0.85rem;
    letter-spacing: 0.3px;
}
.nav-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green-light);
    box-shadow: 0 0 6px var(--green-light);
    animation: blink 2.2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ════════════════════
   MAIN CONTENT AREA
════════════════════ */
.main-content {
    max-width: 860px;
    margin: 0 auto;
    padding: 2rem 2.5rem 140px;
}

/* ════════════════════
   WELCOME SCREEN
════════════════════ */
.welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
}
.welcome-icon {
    width: 76px; height: 76px;
    background: linear-gradient(135deg, #4ade80 0%, #16a34a 100%);
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(74,222,128,0.35), 0 2px 8px rgba(0,0,0,0.08);
}
.welcome-title {
    font-size: clamp(1.7rem, 3.5vw, 2.3rem);
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--ink);
    margin-bottom: 0.6rem;
    line-height: 1.15;
}
.welcome-title span { color: var(--green); }
.welcome-desc {
    font-size: 0.92rem;
    color: var(--muted);
    line-height: 1.7;
    max-width: 380px;
    margin-bottom: 1rem;
    font-weight: 400;
}
.welcome-bisaya {
    display: inline-block;
    font-size: 0.78rem;
    color: var(--green);
    background: var(--green-dim);
    border: 1px solid rgba(22,163,74,0.2);
    border-radius: var(--pill);
    padding: 0.35rem 1rem;
    font-style: italic;
    margin-bottom: 2.5rem;
    font-weight: 500;
}

/* ── SUGGESTION CARDS ── */
.suggest-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted-light);
    margin-bottom: 0.85rem;
    text-align: center;
}

/* Streamlit button overrides for suggestion cards */
.stButton > button {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: var(--radius) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.8rem 1rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    line-height: 1.45 !important;
    white-space: normal !important;
    height: auto !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: var(--surface2) !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Clear button override */
.clear-btn .stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: var(--pill) !important;
    width: auto !important;
    box-shadow: none !important;
}
.clear-btn .stButton > button:hover {
    border-color: #fca5a5 !important;
    color: #ef4444 !important;
    background: #fff5f5 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ════════════════════
   CHAT MESSAGES
════════════════════ */
.chat-feed {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}

/* User message */
.msg-user-row {
    display: flex;
    justify-content: flex-end;
    animation: msgIn 0.22s ease;
}
.msg-user-bubble {
    background: var(--user-bg);
    color: var(--user-text);
    border-radius: var(--radius-lg) var(--radius-lg) 5px var(--radius-lg);
    padding: 0.9rem 1.2rem;
    max-width: min(70%, 540px);
    font-size: 0.92rem;
    line-height: 1.65;
    box-shadow: var(--shadow-md);
    font-weight: 400;
}

/* Bot message */
.msg-bot-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    animation: msgIn 0.22s ease;
}
.bot-avi {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #4ade80, #16a34a);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(74,222,128,0.3);
    margin-top: 2px;
}
.msg-bot-bubble {
    background: var(--surface);
    border: 1.5px solid var(--border);
    color: var(--ink);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 5px;
    padding: 0.95rem 1.2rem;
    max-width: min(78%, 580px);
    font-size: 0.92rem;
    line-height: 1.75;
    box-shadow: var(--shadow-sm);
}

/* Typing indicator */
.typing-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    animation: msgIn 0.22s ease;
    padding-bottom: 0.5rem;
}
.typing-bubble {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 5px;
    padding: 0.85rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 5px;
    box-shadow: var(--shadow-sm);
}
.typing-dot {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    opacity: 0.35;
    animation: typingPulse 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingPulse {
    0%, 60%, 100% { opacity: 0.35; transform: scale(1); }
    30%            { opacity: 1;   transform: scale(1.3); }
}

/* Sources */
.sources-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
}
.source-pill {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--green);
    background: var(--green-dim);
    border: 1px solid rgba(22,163,74,0.2);
    border-radius: var(--pill);
    padding: 0.2rem 0.65rem;
    letter-spacing: 0.2px;
}

/* ════════════════════
   INPUT BAR (sticky bottom)
════════════════════ */
.input-bar-wrap {
    position: fixed;
    bottom: 0;
    left: 0; right: 0;
    background: rgba(247,248,252,0.96);
    backdrop-filter: blur(16px);
    border-top: 1px solid var(--border);
    padding: 1rem 2rem 1.5rem;
    z-index: 100;
}
.input-bar-inner {
    max-width: 860px;
    margin: 0 auto;
}
.input-hint {
    text-align: center;
    font-size: 0.67rem;
    color: var(--muted-light);
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
}

/* Chat input overrides */
[data-testid="stChatInput"] > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.1), var(--shadow-md) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--ink) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important;
    resize: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted-light) !important;
}
[data-testid="stChatInput"] button {
    background: var(--green) !important;
    border-radius: 10px !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.3) !important;
}
[data-testid="stChatInput"] button:hover {
    background: #15803d !important;
}

@keyframes msgIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
}

/* ── MOBILE ── */
@media (max-width: 640px) {
    .topnav-inner { padding: 0.75rem 1.25rem; }
    .main-content { padding: 1.5rem 1.25rem 140px; }
    .input-bar-wrap { padding: 0.75rem 1.25rem 1.25rem; }
    .msg-user-bubble, .msg-bot-bubble { max-width: 88%; font-size: 0.88rem; }
    .welcome { padding: 2.5rem 0.5rem 2rem; }
    .welcome-icon { width: 62px; height: 62px; font-size: 1.8rem; }
    .welcome-title { font-size: 1.5rem; }
    .nav-badge { display: none; }
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
nav_l, nav_m, nav_r = st.columns([2, 3, 2])

with nav_l:
    st.markdown("""
    <div style="padding: 0.85rem 0 0.85rem 2rem;">
        <div style="display:flex;align-items:center;gap:0.65rem;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#4ade80,#16a34a);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:1.1rem;box-shadow:0 2px 12px rgba(74,222,128,0.35);">🌴</div>
            <span style="font-size:1.15rem;font-weight:700;letter-spacing:-0.5px;color:#111827;">
                Gala<span style="color:#16a34a;">.AI</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_m:
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;padding-top:1rem;">
        <div style="display:flex;align-items:center;gap:0.4rem;font-size:0.7rem;font-weight:600;
                    color:#16a34a;background:rgba(22,163,74,0.08);border:1px solid rgba(22,163,74,0.2);
                    border-radius:999px;padding:0.28rem 0.85rem;letter-spacing:0.3px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#4ade80;
                        box-shadow:0 0 6px #4ade80;animation:blink 2.2s infinite;"></div>
            Cebu Tourism AI &nbsp;·&nbsp; Online
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_r:
    st.markdown('<div style="display:flex;justify-content:flex-end;align-items:center;padding-top:0.7rem;padding-right:2rem;">', unsafe_allow_html=True)
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("✕ Clear chat", key="clear_top"):
        st.session_state.messages = []
        st.session_state.is_typing = False
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid #e4e7f0;margin:0 0 0 0;">', unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
if not rag_ready:
    st.error("⚠️ Could not load **cebu_tourism.txt** — make sure it's in the same folder as app.py.")
    st.stop()

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ── WELCOME ──
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🌴</div>
        <div class="welcome-title">Hello! I'm Gala<span>.AI</span></div>
        <div class="welcome-desc">
            Your personal AI travel guide to the best of Cebu, Philippines.
            Ask me anything — beaches, history, food, festivals, hidden gems.
        </div>
        <div class="welcome-bisaya">Wagtang kalaay, decide your destination through Gala.AI!</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="suggest-label">✦ &nbsp; Try asking</p>', unsafe_allow_html=True)

    suggestions = [
        ("🐋", "Whale sharks in Oslob",     "Watch butanding up close"),
        ("🎊", "What is Sinulog Festival?",  "Cebu's biggest celebration"),
        ("💰", "Budget travel tips",         "Explore Cebu for less"),
        ("🏝️", "Hidden gems in Cebu",        "Off the beaten path"),
        ("⚔️", "Who was Lapu-Lapu?",         "Cebu's greatest hero"),
        ("🍖", "What to eat in Cebu?",       "Food you can't miss"),
    ]

    col1, col2 = st.columns(2, gap="medium")
    for i, (icon, label, sub) in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"{icon}  {label}\n{sub}", key=f"chip_{i}"):
                st.session_state.pending = label
                st.rerun()

# ── CHAT MESSAGES ──
else:
    st.markdown('<div class="chat-feed">', unsafe_allow_html=True)

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
        <div class="typing-row">
            <div class="bot-avi">🌴</div>
            <div class="typing-bubble">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Spacer so sticky input doesn't cover messages
st.markdown('<div style="height:120px;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
user_input = st.chat_input("Ask anything about Cebu, Philippines...")

if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None


# ─────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_typing = True
    st.rerun()

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