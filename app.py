import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Gala.AI — Cebu Travel Guide",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# DESIGN SYSTEM & CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,700;0,900;1,300;1,700&family=Cabinet+Grotesk:wght@300;400;500;700&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --ink:        #0e1a12;
    --forest:     #1b4332;
    --jade:       #2d6a4f;
    --seafoam:    #52b788;
    --mist:       #d8f3dc;
    --sand:       #fdf8f0;
    --gold:       #e9c46a;
    --coral:      #e76f51;
    --white:      #ffffff;
    --radius-lg:  16px;
    --radius-xl:  24px;
    --radius-pill:50px;
    --shadow-sm:  0 2px 8px rgba(14,26,18,0.08);
    --shadow-md:  0 8px 32px rgba(14,26,18,0.12);
    --shadow-lg:  0 20px 60px rgba(14,26,18,0.18);
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── RESET & BASE ── */
html, body, [class*="css"] {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    background: var(--sand) !important;
    color: var(--ink) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--forest) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(14,26,18,0.2) !important;
}
section[data-testid="stSidebar"] * { color: var(--white) !important; }

.sidebar-brand {
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.5rem;
}
.sidebar-logo { font-size: 2.8rem; line-height: 1; margin-bottom: 0.5rem; }
.sidebar-name {
    font-family: 'Fraunces', serif !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    letter-spacing: -1px;
    line-height: 1;
    color: var(--white) !important;
}
.sidebar-name span { color: var(--seafoam) !important; }
.sidebar-sub {
    font-size: 0.75rem !important;
    opacity: 0.55;
    margin-top: 0.3rem;
    font-weight: 300 !important;
}
.sidebar-section { padding: 0 1.5rem; margin-bottom: 1.5rem; }
.sidebar-label {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.4;
    margin-bottom: 0.75rem;
    display: block;
}
.sidebar-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: var(--radius-pill);
    padding: 0.35rem 0.75rem;
    font-size: 0.78rem !important;
    margin: 0.2rem 0.2rem 0.2rem 0;
    color: rgba(255,255,255,0.85) !important;
}
.sidebar-tip {
    font-size: 0.8rem !important;
    opacity: 0.6;
    line-height: 1.6;
    font-style: italic;
    border-left: 2px solid var(--seafoam);
    padding-left: 0.75rem;
    margin-bottom: 0.5rem;
}
.sidebar-footer {
    padding: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 0.7rem !important;
    opacity: 0.35;
    text-align: center;
    line-height: 1.8;
}

section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--white) !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1rem !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: rgba(255,255,255,0.35) !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--seafoam) !important;
    box-shadow: 0 0 0 3px rgba(82,183,136,0.2) !important;
    outline: none !important;
}
section[data-testid="stSidebar"] label {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.4;
}
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: var(--radius-lg) !important;
    font-size: 0.8rem !important;
    width: 100%;
    padding: 0.5rem !important;
    transition: var(--transition) !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.12) !important;
    color: var(--white) !important;
}

/* ════════════════════════════════════════
   MAIN LAYOUT
════════════════════════════════════════ */
.main-wrapper {
    max-width: 860px;
    margin: 0 auto;
    padding: 2rem 2rem 6rem;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, var(--forest) 0%, #1e5631 40%, #2d7a4f 100%);
    border-radius: var(--radius-xl);
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 80% 20%, rgba(82,183,136,0.25) 0%, transparent 60%),
        radial-gradient(ellipse at 20% 80%, rgba(233,196,106,0.15) 0%, transparent 50%);
}
.hero::after {
    content: '🌴';
    position: absolute;
    font-size: 10rem;
    right: 1.5rem;
    bottom: -1.5rem;
    opacity: 0.12;
    line-height: 1;
}
.hero-inner { position: relative; z-index: 1; }
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(82,183,136,0.2);
    border: 1px solid rgba(82,183,136,0.35);
    border-radius: var(--radius-pill);
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--seafoam);
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.5rem, 5vw, 3.8rem);
    font-weight: 900;
    color: var(--white);
    margin: 0 0 0.4rem;
    line-height: 1;
    letter-spacing: -2px;
}
.hero-title span { color: var(--seafoam); }
.hero-tagline {
    font-size: 1rem;
    color: rgba(255,255,255,0.65);
    font-weight: 300;
    margin-bottom: 1.5rem;
}
.hero-bisaya {
    display: inline-block;
    background: rgba(233,196,106,0.15);
    border: 1px solid rgba(233,196,106,0.3);
    border-radius: var(--radius-pill);
    padding: 0.4rem 1.1rem;
    font-size: 0.82rem;
    color: var(--gold);
    font-style: italic;
}

/* ── SUGGEST SECTION ── */
.suggest-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ab;
    margin-bottom: 0.75rem;
    display: block;
}

/* ── CHAT ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    margin-bottom: 1.5rem;
}
.msg-row-user {
    display: flex;
    justify-content: flex-end;
    animation: fadeSlideUp 0.3s ease;
}
.bubble-user {
    background: var(--forest);
    color: var(--white);
    border-radius: 20px 20px 4px 20px;
    padding: 0.9rem 1.25rem;
    max-width: 68%;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: var(--shadow-md);
}
.msg-row-bot {
    display: flex;
    justify-content: flex-start;
    gap: 0.75rem;
    align-items: flex-start;
    animation: fadeSlideUp 0.3s ease;
}
.bot-avatar {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--seafoam), var(--jade));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: var(--shadow-sm);
    margin-top: 2px;
}
.bubble-bot {
    background: var(--white);
    border: 1.5px solid #e8e4dc;
    color: var(--ink);
    border-radius: 20px 20px 20px 4px;
    padding: 1rem 1.25rem;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.65;
    box-shadow: var(--shadow-sm);
}
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid #f0ece4;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--mist);
    color: var(--jade);
    border-radius: var(--radius-pill);
    padding: 0.2rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 600;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    color: #b0bbb5;
}
.empty-icon { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-title {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #8a9e94;
    margin-bottom: 0.4rem;
}
.empty-sub { font-size: 0.88rem; line-height: 1.6; }

/* ── SUGGESTION BUTTONS ── */
.stButton button {
    background: var(--white) !important;
    border: 1.5px solid #e0dbd0 !important;
    color: var(--ink) !important;
    border-radius: var(--radius-pill) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton button:hover {
    background: var(--forest) !important;
    border-color: var(--forest) !important;
    color: var(--white) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #ddd8ce 20%, #ddd8ce 80%, transparent);
    margin: 1.5rem 0;
}

/* ── STATUS ── */
.status-badge { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.72rem; font-weight: 600; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--seafoam); box-shadow: 0 0 0 2px rgba(82,183,136,0.3); }

/* ── ANIMATIONS ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
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


def generate_response(query, retrieved_docs, client):
    SYSTEM_PROMPT = """You are Gala.AI 🌴 — a warm, knowledgeable AI travel guide for Cebu, Philippines.
Tagline: "Your AI guide to the best of Cebu."

Answer using ONLY the provided context. If insufficient, say so and suggest another Cebu topic.

Style:
- Conversational, like a friendly local Cebuano
- Sprinkle Bisaya naturally (Dali!, Maayong biyahe!, Nindot kaayo!)
- Concise but complete — use short paragraphs
- End with one follow-up suggestion
- No bullet-point walls — write naturally"""

    context = "\n\n".join([
        f"[{doc['category']}] {doc['title']}:\n{doc['content']}"
        for doc in retrieved_docs
    ])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🌴</div>
        <div class="sidebar-name">Gala<span>.AI</span></div>
        <div class="sidebar-sub">Your AI guide to the best of Cebu</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">🔑 Groq API Key</span>', unsafe_allow_html=True)
    api_key = st.text_input(" ", type="password", placeholder="gsk_...", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <span class="sidebar-label">📚 Knowledge Base</span>
        <div>
            <span class="sidebar-tag">📍 Tourist Spots</span>
            <span class="sidebar-tag">🏛️ History & Culture</span>
            <span class="sidebar-tag">🎉 Festivals</span>
            <span class="sidebar-tag">💡 Travel Tips</span>
            <span class="sidebar-tag">🌟 Hidden Gems</span>
        </div>
    </div>
    <div class="sidebar-section">
        <span class="sidebar-label">💬 Try asking</span>
        <div class="sidebar-tip">"Where to go for only 2 days?"</div>
        <div class="sidebar-tip">"Best time to visit Cebu?"</div>
        <div class="sidebar-tip">"Tell me about Sinulog Festival"</div>
        <div class="sidebar-tip">"Hidden gems tourists miss"</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div class="sidebar-footer">
        Powered by Groq · LLaMA 3.3 · FAISS<br>
        Sentence Transformers · Streamlit<br><br>
        Wagtang kalaay — Gala na! 🌴
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD RAG
# ─────────────────────────────────────────
try:
    kb, embedder, faiss_index = load_rag_system()
    rag_ready = True
except Exception as e:
    st.error(f"⚠️ Could not load **cebu_tourism.txt** — make sure it's in the same folder as app.py.\n\n`{e}`")
    rag_ready = False


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# HERO
status_html = '<span class="status-dot"></span> Ready' if rag_ready else '<span style="background:#e76f51" class="status-dot"></span> Error'
st.markdown(f"""
<div class="hero">
    <div class="hero-inner">
        <div class="hero-eyebrow">
            <span class="status-badge">{status_html}</span>
            &nbsp;·&nbsp; Cebu Tourism AI
        </div>
        <div class="hero-title">Gala<span>.AI</span></div>
        <div class="hero-tagline">Your AI guide to the best of Cebu, Philippines 🇵🇭</div>
        <div class="hero-bisaya">Wagtang kalaay, decide your destination through Gala.AI!</div>
    </div>
</div>
""", unsafe_allow_html=True)

# SUGGESTED QUESTIONS
if not st.session_state.messages and rag_ready:
    st.markdown('<span class="suggest-label">✦ Start with a question</span>', unsafe_allow_html=True)
    suggestions = [
        ("🐋", "Whale sharks in Oslob"),
        ("🎊", "Sinulog Festival"),
        ("💰", "Budget travel tips"),
        ("🏝️", "Hidden gems in Cebu"),
        ("⚔️", "Lapu-Lapu history"),
        ("🍖", "What to eat in Cebu"),
    ]
    cols = st.columns(len(suggestions))
    for col, (icon, label) in zip(cols, suggestions):
        with col:
            if st.button(f"{icon} {label}", key=f"s_{label}"):
                st.session_state.pending = label
                st.rerun()
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# CHAT MESSAGES
if st.session_state.messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row-user">
                <div class="bubble-user">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            sources_html = "".join([
                f'<span class="source-chip">📍 {s}</span>'
                for s in msg.get("sources", [])
            ])
            content = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-row-bot">
                <div class="bot-avatar">🌴</div>
                <div class="bubble-bot">
                    {content}
                    {f'<div class="sources-row">{sources_html}</div>' if sources_html else ''}
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    if rag_ready:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🗺️</div>
            <div class="empty-title">Where do you want to go?</div>
            <div class="empty-sub">Ask me about beaches, history, food, festivals,<br>hidden gems — anything about Cebu.</div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
user_input = st.chat_input("Ask anything about Cebu...")

if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None

if user_input and rag_ready:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Gala.AI is thinking..."):
            try:
                client = Groq(api_key=api_key)
                retrieved = retrieve(user_input, kb, embedder, faiss_index, top_k=3)
                response = generate_response(user_input, retrieved, client)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": [d["title"] for d in retrieved]
                })
            except Exception as e:
                st.error(f"❌ {e}")
        st.rerun()