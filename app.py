import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import os
from dotenv import load_dotenv
from streamlit_lucide_icons import lucide

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
# THEME STATE
# ─────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ─────────────────────────────────────────
# SIDEBAR THEME TOGGLE
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## Gala.AI")
    st.caption("Cebu Tourism Chatbot")

    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1
    )
    st.session_state.theme = theme_choice

    st.markdown("---")

# ─────────────────────────────────────────
# THEME CSS
# ─────────────────────────────────────────
if st.session_state.theme == "Dark":
    THEME_CSS = """
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
        --shadow-soft: 0 8px 24px rgba(0,0,0,0.35);
    }

    html, body, [class*="css"] {
        background: radial-gradient(circle at top, #0f172a, #0b0f17) !important;
        color: var(--ink) !important;
    }
    """
else:
    THEME_CSS = """
    :root {
        --bg: #f7f8fc;
        --surface: #ffffff;
        --surface2: #f1f5f9;
        --border: rgba(0,0,0,0.08);
        --accent: #16a34a;
        --accent-dim: rgba(22,163,74,0.12);
        --ink: #111827;
        --muted: rgba(0,0,0,0.65);
        --muted2: rgba(0,0,0,0.45);
        --radius: 16px;
        --radius-lg: 22px;
        --pill: 999px;
        --shadow: 0 12px 40px rgba(0,0,0,0.10);
        --shadow-soft: 0 8px 24px rgba(0,0,0,0.08);
    }

    html, body, [class*="css"] {
        background: radial-gradient(circle at top, #ffffff, #f7f8fc) !important;
        color: var(--ink) !important;
    }
    """

# ─────────────────────────────────────────
# GLOBAL MODERN CSS (CHATGPT / CLAUDE STYLE)
# ─────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

{THEME_CSS}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* Hide Streamlit junk but KEEP sidebar toggle */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{
    display: none !important;
}}

/* Main container */
.block-container {{
    padding-top: 0.6rem !important;
    max-width: 900px !important;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background: var(--surface2) !important;
    border-right: 1px solid var(--border) !important;
}}

/* Sidebar buttons */
.stButton > button {{
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.04) !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: var(--accent-dim) !important;
    border-color: rgba(34,197,94,0.35) !important;
    transform: translateY(-1px);
}}

/* Header */
.header-wrap {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0.8rem 0.2rem 1rem 0.2rem;
}}

/* Chat wrapper */
.chat-wrap {{
    padding-bottom: 110px;
}}

/* User message */
.user-row {{
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0;
}}
.user-bubble {{
    background: linear-gradient(135deg, var(--accent), #15803d);
    color: white;
    padding: 0.85rem 1.15rem;
    border-radius: var(--radius-lg) var(--radius-lg) 6px var(--radius-lg);
    max-width: 75%;
    line-height: 1.7;
    font-size: 0.95rem;
    box-shadow: var(--shadow-soft);
}}

/* Assistant message */
.bot-row {{
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin: 1rem 0;
}}
.bot-avi {{
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}
.bot-bubble {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0.95rem 1.15rem;
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 6px;
    max-width: 80%;
    line-height: 1.75;
    font-size: 0.95rem;
    color: var(--ink);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
}}

/* Sources */
.sources {{
    margin-top: 0.9rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}}
.source-pill {{
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: var(--pill);
    background: var(--accent-dim);
    border: 1px solid rgba(22,163,74,0.20);
    color: var(--accent);
    font-weight: 600;
}}

/* Chat input - remove weird highlight */
[data-testid="stChatInput"] > div {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: none !important;
}}

[data-testid="stChatInput"] > div:focus-within {{
    border: 1px solid rgba(34,197,94,0.35) !important;
    box-shadow: none !important;
    outline: none !important;
}}

[data-testid="stChatInput"] textarea {{
    color: var(--ink) !important;
    font-size: 0.95rem !important;
    caret-color: var(--accent) !important;
    outline: none !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    outline: none !important;
    box-shadow: none !important;
}}

[data-testid="stChatInput"] textarea::selection {{
    background: rgba(34,197,94,0.25) !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: var(--muted2) !important;
}}

[data-testid="stChatInput"] button {{
    background: var(--accent) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}}

/* Welcome screen */
.welcome {{
    padding: 2.2rem 0 2rem 0;
    text-align: center;
}}
.welcome-title {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -1px;
}}
.welcome-sub {{
    margin-top: 0.65rem;
    color: var(--muted);
    font-size: 0.95rem;
}}
.welcome-pill {{
    margin-top: 1.2rem;
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(34,197,94,0.25);
    background: var(--accent-dim);
    color: var(--accent);
    font-size: 0.8rem;
    font-weight: 600;
}}

/* Mobile */
@media (max-width: 700px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}
    .user-bubble, .bot-bubble {{
        max-width: 92%;
    }}
}}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# KNOWLEDGE BASE PARSER
# ─────────────────────────────────────────
def parse_knowledge_base(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
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
# STREAMING RESPONSE
# ─────────────────────────────────────────
def stream_groq_response(query, retrieved_docs, placeholder):
    client = Groq(api_key=GROQ_API_KEY)

    SYSTEM_PROMPT = """
You are Gala.AI — a warm, knowledgeable AI travel guide for Cebu, Philippines.
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
# SIDEBAR CONTENT
# ─────────────────────────────────────────
with st.sidebar:
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query = None
        st.session_state.pending_regen = False
        st.rerun()

    st.markdown("---")
    st.markdown("### Recent Chats")

    if not st.session_state.chat_titles:
        st.caption("No chat history yet.")
    else:
        for title in reversed(st.session_state.chat_titles[-8:]):
            st.markdown(f"- {title}")

    st.markdown("---")
    st.caption("Powered by Groq + RAG")


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="header-wrap">
    <div>
        <div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.5px;">
            Gala<span style="color:var(--accent);">.AI</span>
        </div>
        <div style="color:var(--muted);font-size:0.85rem;margin-top:0.15rem;">
            Cebu travel guide — friendly local vibes
        </div>
    </div>

    <div style="padding:0.3rem 0.8rem;border-radius:999px;
                border:1px solid var(--border);
                background:rgba(255,255,255,0.03);
                font-size:0.75rem;font-weight:600;color:var(--accent);">
        Online
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# WELCOME SCREEN BACK
# ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">Kamusta! 👋</div>
        <div class="welcome-sub">
            Ask me about beaches, food, history, festivals, and hidden gems in Cebu.
        </div>
        <div class="welcome-pill">Wagtang kalaay — plan your Cebu trip with Gala.AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Suggested questions")

    col1, col2 = st.columns(2)

    suggestions = [
        "Where can I see whale sharks in Cebu?",
        "What is Sinulog Festival all about?",
        "What food should I try in Cebu?",
        "Suggest hidden beaches in Cebu",
        "Who was Lapu-Lapu?",
        "Give me budget travel tips in Cebu"
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

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-row">
            <div class="user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        sources_html = ""
        if msg.get("sources"):
            pills = "".join([f'<span class="source-pill">{s}</span>' for s in msg["sources"]])
            sources_html = f'<div class="sources">{pills}</div>'

        content = msg["content"].replace("\n", "<br>")

        st.markdown(f"""
        <div class="bot-row">
            <div class="bot-avi">{lucide("sparkles", color="var(--accent)", size=18)}</div>
            <div class="bot-bubble">
                {content}
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# ICON REGENERATE BUTTON (REAL ICON)
# ─────────────────────────────────────────
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant":
        col1, col2 = st.columns([1, 12])
        with col1:
            if st.button(" ", help="Regenerate last response"):
                st.session_state.pending_regen = True
                st.rerun()

        st.markdown(
            f"""
            <div style="margin-top:-42px;margin-left:8px;">
                {lucide("refresh-cw", color="var(--muted)", size=18)}
            </div>
            """,
            unsafe_allow_html=True
        )


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
# GENERATE RESPONSE
# ─────────────────────────────────────────
if st.session_state.last_query and (
    (len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user")
    or st.session_state.pending_regen
):

    # if regenerate, remove last assistant response
    if st.session_state.pending_regen:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()

    query = st.session_state.last_query
    retrieved = retrieve(query, kb, embedder, faiss_index, top_k=3)

    st.markdown(f"""
    <div class="bot-row">
        <div class="bot-avi">{lucide("sparkles", color="var(--accent)", size=18)}</div>
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