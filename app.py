import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Gala.AI — Your Cebu Travel Guide",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ocean: #0a4f6e;
    --sand: #f5e6c8;
    --coral: #e8643a;
    --leaf: #2d7a4f;
    --sky: #d4edf7;
    --dark: #0d1f2d;
    --white: #fffef9;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--white);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, var(--ocean) 0%, var(--dark) 100%);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: var(--white) !important; }
section[data-testid="stSidebar"] .stMarkdown p { opacity: 0.85; font-size: 0.85rem; line-height: 1.6; }

/* ── HERO HEADER ── */
.hero {
    background: linear-gradient(135deg, var(--ocean) 0%, #1a7fa0 50%, var(--leaf) 100%);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌊";
    position: absolute;
    font-size: 8rem;
    opacity: 0.08;
    right: -1rem;
    top: -1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: var(--white);
    margin: 0;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-title span { color: #7dd4f0; }
.hero-tagline {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-informal {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    color: var(--sand);
    margin-top: 0.8rem;
    font-style: italic;
}

/* ── CHAT MESSAGES ── */
.chat-wrapper { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1rem; }

.msg-user {
    display: flex;
    justify-content: flex-end;
}
.msg-user .bubble {
    background: var(--ocean);
    color: var(--white);
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    max-width: 70%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(10,79,110,0.2);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    gap: 0.6rem;
    align-items: flex-start;
}
.bot-avatar {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--coral), #f0943a);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-bot .bubble {
    background: var(--white);
    border: 1.5px solid #e8e0d0;
    color: var(--dark);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.sources-pill {
    display: inline-block;
    background: var(--sky);
    color: var(--ocean);
    border-radius: 50px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 0.3rem;
    margin-top: 0.5rem;
    border: 1px solid #b8dded;
}

/* ── SUGGESTED QUESTIONS ── */
.suggest-label {
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── INPUT AREA ── */
.stTextInput input {
    border-radius: 50px !important;
    border: 2px solid #ddd !important;
    padding: 0.7rem 1.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: var(--ocean) !important;
    box-shadow: 0 0 0 3px rgba(10,79,110,0.1) !important;
}

/* ── BUTTONS ── */
.stButton button {
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border: 2px solid var(--ocean) !important;
    color: var(--ocean) !important;
    background: transparent !important;
    transition: all 0.2s !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.9rem !important;
}
.stButton button:hover {
    background: var(--ocean) !important;
    color: var(--white) !important;
}

/* ── DIVIDER ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #ddd, transparent);
    margin: 1rem 0;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #aaa;
}
.empty-state .big-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.95rem; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD KNOWLEDGE BASE FROM TXT
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
            category = ""
            title = ""
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
# RAG SETUP (cached so it only runs once)
# ─────────────────────────────────────────
@st.cache_resource
def load_rag_system():
    kb = parse_knowledge_base("cebu_tourism.txt")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    documents = [
        f"[{item['category']}] {item['title']}: {item['content']}"
        for item in kb
    ]
    embeddings = embedder.encode(documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dimension)
    idx.add(embeddings)
    return kb, embedder, idx


def retrieve(query, kb, embedder, index, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "title": kb[idx]['title'],
            "category": kb[idx]['category'],
            "content": kb[idx]['content']
        })
    return results


def generate_response(query, retrieved_docs, client):
    SYSTEM_PROMPT = """
You are Gala.AI 🌴 — a friendly, knowledgeable, and enthusiastic AI travel guide for Cebu, Philippines.
Your tagline is: "Your AI guide to the best of Cebu."

You answer questions about Cebu tourism using ONLY the context provided to you.
If the context does not contain enough information, say so honestly and suggest the user ask something else about Cebu.

Guidelines:
- Be warm, friendly, and conversational — like a local Cebuano friend giving advice
- Occasionally use Bisaya words naturally (e.g., "Dali, let's explore!", "Maayong biyahe!")
- Keep answers helpful and concise but complete
- Use emojis sparingly to make responses feel lively
- Always end with a follow-up suggestion or question to keep the conversation going
"""
    context = "\n\n".join([
        f"[Source {i+1}: {doc['title']}]\n{doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    user_message = f"""Based on the following information about Cebu:

{context}

Answer this question: {query}"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>🌴</div>
        <div style='font-family: Playfair Display, serif; font-size:1.6rem; font-weight:900; letter-spacing:-0.5px;'>Gala.AI</div>
        <div style='font-size:0.78rem; opacity:0.7; margin-top:0.2rem;'>Your AI guide to the best of Cebu</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔑 API Key")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.markdown("### 📂 What I Know")
    st.markdown("""
    - 📍 Tourist Spots & Landmarks
    - 🏛️ History & Culture  
    - 🎉 Festivals & Events
    - 💡 Tips & Travel Advice
    - 🌟 Popular + Hidden Gems
    """)

    st.markdown("---")
    st.markdown("### 💡 Try asking...")
    st.markdown("""
    *"What are hidden gems in Cebu?"*  
    *"Tell me about Sinulog Festival"*  
    *"I'm on a budget, any tips?"*  
    *"History of Lapu-Lapu"*  
    *"Best beaches in Cebu"*  
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style='text-align:center; font-size:0.72rem; opacity:0.5; margin-top:1rem;'>
        Powered by Groq + LLaMA 3<br>Built with ❤️ for Cebu
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Gala<span>.AI</span></div>
    <div class="hero-tagline">Your AI guide to the best of Cebu, Philippines 🇵🇭</div>
    <div class="hero-informal">Wagtang kalaay, decide your destination through Gala.AI!</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INIT SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────
# LOAD RAG
# ─────────────────────────────────────────
try:
    kb, embedder, faiss_index = load_rag_system()
    rag_ready = True
except Exception as e:
    st.error(f"⚠️ Could not load cebu_tourism.txt — make sure it's in the same folder as app.py. Error: {e}")
    rag_ready = False

# ─────────────────────────────────────────
# SUGGESTED QUESTIONS (only when chat is empty)
# ─────────────────────────────────────────
if not st.session_state.messages and rag_ready:
    st.markdown('<div class="suggest-label">✨ Suggested questions</div>', unsafe_allow_html=True)
    suggestions = [
        "🐋 Whale shark watching in Oslob",
        "🎊 Tell me about Sinulog Festival",
        "💰 Budget travel tips for Cebu",
        "🏝️ Hidden gems in Cebu",
        "⚔️ History of Lapu-Lapu",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, key=f"suggest_{suggestion}"):
                st.session_state.pending_query = suggestion.split(" ", 1)[1]
                st.rerun()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = "".join([
                f'<span class="sources-pill">📍 {s}</span>'
                for s in msg.get("sources", [])
            ])
            st.markdown(f"""
            <div class="msg-bot">
                <div class="bot-avatar">🌴</div>
                <div class="bubble">
                    {msg["content"]}
                    <div style="margin-top:0.5rem;">{sources_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    if rag_ready:
        st.markdown("""
        <div class="empty-state">
            <div class="big-emoji">🌴</div>
            <p>Ask me anything about Cebu!<br>Beaches, history, festivals, food, hidden gems — I've got you.</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask anything about Cebu... (e.g. 'Where should I go for 2 days?')")

# Handle suggested question clicks
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

# ─────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────
if user_input and rag_ready:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Retrieve + Generate
        with st.spinner("🌴 Gala.AI is thinking..."):
            try:
                client = Groq(api_key=api_key)
                retrieved = retrieve(user_input, kb, embedder, faiss_index, top_k=3)
                response = generate_response(user_input, retrieved, client)
                source_titles = [doc["title"] for doc in retrieved]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": source_titles
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")

        st.rerun()