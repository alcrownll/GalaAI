"""
app.py — Streamlit UI for Gala.AI.
Dark mode only. Chat history persisted via local JSON file.
"""

import uuid
import streamlit as st

from theme      import get_tokens, build_css, build_sidebar_toggle_js
from templates  import (
    sidebar_logo, sidebar_section_label, sidebar_recents_label,
    sidebar_compact_list_css, sidebar_chat_item_css,
    sidebar_empty_chats, sidebar_footer,
    main_header, welcome_hero, suggestions_label,
    user_bubble, bot_bubble, bot_bubble_streaming, bot_bubble_streaming_close,
)
from backend    import load_rag_system, retrieve, stream_groq_response
from persistence import load_chats, save_chats
import streamlit.components.v1 as components


# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gala.AI",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────
# THEME + CSS  (dark only)
# ─────────────────────────────────────────────────────

t = get_tokens(is_dark=True)
st.markdown(build_css(t), unsafe_allow_html=True)
components.html(build_sidebar_toggle_js(t), height=0, scrolling=False)


# ─────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────

defaults = {
    "messages":          [],
    "all_chats":         [],
    "active_chat_id":    None,
    "last_query":        None,
    "pending_regen":     False,
    "chats_loaded":      False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────
# LOAD CHAT HISTORY (once per session)
# ─────────────────────────────────────────────────────

if not st.session_state.chats_loaded:
    st.session_state.all_chats  = load_chats()
    st.session_state.chats_loaded = True


# ─────────────────────────────────────────────────────
# LOAD RAG (cached)
# ─────────────────────────────────────────────────────

try:
    kb, embedder, faiss_index = load_rag_system()
except Exception:
    st.error("⚠️  Could not load cebu_tourism.txt. Make sure it lives in the same folder as app.py.")
    st.stop()


# ─────────────────────────────────────────────────────
# CHAT HELPERS
# ─────────────────────────────────────────────────────

def _upsert_active_chat(messages: list, move_to_top: bool = False):
    """
    Insert or update the active chat in all_chats, then persist to disk.
    """
    if not messages:
        return

    raw   = messages[0]["content"]
    title = raw[:34] + ("…" if len(raw) > 34 else "")
    cid   = st.session_state.active_chat_id

    if cid:
        chats = st.session_state.all_chats
        idx   = next((i for i, c in enumerate(chats) if c["id"] == cid), None)
        if idx is not None:
            chats[idx]["messages"] = messages.copy()
            chats[idx]["title"]    = title
            if move_to_top and idx != 0:
                chats.insert(0, chats.pop(idx))
            save_chats(chats)
            return

    new_id = str(uuid.uuid4())
    st.session_state.all_chats.insert(0, {
        "id":       new_id,
        "title":    title,
        "messages": messages.copy(),
    })
    st.session_state.active_chat_id = new_id
    save_chats(st.session_state.all_chats)


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(sidebar_logo(t), unsafe_allow_html=True)
    st.markdown(sidebar_section_label(t, "Chats"), unsafe_allow_html=True)

    if st.button("＋  New conversation", use_container_width=True, key="new_chat"):
        _upsert_active_chat(st.session_state.messages)
        st.session_state.messages       = []
        st.session_state.last_query     = None
        st.session_state.pending_regen  = False
        st.session_state.active_chat_id = None
        st.rerun()

    if st.session_state.all_chats:
        st.markdown(sidebar_recents_label(t), unsafe_allow_html=True)
        st.markdown(sidebar_compact_list_css(), unsafe_allow_html=True)

        for chat in st.session_state.all_chats[:10]:
            cid       = chat["id"]
            is_active = st.session_state.active_chat_id == cid

            st.markdown(sidebar_chat_item_css(t, cid, is_active), unsafe_allow_html=True)

            if st.button(f"💬  {chat['title']}", key=f"chat_{cid}", use_container_width=True):
                st.session_state.messages       = chat["messages"].copy()
                st.session_state.active_chat_id = cid
                st.session_state.last_query     = None
                st.session_state.pending_regen  = False
                st.rerun()
    else:
        st.markdown(sidebar_empty_chats(t), unsafe_allow_html=True)

    st.markdown(sidebar_footer(t), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────

st.markdown(main_header(t), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# WELCOME / SUGGESTIONS  (empty state)
# ─────────────────────────────────────────────────────

SUGGESTIONS = [
    ("🐋", "Where can I see whale sharks in Cebu?"),
    ("🎊", "What is Sinulog Festival all about?"),
    ("🍖", "What food should I try in Cebu?"),
    ("🏝️", "Suggest hidden beaches in Cebu"),
    ("⚔️", "Who was Lapu-Lapu?"),
    ("💰", "Give me budget travel tips in Cebu"),
]

if not st.session_state.messages:
    st.markdown(welcome_hero(t), unsafe_allow_html=True)
    st.markdown(suggestions_label(t), unsafe_allow_html=True)

    st.markdown('<div class="suggestion-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="small")
    for i, (emoji, text) in enumerate(SUGGESTIONS):
        with (col1 if i % 2 == 0 else col2):
            if st.button(f"{emoji}  {text}", use_container_width=True, key=f"suggest_{i}"):
                st.session_state.messages.append({"role": "user", "content": f"{emoji}  {text}"})
                st.session_state.last_query = f"{emoji}  {text}"
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CHAT FEED
# ─────────────────────────────────────────────────────

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(user_bubble(msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(bot_bubble(msg["content"], msg.get("sources", [])), unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# REGENERATE BUTTON
# ─────────────────────────────────────────────────────

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    col_r, _ = st.columns([3, 1])
    with col_r:
        if st.button("🔄  Regenerate response", key="regen_btn"):
            st.session_state.pending_regen = True
            st.rerun()


# ─────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────

user_input = st.chat_input("Ask anything about Cebu, Philippines…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_query = user_input
    st.rerun()


# ─────────────────────────────────────────────────────
# GENERATE / STREAM RESPONSE
# ─────────────────────────────────────────────────────

last_is_user = (
    st.session_state.messages and
    st.session_state.messages[-1]["role"] == "user"
)
should_generate = st.session_state.last_query and (
    last_is_user or st.session_state.pending_regen
)

if should_generate:
    if st.session_state.pending_regen:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()

    query     = st.session_state.last_query
    retrieved = retrieve(query, kb, embedder, faiss_index)

    st.markdown(bot_bubble_streaming(), unsafe_allow_html=True)
    placeholder   = st.empty()
    full_response = stream_groq_response(query, retrieved, placeholder)
    st.markdown(bot_bubble_streaming_close(), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": [d["title"] for d in retrieved],
    })

    _upsert_active_chat(st.session_state.messages, move_to_top=True)
    st.session_state.pending_regen = False
    st.rerun()