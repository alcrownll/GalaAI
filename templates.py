"""
templates.py — HTML rendering functions for Gala.AI.
Dark mode only. localStorage helpers removed (replaced by persistence.py).
"""

from theme import ACCENT, ACCENT_DARK


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────

def sidebar_logo(t: dict) -> str:
    return f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 1.2rem 0.75rem 1.2rem;
        border-bottom: 1px solid {t['border']};
        background: {t['sidebar_bg']};
        position: sticky;
        top: 0;
        z-index: 10;
    ">
        <div style="
            width: 38px; height: 38px;
            border-radius: 12px;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
            display: flex; align-items: center; justify-content: center;
            font-size: 1.15rem;
            box-shadow: 0 4px 18px rgba(34,197,94,0.40);
            flex-shrink: 0;
        ">🌴</div>
        <div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-weight: 700;
                font-size: 1.1rem;
                color: {t['ink']};
                letter-spacing: -0.4px;
                line-height: 1.15;
            ">Gala<span style="color:{ACCENT};">.AI</span></div>
            <div style="
                font-size: 0.65rem;
                color: {t['muted']};
                font-weight: 500;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            ">Cebu Travel Guide</div>
        </div>
    </div>
    """


def sidebar_section_label(t: dict, label: str) -> str:
    return f"""
    <div style="padding: 0.8rem 1.2rem 0.3rem 1.2rem;">
        <div style="
            font-size: 0.68rem;
            font-weight: 700;
            color: {t['muted']};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        ">{label}</div>
    </div>
    """


def sidebar_recents_label(t: dict) -> str:
    return f"""
    <div style="
        padding: 0.7rem 1.2rem 0.25rem 1.2rem;
        font-size: 0.68rem;
        font-weight: 700;
        color: {t['muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
    ">Recent</div>
    """


def sidebar_compact_list_css() -> str:
    return """
    <style>
    [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] > div:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    [data-testid="stSidebarContent"] .stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebarContent"] .stButton > button {
        margin: 0 !important;
    }
    </style>
    """


def sidebar_chat_item_css(t: dict, cid: str, is_active: bool) -> str:
    ink_col = t['ink'] if is_active else t['muted']
    bg_col  = t['chat_item_active'] if is_active else "transparent"
    bdr_col = ACCENT if is_active else "transparent"
    weight  = '600' if is_active else '400'
    return f"""
    <style>
    [data-testid="stSidebar"] div[data-testid="stButton"]:has(button[key="chat_{cid}"]) {{
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stButton"]:has(button[key="chat_{cid}"]) button {{
        background: {bg_col} !important;
        border-top: none !important;
        border-right: none !important;
        border-bottom: none !important;
        border-left: 2px solid {bdr_col} !important;
        border-radius: 0px !important;
        color: {ink_col} !important;
        font-weight: {weight} !important;
        font-size: 0.82rem !important;
        padding: 0.44rem 0.9rem 0.44rem 1.05rem !important;
        text-align: left !important;
        width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stButton"]:has(button[key="chat_{cid}"]) button:hover {{
        background: {t['chat_item_active']} !important;
        color: {ACCENT} !important;
        border-left-color: {ACCENT} !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    </style>
    """


def sidebar_empty_chats(t: dict) -> str:
    return f"""
    <div style="
        padding: 1.2rem 1.2rem;
        color: {t['muted2']};
        font-size: 0.82rem;
        line-height: 1.6;
        text-align: center;
    ">
        <div style="font-size: 1.6rem; margin-bottom: 0.4rem; opacity: 0.4;">💬</div>
        No previous chats yet.<br>
        <span style="color:{t['muted']};">Start a conversation below!</span>
    </div>
    """


def sidebar_footer(t: dict) -> str:
    return f"""
    <div id="gala-footer">
        <div class="top-row">
            <div class="status-group">
                <div class="dot"></div>
                <span class="status-text">Groq · RAG · Llama 3.3 70B</span>
            </div>
        </div>
        <div class="kb-text">📍 Cebu Knowledge Base v1.0</div>
    </div>
    """


# ─────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────

def main_header(t: dict) -> str:
    return f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 0.1rem 1rem 0.1rem;
        border-bottom: 1px solid {t['border']};
        margin-bottom: 0.5rem;
    ">
        <div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-weight: 800;
                font-size: 1.4rem;
                color: {t['ink']};
                letter-spacing: -0.6px;
                line-height: 1.15;
            ">Kumusta! 👋</div>
            <div style="color:{t['muted']}; font-size: 0.84rem; margin-top: 0.2rem; line-height: 1.5;">
                Ask me anything about Cebu — beaches, food, festivals, history &amp; more.
            </div>
        </div>
        <div style="
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.32rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,0.22);
            background: rgba(34,197,94,0.10);
            font-size: 0.7rem;
            font-weight: 700;
            color: {ACCENT};
            letter-spacing: 0.04em;
            white-space: nowrap;
        ">
            <span style="
                display:inline-block;
                width:6px;height:6px;
                border-radius:50%;
                background:{ACCENT};
                box-shadow:0 0 6px {ACCENT};
                animation: pulse 2s infinite;
            "></span>
            ONLINE
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.4; }}
    }}
    </style>
    """


def welcome_hero(t: dict) -> str:
    return f"""
    <div style="padding: 2rem 0 0.8rem 0; text-align: center;">
        <div style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 72px; height: 72px;
            border-radius: 22px;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
            font-size: 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(34,197,94,0.35);
        ">🌴</div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            color: {t['ink']};
            letter-spacing: -0.6px;
            line-height: 1.2;
        ">Your Cebu Travel Guide</div>
        <div style="
            color: {t['muted']};
            font-size: 0.9rem;
            margin-top: 0.55rem;
            max-width: 400px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        ">Powered by local knowledge and AI.<br>Friendly, fast, and a little bit Bisaya.</div>
        <div style="
            display: inline-block;
            margin-top: 1.1rem;
            padding: 0.35rem 1.1rem;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,0.25);
            background: rgba(34,197,94,0.10);
            color: {ACCENT};
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        ">Wagtang kalaay — let's explore Cebu! 🌊</div>
    </div>
    """


def suggestions_label(t: dict) -> str:
    return f"""
    <div style="
        font-size: 0.7rem;
        font-weight: 700;
        color: {t['muted']};
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin: 1.4rem 0 0.7rem 0;
    ">✨ Try asking</div>
    """


# ─────────────────────────────────────────────────────
# CHAT BUBBLES
# ─────────────────────────────────────────────────────

def user_bubble(content: str) -> str:
    return f"""
    <div class="user-row">
        <div class="user-bubble">{content}</div>
    </div>
    """


def bot_bubble(content: str, sources: list) -> str:
    sources_html = ""
    if sources:
        pills = "".join(f'<span class="source-pill">📍 {s}</span>' for s in sources)
        sources_html = f'<div class="sources">{pills}</div>'
    safe_content = content.replace("\n", "<br>")
    return f"""
    <div class="bot-row">
        <div class="bot-avi">🌴</div>
        <div class="bot-bubble">{safe_content}{sources_html}</div>
    </div>
    """


def bot_bubble_streaming() -> str:
    return """
    <div class="bot-row">
        <div class="bot-avi">🌴</div>
        <div class="bot-bubble" style="min-width:80px;">
    """


def bot_bubble_streaming_close() -> str:
    return "</div></div>"       