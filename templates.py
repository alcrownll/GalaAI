"""
templates.py — HTML rendering functions for Gala.AI.

All inline HTML/CSS that was previously scattered in app.py lives here.
app.py calls these functions and passes the result to st.markdown().
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
        gap: 0.55rem;
        padding: 1.1rem 1.2rem 1rem 1.2rem;
        border-bottom: 1px solid {t['border']};
        background: {t['sidebar_bg']};
        position: sticky;
        top: 0;
        z-index: 10;
    ">
        <div style="
            width: 36px; height: 36px;
            border-radius: 11px;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem;
            box-shadow: 0 4px 14px rgba(34,197,94,0.35);
            flex-shrink: 0;
        ">🌴</div>
        <div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-weight: 700;
                font-size: 1.05rem;
                color: {t['ink']};
                letter-spacing: -0.3px;
                line-height: 1.2;
            ">Gala<span style="color:{ACCENT};">.AI</span></div>
            <div style="
                font-size: 0.67rem;
                color: {t['muted']};
                font-weight: 500;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            ">Cebu Travel Guide</div>
        </div>
    </div>
    """


def sidebar_section_label(t: dict, label: str) -> str:
    return f"""
    <div style="padding: 0.75rem 1.2rem 0.3rem 1.2rem;">
        <div style="
            font-size: 0.7rem;
            font-weight: 700;
            color: {t['muted']};
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.5rem;
        ">{label}</div>
    </div>
    """


def sidebar_recents_label(t: dict) -> str:
    return f"""
    <div style="
        padding: 0.6rem 1.2rem 0.2rem 1.2rem;
        font-size: 0.7rem;
        font-weight: 700;
        color: {t['muted']};
        text-transform: uppercase;
        letter-spacing: 0.07em;
    ">Recent</div>
    """


def sidebar_compact_list_css() -> str:
    """Zero-gap CSS for the recent chat button list."""
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
    """Per-item scoped CSS for a single chat button."""
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
        font-size: 0.83rem !important;
        padding: 0.42rem 0.9rem 0.42rem 1.0rem !important;
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
    <div style="padding: 0.6rem 1.2rem; color: {t['muted2']}; font-size: 0.82rem; line-height: 1.5;">
        No previous chats yet.<br>
        <span style="color:{t['muted']};">Start a conversation below!</span>
    </div>
    """


def sidebar_footer(t: dict, is_dark: bool) -> str:
    next_theme   = "light" if is_dark else "dark"
    toggle_title = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
    if is_dark:
        toggle_icon = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>"""
    else:
        toggle_icon = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>"""

    return f"""
    <div id="gala-footer">
        <div class="top-row">
            <div class="status-group">
                <div class="dot"></div>
                <span class="status-text">Groq · RAG · Llama 3.3 70B</span>
            </div>
            <a id="gala-mode-btn"
               href="?theme={next_theme}"
               target="_top"
               title="{toggle_title}">
                {toggle_icon}
            </a>
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
        padding: 0.7rem 0.1rem 0.9rem 0.1rem;
        border-bottom: 1px solid {t['border']};
        margin-bottom: 0.4rem;
    ">
        <div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-weight: 800;
                font-size: 1.35rem;
                color: {t['ink']};
                letter-spacing: -0.5px;
                line-height: 1.2;
            ">Kumusta! 👋</div>
            <div style="color:{t['muted']}; font-size: 0.83rem; margin-top: 0.15rem;">
                Ask me anything about Cebu — beaches, food, festivals, history &amp; more.
            </div>
        </div>
        <div style="
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,0.25);
            background: rgba(34,197,94,0.12);
            font-size: 0.72rem;
            font-weight: 700;
            color: {ACCENT};
            letter-spacing: 0.04em;
            white-space: nowrap;
        ">● ONLINE</div>
    </div>
    """


def welcome_hero(t: dict) -> str:
    return f"""
    <div style="padding: 1.4rem 0 0.5rem 0; text-align: center;">
        <div style="font-size: 2.6rem; margin-bottom: 0.3rem;">🌴</div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            color: {t['ink']};
            letter-spacing: -0.5px;
        ">Your Cebu Travel Guide</div>
        <div style="
            color: {t['muted']};
            font-size: 0.88rem;
            margin-top: 0.4rem;
            max-width: 420px;
            margin-left: auto;
            margin-right: auto;
        ">Powered by local knowledge and AI. Friendly, fast, and a little bit Bisaya.</div>
        <div style="
            display: inline-block;
            margin-top: 1rem;
            padding: 0.3rem 1rem;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,0.28);
            background: rgba(34,197,94,0.12);
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
        font-size: 0.72rem;
        font-weight: 700;
        color: {t['muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.65rem 0;
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
    """Empty bot bubble shell used while streaming a response."""
    return """
    <div class="bot-row">
        <div class="bot-avi">🌴</div>
        <div class="bot-bubble" style="min-width:80px;">
    """


def bot_bubble_streaming_close() -> str:
    return "</div></div>"