"""
theme.py — Design tokens and CSS loader for Gala.AI.

Reads static/style.css, substitutes CSS variable values from the
active theme tokens, and returns a <style> block ready for
st.markdown(..., unsafe_allow_html=True).

Public API
----------
get_tokens(is_dark)             →  dict of CSS values
build_css(tokens)               →  <style>…</style> string
build_sidebar_toggle_js(tokens) →  <script>…</script> string
"""

import pathlib

ACCENT      = "#22c55e"
ACCENT_DARK = "#16a34a"
SIDEBAR_W   = "260px"

# Path to the external CSS file
_CSS_PATH = pathlib.Path(__file__).parent / "static" / "style.css"


def get_tokens(is_dark: bool) -> dict:
    if is_dark:
        return dict(
            bg="#0c111d",
            surface="#111827",
            surface2="#1a2235",
            surface3="#1e2a40",
            border="rgba(255,255,255,0.07)",
            border2="rgba(255,255,255,0.12)",
            ink="#f1f5f9",
            muted="rgba(255,255,255,0.55)",
            muted2="rgba(255,255,255,0.35)",
            bubble_user_grad=f"linear-gradient(135deg, {ACCENT}, {ACCENT_DARK})",
            bubble_bot_bg="rgba(255,255,255,0.03)",
            bubble_bot_bdr="rgba(255,255,255,0.07)",
            sidebar_bg="#0a0e17",
            sidebar_bdr="rgba(255,255,255,0.06)",
            input_bg="rgba(255,255,255,0.04)",
            pill_bg="rgba(34,197,94,0.12)",
            pill_bdr="rgba(34,197,94,0.28)",
            chat_item_active="rgba(34,197,94,0.10)",
            shadow="0 16px 48px rgba(0,0,0,0.55)",
            scrollbar_thumb="rgba(255,255,255,0.10)",
            footer_bg="rgba(10,14,23,0.98)",
        )
    return dict(
        bg="#f0f4f8",
        surface="#ffffff",
        surface2="#e8f0e9",
        surface3="#d4e8d8",
        border="rgba(0,0,0,0.08)",
        border2="rgba(0,0,0,0.14)",
        ink="#0f172a",
        muted="rgba(0,0,0,0.55)",
        muted2="rgba(0,0,0,0.38)",
        bubble_user_grad=f"linear-gradient(135deg, {ACCENT}, {ACCENT_DARK})",
        bubble_bot_bg="#ffffff",
        bubble_bot_bdr="rgba(0,0,0,0.08)",
        sidebar_bg="#ffffff",
        sidebar_bdr="rgba(0,0,0,0.08)",
        input_bg="#ffffff",
        pill_bg="rgba(34,197,94,0.12)",
        pill_bdr="rgba(34,197,94,0.30)",
        chat_item_active="rgba(34,197,94,0.10)",
        shadow="0 4px 20px rgba(0,0,0,0.08)",
        scrollbar_thumb="rgba(0,0,0,0.12)",
        footer_bg="#f8fafb",
    )


def build_css(tokens: dict) -> str:
    """
    Read static/style.css, substitute all [[key]] placeholders with
    the matching token value, and wrap in a <style> tag.

    Uses [[key]] markers instead of {key} so Python's str.format()
    doesn't clash with the curly braces in regular CSS rules.
    """
    raw_css = _CSS_PATH.read_text(encoding="utf-8")
    for key, value in tokens.items():
        raw_css = raw_css.replace(f"[[{key}]]", value)
    return f"<style>\n{raw_css}\n</style>"


def build_sidebar_toggle_js(t: dict) -> str:
    """
    Injects a floating hamburger/X button and locks sidebar width.
    """
    return f"""
<script>
(function() {{
    const doc = window.parent.document;

    function lockSidebarWidth() {{
        const sb = doc.querySelector('section[data-testid="stSidebar"]');
        if (!sb) return;
        const inner = sb.querySelector(':scope > div');
        if (inner) {{
            inner.style.minWidth = '260px';
            inner.style.maxWidth = '260px';
            inner.style.width    = '260px';
        }}
        const handle = sb.querySelector('[data-testid="stSidebarResizeHandle"], .resize-handle, [class*="ResizeHandle"]');
        if (handle) {{
            handle.style.display = 'none';
            handle.style.pointerEvents = 'none';
        }}
        sb.style.minWidth = '260px';
        sb.style.maxWidth = '260px';
    }}

    setInterval(lockSidebarWidth, 500);

    if (doc.getElementById('gala-burger')) return;

    const btn = doc.createElement('div');
    btn.id = 'gala-burger';
    btn.title = 'Toggle sidebar';
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="1" y="3" width="16" height="2" rx="1" fill="currentColor"/>
      <rect x="1" y="8" width="16" height="2" rx="1" fill="currentColor"/>
      <rect x="1" y="13" width="16" height="2" rx="1" fill="currentColor"/>
    </svg>`;

    Object.assign(btn.style, {{
        position:       'fixed',
        top:            '12px',
        left:           '12px',
        zIndex:         '999999',
        width:          '36px',
        height:         '36px',
        borderRadius:   '10px',
        background:     '{t['surface']}',
        border:         '1px solid {t['border2']}',
        color:          '{t['muted']}',
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'center',
        cursor:         'pointer',
        boxShadow:      '0 2px 12px rgba(0,0,0,0.18)',
        transition:     'all 0.18s ease',
        userSelect:     'none',
    }});

    btn.onmouseenter = () => {{
        btn.style.background   = '{ACCENT}';
        btn.style.borderColor  = '{ACCENT}';
        btn.style.color        = 'white';
        btn.style.transform    = 'scale(1.06)';
    }};
    btn.onmouseleave = () => {{
        btn.style.background   = '{t['surface']}';
        btn.style.borderColor  = '{t['border2']}';
        btn.style.color        = '{t['muted']}';
        btn.style.transform    = 'scale(1)';
    }};

    doc.body.appendChild(btn);

    const getSidebar = () => doc.querySelector('section[data-testid="stSidebar"]');

    const isSidebarOpen = () => {{
        const sb = getSidebar();
        if (!sb) return false;
        return sb.getBoundingClientRect().width > 60;
    }};

    const findNativeToggle = () => {{
        const sels = [
            '[data-testid="stSidebarCollapseButton"] button',
            '[data-testid="stSidebarCollapseButton"]',
            'button[aria-label="Close sidebar"]',
            'button[aria-label="collapse sidebar"]',
            '[data-testid="collapsedControl"] button',
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarCollapsedControl"]',
            'button[aria-label="Open sidebar"]',
            'button[aria-label="expand sidebar"]',
        ];
        for (const sel of sels) {{
            const el = doc.querySelector(sel);
            if (el) return el;
        }}
        return null;
    }};

    const ICONS = {{
        open:  `<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="1" y="3" width="16" height="2" rx="1" fill="currentColor"/><rect x="1" y="8" width="16" height="2" rx="1" fill="currentColor"/><rect x="1" y="13" width="16" height="2" rx="1" fill="currentColor"/></svg>`,
        close: `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1 1L13 13M13 1L1 13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
    }};

    const syncIcon = () => {{
        btn.innerHTML = isSidebarOpen() ? ICONS.close : ICONS.open;
    }};

    btn.addEventListener('click', () => {{
        const native = findNativeToggle();
        if (native) native.click();
        setTimeout(syncIcon, 350);
    }});

    const poll = setInterval(() => {{
        if (getSidebar()) {{ syncIcon(); clearInterval(poll); }}
    }}, 150);

    const waitObs = setInterval(() => {{
        const sb = getSidebar();
        if (sb) {{
            new ResizeObserver(syncIcon).observe(sb);
            clearInterval(waitObs);
        }}
    }}, 200);
}})();
</script>
"""