"""
theme.py — Design tokens and CSS loader for Gala.AI.
Dark mode only.
"""

import pathlib

ACCENT      = "#22c55e"
ACCENT_DARK = "#16a34a"
SIDEBAR_W   = "260px"

_CSS_PATH = pathlib.Path(__file__).parent / "static" / "style.css"


def get_tokens(is_dark: bool = True) -> dict:
    return dict(
        bg="#0a0d14",
        surface="#111827",
        surface2="#151d2e",
        surface3="#1a2538",
        border="rgba(255,255,255,0.06)",
        border2="rgba(255,255,255,0.10)",
        ink="#e8edf5",
        muted="rgba(232,237,245,0.50)",
        muted2="rgba(232,237,245,0.28)",
        bubble_user_grad=f"linear-gradient(135deg, {ACCENT}, {ACCENT_DARK})",
        bubble_bot_bg="rgba(255,255,255,0.03)",
        bubble_bot_bdr="rgba(255,255,255,0.07)",
        sidebar_bg="#080b11",
        sidebar_bdr="rgba(255,255,255,0.05)",
        input_bg="rgba(255,255,255,0.04)",
        pill_bg="rgba(34,197,94,0.12)",
        pill_bdr="rgba(34,197,94,0.28)",
        chat_item_active="rgba(34,197,94,0.08)",
        shadow="0 20px 60px rgba(0,0,0,0.70)",
        scrollbar_thumb="rgba(255,255,255,0.08)",
        footer_bg="rgba(8,11,17,0.98)",
    )


def build_css(tokens: dict) -> str:
    raw_css = _CSS_PATH.read_text(encoding="utf-8")
    for key, value in tokens.items():
        raw_css = raw_css.replace(f"[[{key}]]", value)
    return f"<style>\n{raw_css}\n</style>"


def build_sidebar_toggle_js(t: dict) -> str:
    return f"""
<script>
(function() {{
    const doc = window.parent.document;

    // ── Inject adaptive CSS once ──────────────────────────────
    if (!doc.getElementById('gala-adaptive-css')) {{
        const style = doc.createElement('style');
        style.id = 'gala-adaptive-css';
        style.textContent = `
            /* Shift main content right when sidebar is collapsed
               so the burger button doesn't overlap chat bubbles */
            body.gala-sidebar-closed [data-testid="stAppViewBlockContainer"],
            body.gala-sidebar-closed .block-container {{
                padding-left: 3.8rem !important;
                max-width: 100% !important;
            }}
            /* Smooth burger button transition */
            #gala-burger {{
                transition: background 0.18s ease, border-color 0.18s ease,
                            color 0.18s ease, transform 0.18s ease !important;
            }}
        `;
        doc.head.appendChild(style);
    }}

    // ── Icons: sidebar-panel style ────────────────────────────
    // "Collapse" = sidebar is open, click will close it  → arrow points left
    const ICON_COLLAPSE = `<svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <rect x="1" y="1" width="16" height="16" rx="3"
              stroke="currentColor" stroke-width="1.5"/>
        <line x1="6.5" y1="1.5" x2="6.5" y2="16.5"
              stroke="currentColor" stroke-width="1.5"/>
        <polyline points="10,6 8,9 10,12"
                  stroke="currentColor" stroke-width="1.5"
                  stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>`;

    // "Expand" = sidebar is closed, click will open it   → arrow points right
    const ICON_EXPAND = `<svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <rect x="1" y="1" width="16" height="16" rx="3"
              stroke="currentColor" stroke-width="1.5"/>
        <line x1="6.5" y1="1.5" x2="6.5" y2="16.5"
              stroke="currentColor" stroke-width="1.5"/>
        <polyline points="8,6 10,9 8,12"
                  stroke="currentColor" stroke-width="1.5"
                  stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>`;

    // ── Helpers ───────────────────────────────────────────────
    const getSidebar    = () => doc.querySelector('section[data-testid="stSidebar"]');
    const isSidebarOpen = () => {{
        const sb = getSidebar();
        return sb ? sb.getBoundingClientRect().width > 60 : false;
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

    // ── Lock sidebar width & hide resize handle ───────────────
    const lockSidebarWidth = () => {{
        const sb = getSidebar();
        if (!sb || !isSidebarOpen()) return;
        const inner = sb.querySelector(':scope > div');
        if (inner) {{
            inner.style.cssText += ';min-width:260px!important;max-width:260px!important;width:260px!important';
        }}
        const handle = sb.querySelector(
            '[data-testid="stSidebarResizeHandle"],.resize-handle,[class*="ResizeHandle"]'
        );
        if (handle) {{ handle.style.display = 'none'; handle.style.pointerEvents = 'none'; }}
        sb.style.cssText += ';min-width:260px!important;max-width:260px!important';
    }};

    // ── Master sync: icon + body class + width lock ───────────
    const syncState = () => {{
        const open = isSidebarOpen();
        const btn  = doc.getElementById('gala-burger');
        if (!btn) return;

        btn.innerHTML = open ? ICON_COLLAPSE : ICON_EXPAND;
        btn.title     = open ? 'Collapse sidebar' : 'Expand sidebar';

        doc.body.classList.toggle('gala-sidebar-open',   open);
        doc.body.classList.toggle('gala-sidebar-closed', !open);

        if (open) lockSidebarWidth();
    }};

    // ── Create button (only once across reruns) ───────────────
    if (doc.getElementById('gala-burger')) {{
        // Button already in DOM from a previous render — just re-sync icon
        syncState();
        return;
    }}

    const btn = doc.createElement('div');
    btn.id = 'gala-burger';
    btn.title = 'Collapse sidebar';

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
        boxShadow:      '0 2px 12px rgba(0,0,0,0.35)',
        userSelect:     'none',
        lineHeight:     '1',
    }});

    btn.onmouseenter = () => {{
        btn.style.background   = '#22c55e';
        btn.style.borderColor  = '#22c55e';
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

    btn.addEventListener('click', () => {{
        const native = findNativeToggle();
        if (native) native.click();
        // Sync icon after CSS transition settles (~350 ms)
        setTimeout(syncState, 380);
        setTimeout(syncState, 700);   // second pass in case transition is slow
    }});

    // ── Watch sidebar for width changes (most reliable) ───────
    const waitForSidebar = setInterval(() => {{
        const sb = getSidebar();
        if (!sb) return;
        clearInterval(waitForSidebar);
        syncState();
        new ResizeObserver(syncState).observe(sb);
    }}, 150);

    // ── Periodic heartbeat: re-sync after Streamlit reruns ────
    // Reruns can swap the native toggle DOM node, so we keep polling.
    setInterval(syncState,     800);
    setInterval(lockSidebarWidth, 600);

}})();
</script>
"""