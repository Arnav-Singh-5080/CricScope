import streamlit as st
import streamlit.components.v1 as components


def init_theme(default="dark"):
    if "theme" not in st.session_state:
        st.session_state.theme = default


def render_theme_bridge():
    theme = st.session_state.get("theme", "dark")
    st.markdown(
        f'<div id="__theme_value__" data-theme-value="{theme}" style="display:none">{theme}</div>',
        unsafe_allow_html=True,
    )
    components.html(
        f"""
        <script>
        (function() {{
            const theme = {theme!r};
            function applyTheme() {{
            const app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (!app) return false;
            app.setAttribute('data-theme', theme);
            window.parent.document.body.setAttribute('data-theme', theme);
            window.parent.document.documentElement.setAttribute('data-theme', theme);
            return true;
        }}

            let attempts = 0;
            const tryApply = () => {{
                if (applyTheme() || attempts++ >= 60) return;
                window.parent.requestAnimationFrame(tryApply);
            }};

            tryApply();
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def render_sidebar_theme_toggle(key="theme_toggle_page"):
    st.sidebar.markdown('<div class="sidebar-section-label">Display</div>', unsafe_allow_html=True)
    label = "Light Mode" if st.session_state.get("theme", "dark") == "dark" else "Dark Mode"
    st.sidebar.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
    if st.sidebar.button(label, key=key):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
