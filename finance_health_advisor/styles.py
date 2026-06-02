"""
Centralized Styles Module
Manages all CSS, themes, and visual styling for the Finance Health Advisor.
"""
import streamlit as st


class ThemeManager:
    """Manages theme toggling and CSS injection."""

    DARK_CSS = """
    <style>
    .stApp { background-color: #0f172a !important; }
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    .card-title, h1, h2, h3, [data-testid="stMetricValue"] { color: #f8fafc !important; }
    .sidebar-text, [data-testid="stMetricLabel"], .stMarkdown p { color: #cbd5e1 !important; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
    .stTabs [aria-selected="true"] { color: #3b82f6 !important; }
    hr { border-color: #334155 !important; }
    </style>
    """

    BASE_CSS = """
    <style>
    /* Animated gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        animation: gradientShift 8s ease infinite;
        background-size: 200% 200%;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Card hover effects */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: all 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    /* Button styling */
    .stButton button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #667eea;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
    }

    /* Progress bars */
    .stProgress > div > div {
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Highlight utility */
    .highlight {
        color: #2563eb;
        font-weight: 600;
        background-color: rgba(37, 99, 235, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
    """

    @classmethod
    def apply_theme(cls, dark_mode: bool) -> None:
        """Inject theme CSS based on dark/light mode."""
        st.markdown(cls.BASE_CSS, unsafe_allow_html=True)
        if dark_mode:
            st.markdown(cls.DARK_CSS, unsafe_allow_html=True)
