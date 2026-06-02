"""
Document Upload & Storage Page Module
Secure document vault (demo).
"""
import streamlit as st
from components import UIComponents


def render_document_vault():
    """Render the Document Upload & Storage page."""
    UIComponents.page_header(
        "Secure Document Vault (Demo)",
        "Upload and organize financial documents. (Demo — not persisted)",
        icon="🗂️"
    )
    st.warning("⚠️ This is a demo. In production, use encrypted storage and proper auth.")

    uploaded_files = st.file_uploader(
        "Upload financial documents (PDF, CSV, images)",
        type=["pdf", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s) (demo only — not persisted).")
        for f in uploaded_files:
            st.write(f"• {f.name} ({f.size} bytes)")
