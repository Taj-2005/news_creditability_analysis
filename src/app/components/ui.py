"""
Reusable UI components: metric cards, section cards, page header.
Pure Streamlit + optional HTML for premium layout.
"""

import streamlit as st


def page_header(title: str, subtitle: str = ""):
    """Render a consistent page title and optional subtitle."""
    if subtitle:
        st.markdown(
            f"""
            <div style="margin-bottom: 2rem;">
                <h1 style="margin: 0; font-size: 1.75rem; font-weight: 600; color: #111827;">
                    {title}
                </h1>
                <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 1rem;">
                    {subtitle}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {title}")


def metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """
    Render a single metric in a styled block. Use within st.columns().
    For big KPI cards, use value like "0.93" and label "ROC-AUC".
    """
    with st.container():
        if help_text:
            st.metric(label=label, value=value, delta=delta, help=help_text)
        else:
            st.metric(label=label, value=value, delta=delta)


def section_card(title: str, content_md: str = None):
    """Wrap a section with a subtle card feel via spacing and optional title."""
    st.markdown(f"### {title}")
    if content_md:
        st.markdown(content_md)


def kpi_row(metrics: list, num_columns: int = 4):
    """
    metrics: list of dicts with keys label, value, (optional) delta, help
    Renders a row of st.metric with equal columns.
    """
    n = len(metrics)
    cols = st.columns(min(n, num_columns))
    for i, m in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(
                label=m.get("label", ""),
                value=m.get("value", ""),
                delta=m.get("delta"),
                help=m.get("help"),
            )
