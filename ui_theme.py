# ui_theme.py

# MAP-like palette (calm, instructional)
MAP_BLUE = "#2F6FB2"     # on/above track
MAP_YELLOW = "#F2C94C"   # monitor
MAP_RED = "#C0392B"      # risk
MAP_GREY = "#6B7280"
MAP_BG = "#F9FAFB"

def inject_map_css():
    import streamlit as st

    st.markdown(
        f"""
        <style>
          .stApp {{
            background: {MAP_BG};
          }}
          /* Titles slightly tighter, more report-like */
          h1, h2, h3 {{
            letter-spacing: -0.2px;
          }}
          /* MAP-style “pill” badges */
          .map-pill {{
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.35rem;
            border: 1px solid rgba(0,0,0,0.08);
          }}
          .map-blue {{ background: rgba(47,111,178,0.12); color: {MAP_BLUE}; }}
          .map-yellow {{ background: rgba(242,201,76,0.18); color: #8a6a00; }}
          .map-red {{ background: rgba(192,57,43,0.12); color: {MAP_RED}; }}

          /* Make dataframe header calmer */
          div[data-testid="stDataFrame"] thead tr th {{
            background: white !important;
          }}
        </style>
        """,
        unsafe_allow_html=True
    )

def percentile_band(pctl: float | None, pctl_risk=25, pctl_high=10):
    """MAP-aligned interpretation for percentile colours (risk/monitor/on-track)."""
    if pctl is None:
        return ("No Data", "map-yellow")
    try:
        p = float(pctl)
    except Exception:
        return ("No Data", "map-yellow")

    if p < pctl_high:
        return ("High Risk (Achievement)", "map-red")
    if p < pctl_risk:
        return ("At Risk (Achievement)", "map-yellow")
    return ("On Track (Achievement)", "map-blue")
