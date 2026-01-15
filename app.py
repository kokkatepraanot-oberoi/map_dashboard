# app.py
import streamlit as st
import pandas as pd

from data_loader import load_map_excel
from flags import apply_flags
from ui_theme import inject_map_css, percentile_band

# ---------------- Page config + MAP styling ----------------
st.set_page_config(page_title="MAP Dashboard", layout="wide")
inject_map_css()

st.title("MAP Dashboard")
st.caption(
    "MAP notes: Percentile indicates relative achievement vs norms; growth is evaluated using RIT change "
    "(Observed Growth) once Spring data is available."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Data Uploads")
fall_file = st.sidebar.file_uploader("Upload Fall (Sep 2025) MAP Excel", type=["xlsx"])
spring_file = st.sidebar.file_uploader("Upload Spring (Mar 2026) MAP Excel (optional)", type=["xlsx"])

st.sidebar.header("View")
view_mode = st.sidebar.radio("Select View", ["Teacher View", "Admin View"], index=0)

st.sidebar.header("Intervention Priority Rules (Achievement)")
pctl_risk = st.sidebar.slider("At Risk threshold (percentile)", 5, 50, 25, 1)
pctl_high = st.sidebar.slider("High Risk threshold (percentile)", 1, 25, 10, 1)

# MAP-style legend pills
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <span class="map-pill map-blue">On Track</span><br>
    <span class="map-pill map-yellow">At Risk (below {pctl_risk}th)</span><br>
    <span class="map-pill map-red">High Risk (below {pctl_high}th)</span>
    """,
    unsafe_allow_html=True,
)

# ---------------- Data loading ----------------
@st.cache_data(show_spinner=False)
def load_all(fall_upload, spring_upload) -> pd.DataFrame:
    frames = []
    if fall_upload is not None:
        frames.append(load_map_excel(fall_upload.getvalue(), term_label="Sep 2025"))
    if spring_upload is not None:
        frames.append(load_map_excel(spring_upload.getvalue(), term_label="Mar 2026"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


data = load_all(fall_file, spring_file)

if data.empty:
    st.info("Upload at least the Fall (Sep 2025) MAP file to begin.")
    st.stop()

# Apply MAP-aligned intervention priority
data = apply_flags(data, pctl_risk=pctl_risk, pctl_high=pctl_high)

# Add percentile status (MAP-style interpretation)
data["percentile_status"] = data["percentile"].apply(
    lambda x: percentile_band(x, pctl_risk=pctl_risk, pctl_high=pctl_high)[0]
)

# ---------------- Common filter options ----------------
subjects = ["All"] + sorted(data["subject"].dropna().unique().tolist())
terms = ["All"] + sorted(data["term"].dropna().unique().tolist())

grade_vals = sorted([int(x) for x in data["grade"].dropna().unique().tolist()]) if "grade" in data.columns else []
grades = ["All"] + grade_vals

teacher_vals = sorted(data["teacher"].dropna().unique().tolist()) if "teacher" in data.columns else []
teachers = ["All"] + teacher_vals


def map_badge_row(pctl_risk_val: int, pctl_high_val: int):
    st.markdown(
        f"""
        <span class="map-pill map-blue">On Track</span>
        <span class="map-pill map-yellow">At Risk (below {pctl_risk_val}th percentile)</span>
        <span class="map-pill map-red">High Risk (below {pctl_high_val}th percentile)</span>
        """,
        unsafe_allow_html=True,
    )


def metric_row(df: pd.DataFrame):
    total_rows = len(df)
    priority_rows = int(df["intervention_priority"].sum()) if "intervention_priority" in df.columns else 0
    pct_priority = (priority_rows / total_rows * 100) if total_rows else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Records (rows)", f"{total_rows}")
    c2.metric("Intervention Priority (rows)", f"{priority_rows}")
    c3.metric("% Intervention Priority", f"{pct_priority:.1f}%")


# ----------------- NEW: Table styling (color-coded) -----------------
def _row_bg_color(row: pd.Series):
    """Return a list of CSS styles for the whole row based on percentile_status."""
    status = row.get("percentile_status", "")
    if status == "High Risk (Achievement)":
        bg = "rgba(192,57,43,0.12)"   # MAP red tint
    elif status == "At Risk (Achievement)":
        bg = "rgba(242,201,76,0.20)"  # MAP yellow tint
    elif status == "On Track (Achievement)":
        bg = "rgba(47,111,178,0.12)"  # MAP blue tint
    else:
        bg = "transparent"
    return [f"background-color: {bg}"] * len(row)


def style_map_table(df: pd.DataFrame, highlight_status_col: bool = True):
    """
    Return a pandas Styler with MAP-style row colour coding.
    Note: works with st.dataframe(styler)
    """
    styler = (
        df.style
        .apply(_row_bg_color, axis=1)
        .format(
            {
                "rit": "{:.0f}",
                "percentile": "{:.0f}",
            },
            na_rep=""
        )
    )

    if highlight_status_col and "percentile_status" in df.columns:
        # slightly stronger emphasis on the status cell itself
        def _status_cell(val):
            if val == "High Risk (Achievement)":
                return "font-weight: 700; color: #C0392B;"
            if val == "At Risk (Achievement)":
                return "font-weight: 700; color: #8a6a00;"
            if val == "On Track (Achievement)":
                return "font-weight: 700; color: #2F6FB2;"
            return ""
        styler = styler.map(_status_cell, subset=["percentile_status"])

    # Cleaner header
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "white"), ("color", "#111827")]},
            {"selector": "td", "props": [("border-color", "rgba(0,0,0,0.06)")]},
        ]
    )

    return styler


def show_styled_table(df: pd.DataFrame, use_container_width: bool = True):
    """Helper to render styled tables safely."""
    if df.empty:
        st.info("No records match the current filters.")
        return
    st.dataframe(style_map_table(df), use_container_width=use_container_width)


# ---------------- Teacher View ----------------
if view_mode == "Teacher View":
    st.subheader("Teacher View")

    if not teacher_vals:
        st.warning("No teacher column detected in the data.")
        st.stop()

    teacher = st.selectbox("Select Teacher", teacher_vals)

    c1, c2, c3 = st.columns(3)
    term_sel = c1.selectbox("Term", terms, index=0)
    grade_sel = c2.selectbox("Grade", grades, index=0)
    subject_sel = c3.selectbox("Subject", subjects, index=0)

    df = data[data["teacher"] == teacher].copy()

    if term_sel != "All":
        df = df[df["term"] == term_sel]
    if grade_sel != "All":
        df = df[df["grade"] == grade_sel]
    if subject_sel != "All":
        df = df[df["subject"] == subject_sel]

    metric_row(df)
    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    st.markdown("### Intervention Priority Students")

    # Sort so High Risk appears first, then At Risk
    priority_sort_key = {"High Risk": 0, "At Risk": 1, "On Track": 2}
    df["_priority_sort"] = df["priority_level"].map(priority_sort_key).fillna(9)

    priority_df = df[df["intervention_priority"]].sort_values(
        ["_priority_sort", "subject", "percentile", "rit"],
        ascending=[True, True, True, True]
    )

    show_cols = [
        "student_id", "student_name", "grade", "section", "subject",
        "rit", "percentile", "percentile_status", "priority_reason"
    ]

    show_styled_table(priority_df[show_cols])

    st.download_button(
        "Download intervention list (CSV)",
        data=priority_df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"intervention_priority_{teacher.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### All Students (filtered)")
    show_styled_table(df[show_cols])

    df.drop(columns=["_priority_sort"], inplace=True, errors="ignore")


# ---------------- Admin View ----------------
else:
    st.subheader("Admin View")

    c1, c2, c3, c4 = st.columns(4)
    term_sel = c1.selectbox("Term", terms, index=0)
    grade_sel = c2.selectbox("Grade", grades, index=0)
    subject_sel = c3.selectbox("Subject", subjects, index=0)
    teacher_sel = c4.selectbox("Teacher", teachers, index=0)

    df = data.copy()
    if term_sel != "All":
        df = df[df["term"] == term_sel]
    if grade_sel != "All":
        df = df[df["grade"] == grade_sel]
    if subject_sel != "All":
        df = df[df["subject"] == subject_sel]
    if teacher_sel != "All":
        df = df[df["teacher"] == teacher_sel]

    metric_row(df)
    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    st.markdown("### Intervention Priority (Schoolwide)")

    priority_sort_key = {"High Risk": 0, "At Risk": 1, "On Track": 2}
    df["_priority_sort"] = df["priority_level"].map(priority_sort_key).fillna(9)

    priority_df = df[df["intervention_priority"]].sort_values(
        ["_priority_sort", "grade", "subject", "percentile", "rit"],
        ascending=[True, True, True, True, True],
    )

    show_cols_admin = [
        "teacher", "student_id", "student_name", "grade", "section", "subject",
        "rit", "percentile", "percentile_status", "priority_reason", "term"
    ]

    show_styled_table(priority_df[show_cols_admin])

    st.download_button(
        "Download intervention list (CSV)",
        data=priority_df[show_cols_admin].to_csv(index=False).encode("utf-8"),
        file_name="intervention_priority_schoolwide.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### Summary by Grade × Subject")

    summary = (
        df.groupby(["grade", "subject"], dropna=False)
          .agg(
              records=("student_id", "count"),
              intervention_priority=("intervention_priority", "sum"),
              avg_rit=("rit", "mean"),
              avg_percentile=("percentile", "mean"),
          )
          .reset_index()
    )
    summary["pct_intervention_priority"] = (
        (summary["intervention_priority"] / summary["records"] * 100).round(1)
    )

    summary = summary.sort_values(["grade", "subject"])

    # Summary also gets colour coding, but it doesn't have percentile_status.
    st.dataframe(summary, use_container_width=True)

    df.drop(columns=["_priority_sort"], inplace=True, errors="ignore")
